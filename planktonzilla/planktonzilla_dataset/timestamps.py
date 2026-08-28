"""
(c) Inria

Recover the capture time of a sample from its file name.

Plankton imagers name their output after the moment the frame was taken — it is the
one piece of provenance that survives every re-packaging of a dataset, because it is
carried by the file name rather than by a sidecar the archive may not ship. Six
conventions are recognised (see :data:`PATTERNS`) — five verified against real data,
one reserved for modern IFCB exports — and together they cover five of the seventeen
registry sources at a precision the pipeline previously discarded.

Why this exists at all: before it, ``timestamp`` was populated only by
``EcoTaxaRedefiner`` and ``WHOIRedefiner``, each through a live HTTP call made while
the dataset is being built, and each truncating the result to ``YYYY-MM-DD``. That has
three costs this module removes:

  - the nine sources wired to ``redefiner: none`` got a null ``timestamp`` even when
    the capture time was sitting in every one of their file names (zoolake: 17942 of
    17942 rows);
  - WHOI paid an API round-trip per bin to learn a date its own file names already
    state to the second (``IFCB1_2008_073_152345``);
  - a failed fetch is indistinguishable from "this sample has no recorded time",
    because both end as null.

Reading the file name is offline, deterministic and costs no request, so a source that
matches a known convention keeps its capture time even when the upstream API is down.

INVARIANT: this module never *replaces* a recorded timestamp with a different date.
:func:`merge_timestamp` upgrades a value only when the path agrees with it to the day;
a disagreement keeps the recorded value and is counted as a conflict. A regex that
starts matching the wrong digits therefore shows up as a conflict count in the build
log rather than as silently rewritten provenance.

Adding a source: nothing here is keyed by dataset name, so a new entry in
``cfg.datasets`` whose imager uses one of these conventions gets its timestamps with no
wiring at all. A genuinely new convention is one :class:`TimestampPattern` appended to
:data:`PATTERNS`.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# Plausibility window for a parsed capture time. The lower bound predates every
# instrument represented here (the oldest real sample is WHOI's 2006 IFCB run); the
# upper bound allows for a clock slightly ahead of ours rather than assuming the build
# host is authoritative. Anything outside is a mis-parse, not a measurement: it is what
# an object ID or a serial number looks like when read as a date.
EARLIEST_PLAUSIBLE = datetime(1990, 1, 1, tzinfo=UTC)
FUTURE_TOLERANCE = timedelta(days=1)


@dataclass(frozen=True)
class TimestampPattern:
    """One file-naming convention that encodes a capture time.

    ``regex`` is matched with ``search`` against the whole ``original_path``, so each
    pattern must anchor itself to something structural (an instrument prefix, a fixed
    separator). None of them may be a bare run of digits: EcoTaxa-derived sources name
    files after an object ID (``257738613.png``), and a pattern loose enough to read
    that as an epoch would fabricate a timestamp for three sources at once.
    """

    name: str
    regex: re.Pattern
    build: Callable[[re.Match], datetime]
    sources: str


def _from_epoch_microseconds(match: re.Match) -> datetime:
    """SPC/DSPC: the frame's epoch time in microseconds."""
    return datetime.fromtimestamp(int(match.group(1)) / 1_000_000, UTC)


def _from_day_of_year(match: re.Match) -> datetime:
    """IFCB legacy: year + zero-padded day-of-year + HHMMSS."""
    year, doy, hour, minute, second = (int(g) for g in match.groups())
    # strptime would accept day 000; the explicit offset makes an out-of-range day of
    # year raise here rather than silently landing in the previous December.
    if not 1 <= doy <= 366:
        raise ValueError(f"day-of-year {doy} out of range")
    start = datetime(year, 1, 1, hour, minute, second, tzinfo=UTC)
    return start + timedelta(days=doy - 1)


def _from_compact(match: re.Match) -> datetime:
    """ISIIS: YYYYMMDDHHMMSS with milliseconds in the next dotted field."""
    stamp, millis = match.groups()
    parsed = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return parsed.replace(microsecond=int(millis) * 1000)


def _from_underscored(match: re.Match) -> datetime:
    """CPICS: YYYYMMDD_HHMMSS with milliseconds in the next dotted field."""
    date_part, time_part, millis = match.groups()
    parsed = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return parsed.replace(microsecond=int(millis) * 1000)


def _from_dashed(match: re.Match) -> datetime:
    """PlanktoScope: YYYY-MM-DD_HH-MM-SS-ffffff."""
    year, month, day, hour, minute, second, micro = (int(g) for g in match.groups())
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=UTC)


def _from_iso_tag(match: re.Match) -> datetime:
    """IFCB modern / generic: DYYYYMMDDTHHMMSS."""
    date_part, time_part = match.groups()
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


# Order is significance, not preference: the first pattern that matches wins, so the
# most structurally specific conventions come first. They are mutually exclusive on all
# data verified so far, and :func:`audit_paths` is how that stays true for a new source.
PATTERNS = (
    TimestampPattern(
        name="spc-epoch-us",
        # SPC-<site>-<optics>-<epoch microseconds>-<...>. Verified on zoolake, whose
        # Eawag DSPC writes SPC-EAWAG-0P5X-1589537024858496-...; the SPC- prefix is what
        # keeps the 16-digit group from matching anything else.
        regex=re.compile(r"(?:^|/)SPC-[^/]*?(\d{16})(?:\D|$)"),
        build=_from_epoch_microseconds,
        sources="zoolake",
    ),
    TimestampPattern(
        name="ifcb-day-of-year",
        # IFCB<n>_<YYYY>_<DDD>_<HHMMSS>_<roi>. Verified on whoi (IFCB1_2008_073_152345).
        regex=re.compile(r"IFCB\d+_(\d{4})_(\d{3})_(\d{2})(\d{2})(\d{2})"),
        build=_from_day_of_year,
        sources="whoi",
    ),
    TimestampPattern(
        name="ifcb-iso-tag",
        # D<YYYYMMDD>T<HHMMSS>_IFCB<n>, the convention that replaced the day-of-year one
        # upstream. No source in the registry uses it yet; it is here so an IFCB source
        # added later is covered without a code change.
        regex=re.compile(r"(?:^|/)D(\d{8})T(\d{6})_IFCB\d+"),
        build=_from_iso_tag,
        sources="(none yet — modern IFCB exports)",
    ),
    TimestampPattern(
        name="isiis-compact-ms",
        # <YYYYMMDDHHMMSS>.<mmm>.<frame>_... Verified on planktonset1.0
        # (20140602073348.076.0131_000_crop_463.jpg). The two dotted fields after the
        # stamp are what distinguish it from an arbitrary 14-digit run.
        regex=re.compile(r"(?:^|/)(\d{14})\.(\d{3})\.\d"),
        build=_from_compact,
        sources="planktonset1.0",
    ),
    TimestampPattern(
        name="cpics-compact-ms",
        # <YYYYMMDD>_<HHMMSS>.<mmm>.<n>. Verified on jedioceans
        # (20151103_102539.630.0.png), whose CPICS frames carry the only capture time
        # that source has — JediRedefiner assigns one fixed lat/lon/depth and no date.
        regex=re.compile(r"(?:^|/)(\d{8})_(\d{6})\.(\d{3})\.\d"),
        build=_from_underscored,
        sources="jedioceans",
    ),
    TimestampPattern(
        name="planktoscope-dashed-us",
        # <YYYY-MM-DD>_<HH-MM-SS>-<ffffff>_<n>. Verified on planktoscope
        # (2024-03-12_11-55-35-346253_1.jpg).
        regex=re.compile(r"(?:^|/)(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{6})"),
        build=_from_dashed,
        sources="planktoscope",
    ),
)


def extract_path_timestamp(path: str) -> tuple[str, str] | tuple[None, None]:
    """Read the capture time out of one ``original_path``.

    Returns:
        ``(iso_timestamp, pattern_name)``, or ``(None, None)`` when no pattern matches
        or the value it yields is not a plausible capture time. Never raises: a path is
        untrusted input, and one malformed name must not abort a build of millions of
        rows.
    """
    if not path:
        return None, None

    for pattern in PATTERNS:
        match = pattern.regex.search(path)
        if not match:
            continue
        try:
            when = pattern.build(match)
        except (ValueError, OverflowError, OSError):
            # A structurally matching name whose numbers are not a date (day-of-year
            # 999, month 13, an epoch beyond the platform's range). Try the remaining
            # patterns rather than giving up on the path.
            continue
        if not EARLIEST_PLAUSIBLE <= when <= datetime.now(UTC) + FUTURE_TOLERANCE:
            continue
        return when.isoformat(), pattern.name

    return None, None


def merge_timestamp(recorded: str | None, derived: str | None) -> tuple[str | None, str]:
    """Combine an API-recorded timestamp with the one read from the path.

    The two are not rivals: for WHOI they are the same bin, one reported to the day and
    one to the second, so the path value is the recorded value at higher resolution.
    Preferring it is an upgrade, not an override — provided they agree on the day.

    Returns:
        ``(value, outcome)`` where outcome is one of ``kept`` (nothing to add),
        ``filled`` (there was no recorded value), ``refined`` (same day, better
        precision) or ``conflict`` (different days — the recorded value wins).
    """
    if derived is None:
        return recorded, "kept"
    if recorded in (None, ""):
        return derived, "filled"
    if recorded[:10] == derived[:10]:
        return derived, "refined"
    return recorded, "conflict"


def audit_paths(paths, *, sample_name: str = "") -> dict:
    """Report which pattern would claim each of ``paths``, without building anything.

    The tool for wiring up a new source: it answers "does this dataset's naming
    convention parse, and does exactly one pattern claim it?" before a multi-hour
    import depends on the answer.
    """
    counts: dict[str, int] = {}
    unmatched = []
    earliest = latest = None

    for path in paths:
        stamp, pattern = extract_path_timestamp(path)
        key = pattern or "(no match)"
        counts[key] = counts.get(key, 0) + 1
        if stamp is None:
            if len(unmatched) < 5:
                unmatched.append(path)
            continue
        earliest = stamp if earliest is None or stamp < earliest else earliest
        latest = stamp if latest is None or stamp > latest else latest

    total = sum(counts.values())
    matched = total - counts.get("(no match)", 0)
    return {
        "name": sample_name,
        "total": total,
        "matched": matched,
        "coverage": matched / total if total else 0.0,
        "patterns": counts,
        "earliest": earliest,
        "latest": latest,
        "unmatched_examples": unmatched,
    }
