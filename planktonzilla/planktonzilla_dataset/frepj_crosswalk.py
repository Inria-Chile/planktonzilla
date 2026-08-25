"""
(c) Inria

Build and load the committed FREPJ romanized-site -> lat/lon crosswalk.

The per-image ``Sampling site`` cells in Table_S3/S4 are romanized Japanese tokens
(``biwako``, ``akanko``, ``akigawadam``); the coordinates live in Table_S1 under
~85 English site names. Only ~2 tokens match a Table_S1 name directly. This module
resolves the rest with a strict, auditable precedence and writes a tiny static
crosswalk the ``FrepjRedefiner`` (Plan 17-02) joins against offline:

    trivial-normalize exact match
      -> difflib auto-fuzzy (high-confidence romanization variants only)
        -> hand-curated override (:data:`frepj_tables.DEFAULT_OVERRIDES_PATH`)
          -> null   (empty Latitude/Longitude — an ambiguous token is NEVER guessed)

Every crosswalk row records the winning ``method`` (``trivial|fuzzy|override|null``)
and the token's ``n_images`` (S3+S4 row count) so downstream coverage can be audited
from the committed file alone, with no access to the raw tables.

Only :func:`main` touches the network (via :func:`frepj_tables.ensure_frepj_tables`);
it is guarded by ``if __name__ == "__main__"`` and is never imported by the redefiner
or the tests. ``difflib`` is stdlib — no new third-party dependency is introduced.
"""

import difflib
import re
from collections import Counter
from pathlib import Path

import polars as pl

from planktonzilla.planktonzilla_dataset import frepj_tables
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


# Conservative auto-fuzzy cutoff: only high-confidence romanization variants resolve
# automatically; everything below falls through to hand-curated override or null, so
# the pass never manufactures a wrong coordinate (CONTEXT: ambiguous -> null).
FUZZY_CUTOFF = 0.85

# WR-01: `difflib.get_close_matches(..., n=1, ...)` only ever returns the single best
# match, even when a runner-up scores within noise of it (e.g. "sakurajo" scores
# identically against both "Sakurajo_1" and "Sakurajo_2"). A score tie alone is not
# proof of ambiguity -- near-identical real-world coordinates (a few tens of metres,
# the sakurajo case) should still resolve. Only treat a tie as ambiguous (-> fall
# through to override/null) when the tied candidates ALSO map to materially different
# real-world locations, reusing :func:`frepj_tables.haversine_km` -- the same distance
# helper as CR-01, but a DELIBERATELY SEPARATE, DECOUPLED threshold: this guards a tie
# between two DIFFERENT Table_S1 site names (never loosen it), whereas CR-01's
# ``_SITE_COORD_CONFLICT_KM`` (10 km) governs same-name duplicate-row centroiding. Do
# not fold these two constants back into one -- they answer different questions.
FUZZY_TIE_SCORE_EPSILON = 0.02
FUZZY_TIE_CONFLICT_KM = 1.0  # intentionally stays at the tight 1 km tie-rejection bar

CROSSWALK_COLUMNS = ["site_token", "resolved_site", "Latitude", "Longitude", "method", "n_images"]

_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def normalize_token(token: str) -> str:
    """Lowercase and drop all non-alphanumeric separators for cross-form comparison.

    ``"Akigawa Dam"`` and ``"akigawadam"`` both normalize to ``"akigawadam"``;
    ``"(Baiyou)"`` normalizes to ``"baiyou"``. Romanization variants that share no
    common alphanumeric skeleton (``"Lake Biwa"`` vs ``"biwako"``) intentionally do
    NOT collide — those are resolved by the fuzzy or override pass, not trivially.
    """
    return _SEPARATOR_RE.sub("", str(token).strip().lower())


def _to_float_or_none(raw: str | float | None) -> float | None:
    """Parse a CSV coordinate cell to float; blank / unparseable -> None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_fuzzy_match(
    norm: str,
    candidates: list[str],
    norm_to_site: dict[str, str],
    site_coords: dict[str, tuple[float | None, float | None]],
    fuzzy_cutoff: float,
) -> str | None:
    """Return the auto-fuzzy Table_S1 site match for ``norm``, or ``None`` if ambiguous.

    Requests the top-2 ``difflib`` candidates (not just the best) so a SCORE TIE between
    two distinct Table_S1 sites can be detected (WR-01) — ``get_close_matches(n=1, ...)``
    silently picks one via Python's internal tie-break, which is not a confidence signal.
    A tie is only treated as ambiguous (returns ``None``, falling through to
    override/null) when the tied candidates ALSO map to coordinates more than
    :data:`FUZZY_TIE_CONFLICT_KM` apart — near-identical real-world duplicates (e.g. the
    real ``sakurajo`` case, two site rows ~60 m apart) still resolve normally.
    """
    close = difflib.get_close_matches(norm, candidates, n=2, cutoff=fuzzy_cutoff)
    if not close:
        return None

    if len(close) > 1:
        top_ratio = difflib.SequenceMatcher(None, norm, close[0]).ratio()
        runner_up_ratio = difflib.SequenceMatcher(None, norm, close[1]).ratio()
        if (top_ratio - runner_up_ratio) < FUZZY_TIE_SCORE_EPSILON:
            top_coord = site_coords.get(norm_to_site[close[0]], (None, None))
            runner_coord = site_coords.get(norm_to_site[close[1]], (None, None))
            if None in top_coord or None in runner_coord:
                # Can't establish the two candidates agree -- never guess (WR-01).
                return None
            if frepj_tables.haversine_km(top_coord, runner_coord) > FUZZY_TIE_CONFLICT_KM:
                return None

    return norm_to_site[close[0]]


def build_crosswalk(
    site_coords: dict[str, tuple[float | None, float | None]],
    tokens,
    overrides: dict[str, str],
    fuzzy_cutoff: float = FUZZY_CUTOFF,
) -> list[dict]:
    """Resolve every sampling-site token to a Table_S1 coordinate (or null).

    Args:
        site_coords: ``{site_name: (lat, lon)}`` from :func:`frepj_tables.read_site_coordinates`.
        tokens: mapping ``{site_token: n_images}`` (a plain iterable is tolerated with
            ``n_images`` defaulting to 0).
        overrides: ``{site_token: resolved_site_name}`` hand-curated table; an override
            only fires when NO high-confidence trivial/fuzzy match exists.
        fuzzy_cutoff: conservative ``difflib`` ratio gate for the auto-fuzzy pass.

    Returns:
        One row dict per distinct token (sorted by ``site_token``) with keys
        ``site_token, resolved_site, Latitude, Longitude, method, n_images``. A token
        with no confident match and no override is emitted with ``method="null"`` and
        ``Latitude``/``Longitude`` set to ``None`` — never a guessed coordinate.

    Raises:
        ValueError: an override fires for a token whose target site name is not a key
            of ``site_coords`` (a typo'd/stale entry in ``frepj_site_overrides.csv``) --
            fails loud at build time rather than silently emitting a non-null method
            with blank coordinates (WR-02).
    """
    # Deterministic normalized-name -> Table_S1-site lookup (first site wins on a tie).
    norm_to_site: dict[str, str] = {}
    for site in sorted(site_coords):
        norm_to_site.setdefault(normalize_token(site), site)
    candidates = list(norm_to_site)

    counts = tokens if hasattr(tokens, "get") else {}

    rows: list[dict] = []
    for raw_token in sorted(tokens):
        norm = normalize_token(raw_token)
        resolved_site = ""
        method = "null"

        if norm in norm_to_site:
            resolved_site, method = norm_to_site[norm], "trivial"
        else:
            fuzzy_site = _resolve_fuzzy_match(norm, candidates, norm_to_site, site_coords, fuzzy_cutoff)
            if fuzzy_site:
                resolved_site, method = fuzzy_site, "fuzzy"
            elif raw_token in overrides:
                candidate = overrides[raw_token]
                if candidate not in site_coords:
                    # WR-02: a hand-curated override with a typo'd/stale target would
                    # otherwise silently emit method="override" with blank coords,
                    # indistinguishable from a correctly-resolved override except for
                    # the empty cells. Fail loud instead of letting it through quietly.
                    raise ValueError(
                        f"Override for «{raw_token}» names unknown Table_S1 site «{candidate}»; "
                        "fix planktonzilla_dataset/frepj_site_overrides.csv (WR-02)."
                    )
                resolved_site, method = candidate, "override"

        latitude, longitude = site_coords.get(resolved_site, (None, None)) if resolved_site else (None, None)
        if resolved_site and (latitude is None or longitude is None):
            # The token's name resolved (trivial/fuzzy/override), but Table_S1 itself
            # has no reliable coordinate for that site -- e.g. a same-name collision
            # nulled by :func:`frepj_tables.read_site_coordinates` (CR-01). Downgrade to
            # a full null row rather than emitting a non-null method with blank coords.
            resolved_site, method = "", "null"

        rows.append(
            {
                "site_token": raw_token,
                "resolved_site": resolved_site,
                "Latitude": latitude,
                "Longitude": longitude,
                "method": method,
                "n_images": int(counts.get(raw_token, 0)),
            }
        )
    return rows


def write_crosswalk(rows: list[dict], path: str | Path) -> None:
    """Write crosswalk rows to CSV (fixed column order, null coords -> empty cells)."""
    frame = pl.DataFrame(rows, schema_overrides={"Latitude": pl.Float64, "Longitude": pl.Float64})
    frame.select(CROSSWALK_COLUMNS).write_csv(str(path))


def load_crosswalk(path: str | Path) -> dict[str, tuple[float | None, float | None]]:
    """Load the committed crosswalk as ``{site_token: (lat, lon)}`` for the redefiner.

    Empty ``Latitude``/``Longitude`` cells (null rows) map to ``(None, None)``. Reads
    only the committed CSV — never the raw tables — so it is fully offline.
    """
    df = pl.read_csv(path, infer_schema_length=0)
    mapping: dict[str, tuple[float | None, float | None]] = {}
    for row in df.iter_rows(named=True):
        token = row["site_token"]
        if token is None:
            continue
        mapping[str(token)] = (_to_float_or_none(row.get("Latitude")), _to_float_or_none(row.get("Longitude")))
    return mapping


def load_crosswalk_sites(path: str | Path) -> dict[str, str]:
    """Load the committed crosswalk as ``{site_token: resolved_site}`` (resolved tokens only).

    The Table_S1 site name a token was matched to — the key into
    :func:`frepj_tables.read_site_sampling_dates` when a three-digit sampling day needs
    disambiguating. Tokens with no resolution (empty ``resolved_site``) are omitted, so a
    ``.get`` on the result is None for them. Offline like :func:`load_crosswalk`.
    """
    df = pl.read_csv(path, infer_schema_length=0)
    sites: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        token, site = row.get("site_token"), row.get("resolved_site")
        if token is None or site is None or not str(site).strip():
            continue
        sites[str(token)] = str(site).strip()
    return sites


def load_overrides(path: str | Path) -> dict[str, str]:
    """Load the hand-curated override table, skipping ``#`` comments and blank tokens."""
    path = Path(path)
    if not path.exists():
        return {}
    df = pl.read_csv(path, infer_schema_length=0, comment_prefix="#")
    overrides: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        token = row.get("site_token")
        if token is None or str(token).strip() == "":
            continue
        site = row.get("resolved_site")
        overrides[str(token).strip()] = "" if site is None else str(site).strip()
    return overrides


def main() -> list[dict]:
    """Regenerate the committed crosswalk from the real sidecar tables (networked).

    Fetches Table_S1/S3/S4 (md5-verified), derives per-token image counts from the
    per-image index, applies the curated overrides, builds the crosswalk, and writes
    :data:`frepj_tables.DEFAULT_CROSSWALK_PATH` sorted by token. This is the ONLY
    networked path in the module; the redefiner and tests never call it.
    """
    tables = frepj_tables.ensure_frepj_tables(frepj_tables.DEFAULT_TABLES_DIR)
    site_coords = frepj_tables.read_site_coordinates(tables["Table_S1.csv"])
    index = frepj_tables.read_per_image_site_index(tables["Table_S3.csv"], tables["Table_S4.csv"])
    token_counts = dict(Counter(token for token, _date in index.values()))
    overrides = load_overrides(frepj_tables.DEFAULT_OVERRIDES_PATH)

    rows = build_crosswalk(site_coords, token_counts, overrides)
    write_crosswalk(rows, frepj_tables.DEFAULT_CROSSWALK_PATH)

    by_method = Counter(r["method"] for r in rows)
    resolved_rows = [r for r in rows if r["Latitude"] is not None]
    total_images = sum(r["n_images"] for r in rows)
    resolved_images = sum(r["n_images"] for r in resolved_rows)
    coverage = (resolved_images / total_images) if total_images else 0.0
    logger.info(
        f"Wrote {len(rows)} crosswalk rows to {frepj_tables.DEFAULT_CROSSWALK_PATH} "
        f"(methods: {dict(by_method)}); {len(resolved_rows)} tokens resolved covering "
        f"{resolved_images}/{total_images} images ({coverage:.1%})."
    )
    return rows


if __name__ == "__main__":
    main()
