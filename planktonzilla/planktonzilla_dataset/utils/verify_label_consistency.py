"""
(c) Inria

verify_label_consistency.py
===========================
HORIZONTAL verification of ``planktonzilla_taxonomy.csv``: do the rows that share a source
label agree about what that label means?

Why this exists
---------------
``verify_taxonomy_ids.py`` checks each row *vertically* — it takes the taxon a row claims to
be and asks WoRMS / NCBI / Wikidata whether the identifiers agree. Twenty-two of its
twenty-three checks work that way, and the twenty-third works on the identifier axis. None of
them compares two rows to each other along the axis that actually carries the images.

That axis is ``Raw_Labels``. It is the join key ``build_taxonomy_lookup`` uses to attach a
taxonomy to an imagefolder class directory, so two rows sharing a label are describing the
same folder name in two different source datasets. When they publish different taxa, at least
one of them mislabels its images — and no vertical check can see it, because each row is
independently defensible: it groups by ``proposed_label``, which files the disagreeing rows
under *different* taxa and so never puts them side by side.

The concrete case that motivated this (KI-31): ``Raw_Labels=Harpacticoida`` is published as
the order ``harpacticoida`` (aphia 1102) by five datasets and as the genus ``euterpina``
(aphia 115348) by three. Both rows pass every authority check, because aphia 115348 genuinely
*is* Euterpina — it is simply attached to a class directory that says Harpacticoida.

What is reported
----------------
Rows are grouped by case- and whitespace-normalized ``Raw_Labels``; a group whose rows do not
all publish the same taxon yields exactly one finding, classified by how they disagree:

    * ``lineage_contradiction`` (ERROR) — two lineages name DIFFERENT taxa at a rank both
      populate. The label cannot be both; one side is wrong, or the two are nomenclatural
      synonyms (which this check cannot tell apart, so it reports and a human adjudicates).
    * ``rank_inflation`` (ERROR) — one lineage extends the other with no disagreement where
      they overlap. The label is published at a rank finer than it supports, so those images
      claim a precision the source label never had. The limiting case is a lineage against no
      lineage at all: a non-taxonomic bucket (``unknown``, ``other_living``) given a real taxon
      by one dataset.
    * ``label_disagreement`` (WARN) — no rank cell is populated on either side and the
      ``proposed_label`` values differ. A curation inconsistency between two non-taxonomic
      buckets, not a taxonomic error.

Network-free BY CONSTRUCTION: one pass over the committed CSV, no snapshot, no HTTP, and no
import of anything that speaks HTTP. That is the whole cost of the check.

Frozen table, so findings are waived rather than corrected
----------------------------------------------------------
The published table and the datasets derived from it are frozen on HuggingFace Hub under the
zero-behavioural-drift rule, so every finding standing today is adjudicated in
``LABEL_CONSISTENCY_WAIVERS.json`` with its reason. Adding a waiver is a reviewed act: a NEW
disagreement — a CSV edit, or a new source dataset reusing an existing label for a different
taxon — arrives unwaived and turns ``tests/test_taxonomy_label_consistency.py`` red.

Usage:
    python -m planktonzilla.planktonzilla_dataset.utils.verify_label_consistency
    python -m ...verify_label_consistency --all       # include waived findings
    python -m ...verify_label_consistency --json      # machine-readable
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from planktonzilla.planktonzilla_dataset.constants import DEFAULT_TAXONOMY_CSV_FILENAME, TAXONOMY_RANKS

RANKS = TAXONOMY_RANKS
SEVERITIES = ("ERROR", "WARN")
DEFAULT_WAIVERS_PATH = Path(__file__).parent / "LABEL_CONSISTENCY_WAIVERS.json"

CONTRADICTION = "lineage_contradiction"
RANK_INFLATION = "rank_inflation"
LABEL_DISAGREEMENT = "label_disagreement"

SEVERITY_OF = {CONTRADICTION: "ERROR", RANK_INFLATION: "ERROR", LABEL_DISAGREEMENT: "WARN"}


@dataclass(frozen=True)
class Disagreement:
    """One source label whose rows do not agree on the taxon they publish.

    Attributes:
        check: ``lineage_contradiction`` / ``rank_inflation`` / ``label_disagreement``.
        severity: One of :data:`SEVERITIES`.
        raw_label: The normalized ``Raw_Labels`` value the rows share.
        labels: The distinct ``proposed_label`` values published under it, sorted.
        variants: One ``(proposed_label, deepest_rank, datasets)`` triple per distinct lineage.
        conflicts: For a contradiction, the ``(rank, name_a, name_b)`` triples that clash.
        n_rows: How many CSV rows the finding touches.
    """

    check: str
    severity: str
    raw_label: str
    labels: tuple[str, ...]
    variants: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    conflicts: tuple[tuple[str, str, str], ...] = ()
    n_rows: int = 0

    @property
    def finding_id(self) -> str:
        """Stable 12-hex-char identity for waiver matching.

        Keyed on the label, the check and the set of taxa published under it — deliberately
        NOT on ``variants``/``n_rows``, so a waiver survives a new source dataset adopting a
        label with a taxon already seen, but does NOT survive a new taxon appearing under it.
        """
        payload = "|".join((self.check, self.raw_label, ",".join(self.labels)))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> str:
        """One-line human summary, as printed by the CLI."""
        parts = [f"{self.severity:<5} {self.check:<22} {self.raw_label!r} -> {list(self.labels)}"]
        for label, rank, datasets in self.variants:
            slot = rank or "(no rank)"
            parts.append(f"        {label:<32} {slot:<10} {list(datasets)}")
        for rank, left, right in self.conflicts:
            parts.append(f"      ! {rank}: {left} vs {right}")
        return "\n".join(parts)


def norm(value: str | None) -> str:
    """Normalize a cell for comparison: whitespace-collapsed, stripped, lowercased."""
    if not value:
        return ""
    return " ".join(str(value).split()).strip().lower()


def read_rows(csv_path: Path = DEFAULT_TAXONOMY_CSV_FILENAME) -> list[dict]:
    """Read the taxonomy CSV as a list of string-valued dicts."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lineage(row: dict) -> tuple[str, ...]:
    """The row's seven rank cells, normalized, in rank order."""
    return tuple(norm(row.get(rank)) for rank in RANKS)


def rank_depth(cells: tuple[str, ...]) -> int:
    """Index of the deepest populated rank in a lineage tuple; ``-1`` when none is."""
    for index in reversed(range(len(RANKS))):
        if cells[index]:
            return index
    return -1


def deepest_rank(cells: tuple[str, ...]) -> str:
    """Name of the deepest populated rank in a lineage tuple, or ``""`` if none is."""
    depth = rank_depth(cells)
    return RANKS[depth] if depth >= 0 else ""


def group_by_raw_label(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by normalized ``Raw_Labels``.

    Normalizing the case matters: ``Harpacticoida`` and ``harpacticoida`` are the same source
    label wearing two datasets' casing conventions, and grouping them apart would hide exactly
    the disagreement this module exists to find. (``Raw_Labels`` legitimately preserves source
    casing in the table itself — see KI-9.)
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[norm(row.get("Raw_Labels"))].append(row)
    return dict(grouped)


def _conflicting_ranks(lineages: list[tuple[str, ...]]) -> list[tuple[str, str, str]]:
    """Ranks where two lineages both name something, and name something different."""
    conflicts = set()
    for i in range(len(lineages)):
        for j in range(i + 1, len(lineages)):
            left, right = lineages[i], lineages[j]
            for index, rank in enumerate(RANKS):
                if left[index] and right[index] and left[index] != right[index]:
                    conflicts.add((rank, left[index], right[index]))
    return sorted(conflicts)


def check_group(raw_label: str, rows: list[dict]) -> Disagreement | None:
    """Classify one ``Raw_Labels`` group, or return ``None`` when its rows agree.

    Args:
        raw_label: The normalized label shared by every row in the group.
        rows: The CSV rows carrying it.

    Returns:
        The single finding for this group, or ``None``.
    """
    # Keyed on BOTH the lineage and the label: two rows can share an all-empty lineage and
    # still publish different labels, which is precisely the `label_disagreement` case.
    by_variant: dict[tuple[tuple[str, ...], str], set[str]] = defaultdict(set)
    for row in rows:
        variant = (lineage(row), norm(row.get("proposed_label")))
        by_variant[variant].add(row.get("Dataset", ""))

    if len(by_variant) == 1:
        return None

    labels = sorted({label for _, label in by_variant})
    lineages = sorted({cells for cells, _ in by_variant})
    conflicts = _conflicting_ranks(lineages)
    if conflicts:
        check = CONTRADICTION
    elif len({deepest_rank(cells) for cells in lineages}) > 1:
        check = RANK_INFLATION
    else:
        check = LABEL_DISAGREEMENT

    # Shallowest first: the ordering that makes an inflation readable as one.
    variants = tuple(
        (label, deepest_rank(cells), tuple(sorted(datasets)))
        for (cells, label), datasets in sorted(by_variant.items(), key=lambda kv: (rank_depth(kv[0][0]), kv[0][1]))
    )
    return Disagreement(
        check=check,
        severity=SEVERITY_OF[check],
        raw_label=raw_label,
        labels=tuple(labels),
        variants=variants,
        conflicts=tuple(conflicts),
        n_rows=len(rows),
    )


def check_label_consistency(rows: list[dict]) -> list[Disagreement]:
    """Every ``Raw_Labels`` group whose rows disagree about the taxon they publish.

    Args:
        rows: CSV rows, as returned by :func:`read_rows`.

    Returns:
        Findings sorted most-severe first, then by label.
    """
    findings = [finding for raw_label, group in group_by_raw_label(rows).items() if (finding := check_group(raw_label, group))]
    return sorted(findings, key=lambda f: (SEVERITIES.index(f.severity), f.check, f.raw_label))


def load_waivers(path: Path = DEFAULT_WAIVERS_PATH) -> dict[str, dict]:
    """Load the adjudicated findings, keyed by ``finding_id``. A missing file means none."""
    if not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["finding_id"]: entry for entry in payload.get("waivers", [])}


def apply_waivers(
    findings: list[Disagreement], waivers: dict[str, dict]
) -> tuple[list[Disagreement], list[Disagreement], list[str]]:
    """Split findings by waiver status and report waivers matching nothing.

    Returns:
        ``(unwaived, waived, stale_waiver_ids)``. A stale waiver means the disagreement it
        described is gone — the waiver should be deleted with the change that fixed it.
    """
    seen = {finding.finding_id for finding in findings}
    unwaived = [finding for finding in findings if finding.finding_id not in waivers]
    waived = [finding for finding in findings if finding.finding_id in waivers]
    return unwaived, waived, sorted(set(waivers) - seen)


def summarize(findings: list[Disagreement]) -> dict:
    """Counts by severity and by check, plus the rows touched."""
    by_severity: dict[str, int] = defaultdict(int)
    by_check: dict[str, int] = defaultdict(int)
    for finding in findings:
        by_severity[finding.severity] += 1
        by_check[finding.check] += 1
    return {
        "total": len(findings),
        "by_severity": {k: by_severity[k] for k in SEVERITIES if k in by_severity},
        "by_check": dict(sorted(by_check.items())),
        "rows_touched": sum(finding.n_rows for finding in findings),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exits non-zero when an unwaived finding stands, or a waiver has gone stale.
    """
    parser = argparse.ArgumentParser(
        description="Report source labels whose rows publish different taxa (KI-31).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_TAXONOMY_CSV_FILENAME, help="taxonomy CSV to check")
    parser.add_argument("--waivers", type=Path, default=DEFAULT_WAIVERS_PATH, help="adjudicated-findings JSON")
    parser.add_argument("--all", action="store_true", help="also print findings that are waived")
    parser.add_argument("--json", action="store_true", help="emit findings as JSON instead of text")
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)
    findings = check_label_consistency(rows)
    unwaived, waived, stale = apply_waivers(findings, load_waivers(args.waivers))

    if args.json:
        payload = {
            "summary": summarize(findings),
            "unwaived": [f.__dict__ | {"finding_id": f.finding_id} for f in unwaived],
            "waived": [f.__dict__ | {"finding_id": f.finding_id} for f in waived],
            "stale_waivers": stale,
        }
        print(json.dumps(payload, indent=2, default=list))
        return 1 if unwaived or stale else 0

    labels = group_by_raw_label(rows)
    print(f"{len(rows)} rows, {len(labels)} distinct source labels, {len(findings)} disagreeing")
    print(f"summary: {summarize(findings)}")
    for finding in unwaived:
        print(f"\nUNWAIVED {finding.finding_id}\n{finding.describe()}")
    if args.all:
        for finding in waived:
            print(f"\nwaived   {finding.finding_id}\n{finding.describe()}")
    for waiver_id in stale:
        print(f"\nSTALE WAIVER {waiver_id}: matches no current finding — delete it")
    if not unwaived and not stale:
        print(f"\nOK: {len(waived)} finding(s), all adjudicated in {args.waivers.name}")
    return 1 if unwaived or stale else 0


if __name__ == "__main__":
    sys.exit(main())
