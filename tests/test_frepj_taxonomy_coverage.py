"""
(c) Inria

Network-free tests pinning the curated FREPJ taxonomy rows appended to
``planktonzilla_taxonomy.csv`` (Plan 18-01).

These five tests close TAX-05 (coverage / the silent left-join guard), TAX-06
(cross-source spelling consistency + the sentinel-cascade fix), and the
append-only / output-preserving milestone invariant. They are offline BY
CONSTRUCTION: every assertion reads only the committed TSV fixture and the
committed CSV — no HTTP, no ``build_frepj_taxonomy`` call, no live lookup.

They must stay green after BOTH Plan 18-01 (this append) and Plan 18-02 (the
external-ID fill), so nothing here asserts anything about the four external-ID
columns (blank until 18-02).
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import csv
import hashlib
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants

FIXTURES = Path(__file__).parent / "fixtures" / "frepj"
CLASS_DIRS_TSV = FIXTURES / "frepj_class_dirs.tsv"
BASELINE_SHA256 = FIXTURES / "pre_frepj_taxonomy.sha256"

# The base CSV (header + 1485 rows) is exactly 1486 lines; frepj rows are strictly
# appended after it. The first 1486 lines are byte-pinned against the committed baseline
# hash — re-baselined 2026-08-27 for the maintainer-directed repair pass and again for the
# retired-taxid blanking the external-ID checker found (see RESOLVED_ISSUES.md and KI-13);
# row count and join keys did not change in either.
PRISTINE_LINE_COUNT = 1486

DATASET = "frepj"

# The 9 class-dirs whose Genus column is a ``Ge._unk*`` sentinel. The cascade must
# null Species AND resolve proposed_label to the lowercased Family — never leaking
# the echoed ``ge._unk`` string. (Locks the sentinel-cascade fix; the generic
# non-null-proposed_label test would happily pass a wrong-but-non-null value.)
GENUS_SENTINEL_EXPECTED = {
    "Arachnida,Trombidiformes,Hydrachnidia,Ge._unk,Ge._unk": "hydrachnidia",
    "Branchiopoda,Diplostraca,Chydoridae,Ge._unk,Ge._unk": "chydoridae",
    "Copepoda,Cyclopoida,Cyclopidae,Ge._unk,Ge._unk": "cyclopidae",
    "Eurotatoria,Bdelloidea,Habrotrochidae,Ge._unk,Ge._unk": "habrotrochidae",
    "Eurotatoria,Bdelloidea,Philodinidae,Ge._unk,Ge._unk": "philodinidae",
    "Eurotatoria,Flosculariaceae,Flosculariidae,Ge._unk,Ge._unk": "flosculariidae",
    "Eurotatoria,Ploima,Lecanidae,Ge._unk,Ge._unk": "lecanidae",
    "Eurotatoria,Ploima,Notommatidae,Ge._unk,Ge._unk": "notommatidae",
    "Insecta,Diptera,Chironomidae,Ge._unk_larva_stage,Ge._unk_larva_stage": "chironomidae",
}

_HIGHER_RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family")


def _load_frozen_class_dirs() -> list[str]:
    """Load the frozen class-dir strings exactly like ``test_frepj_layout.py``.

    Split each line on tab and take field 0 — commas inside the class-dir are
    preserved verbatim (they are the Class,Order,Family,Genus,Species tuple).
    """
    lines = CLASS_DIRS_TSV.read_text().splitlines()
    return [line.split("\t")[0] for line in lines[1:]]


def _read_csv_rows() -> list[dict]:
    """Read the committed master CSV (quoted comma-bearing Raw_Labels handled)."""
    with constants.DEFAULT_TAXONOMY_CSV_FILENAME.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _frepj_rows() -> list[dict]:
    return [r for r in _read_csv_rows() if r["Dataset"] == DATASET]


def test_frepj_coverage_matches_frozen_class_dirs():
    """TAX-05: frepj Raw_Labels set == frozen class-dir set, byte-exact, 229, unique."""
    class_dirs = _load_frozen_class_dirs()
    assert len(class_dirs) == 229

    frepj = _frepj_rows()
    raw_labels = [r["Raw_Labels"] for r in frepj]

    # No duplicate join keys — a duplicate would silently over/under-count examples.
    assert len(raw_labels) == len(set(raw_labels)), "duplicate frepj Raw_Labels found"
    assert len(raw_labels) == 229

    # Byte-exact set equality (commas and the 6-tuple anomaly preserved). This is the
    # guard against the silent left-join at generate_planktonzilla.py:215.
    assert set(raw_labels) == set(class_dirs)


def test_no_null_proposed_label_for_frepj():
    """TAX-05: every frepj row has a non-null, non-empty proposed_label."""
    empty = [r["Raw_Labels"] for r in _frepj_rows() if r["proposed_label"] is None or r["proposed_label"].strip() == ""]
    assert empty == [], f"frepj rows with empty proposed_label: {empty}"


def test_genus_sentinel_cascade_nulls_species():
    """TAX-01: the 9 Ge._unk-echo rows null Species + resolve to a family-level label."""
    by_label = {r["Raw_Labels"]: r for r in _frepj_rows()}

    for class_dir, expected_family in GENUS_SENTINEL_EXPECTED.items():
        assert class_dir in by_label, f"missing frepj row for {class_dir!r}"
        row = by_label[class_dir]

        # Species must be nulled by the cascade (the echoed Ge._unk must not leak).
        assert (row["Species"] or "").strip() == "", f"{class_dir!r} Species not nulled: {row['Species']!r}"

        # proposed_label must be exactly the lowercased Family.
        assert row["proposed_label"] == expected_family, (
            f"{class_dir!r} proposed_label {row['proposed_label']!r} != {expected_family!r}"
        )

        # Neither field may carry the echoed sentinel substring.
        assert "_unk" not in (row["Species"] or "")
        assert "_unk" not in row["proposed_label"]


def test_no_unk_leak_in_any_frepj_rank_or_label():
    """WR-05: NO frepj row leaks a sentinel into ANY rank column or proposed_label.

    The hardcoded ``test_genus_sentinel_cascade_nulls_species`` above only pins 9
    known rows; this asserts the GENERIC invariant over all 229 shipped rows, so a
    regression that leaked ``_unk`` (or a bare ``sp``/``sp2``) into a DIFFERENT row or
    a different rank column (e.g. Order/Family) would be caught too.
    """
    bad = []
    for r in _frepj_rows():
        for col in ("Order", "Family", "Genus", "Species", "proposed_label"):
            val = (r[col] or "").strip().lower()
            if "_unk" in val or val in ("sp", "sp2"):
                bad.append((r["Raw_Labels"], col, r[col]))
    assert bad == [], f"sentinel leaked into a rank/label column: {bad}"


def test_shared_higher_taxa_single_spelling():
    """TAX-06: a genus shared with another source carries one higher-rank spelling."""
    rows = _read_csv_rows()

    # Existing (non-frepj) genus -> set of (Kingdom..Family) spellings.
    existing = {}
    for r in rows:
        if r["Dataset"] == DATASET:
            continue
        gen = (r["Genus"] or "").strip().lower()
        if gen:
            existing.setdefault(gen, set()).add(tuple(r[rk] for rk in _HIGHER_RANKS))

    for r in _frepj_rows():
        gen = (r["Genus"] or "").strip().lower()
        if not gen or gen not in existing:
            continue
        frepj_lineage = tuple(r[rk] for rk in _HIGHER_RANKS)
        # The frepj higher-rank spellings must be byte-identical to an existing
        # spelling for that genus — no shared higher taxon may carry two spellings.
        assert frepj_lineage in existing[gen], (
            f"genus {gen!r}: frepj lineage {frepj_lineage} not among existing {sorted(existing[gen])}"
        )


def test_existing_rows_byte_frozen():
    """Append-only: the first 1486 lines hash to the committed baseline.

    The baseline moves ONLY with a deliberate, ledgered repair of the base rows (as on
    2026-08-27) — never as a side effect of appending a source.
    """
    expected = BASELINE_SHA256.read_text().strip()

    raw = constants.DEFAULT_TAXONOMY_CSV_FILENAME.read_bytes()
    prefix = b"".join(raw.splitlines(keepends=True)[:PRISTINE_LINE_COUNT])
    actual = hashlib.sha256(prefix).hexdigest()

    assert actual == expected, "the pre-existing header + 1485 rows are no longer byte-frozen"
