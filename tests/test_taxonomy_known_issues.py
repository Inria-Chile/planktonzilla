"""
(c) Inria

Pinning tests for `planktonzilla_taxonomy.csv` — they assert the CURRENT (frozen) state of
the known data inconsistencies documented as KI-8..KI-13 in
`planktonzilla/planktonzilla_dataset/utils/KNOWN_ISSUES.md`.

KI-11 is the exception: it was RESOLVED (the `QUALIFIERS` vocabulary was widened, with no CSV
change), so its write-up now lives in the sibling `RESOLVED_ISSUES.md`. Its test stays here
because what it guards is unchanged — that every `qualifier` in the CSV is a recognized value.

These tests PIN behavior; they do NOT fix it. The taxonomy table and the datasets/models
derived from it are published and frozen on HuggingFace Hub, so under the milestone's
zero-behavioral-drift rule these inconsistencies are documented and pinned rather than
corrected. If one of these assertions starts failing, the CSV has changed — update the test
ONLY together with a golden-output diff against the frozen HuggingFace reference, never
silently. The findings were established by a two-method audit (deterministic checks + a
27-agent adversarially-verified multi-lens audit) on 2026-07-13; each was independently
re-verified.

Network-free: reads only the committed CSV.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import csv
import re
from collections import defaultdict

import pytest

from planktonzilla.planktonzilla_dataset.constants import QUALIFIERS

_CSV_PATH = root / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"

RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
# Columns that are normalized to lowercase (Raw_Labels intentionally keeps source casing).
NORMALIZED_COLUMNS = (*RANKS, "proposed_label", "root_class", "qualifier")
ID_COLUMNS = ("wikidata_ID", "aphia_ID", "NCBI_ID", "ecotaxa_ID", "BOLD_ID")


@pytest.fixture(scope="module")
def rows():
    with _CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_frozen_table_anchor(rows):
    """Sanity anchor: 1,485 frozen rows + the 229 frepj rows appended in v1.2, expected schema.

    The 1,485-row base is still byte-frozen — tests/test_frepj_taxonomy_coverage.py pins
    its sha256 — so the KI-8..KI-13 findings below are unchanged; the frepj rows were
    APPENDED after it and are covered by their own tests.
    """
    assert len(rows) == 1485 + 229
    assert {*NORMALIZED_COLUMNS, *ID_COLUMNS, "Dataset", "Raw_Labels", "plankton", "living"} <= set(rows[0])


# --- KI-8: rank-column contamination — a taxon in a rank slot its suffix contradicts ---
def test_ki8_rank_column_contamination(rows):
    def appears_in(name, col):
        return any(r[col].strip().lower() == name for r in rows)

    # `-phyceae` CLASS names mis-slotted (bacillariophyceae also duplicated into Family).
    assert appears_in("bacillariophyceae", "Order")  # row 945 (neomoelleria cornuta)
    assert appears_in("bacillariophyceae", "Family")  # row 945
    assert appears_in("dinophyceae", "Order")  # row 153 (azadinium caudatum)
    # `-ales` ORDER name used as Family.
    assert appears_in("florenciellales", "Family")  # row 1126 (pseudochattonella farcimen)
    # PHYLUM name used as Class (should be `cryptophyceae`).
    assert appears_in("cryptophyta", "Class")  # row 817 (katablepharis remigera)


# --- KI-9: casing — 'Eukaryota' is the ONLY uppercase value in a normalized column ---
def test_ki9_single_uppercase_normalized_value(rows):
    offenders = [(col, r[col]) for r in rows for col in NORMALIZED_COLUMNS if r[col] and r[col] != r[col].lower()]
    # Sole violator across all 1,485 rows: proposed_label='Eukaryota' (row 671); want 'eukaryota'.
    assert offenders == [("proposed_label", "Eukaryota")]


# --- KI-10: contradictory `plankton` flag for identical fish-egg taxa ---
@pytest.mark.parametrize("label", ["clupeiformes", "engraulidae"])
def test_ki10_plankton_flag_contradiction(rows, label):
    # rows 389/390 and 645/646: identical proposed_label/qualifier/living/root_class/IDs,
    # yet plankton takes both True and False for the same fish-egg taxon.
    flags = {
        r["plankton"].strip() for r in rows if r["proposed_label"].strip().lower() == label and r["qualifier"].strip() == "egg"
    }
    assert flags == {"True", "False"}


# --- KI-11 (RESOLVED 2026-07-13): every CSV qualifier is a recognized vocabulary value ---
def test_ki11_qualifier_vocabulary_complete(rows):
    # constants.QUALIFIERS is the authoritative vocabulary; an empty cell = "unqualified".
    # part_carapace/part_skin/part_trunk are now members (the KI-11 widening), so the CSV
    # conforms. Fails if a future CSV adds a qualifier without updating the constant.
    assert len(QUALIFIERS) == len(set(QUALIFIERS)), "constants.QUALIFIERS has duplicate entries"
    seen = {r["qualifier"].strip() for r in rows}
    unrecognized = seen - set(QUALIFIERS) - {""}
    assert not unrecognized, f"CSV qualifier value(s) absent from constants.QUALIFIERS: {sorted(unrecognized)}"


# --- KI-12: integer IDs serialized as floats ('X.0'); wikidata clean 'Qxxxx' ---
def test_ki12_float_serialized_ids(rows):
    float_int = re.compile(r"^\d+\.0+$")
    for col in ("aphia_ID", "NCBI_ID", "BOLD_ID"):
        vals = [r[col].strip() for r in rows if r[col].strip()]
        assert vals, f"{col} unexpectedly empty"
        assert all(float_int.match(v) for v in vals), f"{col} is expected to be entirely 'X.0'-formatted"
    wikidata = [r["wikidata_ID"].strip() for r in rows if r["wikidata_ID"].strip()]
    assert all(re.match(r"^Q\d+$", v) for v in wikidata)


# --- KI-13: one external ID stamped on >1 distinct taxon ---
def test_ki13_ncbi_id_reused_across_distinct_taxa(rows):
    labels = {r["proposed_label"].strip().lower() for r in rows if r["NCBI_ID"].strip() == "418941.0"}
    assert {"discosphaera tubifera", "rhabdosphaera clavigera"} <= labels


# --- Verified NON-issue (guardrail): the FORWARD ID mapping is clean ---
def test_forward_id_mapping_is_clean(rows):
    """No proposed_label carries two distinct values of any single ID column (0 conflicts)."""
    for col in ID_COLUMNS:
        by_label = defaultdict(set)
        for r in rows:
            label, value = r["proposed_label"].strip().lower(), r[col].strip()
            if label and value:
                by_label[label].add(value)
        conflicting = {k: sorted(v) for k, v in by_label.items() if len(v) > 1}
        assert not conflicting, f"{col} unexpectedly maps a taxon to >1 id: {conflicting}"
