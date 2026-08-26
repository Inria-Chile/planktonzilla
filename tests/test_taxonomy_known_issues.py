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

KI-29..KI-35 below were added by the 2026-08-26 full-table audit — the first re-run of the
deterministic battery since the 229 frepj rows (Plan 18) and the 600 Tara Pacific rows
(issue #10) were appended. Same contract: they PIN the current state of the grown table,
they do not fix it.

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


def _one(rows, dataset, raw_label):
    """The unique row keyed ``(Dataset, Raw_Labels)`` — uniqueness itself is a pinned invariant."""
    matches = [r for r in rows if r["Dataset"] == dataset and r["Raw_Labels"] == raw_label]
    assert len(matches) == 1, (dataset, raw_label, len(matches))
    return matches[0]


def test_frozen_table_anchor(rows):
    """Sanity anchor: 1,485 frozen rows + the v1.2 appends, expected schema.

    The 1,485-row base is still byte-frozen — tests/test_frepj_taxonomy_coverage.py pins
    its sha256 — so the KI-8..KI-13 findings below are unchanged; the 229 frepj rows and
    then the 600 Tara Pacific rows were APPENDED after it and are covered by their own
    tests (tests/test_frepj_taxonomy_coverage.py, tests/test_tara_pacific_taxonomy.py).
    """
    assert len(rows) == 1485 + 229 + 600
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


# ======================================================================================
# KI-29..KI-35 — the 2026-08-26 full-table audit (base + frepj + Tara Pacific).
# ======================================================================================

# The four Tara Pacific sources; KI-34 needs them and this file deliberately avoids
# importing the tara builder (these tests must stay a pure read of the committed CSV).
_TARA_DATASETS = ("tara_pacific_bongo", "tara_pacific_decknet", "tara_pacific_hsn", "tara_pacific_manta")


# --- KI-29: zoocamnet `Cladocera` is mapped to an extinct fossil bivalve genus ---
def test_ki29_cladocera_mapped_to_fossil_bivalve(rows):
    # row 378: the water-flea label carries the Cladoceramus (Inoceramidae) lineage,
    # flagged as living plankton, with CLASS Bivalvia's aphia/NCBI ids stamped on it.
    row = _one(rows, "zoocamnet", "Cladocera")
    assert (row["Phylum"], row["Class"], row["Genus"]) == ("mollusca", "bivalvia", "cladoceramus")
    assert (row["plankton"], row["living"]) == ("True", "True")
    # The ids are the class's, not the genus's — byte-identical to the `bivalvia` label rows.
    bivalvia = next(r for r in rows if r["proposed_label"] == "bivalvia")
    assert (row["aphia_ID"], row["NCBI_ID"]) == (bivalvia["aphia_ID"], bivalvia["NCBI_ID"]) == ("105.0", "6544.0")
    # The same raw string reads as the crustacean superorder everywhere else.
    assert _one(rows, "global_uvp5", "Cladocera")["proposed_label"] == "branchiopoda"


# --- KI-30: the complete same-key `plankton` contradiction set (KI-10 + chordata/hexapoda) ---
def test_ki30_plankton_contradiction_set_is_complete(rows):
    """Exactly four (label, qualifier) keys carry both flag values — a fifth means the CSV moved."""
    flags = defaultdict(set)
    for r in rows:
        flags[(r["proposed_label"].strip().lower(), r["qualifier"].strip())].add(r["plankton"].strip())
    contradictory = {key for key, values in flags.items() if len(values) > 1}
    assert contradictory == {
        ("chordata", "full_body"),  # rows 318-328: True for six sources, False for jedioceans + zoolake
        ("clupeiformes", "egg"),  # KI-10
        ("engraulidae", "egg"),  # KI-10
        ("hexapoda", "full_body"),  # rows 785/786: global_uvp5 Chaeteessa True vs zooscan Insecta False
    }


# --- KI-31: the rank columns are a tree except at exactly four two-parent nodes ---
def test_ki31_two_parent_nodes_are_exactly_the_known_four(rows):
    conflicts = {}
    for child_index in range(1, len(RANKS) - 1):  # Phylum..Genus; species epithets repeat legitimately
        child, parent = RANKS[child_index], RANKS[child_index - 1]
        parents = defaultdict(set)
        for r in rows:
            value = r[child].strip().lower()
            if value:
                parents[value].add(r[parent].strip().lower())
        for value, seen in parents.items():
            if len(seen) > 1:
                conflicts[(child, value)] = seen
    assert conflicts == {
        ("Order", "arcellinida"): {"tubulinea", "lobosa"},  # base Arcella vs frepj Centropyxis
        ("Family", "bosminidae"): {"anomopoda", "diplostraca"},  # frepj Bosminopsis (row 1491)
        ("Family", "daphniidae"): {"anomopoda", "diplostraca"},  # frepj Scapholeberis (row 1527)
        ("Family", "sididae"): {"ctenopoda", "diplostraca"},  # frepj Sida (row 1543)
    }


# --- KI-32: divergent cross-dataset mappings — the four suspect ones, pinned with their contrasts ---
@pytest.mark.parametrize(
    ("dataset", "raw_label", "label"),
    [
        ("zooscan", "other_living", "monstrilloida"),  # catch-all bucket -> one copepod order (row 908)
        ("global_uvp5", "unknown", "thecofilosea"),  # generic unknown -> a concrete cercozoan class (row 1339)
        ("flowcamnet", "Acantharia", "amphibelone"),  # class-level label -> one genus (row 37)
        ("zooscan", "Creseidae", "clio pyramidata"),  # family label -> a species of ANOTHER family (row 387)
    ],
)
def test_ki32_suspect_divergent_mappings(rows, dataset, raw_label, label):
    assert _one(rows, dataset, raw_label)["proposed_label"] == label


def test_ki32_sibling_datasets_read_those_labels_generically(rows):
    others = {r["proposed_label"] for r in rows if r["Raw_Labels"] == "other_living" and r["Dataset"] != "zooscan"}
    assert others == {"other"}
    assert _one(rows, "zoolake", "unknown")["proposed_label"] == "unknown"
    assert _one(rows, "planktoscope", "Acantharia")["proposed_label"] == "acantharia"
    assert _one(rows, "global_uvp5", "Creseidae")["proposed_label"] == "creseidae"
    # zooscan's `Creseidae` row itself records a family that contradicts the label's own name.
    assert _one(rows, "zooscan", "Creseidae")["Family"] == "cliidae"


# --- KI-33: synonym splits — one taxon shipped as two label classes ---
def test_ki33_neoceratium_tripos_split(rows):
    assert {r["proposed_label"] for r in rows if r["Raw_Labels"] == "Neoceratium"} == {"neoceratium", "tripos"}
    # Sharpest inside one dataset: planktoscope's two species-mix classes land apart.
    assert _one(rows, "planktoscope", "neoceratium gibberum concilians mix")["proposed_label"] == "neoceratium"
    assert _one(rows, "planktoscope", "neoceratium falcatum inflatum mix")["proposed_label"] == "tripos"


def test_ki33_heterocapsa_kryptoperidinium_split(rows):
    split = {r["proposed_label"]: r["NCBI_ID"] for r in rows if r["Raw_Labels"] == "Heterocapsa_triquetra"}
    # One species (one NCBI taxid), two label classes depending on the source dataset.
    assert split == {"heterocapsa triquetra": "66468.0", "kryptoperidinium triquetrum": "66468.0"}


# --- KI-34: proposed_label == lowest filled rank everywhere EXCEPT 23 Tara rows ---
def test_ki34_tara_breaks_label_equals_lowest_rank_in_exactly_23_rows(rows):
    violations = []
    for r in rows:
        filled = [(rank, r[rank].strip().lower()) for rank in RANKS if r[rank].strip()]
        if not filled:
            continue  # bucket rows (artefact, detritus, other, ...) carry no ranks at all
        rank, value = filled[-1]
        expected = f"{r['Genus'].strip().lower()} {r['Species'].strip().lower()}" if rank == "Species" else value
        if r["proposed_label"].strip().lower() != expected:
            violations.append(r)
    assert [r["Raw_Labels"] for r in violations if r["Dataset"] not in _TARA_DATASETS] == []
    assert len(violations) == 23
    assert {r["proposed_label"] for r in violations} == {
        "achelata",
        "alciopini",
        "anthozoa",
        "brachyura",
        "chaetoceros inter ciliate",
        "chaetoceros inter. calothrix",
        "cirripedia",
        "coscinodiscids",
        "dinophyceae x",
        "gammaridea",
        "globorotalidae",
        "odontella sp.",
    }


# --- KI-35: comb-jelly raw labels carried on the diatom genus lineage ---
def test_ki35_comb_jelly_raw_labels_carry_the_diatom_lineage(rows):
    tell_tale = [r for r in rows if r["Raw_Labels"] in ("Ctenophora<Animalia", "comb_Ctenophora", "tentacle<Ctenophora")]
    assert len(tell_tale) == 3
    for r in tell_tale:
        # The raw label names the animal (or its comb plates / tentacles); the row says diatom.
        assert (r["Kingdom"], r["Genus"], r["aphia_ID"]) == ("chromista", "ctenophora", "163921.0")
    # While the comb jellies' own subtaxa sit under Phylum ctenophora in animalia...
    beroe = next(r for r in rows if r["proposed_label"] == "beroe")
    assert (beroe["Kingdom"], beroe["Phylum"]) == ("animalia", "ctenophora")
    # ...and the diatom-lineage rows still carry the comb-jelly ecotaxa crosswalk pair.
    assert {r["ecotaxa_ID"] for r in tell_tale} == {"456;559"} == {beroe["ecotaxa_ID"]}
