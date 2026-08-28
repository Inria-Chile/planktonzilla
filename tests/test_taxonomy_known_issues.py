"""
(c) Inria

Pinning tests for `planktonzilla_taxonomy.csv` — the data-issue ledger's test file.

Two kinds of test live here:

- **Open-issue pins** (KI-12, KI-13): the defect is documented in
  `planktonzilla/planktonzilla_dataset/utils/KNOWN_ISSUES.md` and deliberately not fixed;
  the test asserts the CURRENT state so any drift is loud. Fix one only together with its
  ledger entry.
- **Resolved-state guards** (KI-8..KI-10, KI-29..KI-35): the defect was repaired by the
  maintainer-directed 2026-08-27 fix pass (see RESOLVED_ISSUES.md for each entry and the
  full cell-level edit manifest). The guards pin the REPAIRED values, so a regression —
  or an importer re-introducing an old mapping — turns the suite red. The generic
  invariants behind those repairs (one lineage per label, single-parent rank tree, flag
  coherence, label = lowest rank, ...) are enforced in `tests/test_taxonomy_validation.py`;
  the guards here cover only what a generic rule cannot see: WHICH reading each repaired
  row was given.

KI-11 (RESOLVED 2026-07-13) keeps its test here per its original note: what it guards —
that every CSV qualifier is a recognized vocabulary value — is unchanged.

History: KI-8..KI-13 were found by the 2026-07-13 two-method audit of the 1,485-row
table; KI-29..KI-35 by the 2026-08-26 full-table audit after the frepj and Tara Pacific
appends. Until 2026-08-27 this file pinned the BROKEN state of all of them under the
zero-behavioral-drift rule; the repair pass inverted those pins into the guards below.
The published HuggingFace artifacts are unchanged by the repair — the fixed table takes
effect at the next dataset build.

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
ID_COLUMNS = ("wikidata_ID", "aphia_ID", "NCBI_ID", "ecotaxa_ID", "BOLD_ID")


@pytest.fixture(scope="module")
def rows():
    with _CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _one(rows, dataset, raw_label):
    """The unique row keyed ``(Dataset, Raw_Labels)`` — uniqueness itself is enforced."""
    matches = [r for r in rows if r["Dataset"] == dataset and r["Raw_Labels"] == raw_label]
    assert len(matches) == 1, (dataset, raw_label, len(matches))
    return matches[0]


def test_frozen_table_anchor(rows):
    """Sanity anchor: 1,485 base + 229 frepj + 44 daplankton + 600 Tara Pacific, expected schema.

    The base is no longer byte-frozen: the 2026-08-27 repair pass edited 70 rows in
    place (tests/fixtures/frepj/pre_frepj_taxonomy.sha256 was re-baselined with it, and
    again for the retired-taxid blanking). Row COUNT and the (Dataset, Raw_Labels) key
    set are unchanged by the repair — no importer's coverage moved. The later blocks are
    strictly appended and carry their own coverage tests.
    """
    assert len(rows) == 1485 + 229 + 44 + 600
    expected = {*RANKS, "proposed_label", "root_class", "qualifier", *ID_COLUMNS, "Dataset", "Raw_Labels", "plankton", "living"}
    assert expected <= set(rows[0])


# ======================================================================================
# Open-issue pins
# ======================================================================================


# --- KI-11 (RESOLVED 2026-07-13): every CSV qualifier is a recognized vocabulary value ---
def test_ki11_qualifier_vocabulary_complete(rows):
    # constants.QUALIFIERS is the authoritative vocabulary; an empty cell = "unqualified".
    # part_carapace/part_skin/part_trunk are now members (the KI-11 widening), so the CSV
    # conforms. Fails if a future CSV adds a qualifier without updating the constant.
    assert len(QUALIFIERS) == len(set(QUALIFIERS)), "constants.QUALIFIERS has duplicate entries"
    seen = {r["qualifier"].strip() for r in rows}
    unrecognized = seen - set(QUALIFIERS) - {""}
    assert not unrecognized, f"CSV qualifier value(s) absent from constants.QUALIFIERS: {sorted(unrecognized)}"


# --- KI-12 (open): integer IDs serialized as floats ('X.0'); wikidata clean 'Qxxxx' ---
def test_ki12_float_serialized_ids(rows):
    # Canonicalizing to bare integers is deferred: it rewrites ~3,700 cells and shifts
    # dtype inference in both committed readers — gate on the reader-equivalence suite.
    float_int = re.compile(r"^\d+\.0$")
    for col in ("aphia_ID", "NCBI_ID", "BOLD_ID"):
        vals = [r[col].strip() for r in rows if r[col].strip()]
        assert vals, f"{col} unexpectedly empty"
        assert all(float_int.match(v) for v in vals), f"{col} is expected to be entirely 'X.0'-formatted"
    wikidata = [r["wikidata_ID"].strip() for r in rows if r["wikidata_ID"].strip()]
    assert all(re.match(r"^Q\d+$", v) for v in wikidata)


# --- KI-13 (open): one external ID stamped on >1 distinct taxon ---
def test_ki13_ncbi_id_reused_across_distinct_taxa(rows):
    labels = {r["proposed_label"].strip().lower() for r in rows if r["NCBI_ID"].strip() == "418941.0"}
    assert {"discosphaera tubifera", "rhabdosphaera clavigera"} <= labels


# ======================================================================================
# Resolved-state guards — the 2026-08-27 repair pass (RESOLVED_ISSUES.md has each entry)
# ======================================================================================


# --- KI-8 resolved: the four contaminated rank slots carry their verified values ---
def test_ki8_resolved_rank_slots(rows):
    # Azadinium: NCBI/GBIF/Wikidata unanimously place Amphidomataceae in Gonyaulacales
    # (WoRMS abstains: "Dinophyceae incertae sedis" — the origin of the old value).
    assert _one(rows, "medplanktonset", "Azadinium_caudatum")["Order"] == "gonyaulacales"
    row = _one(rows, "planktoscope", "Eucampia cornuta")
    assert (row["Order"], row["Family"]) == ("hemiaulales", "hemiaulaceae")
    # No validly published family exists for Pseudochattonella; the WoRMS-verbatim
    # placeholder keeps the rank ladder gap-free without fabricating a name.
    assert _one(rows, "whoi", "Pseudochattonella_farcimen")["Family"] == "florenciellales incertae sedis"
    row = _one(rows, "syke_ifcb_2022", "Katablepharis_remigera")
    assert (row["Class"], row["Order"]) == ("cryptophyceae", "katablepharidales")


# --- KI-10 + KI-30 resolved: the repaired plankton flags point the ichthyoplankton way ---
def test_ki10_ki30_resolved_plankton_directions(rows):
    def flags(label, qualifier):
        return {r["plankton"] for r in rows if r["proposed_label"] == label and r["qualifier"] == qualifier}

    # Fish eggs and fish larvae ARE plankton; adult-fish rows keep the table's
    # long-standing True; insects are not plankton.
    assert flags("clupeiformes", "egg") == {"True"}
    assert flags("engraulidae", "egg") == {"True"}
    assert flags("chordata", "full_body") == {"True"}
    assert flags("chordata", "larvae") == {"True"}
    assert flags("teleostei", "larvae") == {"True"}
    assert flags("leptocephalus", "larvae") == {"True"}
    assert flags("myctophidae", "larvae") == {"True"}
    assert flags("hexapoda", "full_body") == {"False"}


# --- KI-29 resolved: `Cladocera` reads as the crustacean superorder everywhere ---
def test_ki29_resolved_cladocera_is_branchiopoda(rows):
    for dataset in ("zoocamnet", "global_uvp5"):
        row = _one(rows, dataset, "Cladocera")
        assert row["proposed_label"] == "branchiopoda"
        assert (row["Phylum"], row["Class"]) == ("arthropoda", "branchiopoda")
    assert not any(r["Genus"] == "cladoceramus" for r in rows)


# --- KI-31 resolved: the frepj block joined the table's cladoceran/arcellinid vocabulary ---
def test_ki31_resolved_rank_vocabulary(rows):
    # (The single-parent tree property itself is enforced generically in
    # test_taxonomy_validation.py; this guards WHICH vocabulary won.)
    assert not any(r["Order"] == "diplostraca" for r in rows)
    assert not any(r["Family"] == "daphniida" for r in rows)
    assert not any(r["Class"] == "lobosa" for r in rows)
    assert {r["Order"] for r in rows if r["Family"] == "bosminidae"} == {"anomopoda"}
    assert {r["Order"] for r in rows if r["Family"] == "sididae"} == {"ctenopoda"}
    assert {r["Class"] for r in rows if r["Order"] == "arcellinida"} == {"tubulinea"}


# --- KI-32 resolved (partly): the four misaligned mappings; the judgment tier stays pinned ---
@pytest.mark.parametrize(
    ("dataset", "raw_label", "label"),
    [
        ("zooscan", "other_living", "other"),  # was monstrilloida (row's ecotaxa crosswalk was misaligned)
        ("global_uvp5", "unknown", "unknown"),  # was thecofilosea
        ("flowcamnet", "Acantharia", "acantharia"),  # was amphibelone (a copy from the neighboring row)
        ("zooscan", "Creseidae", "creseidae"),  # was clio pyramidata — a species of ANOTHER family
    ],
)
def test_ki32_resolved_suspect_mappings(rows, dataset, raw_label, label):
    assert _one(rows, dataset, raw_label)["proposed_label"] == label


def test_ki32_remaining_divergences_are_exactly_the_acknowledged_ones(rows):
    """The deliberate per-dataset granularity divergences that REMAIN after the repair.

    Same raw label, different reading, on purpose (UVP 'Annelida' imagery really is
    dominated by Poeobius, and so on). Pinned in both directions so neither a silent new
    divergence nor a silent re-reading can slip in. `build_tara_pacific_taxonomy`'s
    DIVERGENT_DONORS acknowledges the subset of these its verbatim rule can hit.
    """
    by_raw = defaultdict(set)
    for r in rows:
        by_raw[r["Raw_Labels"].strip().lower()].add(r["proposed_label"])
    divergent = {raw: sorted(labels) for raw, labels in by_raw.items() if len(labels) > 1}
    assert divergent == {
        "actinula": ["hydrozoa", "solmundella bitentaculata"],
        "annelida": ["annelida", "poeobius"],
        "darkrods": ["other", "shape"],
        "dinophyceae": ["dinophyceae", "gonyaulacales"],
        "fiber_detritus": ["detritus", "fiber"],
        "filament": ["cyanophyceae", "filament"],
        "foraminifera": ["foraminifera", "globigerinidae"],
        "harpacticoida": ["euterpina", "harpacticoida"],
        "nauplii": ["arthropoda", "copepoda"],
        "ornithocercus": ["ornithocercus", "ornithocercus magnificus"],
        "penilia": ["penilia", "penilia avirostris"],
        "thecosomata": ["cavolinia inflexa", "thecosomata"],
        "trachymedusae": ["botrynema", "trachymedusae"],
    }


# --- KI-33 resolved: one taxon, one label class ---
def test_ki33_resolved_synonym_merges(rows):
    labels = {r["proposed_label"] for r in rows}
    assert "neoceratium" not in labels and "heterocapsa triquetra" not in labels
    assert {r["proposed_label"] for r in rows if r["Raw_Labels"] == "Neoceratium"} == {"tripos"}
    assert {r["proposed_label"] for r in rows if r["Raw_Labels"] == "Heterocapsa_triquetra"} == {"kryptoperidinium triquetrum"}
    # The freshwater genus is untouched by the marine merge.
    assert any(r["proposed_label"] == "ceratium" for r in rows)


# --- KI-34 resolved: the two expressible labels got their rank fills ---
def test_ki34_resolved_rank_fills(rows):
    # (The label-vs-lowest-rank rule and the SUB_RANK_LABELS registry are enforced in
    # test_taxonomy_validation.py; these are the two rows repaired by filling instead.)
    row = _one(rows, "tara_pacific_hsn", "Globorotalidae")
    assert (row["Class"], row["Order"], row["Family"]) == ("globothalamea", "rotaliida", "globorotalidae")
    for dataset in ("tara_pacific_hsn", "tara_pacific_manta"):
        assert _one(rows, dataset, "polype<Anthozoa")["Class"] == "anthozoa"


# --- KI-35 resolved: `ctenophora` is the comb-jelly phylum, verified IDs attached ---
def test_ki35_resolved_ctenophora_is_the_comb_jelly(rows):
    cteno = [r for r in rows if r["proposed_label"] == "ctenophora"]
    assert len(cteno) == 12
    for r in cteno:
        assert (r["Kingdom"], r["Phylum"], r["Genus"]) == ("animalia", "ctenophora", "")
        assert (r["wikidata_ID"], r["aphia_ID"], r["NCBI_ID"]) == ("Q102778", "1248.0", "10197.0")
        assert r["ecotaxa_ID"] == "456;559"
    # The diatom genus reading is gone table-wide.
    assert not any(r["Genus"] == "ctenophora" for r in rows)
