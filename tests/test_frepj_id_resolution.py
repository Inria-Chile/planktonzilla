"""
(c) Inria

Network-free tests pinning the ID-mutating logic of ``resolve_frepj_ids.py`` (Plan
18-02). These close the WR-03 coverage gap: the external-ID resolver had zero unit
tests, yet it is the exact code path that produced the CR-01 wrong-ID defect
(a bare epithet ``"sarsi"`` matching the unrelated genus ``Sarsia``).

Every test here is offline BY CONSTRUCTION: the Wikidata/NCBI resolvers are
monkeypatched or fed in-memory fixtures — no HTTP, no live lookup. They pin the
CURRENT (fixed) behaviour: full-binomial query construction, verbatim overlap reuse,
the Wikidata-label / cross-genus / NCBI-lineage guards that BLANK wrong-taxon hits,
the too-coarse KI-6 blanking, and the frepj-rows-only idempotent CSV backfill.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import copy
import shutil
from pathlib import Path

import polars as pl
import pytest

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR, TaxonomyError, loader, write
from planktonzilla.planktonzilla_dataset.taxonomy.model import read_tsv
from planktonzilla.planktonzilla_dataset.utils import resolve_frepj_ids as rfi

# ── helpers ───────────────────────────────────────────────────────────────────────


def _draft_row(genus, species, raw_label, *, kingdom="animalia", phylum="arthropoda", class_="", order="", family=""):
    """A frepj draft row dict (RANK_COLS + Raw_Labels + proposed_label)."""
    proposed = f"{genus} {species}".strip() if species else (genus or family or order or class_)
    return {
        "Kingdom": kingdom,
        "Phylum": phylum,
        "Class": class_,
        "Order": order,
        "Family": family,
        "Genus": genus,
        "Species": species,
        "Raw_Labels": raw_label,
        "proposed_label": proposed,
    }


def _fake_resolver(preset):
    """Build offline stand-ins for ``fetch_wikidata_ids`` / ``fetch_external_ids``.

    ``preset`` is keyed by the search string the resolver would use — the full
    binomial for a species row, else the genus — and maps to the pretend Wikidata
    hit ``{qid, rank, label, aphia, ncbi, bold}``. This lets the tests assert exactly
    what the guards do with a wrong-label or shared-id hit, with no network.
    """

    def _key(row):
        return row["Species"] or row["Genus"]

    def fetch_wikidata_ids(unique):
        qids, ranks, labels = [], [], []
        for row in unique.iter_rows(named=True):
            hit = preset.get(_key(row))
            qids.append(hit["qid"] if hit else None)
            ranks.append(hit["rank"] if hit else "")
            labels.append(hit["label"] if hit else "")
        return unique.with_columns(
            [
                pl.Series("wikidata_ID", qids, dtype=pl.String),
                pl.Series("Matched Rank", ranks, dtype=pl.String),
                pl.Series("Matched Label", labels, dtype=pl.String),
            ]
        )

    def fetch_external_ids(resolved):
        aphia, ncbi, bold = [], [], []
        for row in resolved.iter_rows(named=True):
            hit = preset.get(_key(row), {})
            aphia.append(hit.get("aphia"))
            ncbi.append(hit.get("ncbi"))
            bold.append(hit.get("bold"))
        return resolved.with_columns(
            [
                pl.Series("aphia_ID", aphia, dtype=pl.String),
                pl.Series("NCBI_ID", ncbi, dtype=pl.String),
                pl.Series("BOLD_ID", bold, dtype=pl.String),
            ]
        )

    return fetch_wikidata_ids, fetch_external_ids


# ── CR-01: full-binomial query construction ───────────────────────────────────────


def test_query_species_uses_full_binomial():
    """A species row is queried as ``Genus species`` — never the bare epithet (CR-01)."""
    assert rfi._query_species("sinodiaptomus", "sarsi") == "sinodiaptomus sarsi"
    assert rfi._query_species("Alona", "affinis ") == "Alona affinis"
    # Genus-only rows keep an empty species cell (resolver falls back to the Genus rank).
    assert rfi._query_species("sida", "") == ""
    assert rfi._query_species("", "") == ""


def test_unique_rank_frame_species_column_is_binomial():
    """The frame handed to the Wikidata resolver carries the binomial in Species."""
    rows = [_draft_row("sinodiaptomus", "sarsi", "RL1"), _draft_row("sida", "", "RL2")]
    frame = rfi._unique_rank_frame(rows)
    species = set(frame["Species"].to_list())
    assert "sinodiaptomus sarsi" in species
    assert "" in species  # the genus-only row


# ── overlap path: verbatim reuse + species/genus tie-break ─────────────────────────


def test_resolve_overlaps_reuses_ids_verbatim_and_ties_break_species_first():
    """Overlap rows copy existing IDs byte-verbatim; species match beats genus match."""
    ex_species = {("daphnia", "pulex"): (("Qsp", "1.0", "2.0", "3.0"), "zoolake")}
    ex_genus = {"daphnia": (("Qgen", "10.0", "20.0", "30.0"), "sykezooscan2024")}

    sp_row = {"Raw_Labels": "RL_SP", "proposed_label": "daphnia pulex", "Genus": "Daphnia", "Species": "pulex"}
    gen_row = {"Raw_Labels": "RL_GEN", "proposed_label": "daphnia", "Genus": "Daphnia", "Species": ""}
    new_row = {"Raw_Labels": "RL_NEW", "proposed_label": "moina micrura", "Genus": "Moina", "Species": "micrura"}

    mapping, drafts = rfi.resolve_overlaps([sp_row, gen_row, new_row], ex_species, ex_genus)

    # Species binomial match wins and copies the SPECIES id set verbatim.
    assert mapping["RL_SP"]["matched_rank"] == "Species"
    assert mapping["RL_SP"][rfi.WIKIDATA_ID] == "Qsp"
    assert mapping["RL_SP"]["aphia_ID"] == "1.0" and mapping["RL_SP"]["provenance"] == "reused:zoolake"
    # Genus-only match copies the GENUS id set verbatim.
    assert mapping["RL_GEN"]["matched_rank"] == "Genus"
    assert mapping["RL_GEN"][rfi.WIKIDATA_ID] == "Qgen"
    assert mapping["RL_GEN"]["provenance"] == "reused:sykezooscan2024"
    # An unmatched taxon is left for the draft path (not invented).
    assert [d["Raw_Labels"] for d in drafts] == ["RL_NEW"]
    assert "RL_NEW" not in mapping


# ── draft path: label lineage guard blanks a homonym ───────────────────────────────


def test_resolve_drafts_blanks_wrong_label_homonym(monkeypatch):
    """A Species hit whose Wikidata label disagrees with the genus is BLANKED (CR-01)."""
    preset = {
        # Bare-epithet homonym: "sinodiaptomus sarsi" resolves to a label "Sarsia".
        "sinodiaptomus sarsi": {
            "qid": "Q_WRONG",
            "rank": "Species",
            "label": "Sarsia",
            "aphia": "117070",
            "ncbi": "6078",
            "bold": "159702",
        },
        # Legitimate hit: label matches the genus.
        "alona guttata": {
            "qid": "Q_OK",
            "rank": "Species",
            "label": "Alona guttata",
            "aphia": "531098",
            "ncbi": "220470",
            "bold": "31188",
        },
    }
    fw, fe = _fake_resolver(preset)
    monkeypatch.setattr(rfi.extract_taxon_ids, "fetch_wikidata_ids", fw)
    monkeypatch.setattr(rfi.extract_taxon_ids, "fetch_external_ids", fe)

    rows = [
        _draft_row("sinodiaptomus", "sarsi", "RL_SARSI", class_="copepoda", order="calanoida", family="diaptomidae"),
        _draft_row("alona", "guttata", "RL_ALONA", class_="branchiopoda", order="diplostraca", family="chydoridae"),
    ]
    mapping = rfi.resolve_drafts(rows)

    wrong = mapping["RL_SARSI"]
    assert wrong["provenance"] == rfi.PROV_GUARD_BLANKED
    assert all(wrong[c] == "" for c in rfi.ID_FIELDS), "wrong-taxon IDs must be blanked, not shipped"

    ok = mapping["RL_ALONA"]
    assert ok["provenance"] == rfi.PROV_DRAFT
    assert ok[rfi.WIKIDATA_ID] == "Q_OK" and ok["NCBI_ID"] == "220470.0"


# ── cross-genus shared-id guard ────────────────────────────────────────────────────


def test_blank_cross_genus_shared_ids():
    """One wikidata_ID shared across different genera is a collision -> all blanked."""
    frepj_rows = [
        {"Raw_Labels": "RL_A", "Genus": "Alona"},
        {"Raw_Labels": "RL_M", "Genus": "Moina"},
        {"Raw_Labels": "RL_S", "Genus": "Sida"},
    ]
    mapping = {
        "RL_A": {
            "raw_label": "RL_A",
            "provenance": rfi.PROV_DRAFT,
            rfi.WIKIDATA_ID: "Q_SHARED",
            "aphia_ID": "1.0",
            "NCBI_ID": "2.0",
            "BOLD_ID": "3.0",
            "cox": "",
        },
        "RL_M": {
            "raw_label": "RL_M",
            "provenance": rfi.PROV_DRAFT,
            rfi.WIKIDATA_ID: "Q_SHARED",
            "aphia_ID": "1.0",
            "NCBI_ID": "2.0",
            "BOLD_ID": "3.0",
            "cox": "",
        },
        "RL_S": {
            "raw_label": "RL_S",
            "provenance": rfi.PROV_DRAFT,
            rfi.WIKIDATA_ID: "Q_UNIQUE",
            "aphia_ID": "9.0",
            "NCBI_ID": "",
            "BOLD_ID": "",
            "cox": "",
        },
    }
    blanked = rfi.blank_cross_genus_shared_ids(mapping, frepj_rows)

    assert set(blanked) == {"RL_A", "RL_M"}
    assert mapping["RL_A"]["provenance"] == rfi.PROV_GUARD_BLANKED
    assert all(mapping["RL_M"][c] == "" for c in rfi.ID_FIELDS)
    # The unique-genus row is untouched.
    assert mapping["RL_S"]["provenance"] == rfi.PROV_DRAFT
    assert mapping["RL_S"][rfi.WIKIDATA_ID] == "Q_UNIQUE"


# ── NCBI lineage guard (WR-01) ─────────────────────────────────────────────────────


def test_ncbi_lineage_consistent():
    """The lineage check confirms the RIGHT taxid, not merely that one exists (WR-01)."""
    sarsia = {"name": "sarsia", "rank": "genus", "lineage": {"cnidaria", "hydrozoa"}}
    assert rfi._ncbi_lineage_consistent(sarsia, "sinodiaptomus", "arthropoda") is False
    correct = {"name": "sinodiaptomus sarsi", "rank": "species", "lineage": {"arthropoda", "copepoda", "sinodiaptomus"}}
    assert rfi._ncbi_lineage_consistent(correct, "sinodiaptomus", "arthropoda") is True
    # A family-rank match on a Ge._unk row (empty genus) with a matching phylum passes.
    family = {"name": "chydoridae", "rank": "family", "lineage": {"arthropoda", "branchiopoda"}}
    assert rfi._ncbi_lineage_consistent(family, "", "arthropoda") is True


def test_verify_ncbi_lineage_blanks_wrong_taxid(monkeypatch):
    """A draft row whose NCBI taxid lineage contradicts the row is BLANKED (WR-01)."""
    info = {
        "6078": {"name": "sarsia", "rank": "genus", "lineage": {"cnidaria", "hydrozoa"}},
        "555048": {"name": "sinodiaptomus sarsi", "rank": "species", "lineage": {"arthropoda", "copepoda", "sinodiaptomus"}},
    }
    monkeypatch.setattr(rfi, "fetch_taxid_info", lambda tax_id: info.get(str(tax_id)))

    mapping = {
        "RL_WRONG": {
            "raw_label": "RL_WRONG",
            "provenance": rfi.PROV_DRAFT,
            rfi.WIKIDATA_ID: "Q1",
            "aphia_ID": "117070.0",
            "NCBI_ID": "6078.0",
            "BOLD_ID": "159702.0",
            "cox": "cox:5",
            "_genus": "sinodiaptomus",
            "_phylum": "arthropoda",
        },
        "RL_OK": {
            "raw_label": "RL_OK",
            "provenance": rfi.PROV_DRAFT,
            rfi.WIKIDATA_ID: "Q2",
            "aphia_ID": "355669.0",
            "NCBI_ID": "555048.0",
            "BOLD_ID": "716607.0",
            "cox": "cox:3",
            "_genus": "sinodiaptomus",
            "_phylum": "arthropoda",
        },
        "RL_REUSED": {
            "raw_label": "RL_REUSED",
            "provenance": "reused:zoolake",
            rfi.WIKIDATA_ID: "Q3",
            "aphia_ID": "1.0",
            "NCBI_ID": "6078.0",
            "BOLD_ID": "3.0",
            "cox": "",
            "_genus": "sarsia",
            "_phylum": "cnidaria",
        },
    }
    blanked = rfi.verify_ncbi_lineage(mapping)

    assert blanked == ["RL_WRONG"]
    assert mapping["RL_WRONG"]["provenance"] == rfi.PROV_GUARD_BLANKED
    assert all(mapping["RL_WRONG"][c] == "" for c in rfi.ID_FIELDS)
    # The correct taxid survives, and a reused (trusted) row is never blanked.
    assert mapping["RL_OK"]["provenance"] == rfi.PROV_DRAFT
    assert mapping["RL_REUSED"]["provenance"] == "reused:zoolake"


# ── too-coarse KI-6 blanking ───────────────────────────────────────────────────────


def test_blank_coarse_matches_scope_and_idempotent():
    """Only Order/Class/Phylum/Kingdom draft matches are blanked; re-running is a no-op."""
    mapping = {
        "RL_ORDER": {
            "raw_label": "RL_ORDER",
            "provenance": rfi.PROV_DRAFT,
            "matched_rank": "Order",
            rfi.WIKIDATA_ID: "Q1",
            "aphia_ID": "1.0",
            "NCBI_ID": "2.0",
            "BOLD_ID": "3.0",
            "cox": "",
        },
        "RL_FAMILY": {
            "raw_label": "RL_FAMILY",
            "provenance": rfi.PROV_DRAFT,
            "matched_rank": "Family",
            rfi.WIKIDATA_ID: "Q2",
            "aphia_ID": "4.0",
            "NCBI_ID": "5.0",
            "BOLD_ID": "6.0",
            "cox": "",
        },
        "RL_SPECIES": {
            "raw_label": "RL_SPECIES",
            "provenance": rfi.PROV_DRAFT,
            "matched_rank": "Species",
            rfi.WIKIDATA_ID: "Q3",
            "aphia_ID": "7.0",
            "NCBI_ID": "8.0",
            "BOLD_ID": "9.0",
            "cox": "",
        },
    }
    blanked = rfi.blank_coarse_matches(mapping)
    assert blanked == ["RL_ORDER"]
    assert mapping["RL_ORDER"]["provenance"] == rfi.PROV_BLANKED
    assert all(mapping["RL_ORDER"][c] == "" for c in rfi.ID_FIELDS)
    # Family- and species-rank matches ship (Family is below the blank threshold).
    assert mapping["RL_FAMILY"][rfi.WIKIDATA_ID] == "Q2"
    assert mapping["RL_SPECIES"][rfi.WIKIDATA_ID] == "Q3"

    # State-idempotent: a second pass leaves the mapping byte-identical. (The blanked
    # raw_label is re-reported because the guard also matches already-blanked rows so a
    # re-run keeps their cells empty — but nothing in the mapping changes.)
    snapshot = copy.deepcopy(mapping)
    again = rfi.blank_coarse_matches(mapping)
    assert again == ["RL_ORDER"]
    assert mapping == snapshot


# ── numeric id formatting ──────────────────────────────────────────────────────────


def test_format_numeric_id():
    """Integer-valued ids normalise to the CSV's ``<int>.0`` float format; blanks stay blank."""
    assert rfi.format_numeric_id("106265") == "106265.0"
    assert rfi.format_numeric_id(106265) == "106265.0"
    assert rfi.format_numeric_id("106265.0") == "106265.0"
    assert rfi.format_numeric_id("") == ""
    assert rfi.format_numeric_id(None) == ""


# ── CSV backfill: frepj-rows-only, four-cells-only, idempotent ─────────────────────

# ── Backfill: identifiers land on the concept, not on the row ───────────────────


@pytest.fixture
def package(tmp_path):
    """A scratch copy of the shipped taxonomy package."""
    work = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, work)
    loader.cache_clear()
    yield work
    loader.cache_clear()


def test_backfill_restores_one_cleared_id_and_touches_nothing_else(package, tmp_path):
    """The re-expression of the line rewriter this replaced, on the real package.

    ``backfill_csv`` split every ``frepj,`` line on its last five commas and rebuilt four fields of
    it. Ids are a property of the concept, not of the row that names it, so they now go on the
    taxon — and the test says so the only way that means anything: clear one id, backfill, and see
    that one identifier row come back with every other byte of the package where it was.
    """
    csv_path = tmp_path / "planktonzilla_taxonomy.csv"
    csv_path.write_bytes(Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME).read_bytes())
    committed = csv_path.read_bytes()

    mapping = rfi.parse_summary()
    raw_label, record = next((label, values) for label, values in mapping.items() if values.get("aphia_ID"))
    taxon_id = next(
        row["taxonID"] for row in read_tsv(package / "mappings" / "frepj.tsv") if row["verbatimIdentification"] == raw_label
    )

    cleared = write.clear_id(package, taxon_id, "aphia_ID", "test: pretend WoRMS withdrew it", apply=True)
    assert len(cleared) == 1
    fingerprint = {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()}

    changes = rfi.backfill_ids(mapping, package_dir=package, csv_path=csv_path, apply=True)

    assert [(change.table, change.key) for change in changes.changes] == [
        ("identifier", f"{taxon_id}|worms:{record['aphia_ID'].removesuffix('.0')}")
    ]
    moved = [path.name for path, blob in fingerprint.items() if path.read_bytes() != blob]
    assert moved == ["identifier.tsv"], f"the backfill touched {moved}"
    assert csv_path.read_bytes() == committed, "the rendered CSV is not back to the committed bytes"


def test_backfill_on_an_unchanged_package_is_a_no_op(package, tmp_path):
    """Every id the summary carries is already on its concept, so a re-run writes nothing."""
    csv_path = tmp_path / "planktonzilla_taxonomy.csv"
    csv_path.write_bytes(Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME).read_bytes())

    assert not rfi.backfill_ids(rfi.parse_summary(), package_dir=package, csv_path=csv_path)


def test_backfill_never_clears_an_id_the_summary_leaves_blank(package):
    """Blanking by omission is what produced the 13 legacy per-row overrides.

    The old rewriter wrote whatever the summary held into the cell, blank included. This one leaves
    an authority the summary says nothing about exactly where it was; ``clear_id`` is the only way.
    """
    before = read_tsv(package / "identifier.tsv")
    blanked = {label: dict.fromkeys(values, "") for label, values in rfi.parse_summary().items()}

    assert not rfi.backfill_ids(blanked, package_dir=package, apply=True)
    assert read_tsv(package / "identifier.tsv") == before


def test_backfill_refuses_a_class_directory_frepj_does_not_map(package):
    """A key that names nothing is a summary that has drifted from the mappings, not a row to skip."""
    with pytest.raises(TaxonomyError, match="does not map"):
        rfi.backfill_ids({"not a frepj class dir": {"aphia_ID": "1"}}, package_dir=package)
