"""
(c) Inria

Network-free tests pinning the 600 Tara Pacific taxonomy rows appended to
``planktonzilla_taxonomy.csv`` by ``build_tara_pacific_taxonomy.py``.

Two things are guarded here that nothing else can guard:

  (a) COVERAGE — the ``Raw_Labels`` of each source are byte-exactly the class dirs its
      importer will create. ``build_taxonomy_lookup`` is a LEFT join: a label present in
      the imagefolder but absent from the CSV silently yields a row with a null taxonomy,
      so set equality here is the only thing standing between a renamed class dir and
      600 000 unlabelled images.

  (b) INVARIANTS — the appended block obeys every rule the pre-existing table obeys
      (lowercase normalized columns, ``X.0``-formatted numeric IDs, a recognized
      qualifier, one lineage and one ID value per ``proposed_label``, no hole in the rank
      ladder), and the rows that came before it are still byte-frozen.

Plus the curation engine's own contract: it is deterministic and idempotent, so
regenerating the block leaves the CSV byte-identical.

Offline BY CONSTRUCTION: reads only the committed class map, the committed frozen EcoTaxa
taxon table and the committed CSV. No HTTP.
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

from planktonzilla.dataset_import import tara_pacific_layout as layout
from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.utils import build_tara_pacific_taxonomy as builder

RANKS = constants.TAXONOMY_RANKS
NORMALIZED_COLUMNS = (*RANKS, "proposed_label", "root_class", "qualifier")

# The pristine CSV (header + 1485 rows) plus the 229 frepj rows and the 44 daplankton
# rows is exactly 1759 lines; the Tara Pacific rows are strictly appended after them.
LINES_BEFORE_TARA_PACIFIC = 1759
EXPECTED_ROWS = 600


@pytest.fixture(scope="module")
def all_rows():
    with constants.DEFAULT_TAXONOMY_CSV_FILENAME.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def tara_rows(all_rows):
    return [row for row in all_rows if row["Dataset"] in builder.DATASET_NAMES]


# --- (a) Coverage ---------------------------------------------------------------------


def test_the_block_has_one_row_per_committed_class_dir(tara_rows):
    assert len(tara_rows) == EXPECTED_ROWS == len(builder.read_class_map())


@pytest.mark.parametrize("source_name", builder.DATASET_NAMES)
def test_raw_labels_match_the_class_dirs_the_importer_will_create(tara_rows, source_name):
    """(a) The guard against ``build_taxonomy_lookup``'s silent left join."""
    labels = [row["Raw_Labels"] for row in tara_rows if row["Dataset"] == source_name]

    # A duplicate join key would silently over- or under-count examples.
    assert len(labels) == len(set(labels))
    # Byte-exact set equality with the map the IMPORTER reads — one file, no second list.
    assert set(labels) == set(layout.class_dirs(source_name))


def test_every_row_has_a_proposed_label(tara_rows):
    assert [row["Raw_Labels"] for row in tara_rows if not (row["proposed_label"] or "").strip()] == []


def test_the_four_sources_are_registered_everywhere(tara_rows):
    """A source in the CSV but not in the constants would raise at build time."""
    assert {row["Dataset"] for row in tara_rows} == set(builder.DATASET_NAMES)
    for name in builder.DATASET_NAMES:
        assert name in constants.DATASET_IMPORT_CONFIGS
        assert constants.license_fields(name) == {"license": layout.LICENSE, "license_url": layout.LICENSE_URL}


# --- (b) Invariants -------------------------------------------------------------------


def test_normalized_columns_are_lowercase(tara_rows):
    """KI-9: 'Eukaryota' is the ONLY uppercase value in the table, and it predates these."""
    offenders = [
        (row["Raw_Labels"], col, row[col]) for row in tara_rows for col in NORMALIZED_COLUMNS if row[col] != row[col].lower()
    ]
    assert offenders == []


def test_numeric_ids_are_float_formatted(tara_rows):
    """KI-12: the table stores integer IDs as ``X.0``; a bare int would break that test."""
    float_int = re.compile(r"^\d+\.0+$")
    for col in ("aphia_ID", "NCBI_ID", "BOLD_ID"):
        bad = [(row["Raw_Labels"], row[col]) for row in tara_rows if row[col].strip() and not float_int.match(row[col].strip())]
        assert bad == [], f"{col}: {bad}"

    wikidata = [row["wikidata_ID"].strip() for row in tara_rows if row["wikidata_ID"].strip()]
    assert wikidata, "no wikidata IDs were inherited at all — the reuse rules stopped working"
    assert all(re.match(r"^Q\d+$", value) for value in wikidata)


def test_qualifiers_are_all_recognized(tara_rows):
    """KI-11: an unrecognized qualifier means the vocabulary constant needs updating."""
    seen = {row["qualifier"].strip() for row in tara_rows}
    assert not seen - set(constants.QUALIFIERS) - {""}


def test_living_agrees_with_root_class(tara_rows):
    mismatched = [row["Raw_Labels"] for row in tara_rows if (row["living"] == "True") != (row["root_class"] == "living")]
    assert mismatched == []


def test_root_class_uses_the_existing_vocabulary(all_rows, tara_rows):
    """No new root_class value: the four are living / detritus / artefact / inert."""
    existing = {row["root_class"] for row in all_rows if row["Dataset"] not in builder.DATASET_NAMES}
    assert {row["root_class"] for row in tara_rows} <= existing


def test_no_row_skips_a_taxonomic_rank(tara_rows):
    """A hole in the ladder silently skips a rank in the published label graph."""
    offenders = []
    for row in tara_rows:
        values = [(row[rank] or "").strip() for rank in RANKS]
        first_gap = next((index for index, value in enumerate(values) if not value), len(values))
        if any(values[first_gap:]):
            offenders.append((row["Raw_Labels"], values))
    assert offenders == []


def test_each_proposed_label_keeps_one_lineage_across_the_whole_table(all_rows):
    lineages = defaultdict(set)
    for row in all_rows:
        label = row["proposed_label"].strip().lower()
        if label:
            lineages[label].add(tuple(row[rank].strip().lower() for rank in RANKS))
    conflicting = {label: sorted(values) for label, values in lineages.items() if len(values) > 1}
    assert not conflicting


def test_no_taxon_acquires_a_second_external_id(all_rows):
    """The reuse rules exist so a label cannot carry two aphia/NCBI/BOLD values."""
    for col in ("wikidata_ID", "aphia_ID", "NCBI_ID", "ecotaxa_ID", "BOLD_ID"):
        by_label = defaultdict(set)
        for row in all_rows:
            label, value = row["proposed_label"].strip().lower(), row[col].strip()
            if label and value:
                by_label[label].add(value)
        conflicting = {label: sorted(values) for label, values in by_label.items() if len(values) > 1}
        assert not conflicting, f"{col}: {conflicting}"


def test_the_microplastic_classes_are_inert_and_separable(tara_rows):
    """Anthropogenic material must be separable from both the organisms and the detritus.

    The Manta deposit is the one that SETS OUT to sample microplastics, and it carries the
    named morphologies. Two more turn up under bongo: EcoTaxa hangs its `rods` node from
    `not-living>plastic>other`, and the Bongo annotators used it — so those rows follow the
    annotators' own classification rather than being re-read as imaging artefacts.
    """
    inert = {(row["Dataset"], row["proposed_label"]) for row in tara_rows if row["root_class"] == "inert"}
    assert inert, "the Manta plastics project contributed no inert rows"

    manta_labels = {label for dataset, label in inert if dataset == "tara_pacific_manta"}
    assert manta_labels >= {"plastic film", "plastic fragment", "plastic pellet", "plastic fiber", "polystyrene"}

    # Never plankton, never living, whichever source they came from.
    for row in tara_rows:
        if row["root_class"] == "inert":
            assert (row["plankton"], row["living"]) == ("False", "False")
        # ...and the converse: nothing on the plastic branch was read as an organism.
        if row["proposed_label"].startswith("plastic") or row["proposed_label"] == "polystyrene":
            assert row["root_class"] == "inert"


def test_the_rows_before_the_block_are_untouched():
    """Append-only: the frozen base and the frepj block still start the file."""
    lines = constants.DEFAULT_TAXONOMY_CSV_FILENAME.read_text(encoding="utf-8").splitlines()
    assert len(lines) == LINES_BEFORE_TARA_PACIFIC + EXPECTED_ROWS
    assert not any(
        line.startswith(tuple(f"{name}," for name in builder.DATASET_NAMES)) for line in lines[:LINES_BEFORE_TARA_PACIFIC]
    )
    assert all(
        line.startswith(tuple(f"{name}," for name in builder.DATASET_NAMES)) for line in lines[LINES_BEFORE_TARA_PACIFIC:]
    )


# --- The curation engine --------------------------------------------------------------


def test_rebuilding_reproduces_the_committed_block_exactly(tara_rows):
    """Deterministic and idempotent: a re-run must leave the CSV byte-identical."""
    rows, _ = builder.build_rows(builder.read_class_map(), builder.read_taxa(), builder.read_master_csv())

    assert len(rows) == len(tara_rows)
    for built, committed in zip(rows, tara_rows):
        assert built == {column: committed[column] for column in builder.CSV_COLUMNS}


def test_every_morphology_token_has_a_rule():
    """A token with no rule falls through to 'artefact', which is a curation decision the
    engine must never make silently."""
    taxa = builder.read_taxa()
    unruled = sorted(
        {
            taxon["name"]
            for entry in builder.read_class_map()
            for taxon in [taxa[entry["taxon_id"]]]
            if taxon["type"] == "M" and builder._branch(taxon) is None and taxon["name"] not in builder.MORPH_RULES
        }
    )
    assert unruled == []


def test_the_rank_gap_guard_rejects_a_hole():
    """The guard that keeps the published label graph walkable."""
    good = [{"Dataset": "d", "Raw_Labels": "x", **{rank: "a" for rank in RANKS}}]
    builder._assert_no_rank_gaps(good)

    holed = [{"Dataset": "d", "Raw_Labels": "x", **{rank: "a" for rank in RANKS}}]
    holed[0]["Order"] = ""
    with pytest.raises(ValueError, match="skip a taxonomic rank"):
        builder._assert_no_rank_gaps(holed)


def test_the_reconciliation_report_is_committed_and_current():
    """The human-verify record of how the derived rows were decided."""
    report = builder.DEFAULT_RECONCILIATION_MD
    assert report.is_file()

    text = report.read_text(encoding="utf-8")
    assert "Tara Pacific taxonomy reconciliation" in text
    # Every hand-made departure from upstream is named in the report, not just in code.
    for class_dir in builder.HOMONYM_NOTES:
        assert class_dir in text
    for label in builder.RANK_GAP_FILLS:
        assert label in text
    # ...including the rank-name decisions, which cover 16 rows between them (finding 1.7).
    for rank, ecotaxa_name, table_name in builder.RANK_DEPARTURES:
        assert f"**{rank}** — EcoTaxa `{ecotaxa_name}`, this table `{table_name}`" in text


def _lineage_departures():
    """Every built row that disagrees with EcoTaxa's tree, at any of the seven ranks.

    Returns ``{class_dir: [(rank, ecotaxa_name, table_name), ...]}``. Only ranks BOTH sides
    populate are compared: EcoTaxa leaving a rank blank is a gap, not a disagreement, and
    the deeper ranks a row carries beyond its anchor are covered by the rank-gap guard.
    """
    taxa = builder.read_taxa()
    rows, _ = builder.build_rows(builder.read_class_map(), taxa, builder.read_master_csv())

    departures = {}
    for row in rows:
        class_dir = row["Raw_Labels"]
        matches = [taxon for taxon in taxa.values() if taxon["display_name"] == class_dir]
        if not matches:
            continue
        derived = builder._derived_ranks(builder.anchor_taxon(matches[0], taxa), taxa)
        diffs = [
            (rank, derived[rank], row[rank])
            for rank in RANKS
            if derived.get(rank) and row.get(rank) and derived[rank] != row[rank]
        ]
        if diffs:
            departures[class_dir] = diffs
    return departures


def test_every_departure_from_the_ecotaxa_lineage_is_documented():
    """A row may depart from EcoTaxa's tree — but only in a way that is written down.

    This compares all seven rank columns. It used to compare `Kingdom` alone, which saw 3
    departures and asserted they were the whole story; there are 19 (docs/CODE_REVIEW.md
    finding 1.7). The 16 it could not see are the one-lineage-per-label rule working as
    designed — the `anchor` / `verbatim` / inherit-higher-ranks paths take the spelling the
    master CSV already uses — but with no record of that, a NEW disagreement anywhere below
    Kingdom was indistinguishable from a deliberate one.

    Two records, because there are two kinds of departure:

      * ``HOMONYM_NOTES`` — this row's WHOLE lineage diverges, a homonym resolved differently.
        Such a row may differ at any rank; the note covers the row.
      * ``RANK_DEPARTURES`` — one rank NAME is spelled differently in this table, wherever it
        appears, keyed by the ``(rank, EcoTaxa name, table name)`` triple so that one decision
        covers however many rows inherit it.
    """
    undocumented = {
        class_dir: [triple for triple in diffs if triple not in builder.RANK_DEPARTURES]
        for class_dir, diffs in _lineage_departures().items()
        if class_dir not in builder.HOMONYM_NOTES
    }
    undocumented = {class_dir: diffs for class_dir, diffs in undocumented.items() if diffs}
    assert not undocumented, (
        "rank(s) departing from EcoTaxa with no entry in RANK_DEPARTURES — document the "
        f"decision or fix the lineage: {undocumented}"
    )


def test_no_documented_departure_has_gone_stale():
    """Both records must still describe the data, or they are decoration.

    A `HOMONYM_NOTES` row that no longer departs, or a `RANK_DEPARTURES` triple nothing
    exercises, means the lineage changed and the note outlived it.
    """
    departures = _lineage_departures()

    silent_homonyms = sorted(set(builder.HOMONYM_NOTES) - set(departures))
    assert not silent_homonyms, f"HOMONYM_NOTES entries that no longer depart: {silent_homonyms}"

    exercised = {triple for diffs in departures.values() for triple in diffs}
    stale = sorted(set(builder.RANK_DEPARTURES) - exercised)
    assert not stale, f"RANK_DEPARTURES entries matching no current departure — delete them: {stale}"


def test_the_shipped_departures_are_exactly_what_was_reviewed():
    """Pins the measured state, so a change in the count cannot pass unnoticed.

    19 rows depart, not the 3 the Kingdom-only comparison reported: 3 whole-lineage homonyms
    plus 16 rows across the 7 rank-name decisions.
    """
    departures = _lineage_departures()
    assert len(departures) == 19
    assert sorted(class_dir for class_dir in departures if class_dir in builder.HOMONYM_NOTES) == sorted(builder.HOMONYM_NOTES)
    assert len([class_dir for class_dir in departures if class_dir not in builder.HOMONYM_NOTES]) == 16
    assert len(builder.RANK_DEPARTURES) == 7
