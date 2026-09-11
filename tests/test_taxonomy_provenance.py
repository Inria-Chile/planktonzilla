"""
(c) Inria

Provenance as data — step 7 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``.

"Why does this row claim this?" used to be answerable by reading two Markdown reports, written by
two builders in two styles, covering two of twenty-one sources. Both builders had already COMPUTED
the answer per row — which rule decided it, which row donated its identifiers, which anchor it hung
from — and then thrown it into prose. This is that computation landing in the columns the mapping
table already had and had always left empty.

Three things are pinned here, and the third is the one that matters most:

1. **The vocabulary is controlled.** Six terms, in ``vocab/method.tsv``, declared in the descriptor.
   Free text would make 2,358 answers incomparable, which is the same as having none.
2. **No published byte moves.** Provenance is curation, not publication; the 19-column CSV has no
   column for any of it and must render identically.
3. **The gap is measured.** 829 rows of 2,358 carry a method and 1,529 do not, and the second number
   is asserted rather than implied. A blank is not a term in the vocabulary and never becomes one:
   writing ``unknown`` on those rows would dress an absence up as provenance.
"""

import shutil
from collections import Counter
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR, TaxonomyError, loader, migrate, validate
from planktonzilla.planktonzilla_dataset.taxonomy.model import BROAD_MATCH, EXACT_MATCH, METHODS, read_tsv, write_tsv

MAPPINGS = PACKAGE_DIR / "mappings"

# Measured on the committed table. The second number is the point of the first.
ROWS_WITH_A_METHOD = 829
ROWS_WITHOUT = 1529


@pytest.fixture(scope="module")
def mapping_rows():
    return [row for path in sorted(MAPPINGS.glob("*.tsv")) for row in read_tsv(path)]


@pytest.fixture
def package(tmp_path):
    work = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, work)
    loader.cache_clear()
    yield work
    loader.cache_clear()


# 1. A controlled vocabulary
def test_every_recorded_method_is_a_vocabulary_term(mapping_rows):
    """Free text here is the same as no provenance: the answers stop being comparable."""
    used = {row["method"] for row in mapping_rows if row["method"]}

    assert used <= METHODS, f"methods outside the vocabulary: {sorted(used - METHODS)}"


def test_the_vocabulary_file_is_the_vocabulary_the_code_enforces():
    """Two statements of one list drift with nothing to reconcile them — the descriptor's own rule."""
    committed = {row["method"] for row in read_tsv(PACKAGE_DIR / "vocab" / "method.tsv")}

    assert committed == METHODS


def test_a_donor_never_appears_without_a_method(mapping_rows):
    """Where a decision came from, with no answer for how, is half a record."""
    orphans = [row["verbatimIdentification"] for row in mapping_rows if row["donor"] and not row["method"]]

    assert orphans == []


def test_blank_is_not_a_term_and_never_becomes_one():
    """``unknown`` or ``migrated`` on 1529 rows would dress an absence up as provenance.

    The absence is recorded as a count instead — a number that can only go down, and that a reader
    can act on. A term meaning "we do not know" is a number that never moves.
    """
    assert "" not in METHODS
    assert not METHODS & {"unknown", "migrated", "none", "n/a"}


# 2. Provenance is not publication
def test_the_render_is_byte_identical_with_every_provenance_column_filled():
    """The 19-column CSV has no column for any of this, and must not notice."""
    assert loader.load_taxonomy().render_wide_csv() == Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME).read_bytes()


def test_the_package_still_validates_with_no_errors():
    """The descriptor is the schema of record, and it sanctioned the new terms before they landed.

    It caught them, in fact: seeding ``semapv:LogicalReasoning`` produced 449 errors until the
    justification enum was extended, which is the descriptor doing its job rather than a formality.
    """
    report = validate.apply_waivers(validate.validate(PACKAGE_DIR), validate.read_waivers(PACKAGE_DIR))
    errors = [finding for finding in report.findings if finding.severity == validate.SEVERITY_ERROR]

    assert errors == []


# 3. The gap, measured
def test_the_rows_that_can_and_cannot_be_answered_for_are_counted(mapping_rows):
    """37 % of sources have coverage evidence (step 5.5a); 35 % of ROWS have provenance. Both pinned.

    Two builders computed this and reported it; nineteen sources never had a builder, so nobody can
    say why their rows claim what they claim. That is a fact about the repository, and it belongs in
    a number rather than in a sentence somebody has to remember to update.
    """
    recorded = [row for row in mapping_rows if row["method"]]

    assert len(recorded) == ROWS_WITH_A_METHOD
    assert len(mapping_rows) - len(recorded) == ROWS_WITHOUT
    assert len(mapping_rows) == 2358


def test_the_two_sources_with_builders_are_the_two_with_provenance(mapping_rows):
    """No third source acquired provenance by accident, and neither of these two lost any."""
    by_source = Counter(row["datasetID"] for row in mapping_rows if row["method"])

    assert set(by_source) == {"frepj", "tara_pacific_bongo", "tara_pacific_decknet", "tara_pacific_hsn", "tara_pacific_manta"}
    assert by_source["frepj"] == 229
    assert sum(count for source, count in by_source.items() if source.startswith("tara_")) == 600


# The rank departures
def test_the_seven_rank_departures_sit_on_the_taxa_they_explain():
    """Prose in a dict in a builder is reachable by reading that file. A column is reachable.

    Each explains why one node is spelled differently from the register the Tara Pacific rows came
    from — a fact about the node, so it lives on the node.
    """
    from planktonzilla.planktonzilla_dataset.utils.build_tara_pacific_taxonomy import RANK_DEPARTURES

    store = loader.load_taxonomy()
    annotated = [taxon for taxon in store.taxa.values() if "departs from EcoTaxa" in taxon.remarks]

    assert len(annotated) == len(RANK_DEPARTURES) == 7
    for _rank, ecotaxa_name, table_name in RANK_DEPARTURES:
        node = next(taxon for taxon in annotated if taxon.scientific_name == table_name)
        assert f"«{ecotaxa_name}»" in node.remarks


def test_the_backfills_are_idempotent(package):
    """Both are pure functions of what is committed, so a second run writes nothing.

    The migration's own standard, and the reason these live beside it rather than in a scratch
    script somebody ran once and deleted.
    """
    assert migrate.backfill_rank_departures(package, apply=True) == []
    assert migrate.backfill_broad_matches(package, apply=True) == []


def test_a_rank_departure_naming_two_nodes_is_refused(package, monkeypatch):
    """A departure keyed by ``(rank, name)`` that matched two nodes would annotate the wrong one."""
    from planktonzilla.planktonzilla_dataset.utils import build_tara_pacific_taxonomy as builder

    monkeypatch.setitem(builder.RANK_DEPARTURES, ("Genus", "somewhere", "no such genus"), "prose")

    with pytest.raises(TaxonomyError, match="matches 0 taxa"):
        migrate.backfill_rank_departures(package)


# Broad matches
def test_the_coarse_identifiers_are_recorded_as_broad_matches():
    """A species carrying its genus's id is not an exact match; the register had nothing finer.

    Published unchanged either way — the wide CSV cannot say "broader than" — so this buys honesty
    in the package and a validator that reports only the 21 findings that are real.
    """
    predicates = Counter(row["predicate_id"] for row in read_tsv(PACKAGE_DIR / "identifier.tsv"))

    assert predicates[BROAD_MATCH] == 449
    assert predicates[EXACT_MATCH] == 3583

    report = validate.validate(PACKAGE_DIR)
    assert [f for f in report.findings if f.check == "coarse_identifier"] == []


def test_relabelling_a_real_collision_as_a_broad_match_is_refused(package):
    """The check that stops re-predication being a way to silence findings.

    A broad match claims the subject is narrower than the id's concept. That is only true if some
    exact-match holder of the id is one of its ancestors — and a cross-branch collision, by
    definition, has none.
    """
    rows = read_tsv(package / "identifier.tsv")
    collision = next(row for row in rows if row["subject_id"] == "pzt:001166" and row["object_id"].startswith("wikidata:"))
    collision["predicate_id"] = BROAD_MATCH
    write_tsv(package / "identifier.tsv", tuple(rows[0]), rows)

    report = validate.validate(package)
    findings = [f for f in report.findings if f.check == "unjustified_broad_match"]

    assert len(findings) == 1
    assert findings[0].severity == validate.SEVERITY_ERROR
    assert "pzt:001166" in findings[0].locator


def test_a_broad_match_still_publishes_its_identifier():
    """The point of doing this at all: the claim gets more honest, the dataset does not change.

    A reader who expects `broadMatch` to withdraw the id would be surprised — so it is stated. The
    19-column CSV has one cell per authority and no way to qualify it; withdrawing 449 published ids
    to express a nuance the format cannot carry would be a worse answer than publishing them.
    """
    store = loader.load_taxonomy()
    broad = [row for row in read_tsv(PACKAGE_DIR / "identifier.tsv") if row["predicate_id"] == BROAD_MATCH]
    subject, object_id = broad[0]["subject_id"], broad[0]["object_id"]
    value = object_id.partition(":")[2]

    published = store.identifiers[subject]
    assert any(value in values for values in published.values()), f"{object_id} vanished from the render"
