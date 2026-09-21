"""
(c) Inria

Per-source coverage, generalised — step 5.5a of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``.

**The taxonomy join is a silent left join.** An image whose ``(Dataset, Raw_Labels)`` pair has no
mapping row does not fail the build: it gets sixteen nulls and the run carries on.
``make_planktonzilla.check_taxonomy_csv`` is per-SOURCE and non-blocking, so a source with 44 class
directories and 3 rows passes it clean. One mistyped class name costs a few thousand untaxonomised
images with nothing red anywhere.

Two sources were guarded against that — ``frepj`` and ``daplankton`` — each by its own hand-written
module. This is the one test both of those coverage checks become, parameterised over whatever
frozen evidence exists, plus the four Tara Pacific sources that had evidence and no coverage test at
all. **Six of twenty-one sources, 873 of 2358 rows.**

What is deliberately NOT adopted is FREPJ's *mechanism*. Its taxonomy half is 1,709 lines of
source-specific curation and Tara Pacific's is another 1,019 written to a different design; cloning
that per source is how the repository came to have three precedents with three methods, and a fourth
would be a fourth dialect with its own bugs. One generic test, not eighteen builders.

**A frozen list is only worth having if it is independent of the CSV.** Deriving one from the table
asserts the table against itself and would pass on any table at all. All three sources here have
external evidence:

===================================  =====  ==========================================
``tests/fixtures/frepj/…``             229  an archive scan, with per-magnification counts
``tests/fixtures/daplankton/…``         44  an archive scan, lab and sea splits
``planktonzilla/dataset_import/…``     600  an EcoTaxa export, four sources
===================================  =====  ==========================================

The other fifteen sources have none, and ``samples.json`` cannot supply it: it records the
*harmonised* ``proposed_label``, not the raw class directory. Minting lists for those is step 5.5b, a
one-time scan of the published artifact — a network job, not a code job. The gap is pinned below so
that landing it turns this module red rather than passing silently at 37 % forever.
"""

import csv
import shutil
import zipfile
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest

from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR, loader
from planktonzilla.planktonzilla_dataset.taxonomy.model import MAPPING_COLUMNS, read_tsv, write_tsv

FIXTURES = Path(__file__).parent / "fixtures"
TARA_CLASSES = Path(root) / "planktonzilla" / "dataset_import" / "tara_pacific_classes.tsv"

# The sources with no independent evidence, pinned so 5.5b landing one is a failing test rather
# than a silent no-op. Was 15; `lensless` moved into FROZEN_CLASS_DIRS below, because it is the one
# source whose archive is COMMITTED — the other fourteen are fetched from upstream, which is why
# 5.5b needs a network scan for them and not for this one.
SOURCES_WITHOUT_EVIDENCE = (
    "flowcamnet",
    "global_uvp5",
    "isiisnet",
    "jedioceans",
    "medplanktonset",
    "planktonset1.0",
    "planktoscope",
    "syke_ifcb_2022",
    "sykezooscan2024",
    "uvp6net",
    "whoi",
    "zoocamnet",
    "zoolake",
    "zooscan",
)


def _first_column(path):
    """Class dirs from a one-source TSV whose first column holds them.

    Split on tab and take field 0 rather than parsing as CSV: a FREPJ class dir is itself a
    comma-separated five-field path, and a reader that treats commas as delimiters silently
    shortens all 229 of them.
    """
    return {line.split("\t")[0] for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]}


def _lensless_dirs():
    """Class dirs for `lensless`, read from the bundled archive the importer itself extracts.

    Stronger evidence than a frozen fixture: this is the source of truth, not a transcription of
    it, so a re-bundled archive that gained or lost a class turns the coverage test red instead of
    agreeing with a stale copy. The layout is `lensless_dataset/{TRAIN,TEST}_IMAGE/<class>/<file>`
    — LenslessDatasetImporter renames the two wrappers to train/ and test/, leaving the class dirs
    where they are, so level 2 is the class name. Byte-exact, double spaces and all
    (`PARAMECIUM  BURSARIA` is a real class dir).
    """
    archive = Path(root) / "planktonzilla" / "dataset_import" / "public_data" / "lensless_dataset.zip"
    with zipfile.ZipFile(archive) as bundle:
        parts = (name.split("/") for name in bundle.namelist())
        return {chunk[2] for chunk in parts if len(chunk) > 3 and chunk[2]}


def _tara_column(dataset):
    """Class dirs for one Tara Pacific source, from the EcoTaxa export covering all four."""
    with TARA_CLASSES.open(newline="", encoding="utf-8") as handle:
        return {row["class_dir"] for row in csv.DictReader(handle, delimiter="\t") if row["dataset"] == dataset}


# dataset -> (how to read its frozen list, the count that evidence recorded).
# The count is spelled out so a truncated or re-generated fixture is a failure here rather than a
# quietly smaller contract that still passes set equality against a quietly smaller table.
FROZEN_CLASS_DIRS = {
    "frepj": (lambda: _first_column(FIXTURES / "frepj" / "frepj_class_dirs.tsv"), 229),
    "lensless": (_lensless_dirs, 10),
    "daplankton": (lambda: _first_column(FIXTURES / "daplankton" / "daplankton_class_dirs.tsv"), 44),
    "tara_pacific_bongo": (lambda: _tara_column("tara_pacific_bongo"), 137),
    "tara_pacific_decknet": (lambda: _tara_column("tara_pacific_decknet"), 132),
    "tara_pacific_hsn": (lambda: _tara_column("tara_pacific_hsn"), 159),
    "tara_pacific_manta": (lambda: _tara_column("tara_pacific_manta"), 172),
}


def coverage(store, dataset, frozen=None):
    """``(missing, extra)`` for one source: class dirs with no mapping row, and rows with no class dir.

    Both directions matter and they fail differently. A **missing** class dir is the silent left
    join — thousands of images published with sixteen nulls. An **extra** row is a mapping that will
    never match anything, which does not corrupt the dataset but does rot: it goes on being
    maintained, re-verified against authorities, and counted, for a class directory that is gone.

    Args:
        store: The loaded taxonomy.
        dataset: The source to check.
        frozen: The class dirs to check against; the registry's own list when omitted. Passed
            explicitly by the gates below, which need a list that disagrees with the table.
    """
    if frozen is None:
        frozen = FROZEN_CLASS_DIRS[dataset][0]()
    mapped = set(store.labels_for(dataset))
    return frozen - mapped, mapped - frozen


@pytest.fixture(scope="module")
def store():
    return loader.load_taxonomy()


@pytest.mark.parametrize("dataset", sorted(FROZEN_CLASS_DIRS))
def test_every_frozen_class_directory_has_a_mapping_row(store, dataset):
    """THE GUARD, over every source whose class dirs are independently known.

    Byte-exact set equality in both directions. The class dirs carry the source's own spelling,
    correct or not — ``Kryptoperidium_foliaceum`` is missing an ``-in-`` and stays that way, because
    the label is a join key against an image folder and not a place to fix names.
    """
    missing, extra = coverage(store, dataset)

    assert not missing, f"{dataset}: {len(missing)} class dir(s) would publish as null taxonomy: {sorted(missing)[:5]}"
    assert not extra, f"{dataset}: {len(extra)} mapping row(s) match no class directory: {sorted(extra)[:5]}"


@pytest.mark.parametrize("dataset", sorted(FROZEN_CLASS_DIRS))
def test_the_frozen_list_is_the_size_its_evidence_recorded(store, dataset):
    """A fixture that shrank is not a contract that was met; set equality alone would not see it."""
    frozen = FROZEN_CLASS_DIRS[dataset][0]()
    expected = FROZEN_CLASS_DIRS[dataset][1]

    assert len(frozen) == expected
    assert len(store.labels_for(dataset)) == expected


def test_an_unmapped_class_directory_turns_this_red(store):
    """The gate, exercised rather than asserted.

    A coverage test nobody has seen fail is a coverage test nobody knows is wired up. The realistic
    failure is an archive that grows a class the curator has not mapped — the frozen list names it,
    the table does not — so that is what this does, rather than deleting a committed row.
    """
    added = "Thalassiosira_baltica"
    assert added not in store.labels_for("daplankton"), "pick a class dir the table really does not map"
    grown = FROZEN_CLASS_DIRS["daplankton"][0]() | {added}

    missing, extra = coverage(store, "daplankton", grown)

    assert missing == {added}
    assert not extra
    assert coverage(store, "frepj") == (set(), set()), "an unrelated source was implicated"


def test_a_mapping_row_for_a_class_directory_that_does_not_exist_turns_this_red(tmp_path):
    """The other direction. A row nobody can reach is not harmless — it is maintained forever.

    Written on the package rather than on the frozen list, because this is the direction a curator
    causes: a row added or kept for a class directory that was renamed upstream. It has to be a
    POST-FREEZE row — one the v1.0 manifest does not name — since a published row that vanishes is
    caught two steps earlier, by the release pin in the loader.
    """
    package = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, package)
    loader.cache_clear()

    path = package / "mappings" / "daplankton.tsv"
    rows = read_tsv(path)
    write_tsv(path, MAPPING_COLUMNS, [*rows, {**rows[0], "verbatimIdentification": "Class_that_was_renamed_upstream"}])

    try:
        missing, extra = coverage(loader.load_taxonomy(package), "daplankton")
        assert extra == {"Class_that_was_renamed_upstream"}
        assert not missing
    finally:
        loader.cache_clear()


def test_a_published_row_that_vanishes_is_caught_before_coverage_ever_runs(tmp_path):
    """Coverage is the second line here, not the first, and it is worth knowing which is which.

    Deleting a row the v1.0 release published does not reach this module: the loader refuses the
    package outright, because the frozen row-order manifest names a key no mapping file carries.
    Coverage catches the case the release layer cannot — a class directory that was never published
    and never mapped.
    """
    from planktonzilla.planktonzilla_dataset.taxonomy import TaxonomyError

    package = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, package)
    loader.cache_clear()

    path = package / "mappings" / "daplankton.tsv"
    write_tsv(path, MAPPING_COLUMNS, read_tsv(path)[1:])

    try:
        with pytest.raises(TaxonomyError, match="which no mapping file carries"):
            loader.load_taxonomy(package)
    finally:
        loader.cache_clear()


# The gap, pinned
def test_the_sources_with_no_independent_evidence_are_exactly_the_fourteen_named(store):
    """The covered share, and the rest named rather than left as an implication.

    Pinned in both directions on purpose. Minting a frozen list for one of these (step 5.5b) turns
    this red, which is the prompt to add it to ``FROZEN_CLASS_DIRS`` — the alternative is a registry
    that quietly never grows and a coverage suite that reports its share as if it were done. It has
    now done that once: `lensless` moved across when its committed archive was read as evidence,
    and this test is the thing that made the move explicit instead of silent.

    The fourteen that remain all fetch their archives from upstream, which is the whole of what
    5.5b still needs — a network scan, not more local reading.
    """
    covered = set(FROZEN_CLASS_DIRS)
    assert set(store.datasets()) == covered | set(SOURCES_WITHOUT_EVIDENCE)
    assert covered.isdisjoint(SOURCES_WITHOUT_EVIDENCE)
    assert len(SOURCES_WITHOUT_EVIDENCE) == 14

    guarded = sum(len(store.labels_for(dataset)) for dataset in covered)
    unguarded = sum(len(store.labels_for(dataset)) for dataset in SOURCES_WITHOUT_EVIDENCE)
    assert (guarded, unguarded) == (883, 1475)


def test_the_evidence_is_independent_of_the_table_it_checks(store):
    """The property that makes any of this mean anything, stated where it can be read.

    Each frozen list came from outside the taxonomy CSV — two archive scans and an EcoTaxa export.
    A list derived FROM the table would make every assertion above a tautology that passes on a
    table with every row wrong. This cannot prove provenance, but it can prove the lists are not
    simply the table's own labels: they carry columns the taxonomy has no idea about.
    """
    frepj_header = (FIXTURES / "frepj" / "frepj_class_dirs.tsv").read_text(encoding="utf-8").split("\n", 1)[0]
    daplankton_header = (FIXTURES / "daplankton" / "daplankton_class_dirs.tsv").read_text(encoding="utf-8").split("\n", 1)[0]
    tara_header = TARA_CLASSES.read_text(encoding="utf-8").split("\n", 1)[0]

    # Per-magnification image counts, per-instrument image counts, and an EcoTaxa node id. None of
    # the three is derivable from the taxonomy table.
    assert "n_40x" in frepj_header and "n_100x" in frepj_header
    assert "n_lab_ifcb" in daplankton_header and "n_sea_ifcb" in daplankton_header
    assert "ecotaxa_taxon_id" in tara_header
