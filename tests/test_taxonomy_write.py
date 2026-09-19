"""
(c) Inria

The write side — step 6 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``.

This is the step the plan schedules alone, because it is the one that can lose data. Three builders
currently write the taxonomy by finding their own block in the byte stream and re-serialising the
file around it; one of them destroys 644 rows on a no-op re-run. ``write.py`` replaces that with a
keyed, per-source, dry-run-by-default API, and this module is the evidence that it holds.

The load-bearing assertion is the first one, and it is measured on the real committed package rather
than on a fixture: **re-deriving all 21 sources from what is committed and upserting them back is a
zero-change no-op.** That single property is what the byte-splicing path never had, and it is what
makes a builder safe to re-run. Everything after it is one named failure mode each — an id blanked
by omission, a row dropped by omission, a spreadsheet boolean, a retirement without a tombstone.
"""

import shutil
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest

from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR, TaxonomyError, loader, write
from planktonzilla.planktonzilla_dataset.taxonomy.model import MAPPING_COLUMNS, read_tsv, write_tsv

COMMITTED_SOURCES = 21
COMMITTED_ROWS = 2358
COMMITTED_TAXA = 1352


@pytest.fixture
def package(tmp_path):
    """A scratch copy of the shipped package. Every test here writes, so none may share one."""
    work = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, work)
    loader.cache_clear()
    yield work
    loader.cache_clear()


def records_for(package_dir, dataset):
    """Re-derive one source's records from what is committed — the input a ported builder produces."""
    store = loader.load_taxonomy(package_dir)
    out = []
    for row in read_tsv(package_dir / "mappings" / f"{dataset}.tsv"):
        node = store.taxon(row["taxonID"])
        lineage = () if node.kind == "bucket" else tuple((n.rank, n.scientific_name) for n in store.ancestors(node.taxon_id))
        record = {
            "verbatim": row["verbatimIdentification"],
            "concept": row["concept"],
            "lineage": lineage,
            "root_class": row["root_class"],
            "qualifier": row["qualifier"],
            "plankton": row["plankton"] == "true",
            "status": row["status"],
            "ids": {column: ";".join(values) for column, values in store.identifiers.get(row["taxonID"], {}).items()},
        }
        record.update({column: row[column] for column in write.PROVENANCE_COLUMNS if row[column]})
        out.append(record)
    return out


def synthetic_records(count=50, dataset_ids=True):
    """A source that shares no concept with the package, so every node it needs must be minted.

    Ten genera of five species — the shape that exercises ancestor sharing (five nodes serve all
    fifty rows) and the ``project7`` rule that a species without a genus ancestor is unrenderable.
    """
    shared = (("kingdom", "synthetica"), ("phylum", "testophyta"), ("class", "fixtura"), ("order", "probata"))
    records = []
    for index in range(count):
        genus, epithet = f"genus{index // 5:02d}", f"species{index % 5}"
        species = f"{genus} {epithet}"
        records.append(
            {
                "verbatim": f"raw/{index:03d}",
                "concept": species,
                "lineage": (*shared, ("family", "assertidae"), ("genus", genus), ("species", species)),
                "root_class": "living",
                "qualifier": "full_body",
                "plankton": True,
                "ids": {"aphia_ID": str(900000 + index)} if dataset_ids else {},
            }
        )
    return records


# The property the byte-splicing write path never had
def test_re_upserting_every_committed_source_changes_nothing(package):
    """The no-op re-run, on all 21 sources and all 2358 rows of the real table.

    A builder is invited to be re-run; today one of them destroys 644 rows when it is. This asserts
    the opposite: reconstruct each source's records from what is committed, write them back, and
    nothing moves — not a taxon id, not an identifier row, not a mapping cell.

    Id minting is the part that makes this non-trivial. Ids are keyed by the path a node sits at,
    not by arrival order, so a re-run resolves to the id already committed. The panel's prototype
    minted in sorted-path order instead and re-pointed 1342 of 1351 ids the first time a row was
    inserted — with every foreign key still resolving.
    """
    datasets = sorted(path.stem for path in (package / "mappings").glob("*.tsv"))
    assert len(datasets) == COMMITTED_SOURCES

    for dataset in datasets:
        changes = write.upsert_source(package, dataset, records_for(package, dataset))
        assert not changes, f"{dataset}: {changes.describe(5)}"


def test_fmt_on_the_committed_package_is_a_no_op(package):
    """Canonical form is what the package is already in; ``fmt`` says so rather than churning it."""
    assert not write.fmt(package)


# The end-to-end path: a source that did not exist before
def test_a_new_source_lands_without_moving_a_published_byte(package):
    """The plan's synthetic 50-class source, end to end.

    Four things at once, because they only mean anything together: the source lands, its 65 new
    nodes mint from the next free id with no gaps, the frozen 1486-line prefix does not move, and
    the second run is a no-op.
    """
    before = loader.load_taxonomy(package).render_wide_csv()
    records = synthetic_records()

    changes = write.upsert_source(package, "synthsource", records, apply=True)
    assert changes.applied
    assert not changes.removals(), "a new source removed something"

    taxa = read_tsv(package / "taxon.tsv")
    minted = [row for row in taxa if row["taxonID"] > f"pzt:{COMMITTED_TAXA:06d}"]
    # 4 shared ranks + 1 family + 10 genera + 50 species. Shared ancestors are minted once.
    assert len(minted) == 65
    assert sorted(row["taxonID"] for row in minted) == [f"pzt:{COMMITTED_TAXA + n:06d}" for n in range(1, 66)]

    store = loader.load_taxonomy(package)
    after = store.render_wide_csv()
    assert after.split(b"\n")[:1486] == before.split(b"\n")[:1486], "a frozen row moved"
    assert len(store.rows()) == COMMITTED_ROWS + 50
    assert {row["Dataset"] for row in store.rows()[COMMITTED_ROWS:]} == {"synthsource"}

    published = write.diff_published(package, before)
    assert len(published) == 50 and all(change.after == "row" for change in published.changes)

    loader.cache_clear()
    assert not write.upsert_source(package, "synthsource", records), "the second run was not a no-op"


def test_an_ancestor_shared_with_the_committed_tree_is_reused_not_re_minted(package):
    """A new source under an existing genus must attach to it, not fork a parallel branch.

    Forking is invisible in the render — both nodes carry the same name — and shows up much later as
    two ids for one concept in the identifier table.
    """
    store = loader.load_taxonomy(package)
    lineage = tuple((node.rank, node.scientific_name) for node in store.ancestors("pzt:000071"))
    assert lineage[-1] == ("species", "bosmina fatalis")

    record = {
        "verbatim": "some/other/spelling",
        "concept": "bosmina fatalis",
        "lineage": lineage,
        "root_class": "living",
        "qualifier": "full_body",
        "plankton": True,
    }
    changes = write.upsert_source(package, "reusesource", [record], apply=True)

    assert not [change for change in changes.changes if change.table == "taxon"], "a shared ancestor was re-minted"
    assert len(read_tsv(package / "taxon.tsv")) == COMMITTED_TAXA
    assert read_tsv(package / "mappings" / "reusesource.tsv")[0]["taxonID"] == "pzt:000071"


def test_a_dry_run_writes_nothing_and_mints_nothing(package):
    """Dry run is the default, and an id burned by a dry run is a gap with no tombstone to explain it."""
    fingerprint = {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()}

    changes = write.upsert_source(package, "synthsource", synthetic_records())
    assert changes and not changes.applied

    assert {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()} == fingerprint
    assert not (package / "mappings" / "synthsource.tsv").exists()

    # And the ids the dry run would have used are still free.
    applied = write.upsert_source(package, "synthsource", synthetic_records(), apply=True)
    assert applied.applied
    minted = [row["taxonID"] for row in read_tsv(package / "taxon.tsv") if row["taxonID"] > f"pzt:{COMMITTED_TAXA:06d}"]
    assert min(minted) == f"pzt:{COMMITTED_TAXA + 1:06d}"


# Ids are never blanked by omission
def test_a_record_that_says_nothing_about_an_authority_leaves_it_alone(package):
    """The 13 legacy overrides exist because ids were cleared per row instead of per taxon."""
    dataset = "zoolake"
    before = read_tsv(package / "identifier.tsv")

    records = records_for(package, dataset)
    assert any(record["ids"] for record in records), "the fixture proves nothing if no row carries an id"
    for record in records:
        record["ids"] = {}

    changes = write.upsert_source(package, dataset, records, apply=True)

    assert not [change for change in changes.changes if change.table == "identifier"]
    assert read_tsv(package / "identifier.tsv") == before


def test_an_authority_named_empty_is_still_not_cleared(package):
    """Silence and an empty string are the same claim: the record has nothing to say here."""
    records = records_for(package, "zoolake")
    with_ids = next(record for record in records if record["ids"].get("aphia_ID"))
    with_ids["ids"] = {"aphia_ID": ""}

    assert not [change for change in write.upsert_source(package, "zoolake", records).changes if change.table == "identifier"]


def test_a_record_contradicting_a_committed_id_is_refused_rather_than_doubled(package):
    """Keeping both is schema-legal and renders as a crash two steps later, survivor by sort order."""
    records = records_for(package, "zoolake")
    target = next(record for record in records if record["ids"].get("aphia_ID"))
    target["ids"]["aphia_ID"] = "999999"

    with pytest.raises(TaxonomyError, match="already holds"):
        write.upsert_source(package, "zoolake", records)


def test_clearing_an_id_is_deliberate_and_carries_a_reason(package):
    """The only way to blank an id, and it will not happen without saying why."""
    store = loader.load_taxonomy(package)
    taxon_id = next(tid for tid, ids in store.identifiers.items() if "aphia_ID" in ids)

    with pytest.raises(TaxonomyError, match="needs a reason"):
        write.clear_id(package, taxon_id, "aphia_ID", "")

    changes = write.clear_id(package, taxon_id, "aphia_ID", "withdrawn by WoRMS", apply=True)
    assert changes.applied and changes.removals()
    assert not [
        row for row in read_tsv(package / "identifier.tsv") if row["subject_id"] == taxon_id and "worms:" in row["object_id"]
    ]


# Rows are never dropped by omission either
def test_omitting_a_committed_row_is_refused_without_allow_delete(package):
    """The failure this whole API exists to prevent looked exactly like an accidental omission."""
    records = records_for(package, "zoolake")

    with pytest.raises(TaxonomyError, match="would drop 1 committed row"):
        write.upsert_source(package, "zoolake", records[1:])

    changes = write.upsert_source(package, "zoolake", records[1:], apply=True, allow_delete=True)
    assert changes.applied
    assert len(read_tsv(package / "mappings" / "zoolake.tsv")) == len(records) - 1


def test_a_source_cannot_touch_another_sources_rows(package):
    """Per-source partition. Not guarded against — unable: only one file is opened for writing."""
    others = {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if path.stem != "synthsource"}

    write.upsert_source(package, "synthsource", synthetic_records(), apply=True)

    assert {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if path.stem != "synthsource"} == others


# Malformed records
def test_two_records_for_one_class_directory_are_refused(package):
    """The per-file primary key. Last-wins would silently lose the row the two disagree about."""
    records = synthetic_records(2)
    records[1]["verbatim"] = records[0]["verbatim"]

    with pytest.raises(TaxonomyError, match="two records both map"):
        write.upsert_source(package, "synthsource", records)


def test_a_concept_that_disagrees_with_its_own_lineage_is_refused(package):
    """The mapping file carries the display name beside the id and the loader hard-errors on a drift."""
    records = synthetic_records(1)
    records[0]["concept"] = "something else"

    with pytest.raises(TaxonomyError, match="the lineage ends at"):
        write.upsert_source(package, "synthsource", records)


def test_a_non_boolean_plankton_is_refused_rather_than_coerced(package):
    """``"false"`` is truthy. Coercing inverts the only boolean the published dataset carries."""
    records = synthetic_records(1)
    records[0]["plankton"] = "false"

    with pytest.raises(TaxonomyError, match="plankton must be a bool"):
        write.upsert_source(package, "synthsource", records)


def test_a_bucket_needs_no_lineage_and_renders_as_seven_blanks(package):
    """A lineage-less concept is a first-class node, not a hole — 43 of them exist today."""
    record = {
        "verbatim": "unidentifiable smudge",
        "concept": "smudge",
        "root_class": "artefact",
        "qualifier": "unqualified",
        "plankton": False,
    }
    write.upsert_source(package, "bucketsource", [record], apply=True)

    store = loader.load_taxonomy(package)
    node = store.taxon(read_tsv(package / "mappings" / "bucketsource.tsv")[0]["taxonID"])
    assert node.kind == "bucket" and node.parent_id == ""
    rendered = next(row for row in store.rows() if row["Dataset"] == "bucketsource")
    assert [rendered[rank] for rank in ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")] == [""] * 7


# Rename, merge, tombstones
def test_a_rename_moves_the_mapping_files_with_the_taxon(package):
    """A rename that stopped at ``taxon.tsv`` would leave the package unloadable, not merely stale."""
    changes = write.rename(package, "pzt:000071", "bosmina renamed", apply=True)

    assert {change.table for change in changes.changes} == {"taxon", "mappings/frepj"}
    store = loader.load_taxonomy(package)
    assert store.taxon("pzt:000071").scientific_name == "bosmina renamed"
    assert any(row["proposed_label"] == "bosmina renamed" for row in store.rows())


def test_a_merge_leaves_a_tombstone_and_repoints_the_mappings(package):
    """An id that simply disappears is an identity lost silently; ``merged.tsv`` is the precedent."""
    changes = write.merge(package, "pzt:000071", "pzt:000072", "duplicate of freyi", apply=True)

    assert changes.applied
    tombstones = read_tsv(package / "merged.tsv")
    assert len(tombstones) == 1
    assert tombstones[0]["retired_id"] == "pzt:000071" and tombstones[0]["replacement_id"] == "pzt:000072"
    assert tombstones[0]["reason"] == "duplicate of freyi" and tombstones[0]["date"]

    store = loader.load_taxonomy(package)
    assert "pzt:000071" not in store.taxa
    assert store.mappings[("frepj", "Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina fatalis")].taxon_id == "pzt:000072"


def test_retiring_a_node_with_children_is_refused(package):
    """Orphaning a subtree is a dangling parent everywhere; re-parent first, deliberately."""
    with pytest.raises(TaxonomyError, match="still has children"):
        write.merge(package, "pzt:000070", "pzt:000069", "genus is redundant")


def test_retiring_an_already_retired_id_is_refused(package):
    """One id, one tombstone. A second would give the same retired id two replacements."""
    write.merge(package, "pzt:000071", "pzt:000072", "duplicate", apply=True)
    loader.cache_clear()

    with pytest.raises(TaxonomyError, match="already retired into pzt:000072"):
        write.merge(package, "pzt:000071", "pzt:000069", "duplicate again")


def test_a_retirement_needs_a_reason_and_cannot_be_into_itself(package):
    with pytest.raises(TaxonomyError, match="needs a reason"):
        write.merge(package, "pzt:000071", "pzt:000072", "")
    with pytest.raises(TaxonomyError, match="into itself"):
        write.merge(package, "pzt:000071", "pzt:000071", "why")


# fmt, the counterpart to merge=union
def test_fmt_repairs_a_union_merge_and_a_spreadsheet_boolean(package):
    """PR 4 made concurrent add-source PRs merge cleanly; the resolution is unsorted by construction.

    Both symptoms are cosmetic until the second one silently inverts ``plankton``.
    """
    path = package / "mappings" / "zoolake.tsv"
    rows = read_tsv(path)
    scrambled = [{**rows[-1], "plankton": "TRUE"}, *rows[:-1]]
    write_tsv(path, MAPPING_COLUMNS, scrambled)

    with pytest.raises(TaxonomyError, match="expected 'true' or 'false'"):
        loader.load_taxonomy(package)

    changes = write.fmt(package, apply=True)
    assert changes.applied
    assert {change.column for change in changes.changes} == {"plankton", "order"}

    loader.cache_clear()
    assert read_tsv(path) == rows, "fmt did not restore canonical form"


def test_fmt_reorders_a_taxon_table_a_union_merge_left_shuffled(package):
    path = package / "taxon.tsv"
    rows = read_tsv(path)
    write_tsv(path, tuple(rows[0]), [*rows[900:], *rows[:900]])

    assert write.fmt(package, apply=True).applied
    assert read_tsv(path) == rows


# The reviewer's question
def test_diff_published_reports_the_cells_that_moved_not_the_rows_that_exist(package):
    """ "Which published cells did this change?" — answered on the legacy view, which is what ships."""
    before = loader.load_taxonomy(package).render_wide_csv()

    write.rename(package, "pzt:000071", "bosmina renamed", apply=True)
    changes = write.diff_published(package, before)

    # Two cells, not one: `proposed_label` carries the name and the `Species` slot carries the
    # epithet project7 derives from it. A diff reporting only the first would understate the change.
    assert [(change.column, change.before, change.after) for change in changes.changes] == [
        ("Species", "fatalis", "renamed"),
        ("proposed_label", "bosmina fatalis", "bosmina renamed"),
    ]


def test_diff_published_is_empty_when_a_write_moved_nothing_published(package):
    """Provenance is curation, not publication; a ``method`` column moving must not read as a data change.

    ``derived`` is a fixture value here, not a claim about zoolake's history — what is being shown is
    that a provenance column moves without a published cell moving.
    """
    before = loader.load_taxonomy(package).render_wide_csv()
    records = records_for(package, "zoolake")
    for record in records:
        record["method"] = "derived"

    changes = write.upsert_source(package, "zoolake", records, apply=True)

    assert changes.applied and all(change.column == "method" for change in changes.changes)
    assert not write.diff_published(package, before)


def test_a_method_outside_the_vocabulary_is_refused(package):
    """Free text defeats the point: 2358 answers to "why does this row claim this?" must be comparable."""
    records = records_for(package, "zoolake")
    records[0]["method"] = "rule_table_v2"

    with pytest.raises(TaxonomyError, match="is not in the vocabulary"):
        write.upsert_source(package, "zoolake", records)


def test_a_donor_with_no_method_is_refused(package):
    """Where a decision came from, with no answer for how, is half a provenance record."""
    records = records_for(package, "zoolake")
    records[0]["donor"] = "syke_ifcb_2022"

    with pytest.raises(TaxonomyError, match="donor with no method"):
        write.upsert_source(package, "zoolake", records)


# The builder-facing adapters
def test_every_committed_source_round_trips_through_the_wide_row_adapter(package):
    """The port's own gate: the shape the builders already produce, back through the new write path.

    ``records_from_wide_rows`` is what makes porting a builder a small diff — their ~2,600 lines of
    curation go on producing 19-column rows and only the write at the end changes. Feeding all 2358
    committed rows back through it must move nothing, which is the guarantee the byte-splicing
    writers could not offer for even one.
    """
    from planktonzilla.planktonzilla_dataset import constants
    from planktonzilla.planktonzilla_dataset.taxonomy import migrate

    wide = migrate.read_legacy_rows(Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME))
    assert len(wide) == COMMITTED_ROWS

    changes = write.upsert_wide_rows(package, wide)

    assert not changes, changes.describe(8)


def test_the_wide_row_adapter_splits_by_the_rows_own_dataset_column(package):
    """A builder that curates four sources writes four files. The split is a partition, not a filter."""
    from planktonzilla.planktonzilla_dataset import constants
    from planktonzilla.planktonzilla_dataset.taxonomy import migrate

    wide = [
        row
        for row in migrate.read_legacy_rows(Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME))
        if row["Dataset"].startswith("tara_pacific_")
    ]
    others = {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if "tara" not in path.stem}

    write.upsert_wide_rows(package, wide, apply=True)

    assert {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if "tara" not in path.stem} == others


def test_a_float_serialised_id_never_becomes_a_second_curie(package):
    """The wide CSV stores three authorities through a float column. A CURIE is not a float.

    Left alone, ``135336.0`` and ``135336`` are two ids for one concept and the renderer crashes on
    the pair two steps later, with the survivor decided by sort order.
    """
    store = loader.load_taxonomy(package)
    taxon_id = next(tid for tid, ids in store.identifiers.items() if ids.get("aphia_ID"))
    held = store.identifiers[taxon_id]["aphia_ID"][0]
    verbatim = next(mapping.verbatim for mapping in store.mappings.values() if mapping.taxon_id == taxon_id)
    dataset = next(mapping.dataset for mapping in store.mappings.values() if mapping.taxon_id == taxon_id)

    assert not write.add_ids(package, dataset, {verbatim: {"aphia_ID": f"{held}.0"}})


def test_a_non_numeric_value_in_a_numeric_authority_is_refused(package):
    """Silently dropping it is how a resolution run reports success and resolves nothing."""
    row = read_tsv(package / "mappings" / "zoolake.tsv")[0]

    with pytest.raises(TaxonomyError, match="numeric but the record gives"):
        write.add_ids(package, "zoolake", {row["verbatimIdentification"]: {"NCBI_ID": "Q12345"}})


def test_add_ids_refuses_a_class_directory_the_source_does_not_map(package):
    with pytest.raises(TaxonomyError, match="does not map"):
        write.add_ids(package, "zoolake", {"no such class dir": {"aphia_ID": "1"}})


def test_add_ids_puts_the_identifier_on_the_concept_not_the_row(package):
    """Two class directories mapping one concept can no longer disagree about its BOLD id.

    ``frepj``'s ``Moina sp`` and ``Sp._unk`` are the same concept spelled twice, and the row-cell
    rewriter this replaced had to be given both. 67 such pairs exist within a single source today.
    """
    dataset, taxon_id = "frepj", "pzt:000139"
    sharing = [
        row["verbatimIdentification"] for row in read_tsv(package / f"mappings/{dataset}.tsv") if row["taxonID"] == taxon_id
    ]
    assert len(sharing) == 2

    changes = write.add_ids(package, dataset, {sharing[0]: {"BOLD_ID": "9999999"}}, apply=True)

    # One identifier row, named once — not one per class directory that happens to point at it.
    assert len(changes) == 1
    store = loader.load_taxonomy(package)
    for verbatim in sharing:
        assert store.identifiers[store.mappings[(dataset, verbatim)].taxon_id]["BOLD_ID"] == ["9999999"]


def test_a_multi_source_write_that_fails_partway_rolls_back(package):
    """A builder that curates four sources must not leave two of them applied.

    A dry run over the groups does not catch this on its own: both groups here agree with what is
    committed and only disagree with EACH OTHER, so the conflict cannot surface until the first has
    been written. The rollback is what makes ``upsert_wide_rows`` an all-or-nothing act.
    """
    from planktonzilla.planktonzilla_dataset.taxonomy.model import LEGACY_HEADER

    fingerprint = {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()}
    shared = dict.fromkeys(LEGACY_HEADER, "")
    shared.update(Kingdom="synthetica", proposed_label="synthetica", plankton="True", living="True", root_class="living")
    wide = [
        {**shared, "Dataset": "asource", "Raw_Labels": "a", "aphia_ID": "111111.0"},
        # Same concept, a different aphia id. Neither row contradicts the committed table.
        {**shared, "Dataset": "zsource", "Raw_Labels": "z", "aphia_ID": "222222.0"},
    ]

    with pytest.raises(TaxonomyError, match="already holds"):
        write.upsert_wide_rows(package, wide, apply=True)

    assert {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()} == fingerprint
    assert not (package / "mappings" / "asource.tsv").exists(), "the first source's file survived the rollback"
