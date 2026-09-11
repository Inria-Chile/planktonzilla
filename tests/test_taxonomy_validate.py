"""
(c) Inria

Gate for step 3 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md`` — the schema of record and the
validator that executes it.

The plan's shape was a Frictionless descriptor plus a polars module that re-implements it, "so a
laptop without Frictionless gives CI's verdict". Two statements of one constraint is a drift
hazard with nothing to make them agree, so ``validate.py`` executes ``datapackage.json`` directly.
One execution path, no re-implementation, and a constraint declared in the descriptor cannot be
silently unenforced. That is checked below rather than asserted: the defect fixtures each violate
a constraint the DESCRIPTOR declares, and the validator has no rule of its own for them.

The seven-defect package the plan calls for is built here by copying the real package and
injecting one defect at a time, rather than committing a second package that would drift from the
first. Each defect is one the panel measured a real tool failing to catch.

The four rules that exist because a tool the design trusted does not enforce them get their own
tests, named after what was measured:

    Frictionless 5.19 parses ``uniqueKeys`` and reported an injected duplicate valid
    a second ``exact`` id is schema-legal in SSSOM and vanishes by numeric sort order
    a foreign source's row passes a per-file primary key unnoticed
    the absence row the design offers is rejected by its own id pattern
"""

import json
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

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import loader, validate
from planktonzilla.planktonzilla_dataset.taxonomy.model import TaxonomyError, read_tsv, write_tsv

PACKAGE = loader.PACKAGE_DIR
REGISTERED = sorted(constants.DATASET_IMPORT_CONFIGS)


@pytest.fixture
def package(tmp_path):
    """A scratch copy of the committed package, for injecting one defect into."""
    work = tmp_path / "data"
    shutil.copytree(PACKAGE, work)
    return work


def _edit_tsv(path: Path, mutate):
    """Read a table, let the caller mutate the row list, write it back in place."""
    rows = read_tsv(path)
    columns = tuple(rows[0]) if rows else ()
    mutate(rows)
    write_tsv(path, columns, rows)


def _checks(report):
    return sorted({finding.check for finding in report.findings})


# The committed package
def test_the_committed_package_validates_clean():
    """THE GATE. Zero errors, every remaining finding adjudicated, no stale adjudication."""
    report = validate.apply_waivers(
        validate.validate(PACKAGE, registered=REGISTERED),
        validate.read_waivers(PACKAGE),
    )

    assert report.errors == [], "\n".join(f.describe() for f in report.errors)
    assert report.stale_waivers == [], f"waiver(s) matching no current finding: {report.stale_waivers}"


def test_the_remaining_findings_are_the_identifier_backlog_and_nothing_else():
    """What is left is a named, countable backlog rather than undifferentiated noise.

    It was 147 coarse identifiers (a species carrying its own genus's id) plus 21 cross-branch
    collisions. **Step 7 worked the first number to zero**: those 449 identifier rows now carry
    ``skos:broadMatch``, which is what they always claimed, so they are no longer reported as a
    state that needs fixing. The published bytes did not move — the 19-column CSV has no way to
    qualify an id — and ``unjustified_broad_match`` now guards the re-predication, so relabelling a
    real collision cannot make it disappear from this report.

    The 21 are the KI-13 backlog and stay pinned: a number for a later step to move.
    """
    report = validate.apply_waivers(
        validate.validate(PACKAGE, registered=REGISTERED),
        validate.read_waivers(PACKAGE),
    )

    assert report.summary()["by_check"] == {"id_shared_across_branches": 21}
    assert report.summary()["waived"] == 6


def test_every_frozen_defect_is_adjudicated_with_a_written_reason():
    """The six waived findings are KI-8 and KI-9, each carrying why it stays."""
    waivers = validate.read_waivers(PACKAGE)

    assert len(waivers) == 6
    assert {row["check"] for row in waivers.values()} == {"lineage_name_repeat", "name_not_lowercase"}
    assert all(row["category"] == "frozen_defect" for row in waivers.values())
    assert all(len(row["reason"]) > 80 for row in waivers.values()), "a waiver without a real reason"


def test_the_ported_adjudications_are_complete_and_resolve_to_real_taxa():
    """The 20 KI-31 and 59 authority adjudications, re-keyed onto stable concept ids.

    Not consumed by this validator — their checks move onto the loader in steps 5 and 7 — so what
    is asserted here is that porting lost nothing and that every id resolves.
    """
    taxa = {row["taxonID"] for row in read_tsv(PACKAGE / "taxon.tsv")}

    horizontal = read_tsv(PACKAGE / "waivers" / "horizontal.tsv")
    assert len(horizontal) == 20
    for row in horizontal:
        assert row["reason"] and row["category"]
        assert set(row["taxon_ids"].split(";")) <= taxa, row["finding_id"]
        assert len(row["taxon_ids"].split(";")) == len(row["labels"].split(";"))

    authority = read_tsv(PACKAGE / "waivers" / "authority.tsv")
    assert len(authority) == 59
    for row in authority:
        assert row["reason"] and row["category"]
        assert row["taxon_id"] in taxa, row["finding_id"]

    # Categories carried over verbatim, not re-derived.
    assert {row["category"] for row in horizontal} == {
        "mislabel_defect",
        "nomenclature_drift",
        "needs_taxonomic_adjudication",
        "rank_treatment",
        "rank_inflation_defect",
        "bucket_naming",
    }
    assert {row["category"] for row in authority} == {
        "defect_wrong_identifier",
        "defect_csv_cell",
        "register_silence",
        "nomenclature_drift",
    }


def test_the_waiver_tables_this_validator_does_not_consume_are_not_reported_stale():
    """79 permanently-stale entries would teach a reader to ignore the stale list entirely."""
    assert validate.STRUCTURAL_WAIVERS == "structural.tsv"
    assert set(validate.read_waivers(PACKAGE)) == {
        row["finding_id"] for row in read_tsv(PACKAGE / "waivers" / "structural.tsv")
    }


# The seven defects
def test_defect_1_a_dangling_foreign_key(package):
    _edit_tsv(package / "mappings" / "zoolake.tsv", lambda rows: rows[0].update(taxonID="pzt:999999"))

    assert "foreign_key" in _checks(validate.validate(package))


def test_defect_2_a_duplicate_primary_key(package):
    _edit_tsv(package / "taxon.tsv", lambda rows: rows.append(dict(rows[0])))

    assert "primary_key" in _checks(validate.validate(package))


def test_defect_3_a_float_serialised_identifier(package):
    """``135336.0`` is a CSV-typing artefact and must not survive into the store."""
    _edit_tsv(package / "identifier.tsv", lambda rows: rows[0].update(object_id="worms:135336.0"))

    assert "pattern" in _checks(validate.validate(package))


def test_defect_4_a_bad_vocabulary_value(package):
    """A term absent from its vocabulary table, caught by the descriptor's ``vocabulary`` constraint."""
    _edit_tsv(package / "mappings" / "zoolake.tsv", lambda rows: rows[0].update(root_class="zoo"))

    assert "vocabulary" in _checks(validate.validate(package))


def test_defect_5_a_duplicate_mapping_key(package):
    _edit_tsv(package / "mappings" / "zoolake.tsv", lambda rows: rows.append(dict(rows[0])))

    assert "primary_key" in _checks(validate.validate(package))


def test_defect_6_a_duplicate_name_rank_parent_triple(package):
    """Frictionless 5.19 parses ``uniqueKeys`` and reports this VALID. Measured by the panel."""

    def mutate(rows):
        clone = dict(rows[5])
        clone["taxonID"] = "pzt:900001"
        rows.append(clone)

    _edit_tsv(package / "taxon.tsv", mutate)
    report = validate.validate(package)

    assert "unique_key" in _checks(report)
    assert any("scientificName" in f.detail for f in report.findings if f.check == "unique_key")


def test_defect_7_a_second_exact_id_for_a_single_valued_authority(package):
    """Schema-legal in any SSSOM table, and dropped by the renderer on numeric sort order."""

    def mutate(rows):
        worms = next(row for row in rows if row["object_id"].startswith("worms:"))
        rows.append({**worms, "object_id": "worms:999999999"})

    _edit_tsv(package / "identifier.tsv", mutate)

    assert "multiple_exact_ids" in _checks(validate.validate(package))


def test_all_seven_defects_are_caught_when_injected_together(package):
    """7/7, the measurement the plan's tooling survey reports for polars."""
    _edit_tsv(package / "mappings" / "zoolake.tsv", lambda rows: rows[0].update(taxonID="pzt:999999", root_class="zoo"))
    _edit_tsv(package / "mappings" / "zoolake.tsv", lambda rows: rows.append(dict(rows[1])))

    def taxon_mutate(rows):
        rows.append(dict(rows[0]))
        clone = dict(rows[5])
        clone["taxonID"] = "pzt:900001"
        rows.append(clone)

    _edit_tsv(package / "taxon.tsv", taxon_mutate)

    def identifier_mutate(rows):
        rows[0]["object_id"] = "worms:135336.0"
        worms = next(row for row in rows if row["object_id"].startswith("worms:") and row is not rows[0])
        rows.append({**worms, "object_id": "worms:999999999"})

    _edit_tsv(package / "identifier.tsv", identifier_mutate)

    caught = set(_checks(validate.validate(package)))

    assert {"foreign_key", "primary_key", "pattern", "vocabulary", "unique_key", "multiple_exact_ids"} <= caught


# The rules a per-file key cannot see
def test_a_foreign_source_row_passes_a_per_file_key_and_is_caught_anyway(package):
    """Its file's primary key is (datasetID, verbatim), so a foreign row is perfectly unique."""
    _edit_tsv(package / "mappings" / "zoolake.tsv", lambda rows: rows[0].update(datasetID="zooscan"))
    report = validate.validate(package)

    assert "dataset_is_not_file_stem" in _checks(report)
    assert "primary_key" not in _checks(report), "the per-file key saw nothing, which is the point"


def test_a_source_on_disk_that_no_registry_knows_about_is_caught(package):
    """Disk and the registry must agree — and the registry is ``constants``, not the descriptor.

    Step 4 removed the descriptor's source enumeration: two registries drift, and both branches of
    a concurrent add-source PR insert into the same line of the second one. The thorough
    both-directions case lives in ``tests/test_taxonomy_concurrency.py``; this is the validator's
    own record that it still refuses an unknown source.
    """
    shutil.copy(package / "mappings" / "zoolake.tsv", package / "mappings" / "newsource.tsv")

    report = validate.validate(package, registered=REGISTERED)
    assert "registry_drift" in _checks(report)
    assert any("not a registered source" in f.detail for f in report.findings)


def test_the_absence_row_the_design_offers_actually_validates(package):
    """The refuters' finding: the pattern and the relation enum rejected the row for "not in this
    register". Both are widened, and this proves it — an absence is a row, not a blank cell."""

    def mutate(rows):
        rows.append(
            {
                **dict.fromkeys(rows[0], ""),
                "subject_id": rows[0]["subject_id"],
                "predicate_id": "sssom:NoTermFound",
                "object_id": "worms:absent",
                "mapping_justification": "semapv:ManualMappingCuration",
                "comment": "freshwater taxon; WoRMS is marine, so absence is expected",
            }
        )

    _edit_tsv(package / "identifier.tsv", mutate)
    report = validate.validate(package)

    assert [f for f in report.findings if f.resource == "identifier" and f.severity == "ERROR"] == []


def test_an_absence_row_is_not_counted_as_a_shared_id(package):
    """Two registers both recording 'absent' must not read as two concepts claiming one id."""

    def mutate(rows):
        for subject in (rows[0]["subject_id"], rows[10]["subject_id"]):
            rows.append(
                {
                    **dict.fromkeys(rows[0], ""),
                    "subject_id": subject,
                    "predicate_id": "sssom:NoTermFound",
                    "object_id": "worms:absent",
                    "mapping_justification": "semapv:ManualMappingCuration",
                }
            )

    _edit_tsv(package / "identifier.tsv", mutate)
    report = validate.validate(package)

    assert not [f for f in report.findings if "absent" in f.locator]


# Findings identify themselves by value, not by position
def test_a_finding_id_is_a_hash_of_its_values_not_its_row_number():
    """The one discipline in the repository where adding a source does not void an adjudication."""
    first = validate.Finding("check", "ERROR", "taxon", "pzt:000001", "detail")
    same = validate.Finding("check", "ERROR", "taxon", "pzt:000001", "detail")
    different = validate.Finding("check", "ERROR", "taxon", "pzt:000002", "detail")

    assert first.finding_id == same.finding_id
    assert first.finding_id != different.finding_id
    assert len(first.finding_id) == 12


def test_a_waiver_that_matches_nothing_is_reported_stale(package):
    """An adjudication that no longer describes anything is a claim that quietly stopped being true."""
    _edit_tsv(
        package / "waivers" / "structural.tsv",
        lambda rows: rows.append({**rows[0], "finding_id": "deadbeefcafe"}),
    )

    report = validate.apply_waivers(validate.validate(package), validate.read_waivers(package))
    assert report.stale_waivers == ["deadbeefcafe"]


# draft status
def test_a_partially_mapped_source_is_committable(package):
    """Design B could not commit one: 150 blank rows produced 150 errors and a renderer crash.

    A ``draft`` row loads, does not count against the row-order manifest, does not reach the
    render, and does not satisfy coverage.

    Since step 4 the new source needs no descriptor edit at all — the mapping resource is a glob —
    so this is now the real flow a curator follows rather than a rehearsal of it.
    """
    template = read_tsv(package / "mappings" / "zoolake.tsv")[0]
    write_tsv(
        package / "mappings" / "newsource.tsv",
        tuple(template),
        [{**template, "datasetID": "newsource", "verbatimIdentification": "not curated yet", "status": "draft"}],
    )

    store = loader.load_taxonomy(package)
    assert ("newsource", "not curated yet") in store.mappings
    assert len(store.rows()) == 2358, "a draft row reached the render"

    report = validate.validate(package, registered=[*REGISTERED, "newsource"])
    coverage = [f for f in report.findings if f.check == "source_not_covered"]
    assert [f.locator for f in coverage] == ["newsource"]
    assert not [f for f in report.findings if f.severity == "ERROR" and f.check != "source_not_covered"]


# The descriptor itself
def test_the_descriptor_declares_every_file_in_the_package():
    """A file nobody declares is a file nobody validates."""
    descriptor = validate.load_descriptor(PACKAGE)
    declared = {
        str(path.relative_to(PACKAGE))
        for resource in descriptor["resources"]
        for path in validate._resource_paths(PACKAGE, resource)
    }
    on_disk = {str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.tsv")}

    assert on_disk - declared == set(), f"undeclared file(s): {sorted(on_disk - declared)}"


def test_the_descriptor_is_plain_json_and_needs_no_dependency():
    """The schema of record parses with the standard library, which is what makes pre-flight local."""
    descriptor = json.loads((PACKAGE / "datapackage.json").read_text(encoding="utf-8"))

    assert descriptor["profile"] == "tabular-data-package"
    assert descriptor["dialect"]["delimiter"] == "\t"
    assert len(descriptor["resources"]) == 15


def test_a_missing_descriptor_is_refused(tmp_path):
    with pytest.raises(TaxonomyError, match="no descriptor at"):
        validate.load_descriptor(tmp_path)
