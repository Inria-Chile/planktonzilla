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

import ast
import csv
import json
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


def _subjects_of(finding):
    """The concept ids an ``id_shared_across_branches`` finding names, back out of its detail."""
    return ast.literal_eval(finding.detail.split("concepts ", 1)[1])


def _lowest_common_ancestor(taxon_ids, taxa):
    """The deepest concept every one of ``taxon_ids`` hangs from — the anchor a coarse id wants."""

    def ancestors(taxon_id):
        out, seen, cursor = [], set(), taxa[taxon_id]["parentNameUsageID"]
        while cursor and cursor in taxa and cursor not in seen:
            seen.add(cursor)
            out.append(cursor)
            cursor = taxa[cursor]["parentNameUsageID"]
        return out

    common = set(ancestors(taxon_ids[0])) | {taxon_ids[0]}
    for taxon_id in taxon_ids[1:]:
        common &= set(ancestors(taxon_id)) | {taxon_id}
    return max(common, key=lambda taxon_id: len(ancestors(taxon_id)))


def _published_labels():
    """Every ``proposed_label`` the frozen CSV publishes — i.e. which concepts have a published row."""
    with constants.DEFAULT_TAXONOMY_CSV_FILENAME.open(newline="", encoding="utf-8") as handle:
        return {(row["proposed_label"] or "").strip().lower() for row in csv.DictReader(handle)}


# The committed package
def test_the_committed_package_validates_clean():
    """THE GATE. Zero errors, every remaining finding adjudicated, no stale adjudication."""
    report = validate.apply_waivers(
        validate.validate(PACKAGE, registered=REGISTERED),
        validate.read_waivers(PACKAGE),
    )

    assert report.errors == [], "\n".join(f.describe() for f in report.errors)
    assert report.stale_waivers == [], f"waiver(s) matching no current finding: {report.stale_waivers}"


def test_nothing_is_left_unadjudicated():
    """The backlog is now zero UNADJUDICATED findings, which is not the same as zero findings.

    It was 147 coarse identifiers (a species carrying its own genus's id) plus 21 cross-branch
    collisions. **Step 7 worked the first number to zero**: those 449 identifier rows now carry
    ``skos:broadMatch``, which is what they always claimed, so they are no longer reported as a
    state that needs fixing. The published bytes did not move — the 19-column CSV has no way to
    qualify an id — and ``unjustified_broad_match`` now guards the re-predication, so relabelling a
    real collision cannot make it disappear from this report.

    The 21 then went two ways. **Two were fixed**: the committed NCBI snapshot names taxid 418932
    as the family Syracosphaeraceae and 418941 as Rhabdosphaeraceae, and in both cases that family
    is exactly the common ancestor of the two holders — so they were the same coarse propagation
    step 7 re-predicated 449 times, missed only because no concept held the id to be broader THAN.
    Anchoring the id on the family and re-predicating the holders moved no published byte (neither
    family is a ``proposed_label``, and the renderer never reads the predicate).

    The remaining **19 are adjudicated**, each with the evidence and the correction written down.
    They stay findings — a waiver records a decision, it does not fix the data — but a NEW
    collision arrives unwaived and is visible against a clean report rather than lost in a list of
    21. The count that matters now is that ``findings`` is empty and no waiver is stale.
    """
    report = validate.apply_waivers(
        validate.validate(PACKAGE, registered=REGISTERED),
        validate.read_waivers(PACKAGE),
    )

    assert report.summary()["by_check"] == {}
    assert report.summary()["waived"] == 25
    assert report.summary()["stale_waivers"] == 0


def test_the_two_anchorable_collisions_were_fixed_rather_than_waived():
    """A waiver for something the committed evidence can actually resolve is a decision not taken.

    NCBI's own record for each of these taxids is the FAMILY, and the family is the common ancestor
    of both holders — so the shape is coarse propagation, not two taxa stamped with one id, and the
    fix is the one step 7 applied 449 times. Guarded here because the alternative is silent: the
    validator reports nothing either way once a waiver exists.
    """
    identifiers = read_tsv(PACKAGE / "identifier.tsv")
    held = {(row["subject_id"], row["object_id"]): row["predicate_id"] for row in identifiers}

    for object_id, anchor, holders in (
        ("ncbi:418932", "pzt:000868", ("pzt:000870", "pzt:000871")),
        ("ncbi:418941", "pzt:000863", ("pzt:000865", "pzt:000867")),
    ):
        assert held[(anchor, object_id)] == "skos:exactMatch", f"{object_id} lost its anchor"
        for holder in holders:
            assert held[(holder, object_id)] == "skos:broadMatch", f"{holder} re-claims {object_id} exactly"

    waivers = validate.read_waivers(PACKAGE)
    waived_ids = {row["locator"] for row in waivers.values()}
    assert not waived_ids & {"ncbi:418932", "ncbi:418941"}, "a resolved collision must not also carry a waiver"


def test_every_adjudicated_collision_states_the_shape_the_package_actually_has():
    """The reasons carry measurements, so a reason that stops describing the data is a failure.

    Nineteen prose paragraphs are exactly the kind of record that rots: the holder count moves, a
    concept is renamed, an ancestor gains a published row, and the adjudication silently starts
    describing a package that no longer exists. Each ``coarse_identifier_no_anchor`` reason names
    its holder count, its anchor concept by id and name, and whether that anchor is published —
    every one of which is re-derived here from the package itself.
    """
    taxa = {row["taxonID"]: row for row in read_tsv(PACKAGE / "taxon.tsv")}
    published = _published_labels()
    unwaived = validate.validate(PACKAGE, registered=REGISTERED)
    collisions = {f.finding_id: f for f in unwaived.findings if f.check == "id_shared_across_branches"}

    coarse = [row for row in validate.read_waivers(PACKAGE).values() if row["category"] == "coarse_identifier_no_anchor"]
    assert len(coarse) == 14, "the fourteen BOLD ids with no anchor to re-predicate onto"

    for row in coarse:
        finding = collisions[row["finding_id"]]
        holders = row["subject"].split(";")
        assert sorted(holders) == sorted(_subjects_of(finding)), f"{row['locator']}: subject drifted from the finding"
        assert f"{len(holders)} concepts hold {row['locator']}" in row["reason"]

        anchor = _lowest_common_ancestor(holders, taxa)
        assert f"{anchor} {taxa[anchor]['scientificName']} ({taxa[anchor]['taxonRank']})" in row["reason"], (
            f"{row['locator']}: the reason names an anchor the tree no longer gives"
        )
        gated = taxa[anchor]["scientificName"] in published
        assert ("IS a published proposed_label" in row["reason"]) == gated, (
            f"{row['locator']}: the reason is wrong about whether correcting it moves a published cell"
        )


def test_every_waived_finding_is_adjudicated_with_a_written_reason():
    """Six KI-8/KI-9 frozen defects plus the nineteen KI-13 collisions, each carrying why it stays.

    The category vocabulary is pinned because it is the machine-readable half of the adjudication:
    ``coarse_identifier_no_anchor`` says the correction is to anchor and re-predicate,
    ``nomenclature_drift`` says the id is right on both rows and the NAMES are what differ, and
    ``defect_wrong_identifier`` says one named row carries an id that is not its taxon's.
    """
    waivers = validate.read_waivers(PACKAGE)

    assert len(waivers) == 25
    assert {row["check"] for row in waivers.values()} == {
        "lineage_name_repeat",
        "name_not_lowercase",
        "id_shared_across_branches",
    }
    assert Counter(row["category"] for row in waivers.values()) == {
        "coarse_identifier_no_anchor": 14,
        "frozen_defect": 6,
        "nomenclature_drift": 4,
        "defect_wrong_identifier": 1,
    }
    assert all(len(row["reason"]) > 80 for row in waivers.values()), "a waiver without a real reason"
    assert all(row["subject"] for row in waivers.values()), "a waiver that does not say what it is about"


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
    assert len(descriptor["resources"]) == 16


def test_a_missing_descriptor_is_refused(tmp_path):
    with pytest.raises(TaxonomyError, match="no descriptor at"):
        validate.load_descriptor(tmp_path)


# A parent cycle must be REPORTED, never hung on
def test_a_parent_cycle_is_reported_rather_than_hung_on(package):
    """``taxon.tsv`` carries ``merge=union``, so two branches re-parenting one node give A->B->A.

    Every walk over ``parentNameUsageID`` therefore needs a guard, and the one that matters is not
    ``_check_no_cycles`` — that one always had a ``seen`` set. It is every check that runs AFTER
    it: an unguarded walk there spins forever, so ``pz_taxonomy check`` hangs instead of printing
    the ``parent_cycle`` finding that would explain the problem. A report is only reachable if the
    checks after the report-producing one terminate.

    The timeout is the assertion. `_check_lineage_name_repeat` was the last unguarded walk, and it
    was missed once already by a review that believed the hang was fixed — so this pins the
    property against the whole check suite rather than against any one walk.
    """
    rows = read_tsv(package / "taxon.tsv")
    by_id = {row["taxonID"]: row for row in rows}
    child = next(row for row in rows if row["parentNameUsageID"] in by_id)
    by_id[child["parentNameUsageID"]]["parentNameUsageID"] = child["taxonID"]
    write_tsv(package / "taxon.tsv", tuple(rows[0]), rows)

    loader.cache_clear()
    report = validate.validate(package)

    assert "parent_cycle" in _checks(report)
    assert any(finding.check == "parent_cycle" for finding in report.errors)


# The two registries that hold the same adjudications
def test_the_ported_waivers_still_agree_with_the_registries_the_tools_read():
    """The TSVs and the JSONs are two copies of one set of adjudications, and nothing compared them.

    ``read_waivers`` deliberately consumes only ``structural.tsv``; ``horizontal.tsv`` and
    ``authority.tsv`` are ports waiting for their checks to move onto the loader. Meanwhile the
    live tools read the JSONs — ``verify_label_consistency`` reads
    ``LABEL_CONSISTENCY_WAIVERS.json`` and ``verify_taxonomy_ids`` reads
    ``AUTHORITY_WAIVERS.json``. So a 21st adjudication added where the tools look leaves the
    ported copy stale, and the test above cannot notice: it asserts the TSV against itself, with
    hard-coded counts that a stale file still satisfies.

    This is the missing edge. It compares the two by ``finding_id`` and by what each says, so
    drift in either direction is a failure rather than a silent divergence.
    """
    import json

    for tsv_name, json_name in (
        ("horizontal.tsv", "LABEL_CONSISTENCY_WAIVERS.json"),
        ("authority.tsv", "AUTHORITY_WAIVERS.json"),
    ):
        ported = {row["finding_id"]: row for row in read_tsv(PACKAGE / "waivers" / tsv_name)}
        live_path = Path(root) / "planktonzilla" / "planktonzilla_dataset" / "utils" / json_name
        live = {entry["finding_id"]: entry for entry in json.loads(live_path.read_text(encoding="utf-8"))["waivers"]}

        assert set(ported) == set(live), (
            f"{tsv_name} and {json_name} adjudicate different findings; "
            f"only in the port: {sorted(set(ported) - set(live))}; only live: {sorted(set(live) - set(ported))}"
        )

        for finding_id, entry in live.items():
            for field in ("check", "category", "reason"):
                assert ported[finding_id][field] == entry[field], (
                    f"{finding_id}: {tsv_name} says {field}={ported[finding_id][field]!r}, {json_name} says {entry[field]!r}"
                )
