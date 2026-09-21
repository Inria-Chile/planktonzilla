"""
(c) Inria

Gate for step 1 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md`` — the normalised table package and
the one-shot migration that generates it.

Nothing in the repository reads the package yet. That is the point of this step: the store is
built beside the wide CSV and proved equivalent to it BEFORE any consumer moves, so the move
itself (step 3) carries no risk it has not already retired. Two properties decide whether the
step is sound, and both have already broken a prototype in the design panel:

**Rendering, not inspection, is the evidence.** The refuters found design A's prototype package
had been hand-finished after its generator ran — re-running the generator produced a different
concept table, and the exporter could not run on its output. So the gate here renders the
19-column CSV back out of the WRITTEN FILES and compares bytes. A package that renders from the
in-memory objects that produced it proves nothing about what was committed.

**Identifier minting is idempotent.** The panel's migration minted ids in sorted-path order with
no seed, so inserting one row re-pointed 1342 of 1351 ids while every foreign key still resolved
— an invisible, total loss of identity. The test below performs that exact edit and asserts zero
existing ids move.

The committed package is also asserted to be current: the render is taken from the files in the
tree, so a CSV edit that is not migrated turns this red rather than leaving a stale store behind.
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
from planktonzilla.planktonzilla_dataset.taxonomy import migrate

REAL_CSV = Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME)
PACKAGE = migrate.PACKAGE_DIR

# Measured on the committed table. Stated as a table rather than as loose asserts so a diff says
# which relation changed size.
EXPECTED_ROWS = {
    "taxon.tsv": 1352,
    # 4,032 at migration; +2 when the two family-level NCBI taxids the committed snapshot names
    # (418932 Syracosphaeraceae, 418941 Rhabdosphaeraceae) were anchored on the family they name,
    # which is what let their four holders be re-predicated to skos:broadMatch. No published byte
    # moved: neither family is a proposed_label, and the renderer never reads the predicate.
    "identifier.tsv": 4034,
    "merged.tsv": 0,
    "release/v1.0/legacy_row_order.tsv": 2358,
    "release/v1.0/legacy_overrides.tsv": 13,
}


@pytest.fixture(scope="module")
def legacy_rows():
    return migrate.read_legacy_rows(REAL_CSV)


@pytest.fixture
def regenerated(tmp_path, legacy_rows):
    """A package generated from scratch into a scratch directory, seeded by the committed one."""
    work = tmp_path / "data"
    shutil.copytree(PACKAGE, work)
    migrate.write_package(work, legacy_rows)
    return work


def _taxa(package):
    return {row["taxonID"]: row for row in migrate.read_tsv(package / "taxon.tsv")}


def test_the_committed_package_renders_the_committed_csv_byte_for_byte():
    """THE GATE. Read the files in the tree, render, compare bytes.

    Doubles as a staleness check: a CSV edit that was not migrated turns this red.
    """
    rendered = migrate.render_legacy_csv(PACKAGE)
    committed = REAL_CSV.read_bytes()

    assert len(rendered) == len(committed), f"rendered {len(rendered)} bytes, committed {len(committed)}"
    assert rendered == committed


def test_verify_reports_the_first_divergent_line_rather_than_a_bare_mismatch(tmp_path, regenerated):
    """A hash mismatch is not a diagnosis. Break one cell and read the message."""
    mapping = regenerated / "mappings" / "zooscan.tsv"
    mapping.write_text(mapping.read_text(encoding="utf-8").replace("\tfull_body\t", "\tegg\t", 1), encoding="utf-8")

    with pytest.raises(migrate.MigrationError) as failure:
        migrate.verify(regenerated, REAL_CSV)

    message = str(failure.value)
    assert "differs from the committed CSV at line" in message
    assert "got:" in message and "want:" in message


def test_running_the_migration_twice_is_a_byte_no_op(tmp_path, legacy_rows):
    """Re-running must change nothing — including the ids."""
    first, second = tmp_path / "first", tmp_path / "second"
    shutil.copytree(PACKAGE, first)
    migrate.write_package(first, legacy_rows)
    shutil.copytree(first, second)
    migrate.write_package(second, legacy_rows)

    for path in sorted(p for p in first.rglob("*") if p.is_file()):
        relative = path.relative_to(first)
        assert path.read_bytes() == (second / relative).read_bytes(), f"{relative} changed on a re-run"


def test_inserting_a_row_mints_only_new_ids_and_moves_none(regenerated, legacy_rows):
    """The failure the plan's risk R1 names, performed.

    A brand-new genus and species inserted as the FIRST row: under sorted-order minting with no
    seed, every id after it shifts and every foreign key still resolves, so nothing catches it.
    """
    before = {row["taxonID"]: row["scientificName"] for row in migrate.read_tsv(regenerated / "taxon.tsv")}

    donor = next(row for row in legacy_rows if row["Family"] == "abylidae" and row["Genus"])
    inserted = dict(donor)
    inserted.update(
        Dataset="synthetic",
        Raw_Labels="Aaa_novum",
        Genus="aaagenus",
        Species="novum",
        proposed_label="aaagenus novum",
        **dict.fromkeys(migrate.LEGACY_ID_COLUMNS, ""),
    )
    migrate.write_package(regenerated, [inserted, *legacy_rows])

    after = {row["taxonID"]: row["scientificName"] for row in migrate.read_tsv(regenerated / "taxon.tsv")}

    assert [i for i in before if i not in after] == [], "an existing id vanished"
    assert [(i, before[i], after[i]) for i in before if before[i] != after[i]] == [], "an existing id moved to another node"
    assert sorted(after[i] for i in set(after) - set(before)) == ["aaagenus", "aaagenus novum"]


def test_every_relation_has_the_measured_number_of_rows():
    for relative, expected in EXPECTED_ROWS.items():
        assert len(migrate.read_tsv(PACKAGE / relative)) == expected, f"{relative} changed size"

    mappings = sorted((PACKAGE / "mappings").glob("*.tsv"))
    assert len(mappings) == 21
    assert sum(len(migrate.read_tsv(path)) for path in mappings) == 2358


def test_the_frozen_prefix_is_in_collation_order(legacy_rows):
    """The positive claim behind the row-order manifest: the prefix is not arbitrary.

    It is non-descending under ``proposed_label.lower().replace("-", "")`` — one normalisation
    that absorbs both documented exceptions (``Eukaryota``'s capital, ``pseudo-nitzschia``'s
    hyphen). The report calls the prefix underivable; only the WITHIN-label order is.
    """
    migrate.check_prefix_collation(legacy_rows)

    keys = [migrate.collation_key(row["proposed_label"]) for row in legacy_rows[:1485]]
    assert keys == sorted(keys)
    assert migrate.collation_key("Eukaryota") == "eukaryota"
    assert migrate.collation_key("pseudo-nitzschia") == "pseudonitzschia"


def test_a_prefix_row_out_of_collation_order_is_refused(legacy_rows):
    """So a corrupted or hand-edited manifest is a failing check, not a silent reordering."""
    scrambled = [legacy_rows[900], *legacy_rows]

    with pytest.raises(migrate.MigrationError) as failure:
        migrate.check_prefix_collation(scrambled)

    assert "not in collation order at row 0" in str(failure.value)


def test_the_within_label_order_is_what_the_manifest_actually_carries(legacy_rows):
    """Measured: the manifest is not redundant, and this is precisely why."""
    from itertools import groupby

    needing, runs = 0, 0
    for _key, group in groupby(legacy_rows[:1485], key=lambda row: migrate.collation_key(row["proposed_label"])):
        block = list(group)
        if len(block) < 2:
            continue
        runs += 1
        if block != sorted(block, key=lambda row: (row["Dataset"], row["Raw_Labels"])):
            needing += 1

    assert (runs, needing) == (240, 56)


def test_every_foreign_key_resolves_and_the_tree_has_no_cycles():
    taxa = _taxa(PACKAGE)

    assert len(taxa) == len(migrate.read_tsv(PACKAGE / "taxon.tsv")), "duplicate taxonID"

    for row in taxa.values():
        parent = row["parentNameUsageID"]
        assert parent == "" or parent in taxa, f"{row['taxonID']} has a dangling parent {parent}"

    for taxon_id in taxa:
        seen, current = set(), taxon_id
        while current:
            assert current not in seen, f"parent cycle through {current}"
            seen.add(current)
            current = taxa[current]["parentNameUsageID"]

    for path in (PACKAGE / "mappings").glob("*.tsv"):
        for row in migrate.read_tsv(path):
            assert row["taxonID"] in taxa, f"{path.name}: mapping points at unknown {row['taxonID']}"

    subjects = {row["subject_id"] for row in migrate.read_tsv(PACKAGE / "identifier.tsv")}
    assert subjects <= set(taxa), "identifier.tsv names a taxon that does not exist"


def test_a_mapping_file_only_ever_holds_its_own_source():
    """``dataset == file stem``. A foreign row here passes a per-file primary key unnoticed."""
    for path in (PACKAGE / "mappings").glob("*.tsv"):
        assert {row["datasetID"] for row in migrate.read_tsv(path)} == {path.stem}


def test_the_identifier_table_carries_typed_ids_not_float_serialised_ones():
    """``135336.0`` is a CSV-typing artefact. It must not survive into the store."""
    prefixes = {prefix for _n, _l, prefix, _s, _m in migrate.AUTHORITIES}

    for row in migrate.read_tsv(PACKAGE / "identifier.tsv"):
        prefix, separator, value = row["object_id"].partition(":")
        assert separator == ":" and prefix in prefixes, f"unrecognised CURIE {row['object_id']}"
        assert not value.endswith(".0"), f"float-serialised id survived: {row['object_id']}"
        assert ";" not in value, f"repeating group survived: {row['object_id']}"


def test_kind_is_derived_from_the_lineage_and_never_from_root_class():
    """Design D typed egg / larvae / mix non-living and mis-pointed 45 living mappings.

    ``kind`` here says only whether the concept has a lineage. Living-ness stays on the mapping,
    which is the only place it is actually a function of anything.
    """
    taxa = _taxa(PACKAGE)
    kinds = Counter(row["kind"] for row in taxa.values())
    assert kinds == {"taxon": 1309, "bucket": 43}

    by_name = {row["scientificName"]: row for row in taxa.values() if row["kind"] == "bucket"}
    assert {"egg", "larvae", "mix", "other", "unknown", "detritus", "plastic"} <= set(by_name)

    root_classes = {}
    for path in (PACKAGE / "mappings").glob("*.tsv"):
        for row in migrate.read_tsv(path):
            root_classes.setdefault(row["concept"], set()).add(row["root_class"])

    # The three that span root classes are exactly why kind cannot encode living-ness.
    assert root_classes["mix"] == {"artefact", "detritus", "living"}
    assert root_classes["other"] == {"artefact", "detritus", "living"}
    assert root_classes["unknown"] == {"artefact", "living"}


def test_every_concept_has_its_own_node():
    """Concept -> taxon is injective, which the raw lineage is not: 10 nodes were shared.

    The 12 concepts finer than their deepest legacy rank get an ``unranked`` child rather than
    being folded into it, so ``brachyura``, ``achelata`` and ``decapoda`` stay three concepts.
    """
    taxa = _taxa(PACKAGE)
    concepts = {}
    for path in (PACKAGE / "mappings").glob("*.tsv"):
        for row in migrate.read_tsv(path):
            concepts.setdefault(row["concept"], set()).add(row["taxonID"])

    assert len(concepts) == 907
    assert all(len(ids) == 1 for ids in concepts.values()), "a concept resolves to two taxa"
    assert len({next(iter(ids)) for ids in concepts.values()}) == 907, "two concepts share a taxon"

    for concept, ids in concepts.items():
        assert taxa[next(iter(ids))]["scientificName"] == concept, f"{concept} names a node called something else"

    unranked = [row for row in taxa.values() if row["taxonRank"] == "unranked"]
    assert len(unranked) == 55  # 43 buckets + the 12 finer-than-their-rank concepts
    assert len([row for row in unranked if row["kind"] == "taxon"]) == 12


def test_species_are_binomials_whose_epithet_is_derivable():
    """The legacy ``Species`` cell is a projection, not an identity: 19 epithets are shared."""
    taxa = _taxa(PACKAGE)
    species = [row for row in taxa.values() if row["taxonRank"] == "species"]
    assert len(species) == 364

    for row in species:
        parent = taxa[row["parentNameUsageID"]]
        assert parent["taxonRank"] == "genus", f"{row['scientificName']} is not parented to a genus"
        assert row["scientificName"].startswith(f"{parent['scientificName']} "), row["scientificName"]

    tautonyms = [row["scientificName"] for row in species if len(set(row["scientificName"].split())) == 1]
    assert "porpita porpita" in tautonyms


def test_the_legacy_id_overrides_are_the_thirteen_frepj_rows():
    """All 13 come from ids blanked per row instead of per taxon (commit 7262085)."""
    overrides = migrate.read_tsv(PACKAGE / "release" / "v1.0" / "legacy_overrides.tsv")

    assert {row["datasetID"] for row in overrides} == {"frepj"}
    assert all(row["reason"] for row in overrides), "an override with no reason"
    # The three mapping-pin columns exist and are empty, so a curator can pin a correction
    # without turning the byte gate red (plan risk R8).
    assert all(row["root_class"] == row["qualifier"] == row["plankton"] == "" for row in overrides)


def test_the_release_sha_covers_the_row_order_keys_only():
    """Not a whole-file hash: design B's gate hashed the render and a new source turned it red."""
    import hashlib

    order = migrate.read_tsv(PACKAGE / "release" / "v1.0" / "legacy_row_order.tsv")
    expected = hashlib.sha256(
        "\n".join(f"{row['datasetID']}\t{row['verbatimIdentification']}" for row in order).encode("utf-8")
    ).hexdigest()

    assert (PACKAGE / "release" / "v1.0" / "sha256").read_text(encoding="utf-8").strip() == expected


def test_a_tab_or_newline_in_a_value_is_refused_rather_than_escaped(tmp_path):
    """Canonical TSV has no quoting, so an unescapable value must fail loudly."""
    with pytest.raises(migrate.MigrationError) as failure:
        migrate.write_tsv(tmp_path / "x.tsv", ("a",), [{"a": "has\ttab"}])

    assert "contains a tab or newline" in str(failure.value)


def test_retiring_an_identifier_is_refused_rather_than_inferred_from_an_absence(regenerated, legacy_rows):
    """Dropping a row must not silently retire the node's id.

    The same class of loss as R1, from the other direction: minting is stable, but a node that
    simply stops appearing would take its identity with it. A retirement is a versioned act and
    belongs in ``merged.tsv``, written by step 6's ``rename`` / ``merge``.
    """
    surviving = [row for row in legacy_rows if row["proposed_label"] != "alciopini"]
    assert len(surviving) < len(legacy_rows)

    with pytest.raises(migrate.MigrationError) as failure:
        migrate.write_package(regenerated, surviving)

    message = str(failure.value)
    assert "absent from this run" in message
    assert "merged.tsv" in message
    assert "alciopini" in message


def test_a_second_id_for_a_single_valued_authority_is_refused(regenerated):
    """Schema-legal in any SSSOM-shaped table, and silently dropped by a naive renderer.

    Which of the two survives would be decided by sort order rather than by the model. The
    renderer refuses instead, so the ambiguity cannot reach the published columns.
    """
    identifiers = regenerated / "identifier.tsv"
    rows = migrate.read_tsv(identifiers)
    worms = next(row for row in rows if row["object_id"].startswith("worms:"))
    rows.append({**worms, "object_id": "worms:999999999"})
    rows.sort(key=lambda row: (row["subject_id"], row["predicate_id"], row["object_id"]))
    migrate.write_tsv(identifiers, migrate.IDENTIFIER_COLUMNS, rows)

    with pytest.raises(migrate.MigrationError) as failure:
        migrate.render_legacy_csv(regenerated)

    assert "single-valued but carries 2 ids" in str(failure.value)


def test_a_species_with_no_genus_ancestor_is_refused(regenerated):
    """The epithet is derived by removing the genus prefix, so a genus-less species is undefined.

    Zero violations today — all 364 species are parented to a genus — which is exactly why the
    case has to fail loudly rather than emit the binomial into the epithet cell.
    """
    taxa = migrate.read_tsv(regenerated / "taxon.tsv")
    by_id = {row["taxonID"]: row for row in taxa}
    species = next(row for row in taxa if row["taxonRank"] == "species")
    genus = by_id[species["parentNameUsageID"]]
    species["parentNameUsageID"] = genus["parentNameUsageID"]  # reparent onto the family
    migrate.write_tsv(regenerated / "taxon.tsv", migrate.TAXON_COLUMNS, taxa)

    with pytest.raises(migrate.MigrationError) as failure:
        migrate.render_legacy_csv(regenerated)

    assert "has no genus ancestor" in str(failure.value)
