"""
(c) Inria

THE GOLDEN GATE — step 2 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``.

Step 1 built the normalised package and proved its own migration could render the committed CSV
back. This module holds the SHIPPED read path to the same standard, and adds the three
contracts a renderer alone cannot cover: the 16-column lookup, the label vocabularies, and the
projection fixed point. Together they are what lets step 3 move nine CSV readers onto one loader
without a single published byte moving.

Five assertions, one per frozen contract:

1. ``load_taxonomy().render_wide_csv()`` equals the committed CSV byte for byte, and says WHICH
   row diverged when it does not.
2. The first 1486 rendered lines match the sha pin in ``tests/fixtures/frepj/``.
3. The 16-column lookup built from the MODEL — not from rendered bytes — equals
   ``build_taxonomy_lookup`` in value and in Python type on all 2358 keys.
4. The 850 / 599 label vocabularies reproduce from the model, in order.
5. ``project7`` is a fixed point: the ranks the model projects equal the ranks the CSV carries.

On (1) and (3), two distinctions are load-bearing and easy to lose:

*Files, not objects.* Everything here loads the package from disk. A store that renders from the
objects that built it proves only that the builder agrees with itself — the refuters found
design A's prototype had been hand-finished after its generator ran.

*Model, not bytes.* The lookup in (3) comes from ``store.rows()``, which :func:`render.legacy_rows`
computes from the taxon tree, the identifier index and the override layer. It never round-trips
through serialised CSV, which would only prove the renderer is its own inverse.

What this gate is NOT is a whole-file hash. Design B's was, and adding a 50-class source turned
it red and crashed its own error reporter. Whole-file hashing belongs only in the per-release sha
over that release's own row-order keys, which step 1 already writes.
"""

import hashlib
import json
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
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as legacy
from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR, TaxonomyError, load_taxonomy, render
from planktonzilla.planktonzilla_dataset.taxonomy.model import LEGACY_RANKS, LOOKUP_COLUMNS

REAL_CSV = Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME)
PREFIX_SHA = Path(__file__).parent / "fixtures" / "frepj" / "pre_frepj_taxonomy.sha256"
# Step 8 moved the released vocabularies into the package, where the build reads them.
VOCABULARY = Path(root) / "planktonzilla" / "planktonzilla_dataset" / "taxonomy" / "data" / "vocab" / "labels"
SAMPLES_JSON = Path(root) / "samples.json"

# The pristine CSV is exactly 1486 lines; every source since is appended after it.
PRISTINE_LINE_COUNT = 1486


@pytest.fixture(scope="module")
def store():
    """The bundled package, loaded from the files in the tree."""
    return load_taxonomy()


@pytest.fixture(scope="module")
def wide_store():
    """The same taxonomy through the legacy wide-CSV backend."""
    return load_taxonomy(REAL_CSV)


def _published_datasets():
    return {row["dataset"] for row in json.loads(SAMPLES_JSON.read_text())["counts"]}


def _committed_vocabulary(tag):
    """The names a release published, in class-id order, straight off the shipped file."""
    lines = (VOCABULARY / f"{tag}_taxpath.tsv").read_text(encoding="utf-8").split("\n")
    return [line.partition("\t")[2] for line in lines[1:-1]]


# 1. The render
def test_the_package_renders_the_committed_csv_byte_for_byte(store):
    """THE GATE, through the shipped loader and renderer rather than the migration's own."""
    rendered = store.render_wide_csv()
    committed = REAL_CSV.read_bytes()

    assert len(rendered) == len(committed), f"rendered {len(rendered)} bytes, committed {len(committed)}"
    assert rendered == committed


def test_a_divergent_render_names_the_row_rather_than_the_hash(tmp_path, store):
    """A bare mismatch is not a diagnosis; the first divergent line is.

    Verified by breaking one cell of a scratch copy of the package, so the message is exercised
    rather than asserted about.
    """
    from planktonzilla.planktonzilla_dataset.taxonomy import migrate

    work = tmp_path / "data"
    __import__("shutil").copytree(PACKAGE_DIR, work)
    mapping = work / "mappings" / "zooscan.tsv"
    mapping.write_text(mapping.read_text(encoding="utf-8").replace("\tfull_body\t", "\tegg\t", 1), encoding="utf-8")

    with pytest.raises(TaxonomyError) as failure:
        migrate.verify(work, REAL_CSV)

    message = str(failure.value)
    assert "differs from the committed CSV at line" in message
    assert "got:" in message and "want:" in message


# 2. The frozen prefix
def test_the_rendered_prefix_matches_the_existing_sha_pin(store):
    """The first 1486 lines are byte-frozen forever, and the render must honour that pin.

    Not a hash of the whole render: a new source appends and must not turn this red.
    """
    rendered_lines = store.render_wide_csv().split(b"\n")[:PRISTINE_LINE_COUNT]
    prefix = b"".join(line + b"\n" for line in rendered_lines)

    assert hashlib.sha256(prefix).hexdigest() == PREFIX_SHA.read_text().split()[0]


# 3. The lookup
def test_the_lookup_built_from_the_model_matches_the_legacy_reader(store):
    """Value AND Python type, on all 2358 keys, over all 16 columns.

    The type half is not pedantry: ``plankton`` is a real ``bool`` and every id is a
    decimal-free ``str``. A column that silently became ``int`` or ``float`` would publish
    different values into 17.4 M rows with every value comparison still passing.
    """
    expected = legacy.build_taxonomy_lookup(str(REAL_CSV))
    actual = store.lookup()

    assert set(actual) == set(expected), "key sets differ"
    assert len(actual) == 2358

    value_diffs, type_diffs = [], []
    for key in expected:
        for column in LOOKUP_COLUMNS:
            got, want = actual[key][column], expected[key][column]
            if got != want:
                value_diffs.append((key, column, got, want))
            elif type(got) is not type(want):
                type_diffs.append((key, column, type(got), type(want)))

    assert not value_diffs, f"{len(value_diffs)} value differences, first 5: {value_diffs[:5]}"
    assert not type_diffs, f"{len(type_diffs)} type differences, first 5: {type_diffs[:5]}"


def test_the_wide_backend_produces_the_same_lookup_as_the_package(store, wide_store):
    """Either backend, one answer — which is what makes ``taxonomy_csv_path`` safe to keep."""
    from_package, from_csv = store.lookup(), wide_store.lookup()
    assert from_csv == from_package

    type_diffs = [
        (key, column)
        for key in from_package
        for column in LOOKUP_COLUMNS
        if type(from_csv[key][column]) is not type(from_package[key][column])
    ]
    assert not type_diffs, f"{len(type_diffs)} type differences, first 5: {type_diffs[:5]}"


def test_the_lookup_column_order_is_not_the_csv_or_published_order():
    """Three orders exist. Conflating two swaps id values between columns, silently."""
    assert LOOKUP_COLUMNS[-5:] == ("wikidata_ID", "ecotaxa_ID", "aphia_ID", "NCBI_ID", "BOLD_ID")
    assert render.LEGACY_ID_COLUMNS == ("wikidata_ID", "aphia_ID", "NCBI_ID", "BOLD_ID", "ecotaxa_ID")
    published = [c for c in constants.CONSOLIDATED_COLUMNS if c in LOOKUP_COLUMNS]
    assert published[-5:] == ["wikidata_ID", "ecotaxa_ID", "aphia_ID", "NCBI_ID", "BOLD_ID"]


# 4. The label vocabularies
def test_the_label_vocabularies_reproduce_from_the_model(store):
    """850 over 21 sources, 599 over the published 15 — the class ids of the released models."""
    rows = store.rows()

    v1_2 = render.label_vocabulary(rows)
    v1_0 = render.label_vocabulary(rows, datasets=_published_datasets())

    assert v1_2 == _committed_vocabulary("v1.2")
    assert v1_0 == _committed_vocabulary("v1.0")
    assert (len(v1_2), len(v1_0)) == (850, 599)
    assert v1_2[0] == "" and v1_0[0] == ""


# 5. The projection fixed point
def test_project7_is_a_fixed_point(store, wide_store):
    """The ranks the tree projects equal the ranks the CSV carries, cell for cell.

    Stronger than the byte render it sits beside: the render could agree while the projection
    was wrong, if the override layer happened to paper over it. This compares the seven slots
    alone, on every row.
    """
    projected = store.rows()
    committed = wide_store.rows()

    assert len(projected) == len(committed) == 2358
    mismatches = [
        (got["Dataset"], got["Raw_Labels"], rank, got[rank], want[rank])
        for got, want in zip(projected, committed)
        for rank in LEGACY_RANKS
        if got[rank] != want[rank]
    ]
    assert not mismatches, f"{len(mismatches)} rank cells differ, first 5: {mismatches[:5]}"


def test_an_unranked_node_contributes_no_legacy_cell(store):
    """Why the 12 finer-than-their-rank concepts cost the frozen columns nothing."""
    unranked = [t for t in store.taxa.values() if t.rank == "unranked" and t.kind == "taxon"]
    assert len(unranked) == 12

    for taxon in unranked:
        slots = render.project7(store, taxon.taxon_id)
        assert taxon.scientific_name not in slots.values(), f"{taxon.scientific_name} leaked into a legacy slot"
        parent_slots = render.project7(store, taxon.parent_id)
        assert slots == parent_slots, "an unranked child must project exactly as its parent"


def test_a_bucket_projects_seven_blanks(store):
    """KI-9's shape: a lineage-less concept renders with every rank column empty."""
    buckets = [t for t in store.taxa.values() if t.kind == "bucket"]
    assert len(buckets) == 43

    for taxon in buckets:
        assert set(render.project7(store, taxon.taxon_id).values()) == {""}


# The store's other views
def test_the_published_projection_uses_the_consolidated_column_order(store):
    """``plankton`` is a bool and ``living`` is excluded — the published contract, not the CSV's."""
    projected = render.published_projection(store.rows()[:50])

    assert "living" not in projected[0]
    assert list(projected[0]) == [c for c in constants.CONSOLIDATED_COLUMNS if c in LOOKUP_COLUMNS]
    assert all(isinstance(row["plankton"], bool) for row in projected)


def test_the_coverage_helpers_answer_what_preflight_asks(store):
    """What ``check_taxonomy_csv`` needs, without it opening the file itself."""
    assert len(store.datasets()) == 21
    assert "zooscan" in store.datasets()
    assert store.has("zooscan", "Acartiidae")
    assert not store.has("zooscan", "no such class dir")
    assert len(store.labels_for("frepj")) == 229


def test_the_frame_view_is_all_strings(store):
    """``verify_taxonomy_ids`` reads ``pl.read_csv(..., infer_schema_length=0)``; this replaces it."""
    import polars as pl

    frame = store.frame()
    assert frame.height == 2358
    assert set(frame.schema.values()) == {pl.Utf8}
    assert frame.columns == list(render.LEGACY_HEADER)


# The wide backend's tolerances — the ten fixture-writing test modules depend on every one
def test_the_wide_backend_accepts_an_eighteen_column_fixture(tmp_path):
    """Capitalised ranks, ``root_class: zoo``, no ``living``, a different id order, a ';' list.

    Transcribed from ``tests/test_gen_planktonzilla_hydra.py``. The loader must keep accepting
    it verbatim, or step 3 breaks ten test modules that have nothing to do with the migration.
    """
    path = tmp_path / "fixture.csv"
    path.write_text(
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID\n"
        "src,lbl,Animalia,Arthropoda,,,,,,Copepoda,True,zoo,,Q3386609,274;1231,135336.0,6854.0,\n"
    )

    values = load_taxonomy(path).lookup()[("src", "lbl")]

    assert values["Kingdom"] == "Animalia"
    assert values["root_class"] == "zoo"
    assert values["plankton"] is True
    assert values["qualifier"] is None
    assert values["ecotaxa_ID"] == "274;1231"
    assert values["aphia_ID"] == "135336"
    assert values["BOLD_ID"] is None
    assert values["Species"] is None


def test_the_wide_backend_is_a_passthrough_not_a_reserialisation(wide_store):
    """Loading and rendering a legacy file must not perturb one byte of it."""
    assert wide_store.render_wide_csv() == REAL_CSV.read_bytes()


def test_a_duplicate_key_keeps_the_last_row_and_warns(tmp_path, caplog):
    """Matching the legacy reader exactly: warn, keep last. The guard belongs at the store."""
    path = tmp_path / "dup.csv"
    path.write_text(
        "Dataset,Raw_Labels,Kingdom,proposed_label,plankton,root_class\n"
        "src,lbl,animalia,first,True,living\nsrc,lbl,chromista,second,False,detritus\n"
    )

    with caplog.at_level("WARNING"):
        values = load_taxonomy(path).lookup()[("src", "lbl")]

    assert values["proposed_label"] == "second"
    assert "duplicate" in caplog.text.lower()


def test_a_file_that_is_not_a_taxonomy_csv_is_refused(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n")

    with pytest.raises(TaxonomyError, match="not a taxonomy CSV"):
        load_taxonomy(path)


def test_a_missing_source_is_refused(tmp_path):
    with pytest.raises(TaxonomyError, match="no taxonomy at"):
        load_taxonomy(tmp_path / "absent")


def test_the_legacy_reader_signature_is_served_by_the_loader():
    """``build_taxonomy_lookup(path)`` keeps working, so step 3 changes no caller's import."""
    from planktonzilla.planktonzilla_dataset.taxonomy import build_taxonomy_lookup

    assert build_taxonomy_lookup(REAL_CSV) == legacy.build_taxonomy_lookup(str(REAL_CSV))


# What the loader refuses, so a malformed package cannot reach a consumer
def _broken(tmp_path, edit):
    import shutil

    work = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, work)
    edit(work)
    return work


def test_a_mapping_whose_display_name_disagrees_with_its_id_is_a_hard_error(tmp_path):
    """The stated conflict rule: blank id resolves by name; both set and disagreeing raises.

    Never an overwrite — a display name silently winning would let a rename half-apply.
    """

    def edit(work):
        path = work / "mappings" / "zooscan.tsv"
        path.write_text(path.read_text(encoding="utf-8").replace("\tacartiidae\t", "\tnot-acartiidae\t", 1), encoding="utf-8")

    with pytest.raises(TaxonomyError, match="names concept"):
        load_taxonomy(_broken(tmp_path, edit))


def test_a_foreign_row_in_a_mapping_file_is_refused(tmp_path):
    """``dataset == file stem``. A per-file primary key cannot see this."""

    def edit(work):
        path = work / "mappings" / "zoolake.tsv"
        rows = path.read_text(encoding="utf-8").split("\n")
        rows[1] = rows[1].replace("zoolake\t", "zooscan\t", 1)
        path.write_text("\n".join(rows), encoding="utf-8")

    with pytest.raises(TaxonomyError, match="holds a row for"):
        load_taxonomy(_broken(tmp_path, edit))


def test_a_spreadsheet_boolean_is_refused_rather_than_coerced(tmp_path):
    """Excel writes TRUE for true. Coercing inverts the only boolean the published set carries."""

    def edit(work):
        path = work / "mappings" / "lensless.tsv"
        path.write_text(path.read_text(encoding="utf-8").replace("\ttrue\t", "\tTRUE\t", 1), encoding="utf-8")

    with pytest.raises(TaxonomyError, match="expected 'true' or 'false'"):
        load_taxonomy(_broken(tmp_path, edit))


def test_a_dangling_parent_is_refused(tmp_path):
    def edit(work):
        path = work / "taxon.tsv"
        rows = path.read_text(encoding="utf-8").split("\n")
        fields = rows[60].split("\t")
        fields[1] = "pzt:999999"
        rows[60] = "\t".join(fields)
        path.write_text("\n".join(rows), encoding="utf-8")

    with pytest.raises(TaxonomyError, match="dangling parent"):
        load_taxonomy(_broken(tmp_path, edit))


def test_an_edited_frozen_row_order_is_refused_by_the_release_pin(tmp_path):
    """A frozen release's order is frozen. Dropping its last key is still dropping a published row.

    The pin is over the manifest's own keys, NOT over the render — design B hashed the render and
    adding one source turned its gate red. Hashing the manifest is what lets the package grow while
    the release it froze stays provably the one that was published.
    """

    def edit(work):
        path = work / "release" / "v1.0" / "legacy_row_order.tsv"
        rows = path.read_text(encoding="utf-8").split("\n")
        path.write_text("\n".join([*rows[:-2], ""]), encoding="utf-8")

    with pytest.raises(TaxonomyError, match="has drifted"):
        load_taxonomy(_broken(tmp_path, edit))


def test_a_frozen_row_order_key_no_mapping_carries_is_refused(tmp_path):
    """The manifest and the mapping files are two halves of one fact; a drift is not renderable."""

    def edit(work):
        path = work / "mappings" / "zoolake.tsv"
        rows = path.read_text(encoding="utf-8").split("\n")
        fields = rows[1].split("\t")
        fields[1] = f"{fields[1]} (renamed upstream)"
        rows[1] = "\t".join(fields)
        path.write_text("\n".join(rows), encoding="utf-8")

    with pytest.raises(TaxonomyError, match="which no mapping file carries"):
        load_taxonomy(_broken(tmp_path, edit))


def test_a_published_row_walked_back_to_draft_is_refused(tmp_path):
    """``draft`` lets a partial curation be committed; it is not a way to unpublish a released row.

    Without this the row simply stops rendering — a published class silently gone from the dataset,
    with nothing in the diff but a one-word status change.
    """

    def edit(work):
        path = work / "mappings" / "zoolake.tsv"
        rows = path.read_text(encoding="utf-8").split("\n")
        fields = rows[1].split("\t")
        fields[7] = "draft"
        rows[1] = "\t".join(fields)
        path.write_text("\n".join(rows), encoding="utf-8")

    with pytest.raises(TaxonomyError, match="published in the frozen row order"):
        load_taxonomy(_broken(tmp_path, edit))


def test_a_source_mapped_after_the_freeze_renders_after_the_frozen_block(tmp_path):
    """The package has to be loadable BETWEEN adding a source and cutting the next release.

    Otherwise every consumer is broken for the length of a curation, and ``upsert_source`` is
    unusable. Rows the frozen manifest does not name follow it, ordered by the collation key the
    frozen prefix already obeys — appended, never interleaved, so a new source can never push a
    frozen row down the file and past the 1486-line prefix pin.
    """
    from planktonzilla.planktonzilla_dataset.taxonomy.model import MAPPING_COLUMNS, read_tsv, write_tsv

    def edit(work):
        template = read_tsv(work / "mappings" / "zoolake.tsv")[0]
        write_tsv(
            work / "mappings" / "newsource.tsv",
            MAPPING_COLUMNS,
            [
                # 'abylidae' sorts first under the collation key and would land at line 2 if
                # post-freeze rows were interleaved rather than appended. The verbatim order is the
                # reverse, so a file that merely kept insertion order would fail too.
                {
                    **template,
                    "datasetID": "newsource",
                    "verbatimIdentification": "b",
                    "concept": "abylidae",
                    "taxonID": "pzt:000382",
                },
                {
                    **template,
                    "datasetID": "newsource",
                    "verbatimIdentification": "a",
                    "concept": "zygodiscales",
                    "taxonID": "pzt:000874",
                },
            ],
        )

    store = load_taxonomy(_broken(tmp_path, edit))
    lines = store.render_wide_csv().decode("utf-8").split("\n")

    assert lines[:1487] == REAL_CSV.read_bytes().decode("utf-8").split("\n")[:1487], "a frozen row moved"
    assert [row["Dataset"] for row in store.rows()[:2358]] == [row["Dataset"] for row in load_taxonomy(REAL_CSV).rows()]
    assert [(row["Dataset"], row["Raw_Labels"]) for row in store.rows()[2358:]] == [
        ("newsource", "b"),
        ("newsource", "a"),
    ], "post-freeze rows are not in collation order"


# 8. The override layer's three attribute columns


def test_a_blank_attribute_override_leaves_the_model_value_alone(store):
    """All thirteen committed override rows are blank in those columns, and must stay inert.

    Blank means one thing for ids and the opposite for attributes, which is the whole design of
    ``_pinned``: a blank id override PUBLISHES a blank (that is what these thirteen rows exist to
    do, one per row rather than per taxon), while a blank attribute override means "no opinion".
    Reading blank the id way would have blanked ``root_class``, ``qualifier`` and ``plankton`` on
    thirteen published rows the moment the columns were wired up. This pins that they did not move.
    """
    assert all(
        not (row.get(column) or "").strip()
        for row in store.overrides.values()
        for column in ("root_class", "qualifier", "plankton")
    ), "an override now pins an attribute — this test's premise, and the sha pin, both need re-checking"

    overridden = {key for key in store.overrides}
    rendered = {(row["Dataset"], row["Raw_Labels"]): row for row in store.rows()}
    for dataset, verbatim in overridden:
        row = rendered[(dataset, verbatim)]
        assert row["root_class"] and row["plankton"] in ("True", "False")


def test_a_pinned_attribute_reaches_the_render(tmp_path):
    """The columns the loader reads and the renderer used to drop.

    ``legacy_overrides.tsv`` declares ``root_class`` / ``qualifier`` / ``plankton`` and the loader
    parses all three, but ``legacy_rows`` applied only ``LEGACY_ID_COLUMNS`` — so a curator pinning
    a mapping attribute for a release had it silently discarded, with the release pin's sha still
    green because the render never changed. Exercised on a scratch copy, because writing a real
    value into the committed table is a published-cell change and gated on the golden diff.
    """

    def edit(work):
        path = work / "release" / "v1.0" / "legacy_overrides.tsv"
        lines = path.read_text(encoding="utf-8").split("\n")
        header = lines[0].split("\t")
        fields = lines[1].split("\t")
        fields[header.index("root_class")] = "artefact"
        fields[header.index("qualifier")] = "part"
        fields[header.index("plankton")] = "False"
        lines[1] = "\t".join(fields)
        path.write_text("\n".join(lines), encoding="utf-8")

    pinned = load_taxonomy(_broken(tmp_path, edit))
    key = next(iter(pinned.overrides))
    row = {(r["Dataset"], r["Raw_Labels"]): r for r in pinned.rows()}[key]

    assert row["root_class"] == "artefact"
    assert row["qualifier"] == "part"
    assert row["plankton"] == "False"
    # `living` is not pinnable: it is not an independent column, it IS root_class == "living",
    # so it must follow the pin rather than keep the model's answer.
    assert row["living"] == "False"

    # And the pin is per row, not per taxon: no other row moved.
    baseline = {(r["Dataset"], r["Raw_Labels"]): r for r in load_taxonomy().rows()}
    after = {(r["Dataset"], r["Raw_Labels"]): r for r in pinned.rows()}
    assert {k for k, r in after.items() if r != baseline[k]} == {key}
