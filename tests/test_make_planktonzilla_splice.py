"""
(c) Inria

Offline, real-data tests for the splice performed by ``pz_planktonzilla``.

Where ``tests/test_make_planktonzilla_hydra.py`` mocks the pipeline to pin the wiring,
this drives the REAL imagefolder import + redefine + splice path over two source
datasets and asserts on the reloaded dataset. The headline test is
``test_incremental_build_equals_full_build``: building both sources at once and
building them one at a time must produce the same rows, with the same values, in the
same ORDER. That is the property the registry-order reassembly exists to provide, and
without it a per-source refresh could not be trusted against the published artifact.

Offline by construction: both fixtures use ``redefiner: none``
(``NoMetadataRedefiner``), which never touches the network, and both imagefolders are
pre-created so ``import_dataset()`` is never invoked. ``requests`` is monkeypatched to
raise, to PROVE it.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import datasets
import huggingface_hub.constants
import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image as PILImage

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp
from planktonzilla.planktonzilla_dataset import make_planktonzilla as mk

# (name, import_name, importer class name) for the two offline sources.
LENSLESS = ("lensless", "lensless", "lenslessdatasetimporter_imagefolder")
ISIIS = ("isiisnet", "isiisnet", "isiisnetdatasetimporter_imagefolder")

CSV_HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,root_class,qualifier,"
    "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
)


def _csv_row(dataset_name, raw_label, proposed="Copepoda", phylum="Arthropoda"):
    return f"{dataset_name},{raw_label},Animalia,{phylum},,,,,,{proposed},True,zoo,,Q3386609,274;1231,135336.0,6854.0,"


def _write_csv(path, rows):
    path.write_text(CSV_HEADER + "\n" + "\n".join(rows) + "\n")


def _write_png(path, color=(120, 160, 200), size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (size, size), color=color).save(path)


def _make_imagefolder(data_dir, folder_name, class_pngs):
    """Create ``<data_dir>/<folder_name>/<class>/img_<i>.png``."""
    imagefolder = data_dir / folder_name
    for class_name, n in class_pngs.items():
        for i in range(n):
            _write_png(imagefolder / class_name / f"img_{i}.png")
    return imagefolder


@pytest.fixture
def offline(monkeypatch):
    """Force the whole run offline and make any network call an immediate failure."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True, raising=False)

    def _no_network(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(gp.requests, "get", _no_network)
    monkeypatch.setattr(gp.requests.Session, "get", _no_network)

    # Determinism: the module-level global read by _serialize / _flatten_metadata.
    monkeypatch.setattr(gp, "num_proc", 1)


def _compose(overrides, job_name):
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name=job_name)
    cfg = hydra.compose(config_name="planktonzilla", overrides=list(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def _restrict_registry(cfg, names):
    """Keep only the named entries of the registry, preserving declaration order."""
    cfg.datasets = [d for d in cfg.datasets if d["name"] in names]
    assert [d["name"] for d in cfg.datasets] == list(names)
    return cfg


def _run(cfg):
    mk.main(cfg)
    GlobalHydra.instance().clear()


def _load(path):
    ds = datasets.load_from_disk(str(path))
    if isinstance(ds, datasets.DatasetDict):
        ds = ds["train"]
    return ds


def _rows(ds, drop_image=True):
    """Materialise rows as plain dicts, replacing the image with a content digest."""
    out = []
    for row in ds:
        row = dict(row)
        image = row.pop("image")
        if not drop_image:
            row["_image_bytes"] = image.tobytes()
        out.append(row)
    return out


@pytest.fixture
def two_source_env(tmp_path):
    """Two pre-built imagefolders plus a taxonomy CSV covering some of their classes."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    _make_imagefolder(data_dir, LENSLESS[2], {"copepoda": 3, "diatom": 2})
    _make_imagefolder(data_dir, ISIIS[2], {"appendicularia": 2, "unknown_blob": 1})

    csv_path = tmp_path / "taxo.csv"
    _write_csv(
        csv_path,
        [
            _csv_row("lensless", "copepoda"),
            _csv_row("isiisnet", "appendicularia", proposed="Appendicularia", phylum="Chordata"),
        ],
    )
    return data_dir, csv_path


def test_registry_order_and_schema_are_what_the_splice_assumes(offline, two_source_env, tmp_path):
    """A full 2-source build produces the canonical column set, in registry order.

    Guards the two assumptions every other test here rests on: that a rebuilt part
    matches ``constants.CONSOLIDATED_COLUMNS`` exactly, and that rows come out grouped
    by source in registry order.
    """
    data_dir, csv_path = two_source_env
    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1", f"output_dir={tmp_path / 'out'}"],
        "test_splice_schema",
    )
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    ds = _load(tmp_path / "out")

    assert set(ds.column_names) == set(constants.CONSOLIDATED_COLUMNS)
    assert [row["dataset"] for row in ds] == ["isiisnet"] * 3 + ["lensless"] * 5


def test_incremental_build_equals_full_build(offline, two_source_env, tmp_path):
    """Building both sources at once == building them one at a time, row for row.

    THE test for the splice. Run A builds isiisnet+lensless together. Run B builds
    isiisnet alone, then splices lensless into it. The two outputs must be identical
    in schema, length, row order and every value — including image bytes.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    # Run A: both at once.
    cfg_a = _compose([*common, f"output_dir={tmp_path / 'A'}"], "test_splice_full")
    _restrict_registry(cfg_a, ["isiisnet", "lensless"])
    _run(cfg_a)

    # Run B, step 1: isiisnet only.
    cfg_b1 = _compose([*common, f"output_dir={tmp_path / 'B1'}", "sources=[isiisnet]"], "test_splice_b1")
    _restrict_registry(cfg_b1, ["isiisnet", "lensless"])
    _run(cfg_b1)

    # Run B, step 2: add lensless on top of that base.
    cfg_b2 = _compose(
        [*common, f"output_dir={tmp_path / 'B2'}", "sources=[lensless]", f"base={tmp_path / 'B1'}"],
        "test_splice_b2",
    )
    _restrict_registry(cfg_b2, ["isiisnet", "lensless"])
    _run(cfg_b2)

    ds_a, ds_b = _load(tmp_path / "A"), _load(tmp_path / "B2")

    assert ds_a.column_names == ds_b.column_names
    assert ds_a.features == ds_b.features
    assert len(ds_a) == len(ds_b) == 8

    rows_a = _rows(ds_a, drop_image=False)
    rows_b = _rows(ds_b, drop_image=False)
    assert rows_a == rows_b, "incremental build diverged from the from-scratch build"


def test_refresh_replaces_exactly_its_own_rows(offline, two_source_env, tmp_path):
    """Refreshing one source rewrites its rows, in place, and leaves the others untouched."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_splice_refresh_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    before = _load(tmp_path / "base")
    isiis_before = [r for r in _rows(before, drop_image=False) if r["dataset"] == "isiisnet"]

    # Mutate the lensless imagefolder: a new class appears, an old image disappears.
    lensless_dir = data_dir / LENSLESS[2]
    _write_png(lensless_dir / "newclass" / "img_0.png", color=(10, 20, 30))
    (lensless_dir / "diatom" / "img_1.png").unlink()

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[lensless]", f"base={tmp_path / 'base'}", "refresh=rebuild"],
        "test_splice_refresh",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    after = _load(tmp_path / "after")
    rows_after = _rows(after, drop_image=False)

    # isiisnet rows survive byte-identically AND stay in their original positions.
    isiis_after = [r for r in rows_after if r["dataset"] == "isiisnet"]
    assert isiis_after == isiis_before
    assert [r["dataset"] for r in rows_after][:3] == ["isiisnet"] * 3

    # lensless now reflects the mutated imagefolder exactly.
    lensless_after = [r for r in rows_after if r["dataset"] == "lensless"]
    assert sorted(r["original_label"] for r in lensless_after) == ["copepoda"] * 3 + ["diatom"] + ["newclass"]
    assert len(lensless_after) == 5


def test_taxonomy_only_run_preserves_rows_and_order(offline, two_source_env, tmp_path):
    """base + sources=[] re-applies the CSV without touching row count or order."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_splice_tax_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    before = _load(tmp_path / "base")
    paths_before = [r["original_path"] for r in before]
    assert {r["proposed_label"] for r in before if r["original_label"] == "copepoda"} == {"Copepoda"}

    # The taxonomy CSV changes: copepoda is reclassified.
    _write_csv(
        csv_path,
        [
            _csv_row("lensless", "copepoda", proposed="Calanoida", phylum="Arthropoda"),
            _csv_row("isiisnet", "appendicularia", proposed="Appendicularia", phylum="Chordata"),
        ],
    )
    gp._build_taxonomy_lookup_cached.cache_clear()

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[]", f"base={tmp_path / 'base'}"],
        "test_splice_tax",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    after = _load(tmp_path / "after")

    assert len(after) == len(before)
    assert [r["original_path"] for r in after] == paths_before, "a taxonomy-only run must not reorder rows"
    assert {r["proposed_label"] for r in after if r["original_label"] == "copepoda"} == {"Calanoida"}

    gp._build_taxonomy_lookup_cached.cache_clear()


def test_sync_unmatched_keep_versus_clear(offline, two_source_env, tmp_path):
    """`keep` preserves an unmatched row's taxonomy; `clear` nulls it.

    Pins the one real semantic gap between the two former scripts: a from-scratch
    build nulls everything for a row with no CSV entry, while the re-sync path kept
    the taxonomy and nulled only the IDs.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_splice_unmatched_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    # Drop lensless/copepoda from the CSV so those rows become unmatched.
    _write_csv(csv_path, [_csv_row("isiisnet", "appendicularia", proposed="Appendicularia", phylum="Chordata")])
    gp._build_taxonomy_lookup_cached.cache_clear()

    results = {}
    for policy in ("keep", "clear"):
        cfg_p = _compose(
            [*common, f"output_dir={tmp_path / policy}", "sources=[]", f"base={tmp_path / 'base'}", f"sync_unmatched={policy}"],
            f"test_splice_unmatched_{policy}",
        )
        _restrict_registry(cfg_p, ["isiisnet", "lensless"])
        _run(cfg_p)
        results[policy] = [r for r in _load(tmp_path / policy) if r["original_label"] == "copepoda"]

    assert all(r["proposed_label"] == "Copepoda" for r in results["keep"])
    assert all(r["Phylum"] == "Arthropoda" for r in results["keep"])
    assert all(r["aphia_ID"] is None for r in results["keep"]), "IDs are always nulled when unmatched"

    assert all(r["proposed_label"] is None for r in results["clear"])
    assert all(r["Phylum"] is None for r in results["clear"])

    gp._build_taxonomy_lookup_cached.cache_clear()


def test_drop_removes_a_source(offline, two_source_env, tmp_path):
    """`drop` removes a source's rows from the output without rebuilding anything."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_splice_drop_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[]", f"base={tmp_path / 'base'}", "drop=[isiisnet]"],
        "test_splice_drop",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    after = _load(tmp_path / "after")
    assert {r["dataset"] for r in after} == {"lensless"}
    assert len(after) == 5


def test_schema_mismatch_in_the_base_is_a_hard_error(offline, two_source_env, tmp_path):
    """A base missing a column fails loudly instead of being silently null-filled.

    concatenate_datasets fills a missing column with nulls rather than raising, so
    without this guard the column would be blanked for exactly the rows just rebuilt.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_splice_schema_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    damaged = _load(tmp_path / "base").remove_columns("qualifier")
    damaged.save_to_disk(str(tmp_path / "damaged"))

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[lensless]", f"base={tmp_path / 'damaged'}"],
        "test_splice_schema_bad",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])

    with pytest.raises(ValueError, match="qualifier"):
        mk.main(cfg2)
    GlobalHydra.instance().clear()


def test_rename_signature_is_refused(offline, two_source_env, tmp_path):
    """Rebuilding a source absent from a base that holds an unknown name is fatal.

    That combination is what a renamed source looks like, and carrying the old rows
    over while appending the rebuilt ones would double the source.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}", "sources=[isiisnet]"], "test_splice_rename_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    # Rewrite the base's dataset column to an old name the registry does not know.
    renamed = _load(tmp_path / "base").map(lambda ex: {"dataset": "isiis-net-old"})
    renamed.save_to_disk(str(tmp_path / "renamed"))

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[isiisnet]", f"base={tmp_path / 'renamed'}"],
        "test_splice_rename",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])

    with pytest.raises(ValueError, match="renamed source"):
        mk.main(cfg2)
    GlobalHydra.instance().clear()


def test_adding_a_genuinely_new_source_is_allowed(offline, two_source_env, tmp_path):
    """A source absent from a base whose other names are all known is a normal addition."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}", "sources=[isiisnet]"], "test_splice_add_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[lensless]", f"base={tmp_path / 'base'}"],
        "test_splice_add",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    after = _load(tmp_path / "after")
    assert [r["dataset"] for r in after] == ["isiisnet"] * 3 + ["lensless"] * 5


def test_in_place_update_survives_writing_over_its_own_source(offline, two_source_env, tmp_path):
    """`base=local` reads and writes the same directory without corrupting it.

    save_to_disk raises PermissionError when the target is the dataset's own source
    directory, so this is staged beside it and swapped in.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]
    target = tmp_path / "inplace"

    cfg = _compose([*common, f"output_dir={target}"], "test_splice_inplace_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    before = _rows(_load(target), drop_image=False)

    cfg2 = _compose(
        [*common, f"output_dir={target}", "sources=[lensless]", "base=local", "refresh=rebuild"],
        "test_splice_inplace",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    after = _rows(_load(target), drop_image=False)
    assert after == before, "an in-place no-change refresh should round-trip exactly"

    residue = [p.name for p in tmp_path.iterdir() if ".new-" in p.name or ".old-" in p.name]
    assert residue == [], f"staging directories left behind: {residue}"


def test_clean_scope_none_skips_the_corrupt_scan(offline, two_source_env, tmp_path, monkeypatch):
    """clean=none performs no integrity pass; clean=all performs exactly one."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    calls = []
    real_clean = gp.clean_corrupt_examples_optimized

    def counting(ds, **kwargs):
        calls.append(kwargs)
        return real_clean(ds, **kwargs)

    monkeypatch.setattr(mk, "clean_corrupt_examples_optimized", counting)

    cfg = _compose([*common, f"output_dir={tmp_path / 'none'}", "clean=none"], "test_splice_clean_none")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)
    assert calls == []

    calls.clear()
    cfg2 = _compose([*common, f"output_dir={tmp_path / 'all'}", "clean=all"], "test_splice_clean_all")
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)
    assert len(calls) == 1
    assert calls[0] == {"batch_size": 1000, "n_jobs": -1}


def test_version_is_embedded_in_the_saved_artifact(offline, two_source_env, tmp_path):
    """A version survives the real save/load round-trip as DatasetInfo.version.

    End-to-end proof that `version=` reaches disk: a copy of the dataset can say which
    version it is without consulting the Hub.
    """
    data_dir, csv_path = two_source_env
    cfg = _compose(
        [
            f"taxonomy_csv_path={csv_path}",
            f"data_dir={data_dir}",
            "num_proc=1",
            f"output_dir={tmp_path / 'out'}",
            "version=1.4.0",
        ],
        "test_splice_version",
    )
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    ds = _load(tmp_path / "out")
    assert str(ds.info.version) == "1.4.0"


def test_unversioned_run_leaves_the_default_version(offline, two_source_env, tmp_path):
    """Not setting a version changes nothing about the artifact."""
    data_dir, csv_path = two_source_env
    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1", f"output_dir={tmp_path / 'out'}"],
        "test_splice_noversion",
    )
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    ds = _load(tmp_path / "out")
    assert ds.info.version is None


def test_non_embeddable_version_still_builds(offline, two_source_env, tmp_path):
    """A Hub-tag-only version does not block the build or corrupt the artifact."""
    data_dir, csv_path = two_source_env
    cfg = _compose(
        [
            f"taxonomy_csv_path={csv_path}",
            f"data_dir={data_dir}",
            "num_proc=1",
            f"output_dir={tmp_path / 'out'}",
            "version=v1.2",
        ],
        "test_splice_version_freeform",
    )
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    ds = _load(tmp_path / "out")
    assert len(ds) == 8
    assert ds.info.version is None, "a non-x.y.z version is not embedded"


def test_built_rows_carry_their_source_license(offline, two_source_env, tmp_path):
    """Every row a build produces is stamped with its source's redistribution terms.

    Guards the merge of the license work into the consolidated command: the license
    columns are part of constants.CONSOLIDATED_COLUMNS, so a build that stopped
    emitting them would also trip the schema guard — but this asserts the VALUES, not
    just the columns' presence.
    """
    data_dir, csv_path = two_source_env
    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1", f"output_dir={tmp_path / 'out'}"],
        "test_splice_license",
    )
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    ds = _load(tmp_path / "out")

    by_source = {}
    for row in ds:
        by_source.setdefault(row["dataset"], set()).add((row["license"], row["license_url"]))

    assert by_source["isiisnet"] == {("cc-by-nc-4.0", "https://creativecommons.org/licenses/by-nc/4.0/")}
    assert by_source["lensless"] == {("cc-by-4.0", "https://creativecommons.org/licenses/by/4.0/")}


def test_spliced_rows_keep_the_licenses_of_the_base(offline, two_source_env, tmp_path):
    """A per-source refresh preserves the license columns of the rows it carries over."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_splice_license_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[lensless]", f"base={tmp_path / 'base'}", "refresh=rebuild"],
        "test_splice_license_refresh",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    rows = list(_load(tmp_path / "after"))
    assert {r["license"] for r in rows if r["dataset"] == "isiisnet"} == {"cc-by-nc-4.0"}
    assert {r["license"] for r in rows if r["dataset"] == "lensless"} == {"cc-by-4.0"}
    assert all(r["license_url"] for r in rows), "no row may carry a null license_url"


def test_dropping_the_only_remaining_source_is_refused_end_to_end(offline, two_source_env, tmp_path):
    """The empty-result guard fires through the real CLI path, not just in isolation.

    Builds a real one-source dataset, then asks for a run that would drop it entirely.
    Nothing is written and the existing output is left alone.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}", "sources=[lensless]"], "test_empty_e2e_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    before = len(_load(tmp_path / "base"))
    assert before == 5

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[]", f"base={tmp_path / 'base'}", "drop=[lensless]"],
        "test_empty_e2e",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])

    with pytest.raises(ValueError, match="EMPTY dataset"):
        mk.main(cfg2)
    GlobalHydra.instance().clear()

    assert not (tmp_path / "after").exists(), "nothing should have been written"
    assert len(_load(tmp_path / "base")) == before, "the base must be untouched"


def test_a_drop_that_leaves_rows_still_works(offline, two_source_env, tmp_path):
    """The guard must not fire on a legitimate drop — regression cover for the fix."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_drop_ok_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[]", f"base={tmp_path / 'base'}", "drop=[isiisnet]"],
        "test_drop_ok",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    after = _load(tmp_path / "after")
    assert {r["dataset"] for r in after} == {"lensless"}
    assert len(after) == 5


def test_every_part_is_conformed_to_one_shared_reference(offline, two_source_env, tmp_path, monkeypatch):
    """All parts are conformed to a single reference before concatenation.

    SCOPE: in this fixture every part already shares a schema, so this cannot show
    WHICH part supplies the reference — that is pinned non-vacuously by
    test_reference_features_come_from_the_largest_part, which uses parts whose features
    genuinely differ. What this adds is that the real pipeline conforms every part
    against one reference and still produces the right rows in the right order.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}"], "test_castref_base")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    seen = []
    real_conform = mk.conform_schema

    def spy(ds, reference_features):
        seen.append((len(ds), reference_features))
        return real_conform(ds, reference_features)

    monkeypatch.setattr(mk, "conform_schema", spy)

    # isiisnet is registry index 0 — the ordering that used to pick the fresh part.
    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[isiisnet]", f"base={tmp_path / 'base'}", "refresh=rebuild"],
        "test_castref",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    assert sorted(size for size, _ in seen) == [3, 5], "both parts should be conformed"
    references = [ref for _, ref in seen]
    assert all(ref == references[0] for ref in references), "every part must share one reference"

    assert [r["dataset"] for r in _load(tmp_path / "after")] == ["isiisnet"] * 3 + ["lensless"] * 5


def test_a_base_predating_the_license_columns_is_migrated_not_rejected(offline, two_source_env, tmp_path):
    """`base=<frozen> sources=[]` adds the license columns instead of failing on them.

    The published planktonzilla-17M was built before per-image licensing, so it has no
    license columns. assert_consolidated_schema requires them, so without deriving them
    first the documented migration —
    `pz_planktonzilla base=hub sources=[] sync_taxonomy=false` — was rejected outright by
    the very command that replaces pz_update_planktonzilla.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'built'}"], "test_licmig_build")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    # Strip the license columns to reproduce the frozen artifact's schema.
    frozen = _load(tmp_path / "built").remove_columns(list(constants.LICENSE_COLS))
    frozen.save_to_disk(str(tmp_path / "frozen"))
    assert not any(c in frozen.column_names for c in constants.LICENSE_COLS)

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'migrated'}", "sources=[]", f"base={tmp_path / 'frozen'}", "sync_taxonomy=false"],
        "test_licmig",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    out = _load(tmp_path / "migrated")

    assert set(out.column_names) == set(constants.CONSOLIDATED_COLUMNS)
    assert len(out) == len(frozen), "a strictly additive run must not change the row count"

    by_source = {}
    for row in out:
        by_source.setdefault(row["dataset"], set()).add((row["license"], row["license_url"]))
    assert by_source["isiisnet"] == {("cc-by-nc-4.0", "https://creativecommons.org/licenses/by-nc/4.0/")}
    assert by_source["lensless"] == {("cc-by-4.0", "https://creativecommons.org/licenses/by/4.0/")}


def test_migrating_a_licenseless_base_while_refreshing_a_source(offline, two_source_env, tmp_path):
    """The migration also works on the splice path, not just the pure-resync fast path."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'built'}"], "test_licmig2_build")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    frozen = _load(tmp_path / "built").remove_columns(list(constants.LICENSE_COLS))
    frozen.save_to_disk(str(tmp_path / "frozen"))

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'out'}", "sources=[lensless]", f"base={tmp_path / 'frozen'}", "refresh=rebuild"],
        "test_licmig2",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    rows = list(_load(tmp_path / "out"))
    assert [r["dataset"] for r in rows] == ["isiisnet"] * 3 + ["lensless"] * 5
    # Carried rows got their licence derived; rebuilt rows got theirs at build time.
    assert {r["license"] for r in rows if r["dataset"] == "isiisnet"} == {"cc-by-nc-4.0"}
    assert {r["license"] for r in rows if r["dataset"] == "lensless"} == {"cc-by-4.0"}
    assert all(r["license_url"] for r in rows)


def test_a_base_predating_custom_metadata_is_filled_not_rejected(offline, two_source_env, tmp_path):
    """`base=<v1.1-shaped> sources=[]` fills custom_metadata with "{}" instead of failing on it.

    Every source published before v1.2 had nothing to put in custom_metadata, so the
    literal empty object is the same value a from-scratch rebuild writes for them —
    derivation, not invention — and the migration stays expressible with this command.
    """
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'built'}"], "test_cmmig_build")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    frozen = _load(tmp_path / "built").remove_columns([constants.CUSTOM_METADATA_COL])
    frozen.save_to_disk(str(tmp_path / "frozen"))
    assert constants.CUSTOM_METADATA_COL not in frozen.column_names

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'migrated'}", "sources=[]", f"base={tmp_path / 'frozen'}", "sync_taxonomy=false"],
        "test_cmmig",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    out = _load(tmp_path / "migrated")
    assert set(out.column_names) == set(constants.CONSOLIDATED_COLUMNS)
    assert len(out) == len(frozen), "a strictly additive run must not change the row count"
    assert out.features[constants.CUSTOM_METADATA_COL].dtype == "string"
    assert set(out[constants.CUSTOM_METADATA_COL]) == {constants.EMPTY_CUSTOM_METADATA}


def test_filling_custom_metadata_on_a_base_while_refreshing_a_source(offline, two_source_env, tmp_path):
    """On the splice path the filled base rows and the rebuilt rows carry the same "{}"."""
    data_dir, csv_path = two_source_env
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'built'}"], "test_cmmig2_build")
    _restrict_registry(cfg, ["isiisnet", "lensless"])
    _run(cfg)

    frozen = _load(tmp_path / "built").remove_columns([constants.CUSTOM_METADATA_COL])
    frozen.save_to_disk(str(tmp_path / "frozen"))

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'out'}", "sources=[lensless]", f"base={tmp_path / 'frozen'}", "refresh=rebuild"],
        "test_cmmig2",
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless"])
    _run(cfg2)

    rows = list(_load(tmp_path / "out"))
    assert [r["dataset"] for r in rows] == ["isiisnet"] * 3 + ["lensless"] * 5
    assert {r[constants.CUSTOM_METADATA_COL] for r in rows} == {constants.EMPTY_CUSTOM_METADATA}


# --- v1.2: frepj joins a base that predates it, as the LAST source --------------------------


def test_frepj_joins_a_base_that_predates_it_as_the_last_source(offline, two_source_env, tmp_path, monkeypatch):
    """`base=<15-source-shaped> sources=[frepj]`: sidecars verified up front, frepj appended LAST.

    The P20 scenario end to end, network-free: synthetic md5-pinned tables under
    <data_dir>/frepj_tables (exactly where FREPJDatasetImporter looks), the fixture
    crosswalk, and file names that join the sample tables. The base rows keep their
    "{}" custom_metadata; the frepj rows carry magnification/site and a normalized date.
    """
    import pathlib

    from planktonzilla.dataset_import.frepj_importer import FREPJDatasetImporter
    from planktonzilla.planktonzilla_dataset import frepj_tables

    data_dir, csv_path = two_source_env
    fixtures = pathlib.Path(__file__).parent / "fixtures" / "frepj"

    tables_dir = data_dir / frepj_tables.TABLES_DIRNAME
    tables_dir.mkdir()
    manifest = []
    for name, fixture in (
        ("Table_S1.csv", "table_s1_sample.csv"),
        ("Table_S3.csv", "table_s3_sample.csv"),
        ("Table_S4.csv", "table_s4_sample.csv"),
    ):
        (tables_dir / name).write_bytes((fixtures / fixture).read_bytes())
        manifest.append(
            {
                "name": name,
                "file_id": 0,
                "url": "https://example.invalid/never",
                "md5": frepj_tables._md5(tables_dir / name),
                "size": 1,
            }
        )
    monkeypatch.setattr(FREPJDatasetImporter, "SIDECAR_MANIFEST", manifest)
    monkeypatch.setattr(FREPJDatasetImporter, "CROSSWALK_PATH", fixtures / "frepj_crosswalk_sample.csv")

    def _boom(*args, **kwargs):
        raise AssertionError("verified sidecars must never be downloaded")

    monkeypatch.setattr(frepj_tables, "DownloadManager", _boom)

    # A built frepj imagefolder whose stems join the sample tables: 40_101 -> akigawadam 2018.03.15
    # (resolved), 100_202 -> "(baiyou)" (no crosswalk row -> null lat/lon; year-only date -> null).
    # Two class dirs: the imagefolder loader only emits a `label` column with more than one.
    frepj_imagefolder = data_dir / "frepjdatasetimporter_imagefolder"
    for filename in ("40_101.png", "40_102.png"):
        _write_png(frepj_imagefolder / "copepoda" / filename)
    for filename in ("100_201.png", "100_202.png"):
        _write_png(frepj_imagefolder / "cladocera" / filename)
    _write_csv(
        csv_path,
        [
            _csv_row("lensless", "copepoda"),
            _csv_row("isiisnet", "appendicularia", proposed="Appendicularia", phylum="Chordata"),
            _csv_row("frepj", "copepoda"),
            _csv_row("frepj", "cladocera", proposed="Cladocera"),
        ],
    )
    common = [f"taxonomy_csv_path={csv_path}", f"data_dir={data_dir}", "num_proc=1"]

    cfg = _compose([*common, f"output_dir={tmp_path / 'base'}", "sources=[isiisnet,lensless]"], "test_frepj_join_base")
    _restrict_registry(cfg, ["isiisnet", "lensless", "frepj"])
    _run(cfg)
    base = _load(tmp_path / "base")
    assert sorted(set(base["dataset"])) == ["isiisnet", "lensless"]

    cfg2 = _compose(
        [*common, f"output_dir={tmp_path / 'after'}", "sources=[frepj]", f"base={tmp_path / 'base'}"], "test_frepj_join"
    )
    _restrict_registry(cfg2, ["isiisnet", "lensless", "frepj"])
    _run(cfg2)

    after = _load(tmp_path / "after")
    assert [r["dataset"] for r in after] == ["isiisnet"] * 3 + ["lensless"] * 5 + ["frepj"] * 4
    assert set(after.column_names) == set(constants.CONSOLIDATED_COLUMNS)

    rows = {r["original_path"].split("/")[-1]: r for r in after if r["dataset"] == "frepj"}
    assert {r[constants.CUSTOM_METADATA_COL] for r in after if r["dataset"] != "frepj"} == {constants.EMPTY_CUSTOM_METADATA}
    assert rows["40_101.png"][constants.CUSTOM_METADATA_COL] == '{"magnification": "40", "site": "akigawadam"}'
    assert rows["40_101.png"]["timestamp"] == "2018-03-15"
    assert rows["40_101.png"]["Latitude"] is not None
    assert rows["100_202.png"]["Latitude"] is None and rows["100_202.png"]["Longitude"] is None
    assert rows["100_202.png"]["timestamp"] is None
    assert {r["license"] for r in after if r["dataset"] == "frepj"} == {"cc-by-4.0"}
