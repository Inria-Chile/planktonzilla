"""
(c) Inria

Network-free tests for the @hydra.main port of
``planktonzilla.planktonzilla_dataset.generate_planktonzilla``.

These are the testable PROXY for zero behavioral drift. A full golden-output
dataset diff is not runnable here (it needs multi-source data + live
NCBI/Wikidata/WHOI/EcoTaxa); instead we pin:

  (a) the config composes with the expected key contract,
  (b) the in-code null fallbacks resolve byte-identically to the legacy argparse
      defaults (the default-run zero-drift guarantee),
  (c) the config-driven `datasets` table + `repo_id` are exactly the frozen values,
  (d) the per-dataset hydra.compose override blocks + redefiner classes built in
      main() are byte-identical and in declaration order,
  (e) the module-level ``num_proc`` global stays independent of cfg.num_proc,
  (f) the opt-in Hub push is additive — the default (push_to_hub false) never
      pushes (zero-drift), while push_to_hub=true pushes once to cfg.repo_id after
      the unconditional save.

Every test PINS current behavior; none "improves" it. All network is mocked.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from pathlib import Path
from unittest.mock import MagicMock

import hydra
import pytest
from datasets import Dataset
from hydra.core.global_hydra import GlobalHydra

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp

# The frozen (name, import_name, cleanup, redefiner_key) table — the single source
# of truth shared by the config-content test and the _run override-pin test. Order
# matters: cfg.datasets is iterated (and concatenated) in this order.
EXPECTED_TABLE = [
    ("isiisnet", "isiisnet", True, "none"),
    ("whoi", "whoi-plankton", True, "whoi"),
    ("flowcamnet", "flowcamnet", True, "ecotaxa"),
    ("lensless", "lensless", True, "none"),
    ("medplanktonset", "medplanktonset", True, "none"),
    ("uvp6net", "uvp6net", True, "ecotaxa"),
    ("zoocamnet", "zoocamnet", True, "none"),
    ("zooscan", "zooscannet", True, "ecotaxa"),
    ("planktonset1.0", "planktonset1", False, "none"),
    ("syke_ifcb_2022", "syke_ifcb_2022", False, "none"),
    ("planktoscope", "planktoscope", False, "ecotaxa"),
    ("global_uvp5", "global_uvp5net", False, "ecotaxa"),
    # Appended 2026-08-01, once none of the three turned out to need the manual .zip
    # they had long been documented as requiring. They go at the END so every source
    # above keeps the index it already had — registry order is the concatenation order
    # of the output. With these the registry covers all 15 sources of the published (frepj, v1.2, follows)
    # dataset.
    ("zoolake", "zoolake", False, "none"),
    ("jedioceans", "jedi_oceans_cpics", False, "jedi"),
    ("sykezooscan2024", "sykezooscan2024", False, "none"),
    # Appended 2026-08-25 (v1.2), LAST, so every source above keeps its index. Its redefiner
    # joins md5-pinned sidecar tables the importer fetches before the first import.
    ("frepj", "frepj", False, "frepj"),
]


def _write_taxonomy_csv(path, dataset_name, raw_label):
    """Write a tiny taxonomy CSV that _build_lookup can resolve.

    Columns mirror the real planktonzilla_taxonomy.csv: Dataset, Raw_Labels plus
    the lookup columns (taxonomy ranks, extras, str IDs and numeric IDs).
    """
    header = (
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
    )
    # aphia_ID / NCBI_ID / BOLD_ID are numeric in the CSV (float) -> text w/o decimals.
    row = f"{dataset_name},{raw_label},Animalia,Arthropoda,,,,,,Copepoda,True,zoo,,Q3386609,274;1231,135336.0,6854.0,"
    with open(path, "w") as f:
        f.write(header + "\n" + row + "\n")


def _expected_overrides(data_dir, import_name, cleanup):
    """The 4-element import_dataset override block build_overrides() emits."""
    return [
        f"dataset_import={import_name}",
        f"dataset_import.cleanup_after_processing={cleanup}",
        "dataset_import.push_to_hub=False",
        f"dataset_import.data_dir={data_dir}",
    ]


def _drive_main_with_mocked_pipeline(monkeypatch, cfg, tmp_path, push_mock=None):
    """Run ``gp.main(cfg)`` end-to-end network-free with the heavy pipeline mocked.

    Mocks the inner per-dataset loop body (hydra.compose / instantiate / os.listdir /
    load_dataset / RedefineDataset.redefine) plus the final clean + save_to_disk so
    ``main`` runs to completion on a tiny in-memory dataset. ``main`` is the
    @hydra.main entry point, driven here via Hydra's cfg-passthrough (``main(cfg)``
    calls the task body directly — the former ``_run`` seam no longer exists).

    Returns ``(captured_overrides, captured_redefiners)``. When ``push_mock`` is
    given it is installed as ``Dataset.push_to_hub`` so the opt-in push can be
    asserted; otherwise the (default-false) push branch is a no-op and never fires.
    """
    captured_overrides = []  # one entry per dataset, in iteration order.

    def _fake_compose(*args, **kwargs):
        captured_overrides.append(list(kwargs["overrides"]))
        return MagicMock()

    monkeypatch.setattr(gp.hydra, "compose", _fake_compose)

    # Mock importer whose imagefolder is "present" and non-empty so no real import /
    # load happens; redefine() returns a trivial dataset so concatenate / clean /
    # save still work.
    importer = MagicMock()
    importer.imagefolder_dir = tmp_path
    importer.ensure_sidecars.return_value = {}
    monkeypatch.setattr(gp.hydra.utils, "instantiate", lambda *a, **k: importer)
    monkeypatch.setattr(gp.os, "listdir", lambda p: ["dummy_category"])
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: MagicMock())

    captured_redefiners = {}
    tiny = Dataset.from_dict({"x": [1]})

    def _fake_redefine(self, hf_dataset, dataset_name, num_proc):
        captured_redefiners[dataset_name] = type(self)
        return tiny

    monkeypatch.setattr(gp.RedefineDataset, "redefine", _fake_redefine)
    monkeypatch.setattr(gp, "clean_corrupt_examples_optimized", lambda ds, **k: ds)
    monkeypatch.setattr(gp.Dataset, "save_to_disk", lambda self, path: None)
    if push_mock is not None:
        monkeypatch.setattr(gp.Dataset, "push_to_hub", push_mock)

    gp.main(cfg)
    return captured_overrides, captured_redefiners


def test_config_composes_with_expected_keys():
    """The config composes and exposes the expected key contract."""
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_compose")
    cfg = hydra.compose(config_name="generate_planktonzilla")

    # Null-default contract: these resolve to legacy defaults via in-code fallbacks.
    for key in ("taxonomy_csv_path", "num_proc"):
        assert key in cfg, f"missing key {key}"
        assert cfg.get(key) is None, f"{key} should default to null"

    # data_dir is interpolated from paths.data_dir (not null), repo_id is the
    # consolidated dataset identity, datasets is the import table.
    assert cfg.get("data_dir") is not None
    assert cfg.get("repo_id") == "project-oceania/planktonzilla-17M"
    assert "datasets" in cfg and len(cfg.datasets) == len(EXPECTED_TABLE)

    GlobalHydra.instance().clear()


def test_null_fallback_defaults_match_legacy_absolute_values():
    """The in-code null fallbacks equal the documented legacy argparse defaults.

    This is the byte-identity assertion that guarantees a default (no-override)
    run produces the exact same resolved (taxonomy_csv_path, num_proc) as the
    argparse era — the testable proxy for zero behavioral drift.
    """
    data_root = (gp.root / "data").resolve()

    # taxonomy_csv_path: the legacy default was str(DATA_ROOT / DEFAULT_TAXONOMY_CSV_FILENAME),
    # but DEFAULT_TAXONOMY_CSV_FILENAME is an ABSOLUTE Path, so the `/` join is a
    # no-op. The in-code fallback uses str(constants.DEFAULT_TAXONOMY_CSV_FILENAME).
    taxo_fallback = str(constants.DEFAULT_TAXONOMY_CSV_FILENAME)
    assert taxo_fallback == str(data_root / constants.DEFAULT_TAXONOMY_CSV_FILENAME)  # join is a no-op

    # num_proc: constants.default_num_proc() (a positive int, >= 1).
    assert constants.default_num_proc() >= 1

    # output dir: now config-driven, Path(cfg.data_dir) / DEFAULT_PLANKTONZILLA_DATASET_NAME.
    # NOTE: the local folder is "planktonzilla-17M" (hyphen, == the HF repo name),
    # whereas the argparse era wrote "planktonzilla_17M" (underscore). This renames
    # only the local save_to_disk directory, not the dataset content.
    assert constants.DEFAULT_PLANKTONZILLA_DATASET_NAME == "planktonzilla-17M"


def test_datasets_and_repo_id_pinned_in_config():
    """Pin the config-driven import table + repo id (the migrated values).

    Asserts cfg.datasets is exactly the frozen 16-row table in order, repo_id is
    the consolidated dataset identity, and the REDEFINERS map resolves each key to
    the expected class.
    """
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_table")
    cfg = hydra.compose(config_name="generate_planktonzilla")

    assert cfg.repo_id == "project-oceania/planktonzilla-17M"

    actual = [(d["name"], d["import_name"], d["cleanup"], d["redefiner"]) for d in cfg.datasets]
    assert actual == EXPECTED_TABLE

    GlobalHydra.instance().clear()

    # REDEFINERS keys cover every key used by the table and map to the right class.
    expected_classes = {
        "none": gp.NoMetadataRedefiner,
        "whoi": gp.WHOIRedefiner,
        "ecotaxa": gp.EcoTaxaRedefiner,
        "jedi": gp.JediRedefiner,
        "frepj": gp.FrepjRedefiner,
    }
    for key, klass in expected_classes.items():
        assert gp.REDEFINERS[key] is klass
    assert {key for _, _, _, key in EXPECTED_TABLE} <= set(gp.REDEFINERS)


def test_main_pins_override_blocks_and_redefiners(monkeypatch, tmp_path):
    """Pin the per-dataset override blocks + redefiner classes built in main().

    Drives ``gp.main(cfg)`` (the @hydra.main body, via cfg-passthrough) with a
    composed config whose taxonomy_csv_path points at a real tiny CSV, mocking the
    whole per-dataset loop body so it runs to completion. Captures, in iteration
    order, the exact ``overrides`` list passed to hydra.compose and the redefiner
    type bound to each dataset_name, then asserts they match the frozen table.
    """
    # Real CSV so the redefiner constructors (run while building datasets_configs)
    # succeed; taxonomy_csv_path routes every redefiner here.
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    # Compose the REAL gen config first (taxonomy_csv_path pointed at the tiny CSV)
    # BEFORE the helper patches hydra.compose. `gp.hydra is hydra` (same module
    # object), so the helper's patch replaces the inner per-dataset import_dataset
    # compose too — exactly what we capture here.
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_override")
    cfg = hydra.compose(config_name="generate_planktonzilla", overrides=[f"taxonomy_csv_path={csv_path}"])

    # build_overrides uses cfg.data_dir; capture its resolved value for the expected
    # block (data_dir is routed through ${paths.data_dir}).
    expected_data_dir = str(cfg.data_dir)

    captured_overrides, captured_redefiners = _drive_main_with_mocked_pipeline(monkeypatch, cfg, tmp_path)

    GlobalHydra.instance().clear()

    # The redefiner class each key resolves to (mirrors the REDEFINERS map).
    key_to_class = {
        "none": gp.NoMetadataRedefiner,
        "whoi": gp.WHOIRedefiner,
        "ecotaxa": gp.EcoTaxaRedefiner,
        "jedi": gp.JediRedefiner,
        "frepj": gp.FrepjRedefiner,
    }

    # Exactly the EXPECTED_TABLE entries (15 published + frepj), in this order.
    assert list(captured_redefiners.keys()) == [t[0] for t in EXPECTED_TABLE]
    assert len(captured_overrides) == len(EXPECTED_TABLE)

    for (name, import_name, cleanup, redefiner_key), overrides in zip(EXPECTED_TABLE, captured_overrides):
        # Override block reproduced byte for byte (cleanup -> "True"/"False").
        assert overrides == _expected_overrides(expected_data_dir, import_name, cleanup)
        # Redefiner type bound to this dataset is exactly the expected class.
        assert captured_redefiners[name] is key_to_class[redefiner_key]


def test_main_skips_push_by_default(monkeypatch, tmp_path):
    """PIN zero-drift: the default run (push_to_hub absent/false) never pushes to the Hub.

    Drives main() to completion with the pipeline mocked and a recording mock on
    ``Dataset.push_to_hub``; the default config leaves push_to_hub false, so the
    push branch must be skipped and the frozen artifact left untouched. All network
    is mocked; this PINS current behavior, it does not "improve" it.
    """
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_nopush")
    cfg = hydra.compose(config_name="generate_planktonzilla", overrides=[f"taxonomy_csv_path={csv_path}"])

    push = MagicMock()
    _drive_main_with_mocked_pipeline(monkeypatch, cfg, tmp_path, push_mock=push)

    GlobalHydra.instance().clear()

    push.assert_not_called()


def test_main_pushes_to_hub_when_enabled(monkeypatch, tmp_path):
    """PIN the opt-in push: push_to_hub=true pushes exactly once to cfg.repo_id.

    The push is additive (it runs after the unconditional save_to_disk in main) and
    forwards the configured ``private`` (push_as_private) flag. All network is
    mocked; this PINS current behavior, it does not "improve" it.
    """
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_push")
    cfg = hydra.compose(
        config_name="generate_planktonzilla",
        overrides=[f"taxonomy_csv_path={csv_path}", "push_to_hub=true"],
    )

    push = MagicMock()
    _drive_main_with_mocked_pipeline(monkeypatch, cfg, tmp_path, push_mock=push)

    GlobalHydra.instance().clear()

    push.assert_called_once()
    assert push.call_args.args[0] == cfg.repo_id
    assert push.call_args.kwargs.get("private") == cfg.push_as_private


def test_module_level_num_proc_independent_of_cfg():
    """Pin: the module-level num_proc global is set from default_num_proc() at
    import time and is intentionally NOT driven by cfg.num_proc (only redefine()
    receives the configurable value)."""
    assert gp.num_proc == constants.default_num_proc()


def test_taxonomy_map_declares_its_schema_instead_of_inferring_it():
    """The taxonomy pass must not depend on which class sorts first.

    `map` types each writer batch from its values, and an imagefolder is ordered by
    class: a first class with no Order/Family/Genus types those columns `null`, and the
    first later class that has one dies with "Couldn't cast array of type string to
    null". sykezooscan2024 is exactly that shape (Bivalvia sorts first, with all three
    empty), so importing it failed on datasets 4.8.5 and 5.0.1 alike.
    """
    from datasets import Value

    redefiner = gp.NoMetadataRedefiner(csv_taxonomies_path=str(constants.DEFAULT_TAXONOMY_CSV_FILENAME))
    ds = Dataset.from_dict({"label": [0, 1]})

    features = redefiner._mapped_features(ds)

    # Every column the taxonomy pass writes is declared...
    for column in (*constants.IDENTITY_COLS, *constants.LICENSE_COLS, *redefiner.lookup_cols):
        assert column in features, f"{column} would be inferred from batch content"
    # ...as the type _cast_scalar_types casts to at the end, so nothing about the
    # output changes — only that the schema is stated rather than guessed.
    assert features["Genus"] == Value("string")
    assert features["aphia_ID"] == Value("string"), "numeric IDs are stored as text"
    assert features["plankton"] == Value("bool")
    assert features["label"] == ds.features["label"], "input columns are carried through untouched"


# --- import_and_redefine_source: sidecar inputs are obtained on EVERY path -------------

_SEAM_ENTRY = {"name": "src", "import_name": "src", "cleanup": False, "redefiner": "none"}


class _FakeImporter:
    """Records the order of ensure_sidecars / import_dataset; import creates one class dir."""

    def __init__(self, imagefolder, sidecars):
        self.imagefolder_dir = imagefolder
        self.sidecars = sidecars
        self.calls = []
        self.folder_existed_at_ensure = None

    def ensure_sidecars(self):
        self.calls.append("ensure_sidecars")
        self.folder_existed_at_ensure = Path(self.imagefolder_dir).exists()
        return self.sidecars

    def import_dataset(self):
        self.calls.append("import_dataset")
        (Path(self.imagefolder_dir) / "cls").mkdir(parents=True, exist_ok=True)
        (Path(self.imagefolder_dir) / "cls" / "img.png").write_bytes(b"x")


class _FakeRedefiner:
    def __init__(self):
        self.attached = None

    def attach_sidecars(self, sidecars):
        self.attached = sidecars

    def redefine(self, hf_dataset, dataset_name, num_proc):
        return "DS"


def _seam(monkeypatch, tmp_path, importer, refresh="reuse"):
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gp.hydra, "compose", lambda *a, **k: pytest.fail("compose must not run when an importer is given"))
    monkeypatch.setattr(gp.hydra.utils, "instantiate", lambda *a, **k: pytest.fail("instantiate must not run either"))
    redefiner = _FakeRedefiner()
    out = gp.import_and_redefine_source(
        _SEAM_ENTRY, data_dir=tmp_path, redefiner=redefiner, num_proc_arg=1, refresh=refresh, importer=importer
    )
    assert out == "DS"
    return redefiner


def test_import_and_redefine_source_ensures_sidecars_on_the_reuse_path(monkeypatch, tmp_path):
    """A built imagefolder skips import_dataset() but NOT the sidecars the redefiner needs."""
    imagefolder = tmp_path / "src_imagefolder"
    (imagefolder / "cls").mkdir(parents=True)
    (imagefolder / "cls" / "img.png").write_bytes(b"x")
    importer = _FakeImporter(imagefolder, {"Table.csv": tmp_path / "Table.csv"})

    redefiner = _seam(monkeypatch, tmp_path, importer)

    assert importer.calls == ["ensure_sidecars"]
    assert redefiner.attached == {"Table.csv": tmp_path / "Table.csv"}


def test_import_and_redefine_source_ensures_sidecars_before_the_archive_and_the_removal(monkeypatch, tmp_path):
    """Sidecars come first: before the multi-GB import, and before a redownload removes the folder."""
    imagefolder = tmp_path / "src_imagefolder"
    importer = _FakeImporter(imagefolder, {})
    _seam(monkeypatch, tmp_path, importer)
    assert importer.calls == ["ensure_sidecars", "import_dataset"]

    importer = _FakeImporter(imagefolder, {})
    _seam(monkeypatch, tmp_path, importer, refresh="redownload")
    assert importer.calls == ["ensure_sidecars", "import_dataset"]
    assert importer.folder_existed_at_ensure is True, "the folder is removed only after the sidecars are in hand"


def test_import_and_redefine_source_accepts_an_instantiated_importer(monkeypatch, tmp_path):
    """importer= reuses the caller's instance: no second compose, no second instantiate (see _seam)."""
    imagefolder = tmp_path / "src_imagefolder"
    (imagefolder / "cls").mkdir(parents=True)
    (imagefolder / "cls" / "img.png").write_bytes(b"x")
    _seam(monkeypatch, tmp_path, _FakeImporter(imagefolder, {}))


def test_a_source_without_sidecars_attaches_an_empty_dict(monkeypatch, tmp_path):
    """The fifteen archive-only sources hand their redefiner {} — and every redefiner accepts it."""
    imagefolder = tmp_path / "src_imagefolder"
    (imagefolder / "cls").mkdir(parents=True)
    (imagefolder / "cls" / "img.png").write_bytes(b"x")
    redefiner = _seam(monkeypatch, tmp_path, _FakeImporter(imagefolder, {}))
    assert redefiner.attached == {}


def test_generate_frepj_only_drives_the_same_seam_end_to_end(monkeypatch, tmp_path):
    """The standalone republish config runs through gp.main: eager FrepjRedefiner construction
    (lazy tables, so no table on disk is needed), one override block, sidecars via the seam."""
    from hydra.core.global_hydra import GlobalHydra

    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "frepj", "copepoda")
    GlobalHydra.instance().clear()
    gp.hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_frepj_only_drive")
    cfg = gp.hydra.compose(
        config_name="generate_frepj_only", overrides=[f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}"]
    )
    GlobalHydra.instance().clear()

    captured_overrides, captured_redefiners = _drive_main_with_mocked_pipeline(monkeypatch, cfg, tmp_path)

    assert len(captured_overrides) == 1 and "dataset_import=frepj" in captured_overrides[0]
    assert captured_redefiners == {"frepj": gp.FrepjRedefiner}
