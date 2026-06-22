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
      _run() are byte-identical and in declaration order,
  (e) the module-level ``num_proc`` global stays independent of cfg.num_proc.

Every test PINS current behavior; none "improves" it. All network is mocked.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from unittest.mock import MagicMock

import hydra
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

    Asserts cfg.datasets is exactly the frozen 12-row table in order, repo_id is
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
    }
    for key, klass in expected_classes.items():
        assert gp.REDEFINERS[key] is klass
    assert {key for _, _, _, key in EXPECTED_TABLE} <= set(gp.REDEFINERS)


def test_run_pins_override_blocks_and_redefiners(monkeypatch, tmp_path):
    """Pin the per-dataset override blocks + redefiner classes built in _run().

    Drives the ported body (gp._run) with a composed config whose
    taxonomy_csv_path points at a real tiny CSV, and mocks the whole per-dataset
    loop body (hydra.compose / instantiate / load_dataset / redefiner.redefine /
    clean / save) so the loop runs to completion. Captures, in iteration order, the
    exact `overrides` list passed to hydra.compose and the redefiner type bound to
    each dataset_name, then asserts they match the frozen table EXACTLY.
    """
    # Real CSV so the redefiner constructors (run while building datasets_configs)
    # succeed; taxonomy_csv_path routes every redefiner here.
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    # Compose the REAL gen config first (taxonomy_csv_path pointed at the tiny CSV)
    # BEFORE patching hydra.compose. `gp.hydra is hydra` (same module object), so the
    # patch below replaces the inner per-dataset import_dataset compose too — exactly
    # what we want to capture. The @hydra.main main() is not directly callable with a
    # cfg, so we drive the ported _run(cfg) seam.
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_override")
    cfg = hydra.compose(config_name="generate_planktonzilla", overrides=[f"taxonomy_csv_path={csv_path}"])

    # build_overrides uses cfg.data_dir; capture its resolved value for the expected
    # block (the user's refactor routes data_dir through ${paths.data_dir}).
    expected_data_dir = str(cfg.data_dir)

    captured_overrides = []  # one entry per dataset, in iteration order.

    def _fake_compose(*args, **kwargs):
        captured_overrides.append(list(kwargs["overrides"]))
        return MagicMock()

    # Patch only AFTER the real gen compose above; this captures the inner
    # per-dataset import_dataset compose calls inside _run.
    monkeypatch.setattr(gp.hydra, "compose", _fake_compose)

    # Make the loop body a no-op: a mock importer whose imagefolder is "present"
    # and non-empty (os.listdir returns a dummy category) so no real import / load
    # happens, and a redefine() that returns a trivial dataset so concatenate /
    # clean / save still work.
    importer = MagicMock()
    importer.imagefolder_dir = tmp_path  # a Path; exists, force has_content below
    monkeypatch.setattr(gp.hydra.utils, "instantiate", lambda *a, **k: importer)
    monkeypatch.setattr(gp.os, "listdir", lambda p: ["dummy_category"])
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: MagicMock())

    # Record (dataset_name -> redefiner type) and short-circuit the heavy work.
    captured_redefiners = {}
    tiny = Dataset.from_dict({"x": [1]})

    def _fake_redefine(self, hf_dataset, dataset_name, num_proc):
        captured_redefiners[dataset_name] = type(self)
        return tiny

    monkeypatch.setattr(gp.RedefineDataset, "redefine", _fake_redefine)
    # Skip corrupt-cleaning and disk save (both would choke on the tiny dataset).
    monkeypatch.setattr(gp, "clean_corrupt_examples_optimized", lambda ds, **k: ds)
    monkeypatch.setattr(gp.Dataset, "save_to_disk", lambda self, path: None)

    gp._run(cfg)

    GlobalHydra.instance().clear()

    # The redefiner class each key resolves to (mirrors the REDEFINERS map).
    key_to_class = {
        "none": gp.NoMetadataRedefiner,
        "whoi": gp.WHOIRedefiner,
        "ecotaxa": gp.EcoTaxaRedefiner,
    }

    # Exactly these 12 active datasets, in this order (commented ones excluded).
    assert list(captured_redefiners.keys()) == [t[0] for t in EXPECTED_TABLE]
    assert len(captured_overrides) == len(EXPECTED_TABLE)

    for (name, import_name, cleanup, redefiner_key), overrides in zip(EXPECTED_TABLE, captured_overrides):
        # Override block reproduced byte for byte (cleanup -> "True"/"False").
        assert overrides == _expected_overrides(expected_data_dir, import_name, cleanup)
        # Redefiner type bound to this dataset is exactly the expected class.
        assert captured_redefiners[name] is key_to_class[redefiner_key]


def test_module_level_num_proc_independent_of_cfg():
    """Pin: the module-level num_proc global is set from default_num_proc() at
    import time and is intentionally NOT driven by cfg.num_proc (only redefine()
    receives the configurable value)."""
    assert gp.num_proc == constants.default_num_proc()
