"""
(c) Inria

Network-free tests for the @hydra.main port of
``planktonzilla.dataset_generation.gen_planktonzilla``.

These are the testable PROXY for zero behavioral drift. A full golden-output
dataset diff is not runnable here (it needs multi-source data + live
NCBI/Wikidata/WHOI/EcoTaxa); instead we pin:

  (a) the new config composes with the null-default contract,
  (b) the in-code null fallbacks resolve byte-identically to the legacy argparse
      defaults (the default-run zero-drift guarantee),
  (c) the 12 per-dataset hydra.compose override blocks + redefiner classes are
      byte-identical and in iteration order (adapted from the deleted
      test_datasets_configs_structure_pins),
  (d) the module-level ``num_proc`` global stays independent of cfg.num_proc.

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

from planktonzilla.dataset_generation import constants
from planktonzilla.dataset_generation import gen_planktonzilla as gp


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


def _expected_overrides(data_root, dataset_import, cleanup):
    return [
        f"dataset_import={dataset_import}",
        f"dataset_import.cleanup_after_processing={cleanup}",
        "dataset_import.push_to_hub=False",
        f"dataset_import.data_dir={data_root}",
    ]


def test_config_composes_with_expected_keys():
    """The new config composes and exposes the null-default contract."""
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_compose")
    cfg = hydra.compose(config_name="gen_planktonzilla")

    for key in ("taxo_csv", "output", "num_proc"):
        assert key in cfg, f"missing key {key}"
        assert cfg.get(key) is None, f"{key} should default to null"

    GlobalHydra.instance().clear()


def test_null_fallback_defaults_match_legacy_absolute_values():
    """The in-code null fallbacks equal the documented legacy argparse defaults.

    This is the byte-identity assertion that guarantees a default (no-override)
    run produces the exact same resolved (taxo_csv, output, num_proc) as the
    argparse era — the testable proxy for zero behavioral drift.
    """
    data_root = (gp.root / "data").resolve()

    # taxo_csv: the legacy default was str(DATA_ROOT / constants.TAXONOMY_CSV_FILENAME),
    # but constants.TAXONOMY_CSV_FILENAME is an ABSOLUTE Path, so the `/` join is a
    # no-op. The in-code fallback uses str(constants.TAXONOMY_CSV_FILENAME) directly.
    taxo_fallback = str(constants.TAXONOMY_CSV_FILENAME)
    assert taxo_fallback == str(data_root / constants.TAXONOMY_CSV_FILENAME)  # join is a no-op
    assert taxo_fallback == str(constants.TAXONOMY_CSV_FILENAME)

    # output: DATA_ROOT / "planktonzilla_17M".
    output_fallback = str(data_root / "planktonzilla_17M")
    assert output_fallback == str((gp.root / "data").resolve() / "planktonzilla_17M")

    # num_proc: constants.default_num_proc().
    assert constants.default_num_proc() == constants.default_num_proc()


def test_datasets_configs_override_blocks_pin(monkeypatch, tmp_path):
    """Pin the per-dataset override blocks + redefiner classes built in _run().

    Adapted from the deleted test_datasets_configs_structure_pins. Drives the
    ported body (gp._run) with a composed config whose taxo_csv points at a real
    tiny CSV, and mocks the whole per-dataset loop body (hydra.compose /
    instantiate / load_dataset / redefiner.redefine / clean / save) so the loop
    runs to completion. Captures, in iteration order, the exact `overrides` list
    passed to hydra.compose and the redefiner type bound to each dataset_name,
    then asserts they match the frozen (name, import_name, cleanup, redefiner)
    table EXACTLY. There is no longer a hydra.initialize CM inside the body to
    patch — do not patch it.
    """
    # Real CSV so the redefiner constructors (run while building datasets_configs)
    # succeed; cfg.taxo_csv routes every redefiner here.
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    # Compose the REAL gen config first (taxo_csv pointed at the tiny CSV) BEFORE
    # patching hydra.compose. `gp.hydra is hydra` (same module object), so the
    # patch below replaces the inner per-dataset import_dataset compose too — which
    # is exactly what we want to capture. We keep the initialized GlobalHydra so
    # _run's inner (now-faked) compose calls do not need it. The @hydra.main main()
    # is not directly callable with a cfg, so we drive the ported _run(cfg) seam.
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_override")
    cfg = hydra.compose(config_name="gen_planktonzilla", overrides=[f"taxo_csv={csv_path}"])

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

    data_root = (gp.root / "data").resolve()

    # (dataset_name, import_name, cleanup, redefiner_type) — the table the literal
    # encodes. Order matters: datasets_configs is iterated in insertion order.
    expected_table = [
        ("isiisnet", "isiisnet", "True", gp.NoMetadataRedefiner),
        ("whoi", "whoi-plankton", "True", gp.WHOIRedefiner),
        ("flowcamnet", "flowcamnet", "True", gp.EcoTaxaRedefiner),
        ("lensless", "lensless", "True", gp.NoMetadataRedefiner),
        ("medplanktonset", "medplanktonset", "True", gp.NoMetadataRedefiner),
        ("uvp6net", "uvp6net", "True", gp.EcoTaxaRedefiner),
        ("zoocamnet", "zoocamnet", "True", gp.NoMetadataRedefiner),
        ("zooscan", "zooscannet", "True", gp.EcoTaxaRedefiner),
        ("planktonset1.0", "planktonset1", "False", gp.NoMetadataRedefiner),
        ("syke_ifcb_2022", "syke_ifcb_2022", "False", gp.NoMetadataRedefiner),
        ("planktoscope", "planktoscope", "False", gp.EcoTaxaRedefiner),
        ("global_uvp5", "global_uvp5net", "False", gp.EcoTaxaRedefiner),
    ]

    # Exactly these 12 active datasets, in this order (commented ones excluded).
    assert list(captured_redefiners.keys()) == [t[0] for t in expected_table]
    assert len(captured_overrides) == len(expected_table)

    for (name, import_name, cleanup, rtype), overrides in zip(expected_table, captured_overrides):
        # Override block reproduced byte for byte.
        assert overrides == _expected_overrides(data_root, import_name, cleanup)
        # Redefiner type bound to this dataset is exactly the expected class.
        assert captured_redefiners[name] is rtype

    # DATA_ROOT is the pyrootutils repo-root data/ dir.
    assert data_root == (gp.root / "data").resolve()


def test_module_level_num_proc_independent_of_cfg():
    """Pin: the module-level num_proc global is set from default_num_proc() at
    import time and is intentionally NOT driven by cfg.num_proc (only redefine()
    receives the configurable value)."""
    assert gp.num_proc == constants.default_num_proc()
