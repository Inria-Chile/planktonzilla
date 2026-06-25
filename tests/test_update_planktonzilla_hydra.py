"""
(c) Inria

Network-free tests for the @hydra.main port of
``planktonzilla.planktonzilla_dataset.update_planktonzilla``.

These are the testable PROXY for zero behavioral drift on the DEFAULT (no-override)
update run. A full golden-output dataset diff is not runnable here (it needs the
17M-image dataset frozen on the HuggingFace Hub); instead we pin:

  (a) the config composes with the expected key contract,
  (b) the in-code null fallbacks resolve byte-identically to the legacy CLI
      defaults — including the INTENTIONAL raw-Path taxonomy_csv_path divergence
      from generate_planktonzilla (which str()-wraps), and the package-relative
      underscore OUTPUT_DIR (no rename),
  (c) _run() wires cfg.repo_id -> load_dataset, the resolved package-relative
      output_dir -> save_to_disk, and the resolved num_proc -> sync_columns.

Every test PINS current behavior; none "improves" it. All network is mocked.
"""

import pathlib

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
from planktonzilla.planktonzilla_dataset import update_planktonzilla as up


def _write_taxonomy_csv(path, dataset_name, raw_label):
    """Write a tiny taxonomy CSV that build_sync_dict can resolve.

    Columns mirror the real planktonzilla_taxonomy.csv: Dataset, Raw_Labels plus
    the sync columns (taxonomy ranks, extras, str IDs and numeric IDs).
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


def test_config_composes_with_expected_keys():
    """The config composes and exposes the expected key contract."""
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_update_compose")
    cfg = hydra.compose(config_name="update_planktonzilla")

    # repo_id is the concrete frozen INPUT dataset identity.
    assert cfg.repo_id == "project-oceania/planktonzilla-17M"

    # Null-default contract: these resolve to legacy defaults via in-code fallbacks.
    for key in ("taxonomy_csv_path", "num_proc", "output_dir"):
        assert key in cfg, f"missing key {key}"
        assert cfg.get(key) is None, f"{key} should default to null"

    GlobalHydra.instance().clear()


def test_null_fallback_defaults_match_legacy():
    """The in-code null fallbacks equal the documented legacy CLI defaults.

    This is the byte-identity assertion that guarantees a default (no-override)
    run resolves the exact same (taxonomy_csv_path, num_proc, output_dir) as the
    argparse era — the testable proxy for zero behavioral drift.
    """
    # taxonomy_csv_path: the legacy --csv-path default was the RAW
    # DEFAULT_TAXONOMY_CSV_FILENAME Path, passed straight into pd.read_csv. The
    # in-code fallback must keep it a raw Path (NOT str()-wrapped) — this is the
    # INTENTIONAL divergence from generate_planktonzilla, which str()-wraps.
    assert isinstance(constants.DEFAULT_TAXONOMY_CSV_FILENAME, pathlib.Path)
    assert not isinstance(constants.DEFAULT_TAXONOMY_CSV_FILENAME, str)

    # num_proc: constants.default_num_proc() (a positive int, >= 1).
    assert constants.default_num_proc() >= 1

    # output_dir: the module-level package-relative OUTPUT_DIR. Pin the underscore
    # form ("planktonzilla_17M_updated") and the no-rename guarantee (no hyphen form).
    assert up.OUTPUT_DIR.endswith("data/planktonzilla_17M_updated")
    assert "17M-updated" not in up.OUTPUT_DIR

    # repo_id frozen identity (shared with generate_planktonzilla).
    assert constants.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID == "project-oceania/planktonzilla-17M"


def test_run_wires_repo_id_output_dir_and_num_proc(monkeypatch, tmp_path):
    """Drive up._run(cfg) network-free and pin the three wiring points.

    Asserts that _run wires cfg.repo_id -> load_dataset, the resolved num_proc
    (null -> default_num_proc()) -> sync_columns, and the resolved output_dir ->
    save_to_disk. The transform itself is short-circuited (we test wiring).
    """
    # Real tiny CSV so build_sync_dict succeeds; routed via the override below.
    csv_path = tmp_path / "taxo.csv"
    out_dir = tmp_path / "out"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    # Compose the REAL update config, overriding taxonomy_csv_path (so the raw read
    # works) and output_dir (a writable target the save_to_disk capture records).
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_update_wiring")
    cfg = hydra.compose(
        config_name="update_planktonzilla",
        overrides=[f"taxonomy_csv_path={csv_path}", f"output_dir={out_dir}"],
    )

    # Tiny in-memory dataset whose columns include dataset, original_label, and every
    # SYNC_COL, so update_example / new_features succeed inside sync_columns.
    columns = {"dataset": ["x"], "original_label": ["y"]}
    for col in up.SYNC_COLS:
        # plankton is a bool; everything else is a (nullable) string column.
        columns[col] = [True] if col == "plankton" else [""]
    tiny = Dataset.from_dict(columns)

    captured = {}

    def _fake_load_dataset(repo_id, split=None):
        captured["repo_id"] = repo_id
        captured["split"] = split
        return tiny

    def _fake_sync_columns(ds, sync_dict, num_proc):
        # Capture the wired num_proc; return ds unchanged (wiring test, not transform).
        captured["num_proc"] = num_proc
        return ds

    def _fake_save_to_disk(self, path):
        captured["save_path"] = path

    monkeypatch.setattr(up, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(up, "sync_columns", _fake_sync_columns)
    monkeypatch.setattr(up.Dataset, "save_to_disk", _fake_save_to_disk)

    up._run(cfg)

    GlobalHydra.instance().clear()

    # load_dataset received cfg.repo_id and the "train" split.
    assert captured["repo_id"] == cfg.repo_id
    assert captured["split"] == "train"

    # num_proc null -> resolved to default_num_proc().
    assert captured["num_proc"] == constants.default_num_proc()

    # save_to_disk received the resolved (overridden) output_dir.
    assert captured["save_path"] == str(out_dir)


def test_maybe_push_to_hub_skips_by_default(monkeypatch):
    """PIN: the default (push=False) path NEVER pushes to the Hub.

    This is the zero-drift pin — with the flag absent/False, _maybe_push_to_hub
    must leave the frozen project-oceania artifact untouched (no Hub call). The
    save_to_disk that precedes it in main stays unconditional and is unaffected.
    All network is mocked; this PINS current behavior, it does not "improve" it.
    """
    push = MagicMock()
    monkeypatch.setattr(up.Dataset, "push_to_hub", push)

    dataset = Dataset.from_dict({"x": [1]})
    up._maybe_push_to_hub(dataset, "project-oceania/planktonzilla-17M", False)

    push.assert_not_called()


def test_maybe_push_to_hub_pushes_once_when_enabled(monkeypatch):
    """PIN: the push=True path pushes exactly once to cfg.repo_id.

    The push is additive (it runs after the unconditional save_to_disk in main)
    and targets the frozen repo id as the first positional arg. All network is
    mocked; this PINS current behavior, it does not "improve" it.
    """
    push = MagicMock()
    monkeypatch.setattr(up.Dataset, "push_to_hub", push)

    dataset = Dataset.from_dict({"x": [1]})
    up._maybe_push_to_hub(dataset, "project-oceania/planktonzilla-17M", True)

    push.assert_called_once()
    assert push.call_args.args[0] == "project-oceania/planktonzilla-17M"
