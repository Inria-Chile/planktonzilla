"""
(c) Inria

Network-free tests for ``planktonzilla.planktonzilla_dataset.make_planktonzilla``
(the ``pz_planktonzilla`` entry point).

These pin the CONTRACT of the consolidated command:

  (a) the config composes with the expected key contract, and inherits the frozen
      `datasets` registry from generate_planktonzilla.yaml intact,
  (b) the DEFAULT no-argument run is byte-equivalent to pz_generate_planktonzilla —
      same per-source override blocks in the same order, same redefiner classes, same
      save target, no Hub read, no push,
  (c) `sources` selection semantics, including the import_name trap and bare strings,
  (d) the guards that stop an expensive or destructive mistake before any I/O,
  (e) the opt-in Hub push.

All network is mocked. (b) is the zero-drift evidence for the consolidation.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import hydra
import pytest
from datasets import Dataset
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from planktonzilla.dataset_import import dataset_importer as di
from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp
from planktonzilla.planktonzilla_dataset import make_planktonzilla as mk

from .test_gen_planktonzilla_hydra import EXPECTED_TABLE, _write_taxonomy_csv


def _compose(overrides=(), job_name="test_make"):
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name=job_name)
    return hydra.compose(config_name="planktonzilla", overrides=list(overrides))


def _drive(monkeypatch, cfg, tmp_path, entry_point, push_mock=None):
    """Run ``entry_point(cfg)`` with the heavy pipeline mocked, capturing the wiring.

    Every patch target lives on ``gp`` on purpose: ``mk`` calls
    ``gp.import_and_redefine_source``, which resolves ``hydra`` / ``os`` /
    ``load_dataset`` from ``gp``'s globals. So the SAME mocks drive both entry points,
    which is what lets the equivalence test compare them fairly.
    """
    captured_overrides = []

    def _fake_compose(*args, **kwargs):
        captured_overrides.append(list(kwargs["overrides"]))
        return MagicMock()

    monkeypatch.setattr(gp.hydra, "compose", _fake_compose)

    importer = MagicMock()
    importer.imagefolder_dir = tmp_path
    # The sidecar protocol: an archive-only source declares nothing and obtains nothing.
    importer.sidecar_targets.return_value = []
    importer.missing_sidecars.return_value = []
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
    monkeypatch.setattr(mk, "clean_corrupt_examples_optimized", lambda ds, **k: ds)

    # The consolidated command asserts the schema of every part; the mocked redefine
    # returns a 1-column stub, so short-circuit that check rather than fake 25 columns.
    monkeypatch.setattr(mk, "assert_consolidated_schema", lambda ds, **k: None)
    monkeypatch.setattr(mk, "log_lookup_coverage", lambda *a, **k: None)

    saved = {}
    monkeypatch.setattr(gp.Dataset, "save_to_disk", lambda self, path: saved.setdefault("path", str(path)))
    if push_mock is not None:
        monkeypatch.setattr(gp.Dataset, "push_to_hub", push_mock)

    entry_point(cfg)
    return captured_overrides, captured_redefiners, saved


def test_config_composes_with_expected_keys():
    """The config exposes the run-shape contract with the documented defaults."""
    cfg = _compose(job_name="test_make_compose")

    assert cfg.sources == "all"
    assert cfg.base is None
    assert cfg.output_dir is None
    assert cfg.refresh == "reuse"
    assert cfg.clean == "fresh"
    assert cfg.sync_taxonomy is True
    assert cfg.sync_unmatched == "keep"
    assert cfg.allow_partial_overwrite is False
    assert cfg.dry_run is False
    assert cfg.check_downloads == "none"
    assert cfg.check_timeout == 30
    assert list(cfg.drop) == []
    assert list(cfg.import_overrides) == []

    # Inherited from generate_planktonzilla.yaml.
    assert cfg.repo_id == "project-oceania/planktonzilla-17M"
    assert cfg.base_repo_id == cfg.repo_id
    assert cfg.push_to_hub is False
    assert cfg.taxonomy_csv_path is None
    assert cfg.num_proc is None
    assert cfg.data_dir is not None

    GlobalHydra.instance().clear()


def test_datasets_table_inherited_intact():
    """The frozen registry reaches the new config unchanged, in order.

    Guards the inheritance: the table has ONE definition in the repo, in
    generate_planktonzilla.yaml, and this config pulls it in through `defaults`.
    """
    cfg = _compose(job_name="test_make_table")

    actual = [(d["name"], d["import_name"], d["cleanup"], d["redefiner"]) for d in cfg.datasets]
    assert actual == EXPECTED_TABLE

    GlobalHydra.instance().clear()


def test_default_invocation_matches_generate_exactly(monkeypatch, tmp_path):
    """ZERO DRIFT: a bare `pz_planktonzilla` does exactly what pz_generate_planktonzilla did.

    Drives both entry points through the same mocked pipeline and compares the
    per-source override blocks, their order, the redefiner class bound to each source,
    and the save target. Also asserts the new command reads nothing from the Hub and
    pushes nothing by default.
    """
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    # Compose BOTH configs before driving anything: _drive patches hydra.compose, and
    # `gp.hydra is hydra`, so a compose after the first drive returns a MagicMock.
    cfg_gen = _compose_generate([f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}"])
    GlobalHydra.instance().clear()
    cfg_make = _compose([f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}"], job_name="test_make_equiv")
    GlobalHydra.instance().clear()

    gen_overrides, gen_redefiners, gen_saved = _drive(monkeypatch, cfg_gen, tmp_path, gp.main)

    hub_reads = []
    monkeypatch.setattr(mk, "load_dataset", lambda *a, **k: hub_reads.append(a) or MagicMock())
    push = MagicMock()
    make_overrides, make_redefiners, make_saved = _drive(monkeypatch, cfg_make, tmp_path, mk.main, push_mock=push)
    GlobalHydra.instance().clear()

    assert make_overrides == gen_overrides, "per-source override blocks differ"
    assert len(make_overrides) == len(EXPECTED_TABLE)
    assert make_redefiners == gen_redefiners, "redefiner class bound to a source differs"
    assert list(make_redefiners) == [name for name, *_ in EXPECTED_TABLE], "source order differs"
    assert make_saved["path"] == gen_saved["path"], "save target differs"
    assert make_saved["path"] == str(tmp_path / constants.DEFAULT_PLANKTONZILLA_DATASET_NAME)

    assert hub_reads == [], "the default run must not read from the Hub"
    push.assert_not_called()


def _compose_generate(overrides):
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_make_gen_ref")
    return hydra.compose(config_name="generate_planktonzilla", overrides=list(overrides))


def test_select_sources_all_and_none():
    """`all` selects the whole registry in order; `[]` and null select nothing."""
    cfg = _compose(job_name="test_make_select")
    GlobalHydra.instance().clear()

    OmegaConf.set_struct(cfg, False)

    assert [e["name"] for e in mk.select_sources(cfg)] == [name for name, *_ in EXPECTED_TABLE]

    cfg.sources = []
    assert mk.select_sources(cfg) == []

    cfg.sources = None
    assert mk.select_sources(cfg) == []


def test_select_sources_returns_registry_order_not_user_order():
    """The user's ordering is ignored: registry order is the concatenation order."""
    cfg = _compose(job_name="test_make_order")
    GlobalHydra.instance().clear()
    OmegaConf.set_struct(cfg, False)

    # whoi is registry index 1, zooscan index 7 — request them backwards.
    cfg.sources = ["zooscan", "whoi"]
    assert [e["name"] for e in mk.select_sources(cfg)] == ["whoi", "zooscan"]


def test_select_sources_accepts_a_bare_string():
    """`sources=whoi` is one source, not five characters."""
    cfg = _compose(job_name="test_make_bare")
    GlobalHydra.instance().clear()
    OmegaConf.set_struct(cfg, False)

    cfg.sources = "whoi"
    assert [e["name"] for e in mk.select_sources(cfg)] == ["whoi"]


def test_select_sources_rejects_an_import_name_by_name():
    """Passing an import_name is called out explicitly, with the name to use instead.

    4 of the 12 entries have an import_name that differs from their name, so this is
    the likeliest typo a user makes.
    """
    cfg = _compose(job_name="test_make_importname")
    GlobalHydra.instance().clear()
    OmegaConf.set_struct(cfg, False)

    for import_name, real_name in [
        ("whoi-plankton", "whoi"),
        ("zooscannet", "zooscan"),
        ("planktonset1", "planktonset1.0"),
        ("global_uvp5net", "global_uvp5"),
    ]:
        cfg.sources = [import_name]
        with pytest.raises(ValueError, match="import_name"):
            mk.select_sources(cfg)
        with pytest.raises(ValueError, match=real_name.replace(".", r"\.")):
            mk.select_sources(cfg)


def test_select_sources_rejects_unknown_and_conflicting_names():
    """An unknown name lists the valid ones; a name in both sources and drop is fatal."""
    cfg = _compose(job_name="test_make_unknown")
    GlobalHydra.instance().clear()
    OmegaConf.set_struct(cfg, False)

    cfg.sources = ["whoiii"]
    with pytest.raises(ValueError, match="Unknown source"):
        mk.select_sources(cfg)

    cfg.sources = ["whoi"]
    cfg.drop = ["whoi"]
    with pytest.raises(ValueError, match="both"):
        mk.select_sources(cfg)


def test_partial_rebuild_over_existing_output_is_refused(monkeypatch, tmp_path):
    """A partial rebuild with no base refuses to overwrite an existing output.

    save_to_disk replaces a different dataset directory silently, so without this
    guard `sources=[lensless]` with a forgotten `base=` would swap the consolidated
    artifact for a one-source fragment and report success. The guard must fire before
    ANY import work happens.
    """
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}", f"output_dir={output_dir}", "sources=[lensless]"],
        job_name="test_make_partial",
    )
    GlobalHydra.instance().clear()

    # Patched only AFTER composing, so the real config is built first.
    composed = []
    monkeypatch.setattr(gp.hydra, "compose", lambda *a, **k: composed.append(k) or MagicMock())

    with pytest.raises(ValueError, match="PARTIAL rebuild"):
        mk.main(cfg)

    assert composed == [], "the guard must fire before any source is composed"


def test_partial_rebuild_allowed_with_explicit_opt_in(monkeypatch, tmp_path):
    """allow_partial_overwrite=true lets the shrink through."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    cfg = _compose(
        [
            f"taxonomy_csv_path={csv_path}",
            f"data_dir={tmp_path}",
            f"output_dir={output_dir}",
            "sources=[lensless]",
            "allow_partial_overwrite=true",
        ],
        job_name="test_make_partial_ok",
    )
    GlobalHydra.instance().clear()

    monkeypatch.setattr(mk, "atomic_replace", lambda ds, path: None)
    overrides, _, _ = _drive(monkeypatch, cfg, tmp_path, mk.main)
    assert len(overrides) == 1


def test_partial_rebuild_into_a_fresh_output_needs_no_opt_in(monkeypatch, tmp_path):
    """The single-source dev build stays one token when the target does not exist."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    cfg = _compose(
        [
            f"taxonomy_csv_path={csv_path}",
            f"data_dir={tmp_path}",
            f"output_dir={tmp_path / 'fresh'}",
            "sources=[lensless]",
        ],
        job_name="test_make_partial_fresh",
    )
    GlobalHydra.instance().clear()

    monkeypatch.setattr(mk, "atomic_replace", lambda ds, path: None)
    overrides, _, _ = _drive(monkeypatch, cfg, tmp_path, mk.main)
    assert len(overrides) == 1


def test_nothing_to_do_is_an_error(tmp_path):
    """No sources and no base is a mistake, not a silent no-op."""
    cfg = _compose([f"data_dir={tmp_path}", "sources=[]"], job_name="test_make_noop")
    GlobalHydra.instance().clear()

    with pytest.raises(ValueError, match="Nothing to do"):
        mk.main(cfg)


@pytest.mark.parametrize(
    "override,message",
    [
        ("refresh=turbo", "refresh must be one of"),
        ("clean=sometimes", "clean must be one of"),
        ("sync_unmatched=maybe", "sync_unmatched must be one of"),
        ("check_downloads=yes", "check_downloads must be one of"),
        ("check_timeout=0", "check_timeout must be"),
        ("check_timeout=null", "check_timeout must be"),
    ],
)
def test_invalid_enum_values_fail_fast(tmp_path, override, message):
    """A mistyped enum fails immediately, not hours into a build."""
    cfg = _compose([f"data_dir={tmp_path}", override], job_name="test_make_enum")
    GlobalHydra.instance().clear()

    with pytest.raises(ValueError, match=message):
        mk.main(cfg)


def test_dry_run_touches_nothing(monkeypatch, tmp_path):
    """dry_run instantiates importers to report the plan but imports/saves/pushes nothing."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}", "dry_run=true"],
        job_name="test_make_dryrun",
    )
    GlobalHydra.instance().clear()

    importer = MagicMock()
    importer.imagefolder_dir = tmp_path
    monkeypatch.setattr(gp.hydra, "compose", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gp.hydra.utils, "instantiate", lambda *a, **k: importer)

    saves, pushes = [], []
    monkeypatch.setattr(mk, "atomic_replace", lambda ds, path: saves.append(path))
    monkeypatch.setattr(gp.Dataset, "push_to_hub", lambda self, *a, **k: pushes.append(a))
    monkeypatch.setattr(gp.RedefineDataset, "redefine", lambda *a, **k: pytest.fail("dry_run must not redefine"))

    mk.main(cfg)

    importer.import_dataset.assert_not_called()
    assert saves == []
    assert pushes == []


def test_push_skipped_by_default(monkeypatch, tmp_path):
    """The default run never pushes to the Hub."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    cfg = _compose([f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}"], job_name="test_make_nopush")
    GlobalHydra.instance().clear()

    push = MagicMock()
    _drive(monkeypatch, cfg, tmp_path, mk.main, push_mock=push)

    push.assert_not_called()


def test_push_when_enabled(monkeypatch, tmp_path):
    """push_to_hub=true pushes exactly once to cfg.repo_id, after the save."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}", "push_to_hub=true"],
        job_name="test_make_push",
    )
    GlobalHydra.instance().clear()

    push = MagicMock()
    _drive(monkeypatch, cfg, tmp_path, mk.main, push_mock=push)

    push.assert_called_once()
    assert push.call_args.args[0] == cfg.repo_id
    assert push.call_args.kwargs.get("private") == cfg.push_as_private


def test_module_has_no_output_dir_constant():
    """The save target comes from cfg, never a module-level path constant."""
    assert not hasattr(mk, "OUTPUT_DIR")


def test_resolve_base_location():
    """base resolves to a Hub repo, the output dir, an explicit path, or nothing."""
    cfg = _compose(job_name="test_make_baseloc")
    GlobalHydra.instance().clear()
    OmegaConf.set_struct(cfg, False)

    out = root / "somewhere"

    cfg.base = None
    assert mk.resolve_base_location(cfg, out) is None

    cfg.base = "hub"
    assert mk.resolve_base_location(cfg, out) == ("hub", cfg.base_repo_id)

    cfg.base = "local"
    assert mk.resolve_base_location(cfg, out) == ("disk", out)

    cfg.base = "/data/staged-pz"
    kind, target = mk.resolve_base_location(cfg, out)
    assert kind == "disk"
    assert str(target) == "/data/staged-pz"


def test_build_overrides_is_module_level_and_frozen_by_default():
    """The default override block is the frozen 4-element one; refresh appends flags."""
    block = gp.build_overrides("/data", "whoi-plankton", True)
    assert block == [
        "dataset_import=whoi-plankton",
        "dataset_import.cleanup_after_processing=True",
        "dataset_import.push_to_hub=False",
        "dataset_import.data_dir=/data",
    ]

    assert gp.build_overrides("/data", "lensless", False, refresh="rebuild")[-1] == (
        "dataset_import.force_imagefolder_preparation=True"
    )

    redownload = gp.build_overrides("/data", "lensless", False, refresh="redownload")
    assert redownload[-2:] == [
        "dataset_import.force_imagefolder_preparation=True",
        "dataset_import.force_download=True",
    ]

    assert gp.build_overrides("/data", "lensless", False, import_overrides=["a=b"])[-1] == "a=b"


def test_version_defaults_to_unset():
    """No version is the default: the run produces an unversioned dataset."""
    cfg = _compose(job_name="test_make_ver_default")
    assert cfg.version is None
    assert cfg.version_strict is False
    assert cfg.version_overwrite is False
    assert cfg.version_message is None
    GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    "value,embeddable",
    [
        ("1.4.0", True),
        ("2026.08.01", True),  # accepted, but normalises to 2026.8.1 when embedded
        ("v1.2", False),  # valid Hub tag, not embeddable
        ("release-candidate", False),
    ],
)
def test_resolve_version_classifies_embeddability(value, embeddable):
    """A version is embeddable only in the x.y.z form datasets.utils.Version accepts."""
    cfg = _compose([f"version={value}"], job_name="test_make_ver_kinds")
    GlobalHydra.instance().clear()

    version, is_embeddable = mk.resolve_version(cfg)
    assert version == value, "the string the user typed is preserved for the Hub tag"
    assert is_embeddable is embeddable


def test_resolve_version_strict_rejects_non_embeddable():
    """version_strict=true refuses anything that cannot be embedded in the artifact."""
    cfg = _compose(["version=v1.2", "version_strict=true"], job_name="test_make_ver_strict")
    GlobalHydra.instance().clear()

    with pytest.raises(ValueError, match="version_strict"):
        mk.resolve_version(cfg)


def test_resolve_version_rejects_a_blank_string():
    """An empty version is a mistake; null is how you ask for unversioned."""
    cfg = _compose(["version=''"], job_name="test_make_ver_blank")
    GlobalHydra.instance().clear()

    with pytest.raises(ValueError, match="empty"):
        mk.resolve_version(cfg)


def test_bad_version_fails_before_any_build_work(monkeypatch, tmp_path):
    """A malformed version costs seconds, not the hours a full build takes."""
    cfg = _compose(
        [f"data_dir={tmp_path}", "version=nope", "version_strict=true"],
        job_name="test_make_ver_earlyfail",
    )
    GlobalHydra.instance().clear()

    composed = []
    monkeypatch.setattr(gp.hydra, "compose", lambda *a, **k: composed.append(k) or MagicMock())

    with pytest.raises(ValueError, match="version_strict"):
        mk.main(cfg)

    assert composed == [], "validation must happen before any source is composed"


def test_version_is_tagged_on_the_hub_after_a_successful_push(monkeypatch, tmp_path):
    """The Hub tag is created only after the push, so it always points at real data."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}", "push_to_hub=true", "version=1.4.0"],
        job_name="test_make_ver_push",
    )
    GlobalHydra.instance().clear()

    order = []
    push = MagicMock(side_effect=lambda *a, **k: order.append("push"))
    monkeypatch.setattr(mk, "tag_hub_release", lambda *a, **k: order.append(("tag", a, k)))

    _drive(monkeypatch, cfg, tmp_path, mk.main, push_mock=push)

    assert order[0] == "push", "the tag must not be created before the push"
    assert order[1][0] == "tag"
    assert order[1][1] == (cfg.repo_id, "1.4.0")
    assert order[1][2]["overwrite"] is False


def test_version_is_not_tagged_when_the_run_does_not_push(monkeypatch, tmp_path):
    """Without a push there is no Hub repo to tag; the run says so instead of failing."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    cfg = _compose(
        [f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}", "version=1.4.0"],
        job_name="test_make_ver_nopush",
    )
    GlobalHydra.instance().clear()

    tags = []
    monkeypatch.setattr(mk, "tag_hub_release", lambda *a, **k: tags.append(a))

    _drive(monkeypatch, cfg, tmp_path, mk.main)

    assert tags == []


def test_tag_hub_release_reports_a_conflict_actionably(monkeypatch):
    """A tag that already exists explains that the push succeeded and how to proceed."""

    class _Api:
        def __init__(self, token=None):
            pass

        def create_tag(self, repo_id, **kwargs):
            raise OSError("409 Conflict: tag already exists")

    monkeypatch.setattr(mk, "HfApi", _Api)

    with pytest.raises(RuntimeError, match="version_overwrite") as excinfo:
        mk.tag_hub_release("org/ds", "1.4.0", token=None)

    # The push already happened, so the message must stop the user re-running the build.
    assert "do not re-run" in str(excinfo.value).lower()


def test_tag_hub_release_overwrite_deletes_then_recreates(monkeypatch):
    """version_overwrite=true moves an existing tag instead of failing."""
    calls = []

    class _Api:
        def __init__(self, token=None):
            pass

        def delete_tag(self, repo_id, **kwargs):
            calls.append(("delete", repo_id, kwargs["tag"]))

        def create_tag(self, repo_id, **kwargs):
            calls.append(("create", repo_id, kwargs["tag"]))

    monkeypatch.setattr(mk, "HfApi", _Api)

    mk.tag_hub_release("org/ds", "1.4.0", token=None, overwrite=True)

    assert calls == [("delete", "org/ds", "1.4.0"), ("create", "org/ds", "1.4.0")]


# --- Empty-result guard (finding 1) -------------------------------------------------


def _bare_cfg(overrides, job_name):
    cfg = _compose(overrides, job_name=job_name)
    GlobalHydra.instance().clear()
    OmegaConf.set_struct(cfg, False)
    return cfg


def test_assemble_refuses_an_empty_result_with_no_base(tmp_path):
    """sources=[] + drop=, with no base, is an empty output — say so, don't crash.

    This slips past main()'s "Nothing to do" guard because `drop` is non-empty, and
    used to reach concatenate_datasets([]) -> "Unable to concatenate an empty list".
    """
    cfg = _bare_cfg([f"data_dir={tmp_path}", "sources=[]", "drop=[whoi]"], "test_empty_nobase")

    with pytest.raises(ValueError, match="EMPTY dataset"):
        mk.assemble(
            base=None,
            fresh={},
            registry=list(cfg.datasets),
            dropped={"whoi"},
            sync_dict={},
            cfg=cfg,
            num_proc_arg=1,
        )


def test_assemble_refuses_when_every_base_source_is_dropped(tmp_path):
    """Dropping every source the base holds leaves nothing; used to be an IndexError."""
    cfg = _bare_cfg([f"data_dir={tmp_path}", "sources=[]", "drop=[whoi]"], "test_empty_alldropped")
    base = Dataset.from_dict({"dataset": ["whoi"] * 3, "x": [1, 2, 3]})

    with pytest.raises(ValueError, match="EMPTY dataset"):
        mk.assemble(
            base=base,
            fresh={},
            registry=list(cfg.datasets),
            dropped={"whoi"},
            sync_dict={},
            cfg=cfg,
            num_proc_arg=1,
        )


def test_empty_result_error_names_what_caused_it(tmp_path):
    """The message identifies the drop set, so the mistake is obvious from the error."""
    cfg = _bare_cfg([f"data_dir={tmp_path}", "sources=[]", "drop=[whoi]"], "test_empty_msg")

    with pytest.raises(ValueError) as excinfo:
        mk.assemble(
            base=None,
            fresh={},
            registry=list(cfg.datasets),
            dropped={"whoi"},
            sync_dict={},
            cfg=cfg,
            num_proc_arg=1,
        )

    message = str(excinfo.value)
    assert "whoi" in message
    assert "nothing was written" in message.lower()


# --- Reference-feature selection (finding 2) ----------------------------------------


def test_reference_features_come_from_the_largest_part():
    """Conforming casts the SMALL side: the reference is the biggest part's features.

    Order-independent on purpose. Taking parts[0] meant that rebuilding the first
    registry entry (isiisnet) made the fresh part the reference, so every carried-over
    base block — up to ~13.6M rows — would be cast instead of the small new one.
    """
    # The two must differ in FEATURES, not just size — otherwise every possible
    # reference choice compares equal and the assertion proves nothing.
    small = Dataset.from_dict({"a": [1]})
    large = Dataset.from_dict({"a": ["x"] * 50})
    assert small.features != large.features, "fixture must be able to tell the two apart"

    # Largest wins whichever position it occupies.
    assert mk._reference_features([small, large]) == large.features
    assert mk._reference_features([large, small]) == large.features
    assert mk._reference_features([small, large]) != small.features


def test_conform_schema_does_not_cast_when_features_already_match(monkeypatch):
    """A part whose features already match is returned untouched, never cast."""
    ds = Dataset.from_dict({"a": [1, 2]})

    def _boom(self, *a, **k):
        raise AssertionError("cast must not run when features already match")

    monkeypatch.setattr(Dataset, "cast", _boom)

    assert mk.conform_schema(ds, ds.features) is ds


# --- atomic_replace staging cleanup (finding 3) -------------------------------------


def test_atomic_replace_cleans_up_staging_when_the_save_fails(tmp_path, monkeypatch):
    """A failed save leaves no .new-<pid> residue and does not touch the existing output."""
    output_dir = tmp_path / "planktonzilla-17M"
    output_dir.mkdir()
    (output_dir / "marker.txt").write_text("original")

    ds = Dataset.from_dict({"a": [1]})

    def _failing_save(self, path):
        # Write something first, so the staging dir genuinely exists when we blow up.
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "partial.arrow").write_bytes(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr(Dataset, "save_to_disk", _failing_save)

    with pytest.raises(OSError, match="No space left"):
        mk.atomic_replace(ds, output_dir)

    residue = [p.name for p in tmp_path.iterdir() if ".new-" in p.name or ".old-" in p.name]
    assert residue == [], f"staging left behind: {residue}"
    assert (output_dir / "marker.txt").read_text() == "original", "existing output must be intact"


def test_assemble_conforms_to_the_largest_part_not_the_first(tmp_path):
    """assemble() takes its reference from the largest part — WIRING, not just the helper.

    Discriminating by construction: the carried-over base part (5 rows) types its `n`
    column int64 while the freshly rebuilt part (3 rows) types it int32, and isiisnet is
    registry index 0. With `reference = parts[0].features` the small fresh part won and
    the result came out int32 — casting the big side on the real dataset. The assertion
    below fails in that case, which a test that only calls _reference_features directly
    cannot detect.
    """
    from datasets import Features, Value

    cfg = _bare_cfg([f"data_dir={tmp_path}", "sync_taxonomy=false"], "test_assemble_ref")

    base_features = Features({"dataset": Value("string"), "n": Value("int64")})
    fresh_features = Features({"dataset": Value("string"), "n": Value("int32")})

    base = Dataset.from_dict({"dataset": ["lensless"] * 5, "n": list(range(5))}, features=base_features)
    fresh = {"isiisnet": Dataset.from_dict({"dataset": ["isiisnet"] * 3, "n": [0, 1, 2]}, features=fresh_features)}

    registry = [{"name": "isiisnet"}, {"name": "lensless"}]

    final = mk.assemble(
        base=base,
        fresh=fresh,
        registry=registry,
        dropped=set(),
        sync_dict={},
        cfg=cfg,
        num_proc_arg=1,
    )

    assert final.features["n"] == Value("int64"), "reference should come from the 5-row base part"
    assert [r["dataset"] for r in final] == ["isiisnet"] * 3 + ["lensless"] * 5


# --- Network pre-flight (dry_run / check_downloads) ---------------------------------


class _StubImporter:
    """An importer for the pre-flight to walk over: no Hydra, no network, no disk.

    Implements exactly the protocol the pre-flight consumes — the imagefolder, the
    manual-download helpers, ``probe_downloads`` and the three sidecar hooks.
    """

    def __init__(self, imagefolder, results=(), sidecar_targets=(), missing_sidecars=()):
        self.imagefolder_dir = imagefolder
        self._results = list(results)
        self._sidecar_targets = list(sidecar_targets)
        self._missing_sidecars = list(missing_sidecars)
        self.probes = []
        self.ensured = 0

    def missing_manual_downloads(self):
        return []

    def manual_download_instructions(self):
        return ""

    def probe_downloads(self, *, timeout=None, session=None):
        self.probes.append(timeout)
        return self._results

    def sidecar_targets(self):
        return self._sidecar_targets

    def missing_sidecars(self):
        return self._missing_sidecars

    def ensure_sidecars(self):
        self.ensured += 1
        return {}


def _stub_importers(monkeypatch, tmp_path, results=(), built=False, sidecar_targets=(), missing_sidecars=()):
    """Replace importer instantiation with one stub per selected source.

    ``built`` gives every stub a non-empty imagefolder, which is what makes a real run
    skip its download — the state the `needed` scope keys off.
    """
    made = []

    def _instantiate(*args, **kwargs):
        imagefolder = tmp_path / f"imagefolder_{len(made)}"
        if built:
            (imagefolder / "a_class").mkdir(parents=True)
            (imagefolder / "a_class" / "img.png").write_bytes(b"x")
        made.append(_StubImporter(imagefolder, results, sidecar_targets, missing_sidecars))
        return made[-1]

    monkeypatch.setattr(gp.hydra, "compose", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gp.hydra.utils, "instantiate", _instantiate)
    return made


def _probe(ok=True, location="https://example.invalid/a.zip", detail="HTTP 200, application/zip", size=1024, warning=None):
    return di.ProbeResult(kind="url", location=location, ok=ok, detail=detail, size=size, warning=warning)


def _preflight_cfg(tmp_path, overrides, job_name):
    """A composed cfg with a valid taxonomy CSV, ready to drive the pre-flight."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "isiisnet", "y")
    cfg = _compose([f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}", *overrides], job_name=job_name)
    GlobalHydra.instance().clear()
    return cfg


def test_a_plain_run_never_pre_flights(monkeypatch, tmp_path):
    """check_downloads=none without dry_run leaves the run exactly as it always was.

    The zero-drift guarantee of the consolidated command is asserted elsewhere against
    pz_generate_planktonzilla; this pins that the new checks cannot creep into it.
    """
    cfg = _preflight_cfg(tmp_path, [], "test_make_preflight_off")

    def _boom(**kwargs):
        raise AssertionError("a plain run must not pre-flight")

    monkeypatch.setattr(mk, "run_preflight", _boom)
    monkeypatch.setattr(mk, "atomic_replace", lambda ds, path: None)

    _drive(monkeypatch, cfg, tmp_path, mk.main)


def test_dry_run_makes_no_network_call(monkeypatch, tmp_path):
    """The default dry run answers from local state only: no HTTP session is opened."""
    cfg = _preflight_cfg(tmp_path, ["dry_run=true"], "test_make_preflight_local")
    importers = _stub_importers(monkeypatch, tmp_path)

    def _no_network(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(mk.requests, "Session", _no_network)
    monkeypatch.setattr(mk, "HfApi", _no_network)

    mk.main(cfg)

    assert len(importers) == len(EXPECTED_TABLE), "every selected source is still resolved"
    assert all(importer.probes == [] for importer in importers), "nothing may be probed"


def test_dry_run_probes_every_download_when_asked(monkeypatch, tmp_path):
    """check_downloads=all probes each source, through the configured timeout."""
    cfg = _preflight_cfg(
        tmp_path,
        ["dry_run=true", "check_downloads=all", "check_timeout=5"],
        "test_make_preflight_probe",
    )
    importers = _stub_importers(monkeypatch, tmp_path, results=[_probe()], built=True)

    mk.main(cfg)

    assert [importer.probes for importer in importers] == [[5]] * len(EXPECTED_TABLE)


def test_an_unreachable_download_stops_a_real_run_before_any_import(monkeypatch, tmp_path):
    """check_downloads on a REAL run refuses to start, rather than failing four sources in.

    ``built=False`` matters: these sources have no imagefolder, so the run really would
    download them, which is what makes an unreachable archive fatal to it.
    """
    cfg = _preflight_cfg(tmp_path, ["check_downloads=needed"], "test_make_preflight_gate")
    _stub_importers(
        monkeypatch,
        tmp_path,
        results=[_probe(ok=False, location="https://example.invalid/gone.tar", detail="HTTP 404, text/html")],
        built=False,
    )

    monkeypatch.setattr(gp.RedefineDataset, "redefine", lambda *a, **k: pytest.fail("the run must not import"))
    saves = []
    monkeypatch.setattr(mk, "atomic_replace", lambda ds, path: saves.append(path))

    with pytest.raises(RuntimeError, match="Pre-flight found") as excinfo:
        mk.main(cfg)

    message = str(excinfo.value)
    assert "gone.tar" in message, "the failing URL must be named"
    assert "HTTP 404" in message
    assert saves == []


def test_a_warning_alone_does_not_stop_the_run(monkeypatch, tmp_path):
    """An HTML body is reported loudly but is not a verdict a machine can make."""
    cfg = _preflight_cfg(tmp_path, ["dry_run=true", "check_downloads=all"], "test_make_preflight_warn")
    _stub_importers(monkeypatch, tmp_path, results=[_probe(warning="the server returned an HTML page")], built=True)

    mk.main(cfg)


def test_needed_scope_skips_the_sources_that_are_already_built():
    """`needed` asks "can THIS run proceed?"; `all` asks "is every source downloadable?"."""
    entries = [{"name": "whoi"}, {"name": "zoolake"}]
    importers = [(entry, _StubImporter(Path("/nowhere"), [_probe()])) for entry in entries]

    checks, total = mk.check_source_downloads(importers, ["whoi"], scope="needed", timeout=1)
    probed = [name for entry, importer in importers for name in [entry["name"]] if importer.probes]
    assert probed == ["whoi"]
    assert total == 1024
    # Named, not counted: a probe covering 1 of 2 sources must not read as a clean bill
    # of health for both.
    assert any("zoolake" in check.detail for check in checks if check.name == "downloads-skipped")

    for _, importer in importers:
        importer.probes.clear()
    mk.check_source_downloads(importers, ["whoi"], scope="all", timeout=1)
    assert all(importer.probes for _, importer in importers)


def test_an_all_scope_audit_blocks_only_what_a_real_run_actually_needs():
    """`all` must be usable as both an audit and a gate, and they differ on one point.

    A dry run builds nothing, so every failure it finds IS the answer — it exits non-zero.
    A real run may only be stopped by a source it would fetch: refusing to build because
    an archive is unreachable for a source whose imagefolder is already on disk would
    stop a run that was going to succeed.
    """
    entries = [{"name": "whoi"}, {"name": "zoolake"}]
    dead = _probe(ok=False, location="https://example.invalid/gone.tar", detail="HTTP 404, text/html")
    importers = [(entry, _StubImporter(Path("/nowhere"), [dead])) for entry in entries]

    audit, _ = mk.check_source_downloads(importers, ["whoi"], scope="all", timeout=1, audit=True)
    assert all(check.blocking for check in audit if check.name.startswith("download:"))

    gate, _ = mk.check_source_downloads(importers, ["whoi"], scope="all", timeout=1, audit=False)
    verdicts = {check.name: check for check in gate if check.name.startswith("download:")}
    assert verdicts["download:whoi"].blocking, "this run would fetch whoi, so whoi stops it"
    assert not verdicts["download:zoolake"].blocking, "zoolake is reused from disk, so it cannot"
    assert "not blocking" in verdicts["download:zoolake"].detail, "and the report must say why"


def test_the_disk_estimate_counts_only_what_would_be_downloaded():
    """An `all`-scope audit must not inflate the free-space check with archives it audits."""
    entries = [{"name": "whoi"}, {"name": "zoolake"}]
    importers = [(entry, _StubImporter(Path("/nowhere"), [_probe(size=1024)])) for entry in entries]

    _, total = mk.check_source_downloads(importers, ["whoi"], scope="all", timeout=1)

    assert total == 1024, "zoolake is probed but never downloaded, so its bytes are not needed"


def test_a_crashing_probe_costs_one_source_not_the_whole_report():
    """probe_downloads only promises to swallow requests' own exceptions.

    Anything else would propagate out of executor.map and lose the verdicts on the other
    fourteen sources — the one thing this report exists to prevent.
    """

    class _ExplodingImporter(_StubImporter):
        def probe_downloads(self, *, timeout=None, session=None):
            raise ValueError("malformed header")

    importers = [
        ({"name": "whoi"}, _ExplodingImporter(Path("/nowhere"))),
        ({"name": "zoolake"}, _StubImporter(Path("/nowhere"), [_probe()])),
    ]

    checks, _ = mk.check_source_downloads(importers, ["whoi", "zoolake"], scope="needed", timeout=1)
    verdicts = {check.name: check for check in checks}

    assert not verdicts["download:whoi"].ok
    assert "ValueError" in verdicts["download:whoi"].detail
    assert verdicts["download:zoolake"].ok, "the other sources are still reported"


def test_check_taxonomy_csv_catches_a_missing_file_and_a_missing_column(tmp_path):
    """A CSV that lost a rank would build the whole dataset with that rank blank."""
    assert not mk.check_taxonomy_csv(tmp_path / "nope.csv", [])[0].ok

    truncated = tmp_path / "taxo.csv"
    truncated.write_text("Dataset,Raw_Labels,Kingdom\nisiisnet,y,Animalia\n")
    check = mk.check_taxonomy_csv(truncated, [])[0]
    assert not check.ok
    assert "Species" in check.detail, "the absent columns must be named"


def test_check_taxonomy_csv_warns_about_a_source_it_does_not_cover(tmp_path):
    """A source with no CSV row builds fine and produces null taxonomy for every image."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "isiisnet", "y")

    checks = mk.check_taxonomy_csv(csv_path, [{"name": "isiisnet"}, {"name": "whoi"}])
    coverage = next(check for check in checks if check.name == "taxonomy-coverage")

    assert not coverage.ok
    assert not coverage.blocking, "adding a source before curating its labels is legitimate"
    assert "whoi" in coverage.detail and "isiisnet" not in coverage.detail


def test_check_base_on_disk_recognises_a_saved_dataset(tmp_path):
    """The base is identified the way load_from_disk identifies it, without loading it."""
    absent = mk.check_base_on_disk(tmp_path / "nothing")
    assert not absent[0].ok and "does not exist" in absent[0].detail

    not_a_dataset = tmp_path / "empty"
    not_a_dataset.mkdir()
    assert not mk.check_base_on_disk(not_a_dataset)[0].ok

    saved = tmp_path / "saved"
    Dataset.from_dict({"dataset": ["whoi"] * 3}).save_to_disk(str(saved))
    check = mk.check_base_on_disk(saved)[0]
    assert check.ok
    assert "1 shard(s)" in check.detail

    # A dataset built by a builder — which is what the real artifact is — also records
    # its splits, and then the row count and version are reported without loading it.
    info_file = saved / "dataset_info.json"
    info = json.loads(info_file.read_text())
    info["splits"] = {"train": {"name": "train", "num_examples": 3}}
    info["version"] = {"version_str": "1.4.0"}
    info_file.write_text(json.dumps(info))

    detailed = mk.check_base_on_disk(saved)[0]
    assert "3 rows" in detailed.detail
    assert "version 1.4.0" in detailed.detail


def test_check_base_on_disk_notices_a_shard_that_is_gone(tmp_path):
    """state.json names the shards; a deleted one is a broken base, not a small one."""
    saved = tmp_path / "saved"
    Dataset.from_dict({"dataset": ["whoi"] * 3}).save_to_disk(str(saved))
    for shard in saved.glob("*.arrow"):
        shard.unlink()

    checks = mk.check_base_on_disk(saved)
    assert any(not check.ok and check.name == "base-shards" for check in checks)


class _FakeHubApi:
    """Stand-in for HfApi: scripted refs, scripted auth verdict, no network."""

    def __init__(self, tags=(), branches=("main",), auth=None):
        self.tags = tags
        self.branches = branches
        self.auth = auth

    def auth_check(self, repo_id, *, repo_type=None, token=None, write=False):
        if isinstance(self.auth, Exception):
            raise self.auth

    def list_repo_refs(self, repo_id, *, repo_type=None):
        return SimpleNamespace(
            tags=[SimpleNamespace(name=name) for name in self.tags],
            branches=[SimpleNamespace(name=name) for name in self.branches],
        )


def _push_cfg(tmp_path, overrides, job_name):
    cfg = _bare_cfg([f"data_dir={tmp_path}", "push_to_hub=true", *overrides], job_name)
    cfg.hf_token = "hf_test_token"  # so the check never falls back to the ambient login
    return cfg


def test_check_push_target_catches_a_tag_that_already_exists(tmp_path):
    """A tag collision is only discovered AFTER the upload today; this moves it to second 0.

    tag_hub_release's own error exists to stop the user re-running an upload that
    succeeded — which is the expensive way to learn the version name was taken.
    """
    cfg = _push_cfg(tmp_path, ["version=1.4.0"], "test_make_preflight_tag")
    checks = mk.check_push_target(cfg, "1.4.0", api=_FakeHubApi(tags=("1.4.0",)))

    conflict = next(check for check in checks if check.name == "version-tag")
    assert not conflict.ok
    assert "version_overwrite=true" in conflict.detail

    # The same collision with the opt-in set is a deliberate move, not a failure.
    cfg.version_overwrite = True
    moved = next(
        check for check in mk.check_push_target(cfg, "1.4.0", api=_FakeHubApi(tags=("1.4.0",))) if check.name == "version-tag"
    )
    assert moved.ok
    assert "MOVED" in moved.detail


def test_check_push_target_reports_a_free_tag_and_a_new_branch(tmp_path):
    """The happy path says so explicitly, including whether push_revision exists yet."""
    cfg = _push_cfg(tmp_path, ["version=1.5.0", "push_revision=v1.1"], "test_make_preflight_tag_ok")
    checks = mk.check_push_target(cfg, "1.5.0", api=_FakeHubApi(tags=("1.4.0",), branches=("main",)))

    assert all(check.ok for check in checks)
    assert any("does not exist yet" in check.detail for check in checks if check.name == "push-revision")


def test_check_push_target_requires_a_token(tmp_path, monkeypatch):
    """push_to_hub=true with no token anywhere is a five-second failure, not an hours-long one.

    ``get_token`` is stubbed rather than trusted: this machine has a cached login and a
    HF_TOKEN in .env, so an unstubbed check would silently pass and prove nothing.
    """
    cfg = _bare_cfg([f"data_dir={tmp_path}", "push_to_hub=true"], "test_make_preflight_token")
    cfg.hf_token = None
    monkeypatch.setattr(mk, "get_token", lambda: None)

    checks = mk.check_push_target(cfg, None, api=_FakeHubApi())

    assert [check.name for check in checks] == ["push-token"], "nothing else can be checked without one"
    assert not checks[0].ok
    assert "HF_TOKEN" in checks[0].detail


def test_check_writable_dir_reports_free_space_and_a_dead_end(tmp_path):
    """Writability is tested by writing, because os.access lies on NFS and ACL mounts."""
    ok = mk.check_writable_dir("output-dir", tmp_path / "not" / "created" / "yet")[0]
    assert ok.ok
    assert "free" in ok.detail
    assert not list(tmp_path.glob(".pz-preflight-*")), "the probe file must be removed"

    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory")
    assert not mk.check_writable_dir("output-dir", blocked / "under-a-file")[0].ok


def test_check_writable_dir_warns_when_the_downloads_will_not_fit(tmp_path):
    """A build needs about three times the download size; saying so beats ENOSPC at hour 6."""
    check = mk.check_writable_dir("data-dir", tmp_path, needed_bytes=10**18)[0]

    assert not check.ok
    assert not check.blocking, "the estimate is a lower bound, not a verdict"
    assert f"{mk.DISK_SPACE_FACTOR}x" in check.detail


# --- Sidecar inputs in the pre-flight and in a plain run ---------------------------------


def _built_stub(tmp_path, **kwargs):
    imagefolder = tmp_path / "imagefolder_built"
    (imagefolder / "a_class").mkdir(parents=True)
    (imagefolder / "a_class" / "img.png").write_bytes(b"x")
    return _StubImporter(imagefolder, **kwargs)


_SRC = {"name": "src", "import_name": "src", "cleanup": False, "redefiner": "none"}


def test_missing_sidecars_make_a_built_source_a_fetcher(tmp_path):
    """A built imagefolder normally skips the probes; a missing sidecar puts the source back on the list."""
    stub = _built_stub(
        tmp_path,
        results=[_probe(location="https://example.invalid/Table_S1.csv")],
        sidecar_targets=[("url", "https://example.invalid/Table_S1.csv")],
        missing_sidecars=[tmp_path / "frepj_tables" / "Table_S1.csv"],
    )
    checks, fetch_names = mk.report_source_state([(_SRC, stub)], SimpleNamespace(refresh="reuse"))

    assert fetch_names == ["src"]
    (sidecar_check,) = [c for c in checks if c.name == "sidecars:src"]
    assert sidecar_check.ok
    assert "Table_S1.csv" in sidecar_check.detail and str(tmp_path / "frepj_tables") in sidecar_check.detail

    download_checks, _ = mk.check_source_downloads([(_SRC, stub)], fetch_names, scope="needed", timeout=1, audit=True)
    assert stub.probes == [1], "the source IS probed under `needed`, built imagefolder or not"
    assert not any(c.name == "downloads-skipped" for c in download_checks)


def test_verified_sidecars_leave_a_built_source_unprobed(tmp_path):
    """Verified sidecars change nothing: a built source is still not one this run fetches."""
    stub = _built_stub(tmp_path, sidecar_targets=[("url", "https://example.invalid/Table_S1.csv")], missing_sidecars=[])
    checks, fetch_names = mk.report_source_state([(_SRC, stub)], SimpleNamespace(refresh="reuse"))

    assert fetch_names == []
    (sidecar_check,) = [c for c in checks if c.name == "sidecars:src"]
    assert sidecar_check.ok and "fetched sidecar target(s) satisfied" in sidecar_check.detail

    download_checks, _ = mk.check_source_downloads([(_SRC, stub)], fetch_names, scope="needed", timeout=1, audit=True)
    assert stub.probes == []
    assert any(c.name == "downloads-skipped" and "src" in c.detail for c in download_checks)


def test_a_drifted_sidecar_is_a_warning_not_a_failure(tmp_path):
    """On disk but failing its pin: re-fetched before the first import, so a WARN, never a FAIL."""
    drifted = tmp_path / "frepj_tables" / "Table_S3.csv"
    drifted.parent.mkdir(parents=True)
    drifted.write_bytes(b"drifted")
    stub = _built_stub(tmp_path, sidecar_targets=[("url", "https://example.invalid/t")], missing_sidecars=[drifted])

    checks, fetch_names = mk.report_source_state([(_SRC, stub)], SimpleNamespace(refresh="reuse"))

    assert fetch_names == ["src"]
    warnings = [c for c in checks if c.name == "sidecars:src" and not c.ok]
    assert len(warnings) == 1 and not warnings[0].blocking and "fails verification" in warnings[0].detail


def test_a_missing_bundled_sidecar_blocks_a_dry_run(monkeypatch, tmp_path):
    """A committed file that is gone cannot be repaired by any run: a blocking FAIL, nothing built."""
    gone = tmp_path / "gone.csv"
    cfg = _preflight_cfg(tmp_path, ["dry_run=true"], "test_make_preflight_bundled_gone")
    _stub_importers(monkeypatch, tmp_path, built=True, sidecar_targets=[("bundled", str(gone))])
    monkeypatch.setattr(mk, "atomic_replace", lambda ds, path: pytest.fail("nothing may be written"))

    with pytest.raises(RuntimeError, match="Pre-flight found") as excinfo:
        mk.main(cfg)
    assert "sidecars:" in str(excinfo.value) and "gone.csv" in str(excinfo.value)


def _drive_plain_run(monkeypatch, cfg, tmp_path, ensure_side_effect, import_side_effect):
    """A plain run (no pre-flight) with the sidecar step and the import step instrumented."""
    monkeypatch.setattr(gp.hydra, "compose", lambda *a, **k: MagicMock())
    importer = MagicMock()
    importer.imagefolder_dir = tmp_path
    importer.ensure_sidecars.side_effect = ensure_side_effect
    monkeypatch.setattr(gp.hydra.utils, "instantiate", lambda *a, **k: importer)
    monkeypatch.setattr(mk, "import_and_redefine_source", import_side_effect)
    monkeypatch.setattr(mk, "assert_consolidated_schema", lambda ds, **k: None)
    monkeypatch.setattr(mk, "log_lookup_coverage", lambda *a, **k: None)
    monkeypatch.setattr(mk, "clean_corrupt_examples_optimized", lambda ds, **k: ds)
    monkeypatch.setattr(gp.Dataset, "save_to_disk", lambda self, path: None)
    mk.main(cfg)
    return importer


def test_a_plain_run_obtains_sidecars_before_the_first_import(monkeypatch, tmp_path):
    """Every source's sidecars are obtained up front — seconds, not at the sixteenth source's turn."""
    cfg = _preflight_cfg(tmp_path, [], "test_make_sidecars_first")
    order = []
    tiny = Dataset.from_dict({"x": [1]})

    def _ensure():
        order.append("ensure")
        return {}

    passed_importers = []

    def _import(entry, **kwargs):
        order.append("import")
        passed_importers.append(kwargs["importer"])
        return tiny

    importer = _drive_plain_run(monkeypatch, cfg, tmp_path, _ensure, _import)

    assert order == ["ensure"] * len(EXPECTED_TABLE) + ["import"] * len(EXPECTED_TABLE)
    assert all(passed is importer for passed in passed_importers), "the up-front importer is reused, not composed again"
    assert importer.ensure_sidecars.call_count == len(EXPECTED_TABLE)


def test_a_plain_run_stops_on_a_sidecar_it_cannot_obtain(monkeypatch, tmp_path):
    """A sidecar that cannot be obtained fails the run before any import — nothing is written."""
    cfg = _preflight_cfg(tmp_path, [], "test_make_sidecars_fail_fast")
    monkeypatch.setattr(mk, "atomic_replace", lambda ds, path: pytest.fail("nothing may be written"))

    def _ensure():
        raise RuntimeError("«frepj» could not obtain its md5-pinned sidecar tables: Table_S3.csv")

    with pytest.raises(RuntimeError, match=r"Table_S3\.csv"):
        _drive_plain_run(monkeypatch, cfg, tmp_path, _ensure, lambda *a, **k: pytest.fail("no import may start"))
