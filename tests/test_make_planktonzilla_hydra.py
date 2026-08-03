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


from pathlib import Path
from unittest.mock import MagicMock

import hydra
import pytest
from datasets import Dataset
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

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
