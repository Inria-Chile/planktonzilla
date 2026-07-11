"""
(c) Inria

Network-free tests for the FREPJ-only selection config (Plan 19-01) and the
reusable frozen-repo guard.

They pin, by construction (Hydra compose only — no build, no network):

  (a) generate_frepj_only.yaml composes to exactly ONE datasets entry
      {name: frepj, import_name: frepj, cleanup: false, redefiner: frepj} targeting
      project-oceania/planktonzilla-frepj with push_to_hub false,
  (b) the DEFAULT generate_planktonzilla.yaml stays unchanged (12 entries,
      repo_id planktonzilla-17M, frepj absent) — the output-preserving pin,
  (c) assert_not_frozen_repo rejects the frozen planktonzilla-17M (full id, bare
      basename, and a defensive case-insensitive basename) and allows the
      intermediate planktonzilla-frepj / a versioned release.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from planktonzilla.planktonzilla_dataset import frozen_repo_guard as guard


def _compose(config_name, overrides=None):
    """Compose a config from ../configs, clearing the GlobalHydra singleton first."""
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_frepj_only")
    return hydra.compose(config_name=config_name, overrides=overrides or [])


def test_frepj_only_config_single_entry():
    """generate_frepj_only composes to the single frepj entry + frepj repo + no push."""
    cfg = _compose("generate_frepj_only")
    try:
        assert cfg.repo_id == "project-oceania/planktonzilla-frepj"
        assert cfg.push_to_hub is False
        assert cfg.push_as_private is True

        assert len(cfg.datasets) == 1
        entry = cfg.datasets[0]
        assert entry["name"] == "frepj"
        assert entry["import_name"] == "frepj"
        assert entry["redefiner"] == "frepj"
        assert entry["cleanup"] is False

        # Null fallbacks preserved (resolve to the in-code defaults at run time).
        assert cfg.get("taxonomy_csv_path") is None
        assert cfg.get("num_proc") is None
        assert cfg.task_name == "generate_frepj_only"
    finally:
        GlobalHydra.instance().clear()


def test_default_config_unchanged_output_preserving_pin():
    """The default generate_planktonzilla stays 12 entries / planktonzilla-17M / no frepj."""
    cfg = _compose("generate_planktonzilla")
    try:
        assert cfg.repo_id == "project-oceania/planktonzilla-17M"
        assert len(cfg.datasets) == 12
        assert not any(d["name"] == "frepj" for d in cfg.datasets)
    finally:
        GlobalHydra.instance().clear()


def test_guard_rejects_frozen_full_repo_id():
    """assert_not_frozen_repo raises on the frozen full owner/name id."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo("project-oceania/planktonzilla-17M")


def test_guard_rejects_frozen_bare_basename():
    """assert_not_frozen_repo raises on the bare frozen basename."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo("planktonzilla-17M")


def test_guard_rejects_frozen_case_insensitive_basename():
    """assert_not_frozen_repo raises on a differently-cased frozen basename."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo("Project-Oceania/Planktonzilla-17M")


def test_guard_allows_frepj_target():
    """assert_not_frozen_repo returns None for the intended frepj target."""
    assert guard.assert_not_frozen_repo("project-oceania/planktonzilla-frepj") is None


def test_guard_allows_versioned_release():
    """assert_not_frozen_repo returns None for a versioned release repo id."""
    assert guard.assert_not_frozen_repo("project-oceania/planktonzilla-v1.2") is None
