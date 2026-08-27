"""
(c) Inria

Network-free tests for the FREPJ-only selection config (Plan 19-01) and the
reusable frozen-repo guard.

They pin, by construction (Hydra compose only — no build, no network):

  (a) generate_frepj_only.yaml composes to exactly ONE datasets entry
      {name: frepj, import_name: frepj, cleanup: false, redefiner: frepj} targeting
      project-oceania/planktonzilla-frepj with push_to_hub false,
  (b) the DEFAULT generate_planktonzilla.yaml keeps the 15 published sources in place
      and appends frepj LAST (v1.2), repo_id planktonzilla-17M — the output-preserving pin,
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


# The fifteen sources of the frozen planktonzilla-17M, in their concatenation order.
_FROZEN_FIFTEEN = [
    "isiisnet",
    "whoi",
    "flowcamnet",
    "lensless",
    "medplanktonset",
    "uvp6net",
    "zoocamnet",
    "zooscan",
    "planktonset1.0",
    "syke_ifcb_2022",
    "planktoscope",
    "global_uvp5",
    "zoolake",
    "jedioceans",
    "sykezooscan2024",
]


def test_default_config_appends_frepj_last_keeping_the_fifteen_in_place():
    """The default registry is the frozen fifteen, in order, then frepj (v1.2), then daplankton.

    Registry order is the concatenation order of the output, so appending — never
    inserting — is what keeps every published source at the index it already had.

    frepj is pinned by its ABSOLUTE index, not by ``[-1]``: it stopped being last the
    moment daplankton was appended after it, and a negative index would have quietly
    re-pointed this assertion at the new source instead of failing.
    """
    cfg = _compose("generate_planktonzilla")
    try:
        assert cfg.repo_id == "project-oceania/planktonzilla-17M"
        assert len(cfg.datasets) == 17
        assert [d["name"] for d in cfg.datasets[:15]] == _FROZEN_FIFTEEN
        assert dict(cfg.datasets[15]) == {"name": "frepj", "import_name": "frepj", "cleanup": False, "redefiner": "frepj"}
        assert dict(cfg.datasets[16]) == {
            "name": "daplankton",
            "import_name": "daplankton",
            "cleanup": False,
            "redefiner": "none",
        }
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


def test_guard_rejects_frozen_id_with_trailing_slash():
    """assert_not_frozen_repo raises on the frozen id decorated with a trailing slash (CR-01)."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo("project-oceania/planktonzilla-17M/")


def test_guard_rejects_frozen_id_with_surrounding_whitespace():
    """assert_not_frozen_repo raises on the frozen id with leading/trailing whitespace (CR-01)."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo(" project-oceania/planktonzilla-17M ")


def test_guard_rejects_frozen_id_with_whitespace_and_trailing_slash():
    """assert_not_frozen_repo raises when both a trailing slash and surrounding whitespace are present (CR-01)."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo(" project-oceania/planktonzilla-17M/ ")


def test_guard_rejects_frozen_bare_basename_with_trailing_whitespace():
    """assert_not_frozen_repo raises on the bare frozen basename with a trailing space (CR-01)."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo("planktonzilla-17M ")


def test_guard_rejects_frozen_id_with_stray_space_after_slash():
    """assert_not_frozen_repo raises when a stray space follows the ``/`` separator (CR-01)."""
    with pytest.raises(ValueError):
        guard.assert_not_frozen_repo("project-oceania/ planktonzilla-17M")


def test_guard_allows_frepj_target():
    """assert_not_frozen_repo returns None for the intended frepj target."""
    assert guard.assert_not_frozen_repo("project-oceania/planktonzilla-frepj") is None


def test_guard_allows_versioned_release():
    """assert_not_frozen_repo returns None for a versioned release repo id."""
    assert guard.assert_not_frozen_repo("project-oceania/planktonzilla-v1.2") is None
