"""
(c) Inria

Tests for experiment hygiene: the held-out test split, and the experiment config group.

Both are about a run measuring what it claims to. A test split read on every iteration is not
held out, and an experiment config that cannot compose cannot be run at all — so neither defect
shows up as a wrong number, only as a number that means less than it appears to.
"""

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import planktonzilla.utils.resolvers  # noqa: F401  -- side-effect: registers strip_yaml_suffix
from planktonzilla.train import should_evaluate_test_split

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
EXPERIMENT_DIR = os.path.join(CONFIG_DIR, "experiment")

# Two configs are irreparable rather than merely broken: they select `dataset/cifar100`,
# `dataset/inaturalist`, `augmentation/cifar100`, `augmentation/inaturalist` and
# `model/timm-resnet50` — none of which exist — and instantiate `deep_plankton.losses.LdamLoss`
# and `deep_plankton.scheduler.CustomLRScheduler` from a predecessor package that is not a
# dependency of this project and is not importable. Repairing them means authoring five config
# files and a corpus loader, so they are recorded here instead of guessed at. `strict=True`, so
# this flips to a failure the moment someone lands those configs.
IRREPARABLE = {
    "base_cifar100": "needs dataset/cifar100, augmentation/cifar100, model/timm-resnet50, and deep_plankton",
    "base_inaturalist": "needs dataset/inaturalist, augmentation/inaturalist, and deep_plankton",
}


def _experiment_names():
    return sorted(f[: -len(".yaml")] for f in os.listdir(EXPERIMENT_DIR) if f.endswith(".yaml"))


def _compose_and_resolve(name: str):
    """Compose `train.yaml` with this experiment selected, then force every interpolation.

    Resolution is the part that matters and the part `compose()` alone does not do: Hydra
    resolves lazily, so a config referencing a key no config defines composes cleanly and only
    fails later, at access. That is exactly how `experiment_metadata.loss` sat broken.
    """
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={name}"], return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        # `hydra` is a read-only node and is not part of the project's own surface.
        resolved = OmegaConf.create({k: v for k, v in cfg.items() if k != "hydra"})
        OmegaConf.resolve(resolved)
        return resolved


@pytest.fixture(autouse=True)
def _project_root(monkeypatch):
    """`configs/paths/default.yaml` interpolates ${oc.env:PROJECT_ROOT}, set by pyrootutils at runtime."""
    monkeypatch.setenv("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --------------------------------------------------------------------------------------
# The experiment config group (finding 1.10)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _experiment_names())
def test_every_experiment_config_composes_and_resolves(name, request):
    """Every member of the experiment group must compose *and* resolve.

    Before this, 9 of 11 failed to compose on missing group members and the other 2 resolved
    into `model.net.model_name` — a PyTorch-Lightning-era key no config defines — so the entire
    group was unusable. An experiment group is the infrastructure an ablation sits on, so it
    fails loudly here rather than at minute forty of a training run.
    """
    if name in IRREPARABLE:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=IRREPARABLE[name]))
    cfg = _compose_and_resolve(name)
    assert cfg.model is not None, f"{name} composed without a model"


def test_the_experiment_group_is_not_silently_empty():
    """Guards the parametrisation itself: an empty glob would make the suite above vacuous."""
    names = _experiment_names()
    assert len(names) >= 10, names
    assert "default" in names


@pytest.mark.parametrize("name", [n for n in _experiment_names() if n not in IRREPARABLE])
def test_experiment_configs_carry_no_stray_model_keys(name):
    """`cfg.model` is splatted into the model constructor, so a stray key there is not inert.

    `train()` calls `hydra.utils.instantiate(cfg.model, ...)`, which passes every key under
    `model` as a keyword argument. The Lightning-era `model.optimizer` / `model.criterion` /
    `model.scheduler` blocks these configs used to carry would therefore have been instantiated
    and handed to `from_pretrained`, not ignored. Optimizer and schedule settings belong in the
    `training_arguments` group.
    """
    cfg = _compose_and_resolve(name)
    stray = sorted({"optimizer", "criterion", "scheduler"} & set(cfg.model))
    assert not stray, f"{name} leaves {stray} under cfg.model, which instantiate() would pass to the constructor"


def test_experiment_metadata_only_references_keys_that_exist():
    """`experiment_metadata` is logged as tracked hyperparameters, so a dead key breaks the run.

    This is the specific failure that took the whole group down: it referenced
    `model.criterion._target_`, `model.net.model_name` and `dataset.sampler_type`, none of which
    any config in the repo defines.
    """
    cfg = _compose_and_resolve("default")
    assert cfg.experiment_metadata
    for key, value in cfg.experiment_metadata.items():
        assert value is not None, f"experiment_metadata.{key} resolved to None"


# --------------------------------------------------------------------------------------
# The held-out test split
# --------------------------------------------------------------------------------------


class _Args:
    def __init__(self, do_eval=True):
        self.do_eval = do_eval


def test_the_test_split_is_held_out_by_default():
    """The default must be OFF. This is the whole point: the old code read test on every run."""
    assert should_evaluate_test_split(OmegaConf.create({}), _Args(do_eval=True)) is False


def test_the_shipped_config_holds_the_test_split_out():
    """Not just the function default — the config a real run composes."""
    cfg = _compose_and_resolve("default")
    assert cfg.eval_test is False
    assert should_evaluate_test_split(cfg, _Args(do_eval=True)) is False


def test_the_test_split_is_read_when_explicitly_requested():
    assert should_evaluate_test_split(OmegaConf.create({"eval_test": True}), _Args(do_eval=True)) is True


def test_eval_test_does_not_override_do_eval():
    """`do_eval=false` means no evaluation at all; eval_test cannot re-enable it."""
    assert should_evaluate_test_split(OmegaConf.create({"eval_test": True}), _Args(do_eval=False)) is False
