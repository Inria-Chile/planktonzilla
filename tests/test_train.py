"""
(c) Inria
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from planktonzilla.train import train

from .shared import dataset_names, model_names, skip_in_github_ci

losses = ["asymmetric", "balanced_meta_softmax", "focal", "ldam", "max_margin", "ral"]


@skip_in_github_ci
@pytest.mark.parametrize("dataset_name", dataset_names)
@pytest.mark.parametrize("model_name", model_names)
def test_training(hydra_conf_path: Path, model_name: str, dataset_name: str):
    with TemporaryDirectory() as tmp_dir:
        with initialize(config_path=hydra_conf_path, job_name="test_training", version_base="1.3"):
            cfg: DictConfig = compose(
                config_name="train",
                overrides=[
                    f"dataset={dataset_name}",
                    f"model={model_name}",
                    "extras.print_config=False",
                    "extras.enforce_tags=False",
                    "training_arguments=test_minirun",
                ],
                return_hydra_config=False,
            )

            cfg.training_arguments.output_dir = tmp_dir
            cfg.paths.output_dir = tmp_dir
            cfg.tracking.use_wandb = False
            cfg.tracking.use_mlflow = False
            cfg.tracking.use_trackio = False
            cfg.model_push_to_hub = False

            metric_dict, _ = train(cfg)

            assert metric_dict


@skip_in_github_ci
@pytest.mark.parametrize("dataset_name", dataset_names)
@pytest.mark.parametrize("model_name", model_names)
@pytest.mark.parametrize("custom_loss", losses)
def test_training_custom_losses(hydra_conf_path: Path, model_name: str, dataset_name: str, custom_loss: str):
    with TemporaryDirectory() as tmp_dir:
        with initialize(config_path=hydra_conf_path, job_name="test_training", version_base="1.3"):
            cfg: DictConfig = compose(
                config_name="train",
                overrides=[
                    f"dataset={dataset_name}",
                    f"model={model_name}",
                    f"custom_loss={custom_loss}",
                    "extras.print_config=False",
                    "extras.enforce_tags=False",
                    "training_arguments=test_minirun",
                ],
                return_hydra_config=False,
            )

            cfg.training_arguments.output_dir = tmp_dir
            cfg.paths.output_dir = tmp_dir
            cfg.tracking.use_wandb = False
            cfg.tracking.use_mlflow = False
            cfg.tracking.use_trackio = False
            cfg.model_push_to_hub = False
            cfg.training_arguments.do_eval = False

            metric_dict, _ = train(cfg)

            assert metric_dict


@skip_in_github_ci
@pytest.mark.parametrize("eval_test", [False, True])
def test_the_test_split_is_only_read_when_asked_for(hydra_conf_path: Path, eval_test: bool):
    """End-to-end proof that the held-out gate actually holds the test split out.

    The unit tests in `tests/test_experiment_protocol.py` pin the predicate; this runs the real
    `train()` and checks the metric dict it returns, which is what a caller — and every wandb
    dashboard — actually sees. Validation must be reported either way: it is what you tune on,
    and it stays governed by `do_eval` alone.
    """
    with TemporaryDirectory() as tmp_dir:
        with initialize(config_path=hydra_conf_path, job_name="test_eval_gate", version_base="1.3"):
            cfg: DictConfig = compose(
                config_name="train",
                overrides=[
                    f"dataset={dataset_names[0]}",
                    f"model={model_names[0]}",
                    "extras.print_config=False",
                    "extras.enforce_tags=False",
                    "training_arguments=test_minirun",
                ],
                return_hydra_config=False,
            )

            cfg.training_arguments.output_dir = tmp_dir
            cfg.paths.output_dir = tmp_dir
            cfg.tracking.use_wandb = False
            cfg.tracking.use_mlflow = False
            cfg.tracking.use_trackio = False
            cfg.model_push_to_hub = False
            cfg.training_arguments.do_eval = True
            cfg.eval_test = eval_test

            metric_dict, _ = train(cfg)

            test_keys = [k for k in metric_dict if k.startswith("test_")]
            val_keys = [k for k in metric_dict if k.startswith("val_")]

            assert val_keys, f"validation must be reported regardless of eval_test; got {sorted(metric_dict)}"
            if eval_test:
                assert test_keys, f"eval_test=true must report the test split; got {sorted(metric_dict)}"
            else:
                assert not test_keys, f"the test split must be held out by default; leaked {test_keys}"
