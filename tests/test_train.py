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
        with initialize(config_path=hydra_conf_path, job_name="test_training"):
            cfg: DictConfig = compose(
                config_name="train",
                overrides=[
                    f"dataset={dataset_name}",
                    f"model={model_name}",
                    "extras.print_config=False",
                    "extras.enforce_tags=False",
                ],
                return_hydra_config=False,
            )

            cfg.training_arguments.output_dir = tmp_dir
            cfg.paths.output_dir = tmp_dir
            cfg.training_arguments.num_train_epochs = 2
            cfg.training_arguments.auto_find_batch_size = False

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
        with initialize(config_path=hydra_conf_path, job_name="test_training"):
            cfg: DictConfig = compose(
                config_name="train",
                overrides=[
                    f"dataset={dataset_name}",
                    f"model={model_name}",
                    "extras.print_config=False",
                    "extras.enforce_tags=False",
                ],
                return_hydra_config=False,
            )

            cfg.training_arguments.output_dir = tmp_dir
            cfg.paths.output_dir = tmp_dir
            cfg.training_arguments.num_train_epochs = 2
            cfg.training_arguments.auto_find_batch_size = False

            cfg.tracking.use_wandb = False
            cfg.tracking.use_mlflow = False
            cfg.tracking.use_trackio = False
            cfg.model_push_to_hub = False

            metric_dict, _ = train(cfg)

            assert metric_dict
