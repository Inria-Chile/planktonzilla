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

from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from planktonzilla.dataset_import.dataset_importer import DatasetImporter
from planktonzilla.train import strip_yaml_suffix
from planktonzilla.utils.hydra import task_wrapper
from planktonzilla.utils.logger import get_pylogger

log = get_pylogger(__name__)

try:
    OmegaConf.register_new_resolver("strip_yaml_suffix", strip_yaml_suffix)
except ValueError:
    pass


@task_wrapper
def import_dataset(cfg: DictConfig) -> None:
    """
    Import a dataset as a HuggingFace

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    """

    log.info(f"Instantiating dataset importer «{cfg.dataset_import._target_}».")

    dataset_importer: DatasetImporter = hydra.utils.instantiate(cfg.dataset_import)

    if cfg.get("action") == "import":
        dataset_importer.import_dataset()
        log.info(f"Done importing dataset «{cfg.dataset_import._target_}».")
    elif cfg.get("action") == "update-metadata":
        dataset_importer.update_dataset_metadata()
        log.info(f"Done updating metadata of dataset «{cfg.dataset_import._target_}».")
    elif cfg.get("action") == "show":
        dataset_importer.show_details()
        log.info(f"Done showing details of dataset «{cfg.dataset_import._target_}».")
    else:
        log.error(f"Unsupported action={cfg.get('action', None)}. Valid values are: import, update-metadata and show.")

    return None, None  # because of Hydra


@hydra.main(
    version_base="1.3",
    config_path=str(root / "configs"),
    config_name="import_dataset.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    import_dataset(cfg)
    return 0


if __name__ == "__main__":
    main()
