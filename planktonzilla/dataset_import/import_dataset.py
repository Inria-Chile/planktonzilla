"""
(c) Inria

Hydra entry point for importing plankton datasets.

Composes a dataset-import configuration with Hydra, instantiates the matching
:class:`~planktonzilla.dataset_import.dataset_importer.DatasetImporter`
subclass, and dispatches on the requested ``action`` (``import``,
``update-metadata`` or ``show``). Run as a script via the ``main`` entry point.
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
from omegaconf import DictConfig

from planktonzilla.dataset_import.dataset_importer import DatasetImporter
from planktonzilla.utils import resolvers as _resolvers  # noqa: F401  -- side-effect: registers strip_yaml_suffix
from planktonzilla.utils.hydra import task_wrapper
from planktonzilla.utils.logger import get_pylogger

log = get_pylogger(__name__)


@task_wrapper
def import_dataset(cfg: DictConfig) -> None:
    """Run a dataset-import action from a Hydra configuration.

    Instantiates the ``cfg.dataset_import`` target as a
    :class:`~planktonzilla.dataset_import.dataset_importer.DatasetImporter` and
    dispatches on ``cfg.action``: ``import`` builds and loads (and optionally
    pushes) the dataset, ``update-metadata`` refreshes its Hub card, and
    ``show`` prints its details. Any other action is logged as an error.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        tuple: ``(None, None)``, the metric/object pair expected by the Hydra
        ``task_wrapper`` decorator.
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
    """Hydra-decorated CLI entry point that runs :func:`import_dataset`.

    Args:
        cfg (DictConfig): Configuration composed by Hydra from
            ``configs/import_dataset.yaml``.

    Returns:
        Optional[float]: ``0`` on completion.
    """
    import_dataset(cfg)
    return 0


if __name__ == "__main__":
    main()
