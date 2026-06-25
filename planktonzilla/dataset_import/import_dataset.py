"""
(c) Inria

``pz_import_dataset`` entry point.

Hydra-driven CLI that instantiates a :class:`DatasetImporter` from
``cfg.dataset_import`` and dispatches on ``cfg.action`` (``import``,
``update-metadata``, or ``show``). ``pyrootutils.setup_root`` runs at module top
level — before the other imports — to set ``sys.path``, find ``configs/``, and load
``.env``.
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
    """Instantiate the configured importer and dispatch on ``cfg.action``.

    Builds the ``DatasetImporter`` from ``cfg.dataset_import`` and runs one action:
    ``import`` (download/normalize/push the dataset), ``update-metadata`` (refresh the
    Hub dataset card), or ``show`` (print the Hub dataset details). An unknown action is
    logged as an error and performs no work. Wrapped by :func:`task_wrapper` for timing,
    exception logging, and logger teardown.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        ``(None, None)`` — the ``(metrics, objects)`` pair the Hydra ``task_wrapper``
        contract expects; this task produces neither.
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
    """Hydra ``pz_import_dataset`` entry point; runs :func:`import_dataset` and returns 0."""
    import_dataset(cfg)
    return 0


if __name__ == "__main__":
    main()
