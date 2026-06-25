"""
(c) Inria

Hydra task-orchestration helpers: the `task_wrapper` decorator, config `extras`, hyperparameter
logging, metric retrieval, and logger teardown shared across entry points.
"""

import time
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from omegaconf import DictConfig

# from pytorch_lightning import Callback
# from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
# from pytorch_lightning.utilities import rank_zero_only
from planktonzilla.utils.logger import get_pylogger
from planktonzilla.utils.rich_utils import enforce_tags, print_config_tree

log = get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Decorator that wraps a Hydra task function in failure-resilient utilities.

    The wrapped callable must have the signature ``(cfg: DictConfig) -> tuple[dict, dict]``
    (returning ``(metric_dict, object_dict)``), and the returned wrapper preserves that
    contract. Hardening the task this way keeps a multirun from aborting on a single failure.

    Side effects (applied around every call):
    - Runs `extras(cfg)` before the task to apply optional pre-task utilities.
    - On exception: logs the full traceback via `log.exception("")` (saved to the `.log` file),
      then re-raises so the failure is not swallowed.
    - In a `finally` block (so it runs even on exception): writes the task execution time to
      `exec_time.log` under `cfg.paths.output_dir`, and calls `close_loggers()` so a crashed
      logger does not take down the rest of a multirun.
    - On success: logs the output dir.

    Args:
        task_func: The Hydra task function to wrap.

    Returns:
        The wrapping callable, which returns the task's ``(metric_dict, object_dict)``.
    """

    def wrap(cfg: DictConfig):
        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! «cfg.extras=null»")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! «cfg.extras.ignore_warnings=True»")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! «cfg.extras.enforce_tags=True»")
        enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! «cfg.extras.print_config=True»")
        print_config_tree(cfg, resolve=True, save_to_file=True)


# @rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


# def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
#     """Instantiates callbacks from config."""
#     callbacks: list[Callback] = []

#     if not callbacks_cfg:
#         log.warning("Callbacks config is empty.")
#         return callbacks

#     if not isinstance(callbacks_cfg, DictConfig):
#         raise TypeError("Callbacks config must be a DictConfig!")

#     for cb_conf in callbacks_cfg.values():
#         if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
#             log.info(f"Instantiating callback «{cb_conf._target_}».")
#             callbacks.append(hydra.utils.instantiate(cb_conf))

#     return callbacks


# def instantiate_loggers(logger_cfg: DictConfig) -> list[LightningLoggerBase]:
#     """Instantiates loggers from config."""
#     logger: list[LightningLoggerBase] = []

#     if not logger_cfg:
#         log.warning("Logger config is empty.")
#         return logger

#     if not isinstance(logger_cfg, DictConfig):
#         raise TypeError("Logger config must be a DictConfig!")

#     for lg_conf in logger_cfg.values():
#         if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
#             log.info(f"Instantiating logger «{lg_conf._target_}».")
#             logger.append(hydra.utils.instantiate(lg_conf))

#     return logger


# @rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    try:
        if not trainer.logger:
            log.warning("Logger not found! Skipping hyperparameter logging.")
            return
    except AttributeError:
        log.warning("Trainer has no attribute 'logger'! Skipping hyperparameter logging.")
        return

    # log experiment metadata if present
    if cfg.get("experiment_metadata"):
        for metadata in cfg.get("experiment_metadata"):
            hparams[metadata] = cfg.get("experiment_metadata")[metadata]

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    hparams["batch_size"] = cfg["datamodule"].batch_size
    hparams["initial_lr"] = cfg["model"].lr

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieve a logged metric value by name (used for hyperparameter-search optimization).

    Args:
        metric_dict: Mapping of metric name to a value exposing `.item()`.
        metric_name: The metric to retrieve; a falsy value short-circuits to ``None``.

    Returns:
        The metric's scalar value, or ``None`` if ``metric_name`` is falsy.

    Raises:
        Exception: If ``metric_name`` is truthy but absent from ``metric_dict``.
    """

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval.")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric name not found! «metric_name={metric_name}»\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! «{metric_name}={metric_value}».")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers.")

    if find_spec("wandb"):  # if wandb is installed
        import wandb  # type: ignore

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
