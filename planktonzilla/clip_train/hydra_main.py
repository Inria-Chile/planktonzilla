"""
(c) Inria

Hydra front-end for the contrastive CLIP pretraining path.

``planktonzilla.clip_train.main`` delegates to upstream ``open_clip_train.main``, which is
argparse-only — so the ``configs/`` tree, and in particular the ``tracking`` group that
``pz_train`` honours, never reached it. This module closes that gap: it composes the Hydra
config, renders it into the upstream argv list, validates that list against upstream's own
parser, and then calls ``clip_train.main.main(argv)``. ``main()`` already accepts
``args: list[str] | None``, so no upstream change is needed — that parameter is the seam.

Deliberately NOT a ``@hydra.main`` entry point. This path runs under
``torchrun --nproc_per_node=4 --nnodes=16``, i.e. 64 processes, and
``configs/hydra/default.yaml`` builds its run directory from ``${now:...}`` — every rank
would evaluate that independently and create its own timestamped output directory. Using
``initialize_config_dir`` + ``compose`` gives the same configuration with no output-directory
creation and no working-directory change, which is what a distributed launcher needs.
Overrides are still taken from the command line in the usual Hydra ``key=value`` form.

Usage::

    torchrun --nproc_per_node=4 -m planktonzilla.clip_train.hydra_main \\
        "clip_pretrain.train_data='/path/shards/train/shard_{00000..01771}.tar'" \\
        tracking.use_wandb=true

Note the *inner* single quotes around the shard pattern. Hydra's override grammar rejects a
bare ``{`` and ``..``, which every webdataset shard pattern contains, so the value has to
reach Hydra already quoted — the outer quotes are consumed by the shell.
"""

from __future__ import annotations

import sys

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from planktonzilla.utils.logger import get_pylogger

log = get_pylogger(__name__)

CONFIG_DIR = str(root / "configs")
CONFIG_NAME = "train_clip.yaml"

# Backends the Hydra `tracking` group offers that upstream open_clip_train cannot reach.
# Upstream's only sink is `--report-to`, which accepts 'wandb' and 'tensorboard'.
UNSUPPORTED_BACKENDS = {
    "use_mlflow": "mlflow",
    "use_trackio": "trackio",
}


def render_argv(params: DictConfig) -> list[str]:
    """Render a config section into upstream ``open_clip_train`` CLI arguments.

    Each key becomes its ``--kebab-case`` flag. Values are rendered by type:

    - ``True`` emits the bare flag, matching upstream's ``store_true`` arguments;
    - ``False`` and ``null`` omit the flag entirely, so the upstream default stands;
    - anything else emits ``--flag`` followed by ``str(value)``. Numbers written in YAML
      as ``1e-4`` reach argparse as the string ``"1e-4"``, which its ``type=float``
      parses correctly.

    Raises:
        ValueError: if a mandatory key (``???``) was never given a value, naming every
            such key at once rather than failing on the first one; or if the section tries
            to set ``report_to``, which is owned by the `tracking` group so that there is
            exactly one place metric logging is configured.
    """
    if "report_to" in params:
        raise ValueError(
            "clip_pretrain.report_to is not accepted: metric logging is configured through the "
            "`tracking` group (tracking.use_wandb / tracking.use_tensorboard) so that pz_train and "
            "pz_train_clip share one switch. Remove it from the clip_pretrain config."
        )

    missing = [key for key in params if OmegaConf.is_missing(params, key)]
    if missing:
        raise ValueError(
            f"clip_pretrain has {len(missing)} mandatory value(s) still unset: {sorted(missing)}. "
            f"Supply them on the command line, e.g. clip_pretrain.{min(missing)}=..."
        )

    argv: list[str] = []
    for key, value in params.items():
        if value is None or value is False:
            continue
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def render_tracking_argv(tracking: DictConfig) -> list[str]:
    """Translate the Hydra `tracking` group into upstream's ``--report-to`` flags.

    Upstream derives its sinks as ``args.wandb = 'wandb' in args.report_to`` (and likewise
    for tensorboard) from a flag that defaults to the empty string — so omitting
    ``--report-to`` silently disables all metric logging, which is what
    ``scripts/train_clip.sh`` was doing.

    Backends the `tracking` group offers but upstream has no sink for are reported with a
    warning rather than dropped silently; a request to log to mlflow that quietly does
    nothing is worse than a noisy one.
    """
    argv: list[str] = []
    report_to = []

    if tracking.get("use_wandb", False):
        report_to.append("wandb")
    if tracking.get("use_tensorboard", False):
        report_to.append("tensorboard")

    for key, name in UNSUPPORTED_BACKENDS.items():
        if tracking.get(key, False):
            log.warning(
                f"⚠️ tracking.{key}=true, but upstream open_clip_train has no {name} sink — it "
                f"reports only to wandb and tensorboard. Metrics will NOT reach {name} on the "
                f"contrastive path. Use tracking.use_wandb or tracking.use_tensorboard instead."
            )

    if report_to:
        argv.extend(["--report-to", ",".join(report_to)])
        if "wandb" in report_to:
            project = tracking.get("wandb_project", None)
            if project:
                argv.extend(["--wandb-project-name", str(project)])
    else:
        log.warning(
            "⚠️ No tracking backend is enabled, so no metrics will be logged. Set "
            "tracking.use_wandb=true or tracking.use_tensorboard=true to record this run."
        )

    return argv


def build_argv(cfg: DictConfig) -> list[str]:
    """Assemble the full upstream argv from the composed config."""
    argv = render_argv(cfg.clip_pretrain)
    argv += render_tracking_argv(cfg.tracking)
    log_dir = cfg.get("paths", {}).get("log_dir", None)
    if log_dir:
        argv += ["--logs", str(log_dir)]
    return argv


def validate_argv(argv: list[str]) -> None:
    """Parse the rendered argv with upstream's own parser, so a bad key fails before launch.

    Cheaper than discovering a typo after a 16-node job has been scheduled, and it keeps
    this module honest about upstream's flag set without duplicating it.
    """
    from open_clip_train.params import parse_args

    parse_args(argv)


def main(overrides: list[str] | None = None) -> None:
    """Compose the Hydra config, render it to upstream argv, and launch pretraining."""
    overrides = list(sys.argv[1:]) if overrides is None else list(overrides)

    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR, job_name="train_clip"):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    argv = build_argv(cfg)
    validate_argv(argv)
    log.info(f"Launching open_clip_train with: {' '.join(argv)}")

    from planktonzilla.clip_train.main import main as clip_main

    clip_main(argv)


if __name__ == "__main__":
    main()
