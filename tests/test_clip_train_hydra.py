"""
(c) Inria

Tests for the Hydra front-end to the contrastive CLIP pretraining path.

The path itself delegates to upstream ``open_clip_train``, which is argparse-only. These
tests pin the translation from the Hydra config into upstream's CLI arguments, and — most
importantly — that a run which asks for metric logging actually gets it. Upstream derives
its sinks from ``--report-to`` (default ``''``), so omitting the flag silently disables all
logging; that is exactly the state ``scripts/train_clip.sh`` was in.
"""

import logging

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from planktonzilla.clip_train.hydra_main import (
    CONFIG_DIR,
    CONFIG_NAME,
    build_argv,
    render_argv,
    render_tracking_argv,
    validate_argv,
)

SHARDS = "/data/train/shard_{00000..01771}.tar"


def _compose(overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR, job_name="test_train_clip"):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


def _flag_value(argv, flag):
    """The value following `flag`, or None when the flag is absent or is a bare switch."""
    if flag not in argv:
        return None
    index = argv.index(flag) + 1
    if index >= len(argv) or argv[index].startswith("--"):
        return None
    return argv[index]


# --------------------------------------------------------------------------------------
# argv rendering
# --------------------------------------------------------------------------------------


def test_keys_become_kebab_case_flags_and_bools_become_bare_switches():
    params = OmegaConf.create(
        {
            "model": "EVA02-L-14",
            "batch_size": 256,
            "local_loss": True,
            "grad_checkpointing": False,
            "val_data": None,
        }
    )

    argv = render_argv(params)

    assert argv == ["--model", "EVA02-L-14", "--batch-size", "256", "--local-loss"]
    # False and None omit the flag entirely, leaving upstream's own default in force.
    assert "--grad-checkpointing" not in argv
    assert "--val-data" not in argv


def test_scientific_notation_survives_to_a_float():
    """`lr: 1e-4` must reach upstream's `type=float` as a parseable string."""
    argv = render_argv(OmegaConf.create({"lr": 1e-4}))
    assert argv == ["--lr", "0.0001"]


def test_missing_mandatory_values_are_named_together():
    params = OmegaConf.create({"train_data": "???", "model": "EVA02-L-14"})
    with pytest.raises(ValueError, match="train_data"):
        render_argv(params)


def test_report_to_cannot_be_set_on_the_params_group():
    """One switch for metric logging, in the `tracking` group."""
    with pytest.raises(ValueError, match="tracking"):
        render_argv(OmegaConf.create({"report_to": "wandb"}))


# --------------------------------------------------------------------------------------
# tracking translation
# --------------------------------------------------------------------------------------


def test_wandb_is_translated_to_report_to_and_project():
    argv = render_tracking_argv(OmegaConf.create({"use_wandb": True, "wandb_project": "planktonzilla-turbo"}))

    assert _flag_value(argv, "--report-to") == "wandb"
    assert _flag_value(argv, "--wandb-project-name") == "planktonzilla-turbo"


def test_both_backends_are_joined_into_one_report_to():
    argv = render_tracking_argv(OmegaConf.create({"use_wandb": True, "use_tensorboard": True}))
    assert _flag_value(argv, "--report-to") == "wandb,tensorboard"


def test_tensorboard_alone_does_not_ask_for_a_wandb_project():
    argv = render_tracking_argv(OmegaConf.create({"use_tensorboard": True, "wandb_project": "unused"}))

    assert _flag_value(argv, "--report-to") == "tensorboard"
    assert "--wandb-project-name" not in argv


def test_no_backend_enabled_warns_that_nothing_will_be_logged(caplog):
    """Silence here is the bug: upstream defaults --report-to to '' and logs nothing."""
    with caplog.at_level(logging.WARNING):
        argv = render_tracking_argv(OmegaConf.create({}))

    assert "--report-to" not in argv
    assert any("no metrics will be logged" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("key", "backend"),
    [("use_mlflow", "mlflow"), ("use_trackio", "trackio")],
)
def test_backends_upstream_cannot_reach_are_reported_not_dropped(key, backend, caplog):
    """The `tracking` group offers three backends; upstream has a sink for one of them."""
    with caplog.at_level(logging.WARNING):
        argv = render_tracking_argv(OmegaConf.create({key: True}))

    messages = [record.message for record in caplog.records]
    assert "--report-to" not in argv
    assert any(backend in message for message in messages), (
        f"asking for {backend} must warn, not silently do nothing; got {messages}"
    )


# --------------------------------------------------------------------------------------
# End to end against the real upstream parser
# --------------------------------------------------------------------------------------


def test_composed_config_renders_argv_upstream_accepts():
    cfg = _compose([f"clip_pretrain.train_data='{SHARDS}'", "tracking.use_wandb=true"])
    argv = build_argv(cfg)

    validate_argv(argv)  # raises / SystemExit if upstream rejects any flag

    from open_clip_train.params import parse_args

    args = parse_args(argv)
    assert args.train_data == SHARDS, "the webdataset brace pattern must survive Hydra and rendering"
    assert args.report_to == "wandb"
    assert args.model == "EVA02-L-14"
    assert args.lr == pytest.approx(1e-4)
    assert args.local_loss is True
    assert args.logs, "--logs should be taken from paths.log_dir"


def test_upstream_flag_derivation_makes_report_to_the_switch():
    """Pins the upstream contract this whole translation exists to satisfy."""
    import inspect

    import open_clip_train.main as upstream

    source = inspect.getsource(upstream.main)
    assert "args.wandb = 'wandb' in args.report_to" in source, (
        "upstream changed how it derives its logging sinks; render_tracking_argv needs revisiting"
    )


def test_an_unknown_parameter_key_fails_before_launch():
    """A typo must not survive until a 16-node job has been scheduled."""
    argv = render_argv(OmegaConf.create({"not_a_real_upstream_flag": 1}))
    with pytest.raises(SystemExit):
        validate_argv(argv)


def test_default_config_still_requires_the_shard_pattern():
    cfg = _compose()
    with pytest.raises(ValueError, match="mandatory"):
        build_argv(cfg)


def test_main_delegates_through_clip_train_main_so_the_patches_still_apply(monkeypatch):
    """The front-end must not call upstream directly.

    `planktonzilla.clip_train.main.main` is what runs `_patch_upstream()`, which installs
    our classification-metric `evaluate` and the trivial-augment `image_transform`. Calling
    `open_clip_train.main.main` straight from here would render the same argv but silently
    drop both.
    """
    from planktonzilla.clip_train import hydra_main

    seen = {}

    def fake_clip_main(argv):
        seen["argv"] = argv

    monkeypatch.setattr("planktonzilla.clip_train.main.main", fake_clip_main)
    hydra_main.main([f"clip_pretrain.train_data='{SHARDS}'", "tracking.use_wandb=true"])

    assert "argv" in seen, "hydra_main.main must delegate to planktonzilla.clip_train.main.main"
    assert "--report-to" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--train-data") + 1] == SHARDS
