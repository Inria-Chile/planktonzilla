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


# --------------------------------------------------------------------------------------
# The csv corpus path — a smoke-testable alternative to terabytes of webdataset shards
# --------------------------------------------------------------------------------------
#
# The shipped `train_data` is a webdataset brace pattern over shards nobody has locally, so
# "does the contrastive path still consume data" has no cheap answer. Upstream's
# `--dataset-type synthetic` only half-covers that: it reuses ONE blank image for every
# sample, so it never opens a file, never decodes an image, and its loss is pinned at exactly
# ln(batch_size) — chance, because every pair is identical. It proves the wiring and nothing
# about the data path.
#
# So these build a real corpus, in-process, using the column names upstream's csv loader
# already defaults to, and push it through upstream's own `CsvDataset`. Asserting against a
# hand-rolled reader would prove nothing — it would pass while the real loader choked, which
# is the exact class of interface defect this file exists to catch.

TOY_IMAGE_SIZE = 96
TOY_TAXA = ("copepod", "diatom chain", "radiolarian", "appendicularian")
TOY_COLOURS = ((200, 80, 80), (80, 160, 200), (120, 200, 120), (200, 180, 90))


def _csv_defaults():
    """Upstream's own --csv-* defaults, read from its parser rather than restated here."""
    from open_clip_train.params import parse_args

    args = parse_args(["--train-data", "x", "--dataset-type", "csv"])
    return args.csv_img_key, args.csv_caption_key, args.csv_separator


def _build_toy_corpus(destination, count=8):
    """Write `count` image/caption pairs and return the index path.

    Deliberately not blank or random: alternating shapes and cycling colours paired with
    matching captions mean the contrastive objective has something to latch onto, so a run
    over this can answer "can the loss move at all", not only "did the process survive".
    Paths are absolute so a run launched from any directory resolves them.
    """
    from PIL import Image, ImageDraw

    img_key, caption_key, separator = _csv_defaults()
    images = destination / "images"
    images.mkdir(parents=True, exist_ok=True)

    rows = [f"{img_key}{separator}{caption_key}"]
    for i in range(count):
        image = Image.new("RGB", (TOY_IMAGE_SIZE, TOY_IMAGE_SIZE), (20, 24, 32))
        draw = ImageDraw.Draw(image)
        inset = 12 + (i % 3) * 6
        box = [inset, inset, TOY_IMAGE_SIZE - inset, TOY_IMAGE_SIZE - inset]
        colour = TOY_COLOURS[i % len(TOY_COLOURS)]
        (draw.ellipse if i % 2 else draw.rectangle)(box, fill=colour)

        path = (images / f"{i:04d}.png").resolve()
        image.save(path)
        rows.append(f"{path}{separator}a microscopy image of a {TOY_TAXA[i % len(TOY_TAXA)]}")

    index = destination / "train.tsv"
    index.write_text("\n".join(rows) + "\n")
    return index


def test_upstream_csv_loader_reads_a_toy_corpus_with_its_own_defaults(tmp_path):
    """Load through the real `CsvDataset`: __getitem__ opens the file and runs the transform.

    Fails on a bad path, an undecodable image, or a column name upstream does not recognise —
    so it pins the whole no-flags-needed contract, not just the file's shape.
    """
    from open_clip_train.data import CsvDataset

    index = _build_toy_corpus(tmp_path / "toy", count=8)
    img_key, caption_key, separator = _csv_defaults()

    dataset = CsvDataset(
        input_filename=str(index),
        transforms=lambda image: image.convert("RGB"),
        img_key=img_key,
        caption_key=caption_key,
        sep=separator,
        tokenizer=lambda texts: texts,
    )

    assert len(dataset) == 8
    seen = set()
    for i in range(len(dataset)):
        image, texts = dataset[i]
        assert image.size == (TOY_IMAGE_SIZE, TOY_IMAGE_SIZE)
        assert texts[0].strip()
        seen.add(image.tobytes())

    assert len(seen) > 1, "every image decoded identically — the corpus cannot drive a loss"


def test_a_csv_corpus_renders_argv_upstream_accepts(tmp_path):
    """The other half: the Hydra front-end must render a csv run upstream really parses.

    `dataset_type=csv` is the smoke-test route into this path, and it goes through the same
    render-and-validate seam as a 64-rank webdataset job.
    """
    from open_clip_train.params import parse_args

    index = _build_toy_corpus(tmp_path / "toy", count=4)
    cfg = _compose([f"clip_pretrain.train_data={index}", "clip_pretrain.dataset_type=csv"])
    argv = build_argv(cfg)

    validate_argv(argv)  # raises / SystemExit if upstream rejects any flag

    args = parse_args(argv)
    assert args.dataset_type == "csv"
    assert args.train_data == str(index)


def test_forcing_an_upstream_flag_absent_from_the_group_needs_the_add_prefix():
    """`+clip_pretrain.force_image_size=64` — the `+` is not optional, and it is easy to miss.

    The group composes in struct mode, so overriding a key it does not define raises rather
    than adding it. Pinning this keeps the documented smoke-test command honest.
    """
    with pytest.raises(Exception, match="force_image_size"):
        _compose([f"clip_pretrain.train_data='{SHARDS}'", "clip_pretrain.force_image_size=64"])

    cfg = _compose([f"clip_pretrain.train_data='{SHARDS}'", "+clip_pretrain.force_image_size=64"])
    argv = build_argv(cfg)
    validate_argv(argv)
    assert _flag_value(argv, "--force-image-size") == "64"
