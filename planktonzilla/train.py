"""
(c) Inria

Hydra entry point for plankton image-classification training and evaluation.

Composes the full experiment from the ``configs/`` tree, instantiates the dataset wrapper,
model (Hugging Face ``AutoModelForImageClassification`` or the CLIP-based
:class:`~planktonzilla.clip_model.ClipClassifier`), optional PEFT/LoRA adapters, and an
optional imbalance-aware loss, then drives the Hugging Face ``Trainer`` through training,
validation, test evaluation, and optional push-to-hub. ``main`` is the ``pz_train`` console
script; it returns the optimized metric so Hydra hyperparameter sweeps can read it.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import os

import hydra
import numpy as np
import torch
from huggingface_hub import DatasetCard, login
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments, set_seed

from planktonzilla.clip_model import ClipClassifier
from planktonzilla.dataset import DatasetWrapper
from planktonzilla.heads import replace_head_with_cosine
from planktonzilla.utils import resolvers as _resolvers  # noqa: F401  -- side-effect: registers strip_yaml_suffix
from planktonzilla.utils.hydra import (
    get_metric_value,
    task_wrapper,
)
from planktonzilla.utils.logger import get_pylogger

log = get_pylogger(__name__)


def validate_environment(cfg: DictConfig | None = None):
    """Check and log important external service environment variables.

    Warns when Hugging Face hub or tracking services are likely unavailable
    and logs presence of common environment variables such as `HF_TOKEN`,
    `WANDB_API_KEY` and `MLFLOW_TRACKING_URI`.

    Per ABSORB-02 / audit Q6: also applies the planktonzilla-specific
    `torch.backends.cuda.matmul.allow_tf32` setting when `cfg.tf32` is True.
    This reproduces the throughput characteristic the vendored
    `open_clip_train.main` had by default; opt-in to preserve strict
    reproducibility with pre-Phase-4 runs. `cfg` is optional for backward
    compat with call sites that don't have a Hydra config in scope.
    """
    if cfg is not None and bool(cfg.get("tf32", False)):
        import torch as _torch  # local import keeps top-level torch state untouched if disabled

        _torch.backends.cuda.matmul.allow_tf32 = True
        log.info("✅ TF32 matmul enabled (cfg.tf32=true; reproduces vendored open_clip_train default).")

    if "HF_HUB_OFFLINE" in os.environ and os.environ["HF_HUB_OFFLINE"] == "1":
        log.warning("⚠️ Environment variable HF_HUB_OFFLINE=1. Hugging Face hub will be offline.")
    else:
        if "HF_TOKEN" in os.environ:
            log.info("✅ HF_TOKEN environment variable is set.")
            try:
                login()
                log.info("✅ Login to Hugging Face hub verified.")
            except ValueError as e:
                log.error(f"🛑 Login to Hugging Face hub failed: {e}.")
            except ImportError:  # If running in a notebook but ipywidgets is not installed.
                log.error("🛑 Running in a notebook but ipywidgets is not installed.")
        else:
            log.warning("⚠️ HF_TOKEN environment variable is not set. Access to private models and datasets will be limited.")

    if "WANDB_MODE" in os.environ and os.environ["WANDB_MODE"] == "offline":
        log.warning("⚠️ Environment variable WANDB_MODE=offline. WandB will be offline. Remember to sync results later on.")
    elif "WANDB_API_KEY" in os.environ:
        log.info("✅ WANDB_API_KEY environment variable is set.")
    else:
        log.warning("⚠️ WANDB_API_KEY environment variable is not set. WandB logging will be disabled.")

    if "MLFLOW_TRACKING_URI" in os.environ:
        log.info("✅ MLFLOW_TRACKING_URI environment variable is set.")
        if "MLFLOW_TRACKING_USERNAME" in os.environ:
            log.info("✅ MLFLOW_TRACKING_USERNAME environment variable is set.")
        if "MLFLOW_TRACKING_PASSWORD" in os.environ:
            log.info("✅ MLFLOW_TRACKING_PASSWORD environment variable is set.")
    else:
        log.warning("⚠️ MLFLOW_TRACKING_URI environment variable is not set, if mlflow is enabled will log to local folder.")


# def compute_metrics(eval_pred):
#     """requires training_args.eval_do_concat_batches = True"""
#     metrics = combine([load("f1"), load("precision"), load("recall")])
#     predictions = np.argmax(eval_pred.predictions, axis=-1)
#     res = metrics.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
#     acc = load("accuracy").compute(predictions=predictions, references=eval_pred.label_ids)
#     return {**res, **acc}


def build_preprocess_logits_for_metrics(top_k: int = 5):
    """Reduce each eval batch's logits to its top-k class indices before they accumulate.

    `Trainer` concatenates whatever `prediction_step` returns across the whole evaluation
    set before handing it to `compute_metrics`, so without this hook it holds an
    ``(n_eval, n_classes)`` float array in memory — for a million validation images over a
    few thousand plankton classes that is gigabytes, and it grows with the class space
    every time a source is added. Only the ranking is needed downstream, so keeping the
    top-k indices bounds the accumulation at ``(n_eval, k)`` int64.

    This does not change any metric: the top-1 index is the same argmax `compute_metrics`
    used to take itself.
    """

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits.topk(min(top_k, logits.shape[-1]), dim=-1).indices

    return preprocess_logits_for_metrics


def shot_group_recall(labels, predictions, cls_num_list, few_shot_max: int, many_shot_min: int) -> dict:
    """Per-class recall averaged within each long-tail shot group.

    A single macro-F1 cannot distinguish "helped the tail, cost a little head accuracy"
    from noise, which is the comparison every imbalance-learning change in this repo is
    actually trying to make. Classes are bucketed by how many *training* images they have
    — the standard many/medium/few-shot split from the long-tail literature — and each
    bucket reports the mean of its per-class recalls.

    Classes absent from the evaluation split contribute nothing (they have no recall to
    measure), and a bucket with no such classes is omitted rather than reported as 0.0.
    """
    counts = np.asarray(cls_num_list)
    buckets: dict[str, list[float]] = {"many_shot": [], "medium_shot": [], "few_shot": []}

    for class_id in np.unique(labels):
        mask = labels == class_id
        recall = float((predictions[mask] == class_id).mean())
        n_train = int(counts[class_id]) if class_id < len(counts) else 0
        if n_train < few_shot_max:
            buckets["few_shot"].append(recall)
        elif n_train <= many_shot_min:
            buckets["medium_shot"].append(recall)
        else:
            buckets["many_shot"].append(recall)

    metrics = {}
    for name, recalls in buckets.items():
        metrics[f"n_classes_{name}"] = len(recalls)
        if recalls:
            metrics[f"recall_{name}"] = float(np.mean(recalls))
    return metrics


def build_compute_metrics(cls_num_list=None, top_k: int = 5, shot_thresholds: tuple[int, int] = (20, 100)):
    """Build the `Trainer`'s `compute_metrics`, optionally with long-tail shot groups.

    Always reports ``accuracy`` plus macro ``f1`` / ``precision`` / ``recall`` (macro recall
    is balanced accuracy, so it is not reported twice under another name). Adds
    ``top{k}_accuracy`` and, when `cls_num_list` is supplied, the per-shot-group recalls
    from :func:`shot_group_recall`.

    Accepts either the top-k index array produced by
    :func:`build_preprocess_logits_for_metrics` (integer dtype) or raw logits (floating
    dtype), so the metrics stay computable if that hook is ever unwired.

    Note:
        requires training_args.eval_do_concat_batches = True
    """
    few_shot_max, many_shot_min = shot_thresholds

    def compute_metrics(eval_pred):
        predictions = np.asarray(eval_pred.predictions)
        labels = np.asarray(eval_pred.label_ids)

        if np.issubdtype(predictions.dtype, np.floating):
            # Raw logits: rank them here instead. Same result, more memory.
            ranked = np.argsort(-predictions, axis=-1)[:, :top_k]
        else:
            ranked = predictions if predictions.ndim == 2 else predictions[:, None]

        top1 = ranked[:, 0]
        metrics = {
            "accuracy": accuracy_score(labels, top1),
            "f1": f1_score(labels, top1, average="macro", zero_division=0),
            "precision": precision_score(labels, top1, average="macro", zero_division=0),
            "recall": recall_score(labels, top1, average="macro", zero_division=0),
        }

        k = ranked.shape[1]
        if k > 1:
            metrics[f"top{k}_accuracy"] = float((ranked == labels[:, None]).any(axis=1).mean())

        if cls_num_list is not None:
            metrics.update(shot_group_recall(labels, top1, cls_num_list, few_shot_max, many_shot_min))

        return metrics

    return compute_metrics


# Module-level default for callers that just want the metric set with no long-tail groups.
compute_metrics = build_compute_metrics()


def resolve_precision(precision, training_args) -> str:
    """Set `training_args.bf16` / `.fp16` from a single `precision` knob, and say what won.

    The shipped default was `fp16: true`. Every loss in `planktonzilla.loss` is built from
    `log`, `exp` and `pow` — the operations where fp16's narrow exponent range produces inf
    and NaN — and bf16 keeps fp32's exponent range at the same memory cost, so it is the
    better choice wherever the hardware has it. But bf16 needs Ampere or newer, and this
    project also targets V100/T4 boxes, so flipping the default outright would break them.

    `auto` therefore picks bf16 when the GPU supports it, fp16 when it does not, and fp32
    when there is no GPU at all (mixed precision on CPU buys nothing). An explicit
    `bf16`/`fp16`/`fp32` forces that choice — `precision=fp16` reproduces the old default
    exactly — and `null` leaves whatever `training_arguments` set, for full manual control.

    Returns:
        The resolved mode, for logging.
    """
    if precision is None:
        return "unchanged"

    choice = str(precision).lower()
    if choice == "auto":
        if not torch.cuda.is_available():
            choice = "fp32"
        else:
            choice = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if choice not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"precision must be one of auto/bf16/fp16/fp32 or null, got {precision!r}")

    training_args.bf16 = choice == "bf16"
    training_args.fp16 = choice == "fp16"
    return choice


def warn_if_label_smoothing_is_ignored(cfg, training_args) -> bool:
    """Report that `TrainingArguments.label_smoothing_factor` cannot take effect here.

    `Trainer.compute_loss` reads ``if self.compute_loss_func is not None: ... elif labels is
    not None: <label smoothing>``, so the smoother is skipped outright whenever a custom loss
    is set — and this project always sets one, because `configs/custom_loss/default.yaml`
    selects `CrossEntropyLossHF`. Setting the `TrainingArguments` field would therefore be a
    silent no-op in *every* configuration.

    The working knob is the loss's own: `custom_loss.label_smoothing` on the cross-entropy
    family, or `custom_loss.eps` on ASL/RAL, which already smooth internally.

    Returns:
        True when a warning was emitted, so callers (and tests) can assert on it.
    """
    factor = getattr(training_args, "label_smoothing_factor", 0.0) or 0.0
    if factor > 0 and cfg.get("custom_loss"):
        log.warning(
            f"⚠️ training_arguments.label_smoothing_factor={factor} has NO effect: the HF Trainer skips its "
            f"label smoother whenever compute_loss_func is set, and a custom loss is always configured here. "
            f"Use custom_loss.label_smoothing instead (cross-entropy, LDAM, balanced-meta-softmax, "
            f"max-margin), or custom_loss.eps for the asymmetric losses, which smooth internally."
        )
        return True
    return False


def should_evaluate_test_split(cfg, training_args) -> bool:
    """Whether this run may read the TEST split. Defaults to False.

    The test split used to share `do_eval` with the validation pass, so every run — every
    hyperparameter sweep, every debugging iteration — read it. A split read on every iteration
    is not held out: the number it reports stops being an unbiased estimate of generalisation
    the moment anyone tunes against it.

    So `do_eval` still governs validation, which is what you tune on, and the test split needs
    `eval_test=true` asked for deliberately, once, on a final configuration. `eval_test` lives on
    `cfg` rather than in the `training_arguments` group precisely so it does not read as one of
    Hugging Face's own `do_*` flags — `TrainingArguments` has no such field.

    Returns:
        True only when evaluation is on *and* the test split was explicitly requested.
    """
    return bool(training_args.do_eval) and bool(cfg.get("eval_test", False))


def freeze_backbone_except_head(model) -> list[str]:
    """Freeze every parameter except the classification head's, and say which survived.

    The head is resolved by identity where the model exposes one (`ClipClassifier.head`),
    falling back to the name match that is correct for Hugging Face image-classification
    models, whose head is named `classifier`. The previous name-only version matched
    `"classifier" in name or "head" in name` against *every* model: on `ClipClassifier`'s
    ViT path the head is `nn.Sequential(visual, nn.Linear(...))`, so its parameters are
    named `1.weight` / `1.bias`, nothing matched, and `freeze_backbone=true` froze the
    freshly-initialised head along with the backbone — leaving no trainable parameter at
    all. `freeze_backbone` is set true in five shipped model configs.

    Raises:
        ValueError: if the selection leaves nothing trainable, which is never a usable
            training run and is exactly how the previous defect stayed silent.

    Returns:
        The names of the parameters left trainable.
    """
    head = getattr(model, "head", None)
    if isinstance(head, torch.nn.Module):
        keep = {id(param) for param in head.parameters()}
        selected = [name for name, param in model.named_parameters() if id(param) in keep]
    else:
        selected = [name for name, _ in model.named_parameters() if "classifier" in name or "head" in name]

    # Checked before anything is mutated, so a caller that handles the error is not left
    # holding a fully-frozen model.
    if not selected:
        raise ValueError(
            f"freeze_backbone=true left no trainable parameters on {type(model).__name__}: no "
            f"classification head could be identified. Expose a `head` module on the model, or "
            f"name the head's parameters so they contain 'classifier' or 'head'."
        )

    selected_set = set(selected)
    for name, param in model.named_parameters():
        param.requires_grad = name in selected_set

    return selected


def build_compute_loss_func(loss_module):
    """Adapt one of the `planktonzilla.loss` modules to the `Trainer.compute_loss_func` contract.

    Hugging Face hands a custom loss function `(outputs, labels, num_items_in_batch=...)`
    and, when one is set, *skips* its own gradient-accumulation normalisation —
    `Trainer.training_step` divides by `current_gradient_accumulation_steps` only
    `if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and
    self.compute_loss_func is None`. The loss is expected to normalise by
    `num_items_in_batch` itself, which counts items across the whole accumulation group.

    Every loss in `planktonzilla.loss` takes `**kwargs` and ignores `num_items_in_batch`,
    returning a mean over the micro-batch. With `gradient_accumulation_steps > 1` the
    micro-batch losses were therefore summed without being divided by the number of
    accumulation steps, inflating the gradient — and so the effective learning rate — by
    that factor, silently. The default config uses 1 step, so this only bites when
    someone raises it to fit a larger backbone.

    Rescaling `mean * micro_batch_size / num_items_in_batch` yields
    `sum / num_items_in_batch`, which is exactly the mean again when accumulation is 1,
    so single-step runs are bit-for-bit unchanged.
    """

    def compute_loss(outputs, labels, num_items_in_batch=None, **kwargs):
        loss = loss_module(outputs, labels)
        if labels is not None and num_items_in_batch is not None:
            total = num_items_in_batch
            if torch.is_tensor(total):
                total = total.to(loss.device, dtype=loss.dtype)
            if total > 0:
                loss = loss * labels.numel() / total
        return loss

    return compute_loss


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random

    validate_environment(cfg)

    if cfg.get("seed"):
        set_seed(cfg.seed, cfg.get("deterministic", False))

    # set proper matmul precision
    # hydra.utils.instantiate(cfg.torch_matmul_precision)

    log.info(f"Instantiating dataset wrapper for «{cfg.dataset.name}».")
    dataset_wrapper: DatasetWrapper = hydra.utils.instantiate(cfg.dataset)

    log.info("Instantiating data augmentation(s).")
    augmentation = hydra.utils.instantiate(cfg.augmentation)

    log.info(f"Preparing data splits for «{cfg.dataset.name}».")
    dataset_wrapper.prepare_datasets(augmentation)

    dataset_card = DatasetCard.load(cfg.dataset.name)
    log.info(
        f"Dataset «{cfg.dataset.name}» {dataset_card.data.dataset_info.get('dataset_name', '')} <https://huggingface.co/datasets/{cfg.dataset.name}>."
    )

    log.info(f"Instantiating base model «{cfg.model._args_[0]}».")

    # FIX-02: explicit cfg.model.type dispatch replaces the broad `except Exception:`
    # pattern that silently misrouted real dispatch failures into the CLIP branch.
    # Pop the dispatch field before instantiate() (Hydra would pass it as a kwarg
    # to the model constructor otherwise — neither HF nor ClipClassifier accept `type=`).
    OmegaConf.set_struct(cfg.model, False)
    model_type = cfg.model.pop("type")
    OmegaConf.set_struct(cfg.model, True)

    if model_type == "hf":
        model: AutoModelForImageClassification = hydra.utils.instantiate(
            cfg.model,
            id2label=dataset_wrapper.id2label,
            label2id=dataset_wrapper.label2id,
            num_labels=len(dataset_wrapper.label2id),
            _convert_="all",
        )
    elif model_type == "clip":
        model: ClipClassifier = hydra.utils.instantiate(
            cfg.model,
            num_features=cfg.num_features,
            id2label=dataset_wrapper.id2label,
            label2id=dataset_wrapper.label2id,
            num_labels=len(dataset_wrapper.label2id),
            _convert_="all",
        )
    else:
        raise ValueError(
            f"Unknown cfg.model.type: {model_type!r} (expected 'hf' or 'clip'). "
            f"Add the `type:` field to your model config; see configs/model/default.yaml "
            f"or configs/model/default_clip.yaml for examples."
        )

    head_type = str(cfg.get("head_type", "linear")).lower()
    if head_type == "cosine":
        head = replace_head_with_cosine(
            model,
            scale=cfg.get("head_scale", None),
            learnable_scale=bool(cfg.get("head_scale_learnable", True)),
        )
        log.info(f"Replaced the linear classification head with a cosine head: {head}.")
    elif head_type != "linear":
        raise ValueError(f"head_type must be 'linear' or 'cosine', got {head_type!r}")

    if cfg.get("peft"):
        log.info("Adding LoRA adapter(s).")
        for adapter_name in cfg.peft:
            adapter = hydra.utils.instantiate(cfg.peft[adapter_name])
            model.add_adapter(adapter, adapter_name=adapter_name)
            log.info(f"Added LoRA adapter «{adapter_name}»: {cfg.peft[adapter_name]}.")

    # freeze backbone
    if cfg.freeze_backbone:
        log.info("Model backbone will not be trained.")
        trainable = freeze_backbone_except_head(model)
        log.info(f"Trainable head parameters: {trainable}.")

    log.info("Instantiating training arguments.")
    training_args: TrainingArguments = hydra.utils.instantiate(cfg.training_arguments, _convert_="all")

    resolved_precision = resolve_precision(cfg.get("precision", None), training_args)
    log.info(f"Mixed precision: {resolved_precision} (bf16={training_args.bf16}, fp16={training_args.fp16}).")
    warn_if_label_smoothing_is_ignored(cfg, training_args)

    if cfg.model_push_to_hub:
        training_args.push_to_hub = False
        training_args.hub_model_id = (
            cfg.model_push_to_hub_org_name
            + "/"
            + cfg.model_push_to_hub_repo_name_prefix
            + "_"
            + model.name_or_path.replace("/", "_")
            + "_"
            + cfg.dataset.name.replace("/", "_")
        )
        # training_args.hub_token = cfg.hf_token
        training_args.hub_private_repo = cfg.model_push_as_private
    else:
        training_args.push_to_hub = False

    if cfg.get("resume_from_ckpt_path"):
        training_args.resume_from_checkpoint = cfg.resume_from_ckpt_path

    # Loss function
    custom_loss = None
    if cfg.custom_loss:
        cfg_loss = cfg.custom_loss.get("custom_loss", cfg.custom_loss)
        log.info(f"Instantiating custom loss function «{cfg_loss._target_}».")
        try:
            loss_instance = hydra.utils.instantiate(cfg_loss, _convert_="all")
        except Exception:
            loss_instance = hydra.utils.instantiate(cfg_loss, cls_num_list=dataset_wrapper.cls_num_list, _convert_="all")
        custom_loss = build_compute_loss_func(loss_instance)
    else:
        log.info("Using default loss function.")

    report_to = []
    # setting up wandb for logging
    if cfg.tracking.get("use_wandb", False):
        report_to += ["wandb"]
        os.environ["WANDB_PROJECT"] = cfg.tracking.wandb_project
        os.environ["WANDB_ENTITY"] = cfg.tracking.wandb_entity
        os.environ["WANDB_LOG_MODEL"] = cfg.tracking.wandb_log_model
        os.environ["WANDB_WATCH"] = cfg.tracking.wandb_watch
        os.environ["WANDB_DIR"] = cfg.tracking.wandb_dir

    if cfg.tracking.get("use_mlflow", False):
        report_to += ["mlflow"]
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = str(cfg.tracking.mlflow_log_artifacts).upper()
        os.environ["MLFLOW_TRACKING_URI"] = cfg.tracking.mlflow_tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.tracking.mlflow_experiment_name
        os.environ["MLFLOW_TAGS"] = str(cfg.tracking.get("mlflow_tags", ""))

    if cfg.tracking.get("use_trackio", False):
        report_to += ["trackio"]
        os.environ["TRACKIO_DIR"] = cfg.tracking.trackio_dir
        os.environ["TRACKIO_DATASET_ID"] = cfg.tracking.trackio_dataset_id

    log.info(f"Logging metrics and/or models to: {report_to}.")
    training_args.report_to = report_to or "none"
    training_args.run_name = model.name_or_path.replace("/", "_") + "__" + cfg.dataset.name.replace("/", "_")

    log.info("Instantiating trainer.")
    top_k = int(cfg.get("eval_top_k", 5))
    shot_thresholds = tuple(cfg.get("eval_shot_thresholds", [20, 100]))
    log.info(
        f"Evaluating with top-{top_k} accuracy and shot groups "
        f"few<{shot_thresholds[0]} <=medium<= {shot_thresholds[1]}<many training images per class."
    )
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_wrapper.dataset["train"],
        eval_dataset=dataset_wrapper.dataset[dataset_wrapper.val_split_name],
        compute_metrics=build_compute_metrics(
            cls_num_list=dataset_wrapper.cls_num_list,
            top_k=top_k,
            shot_thresholds=shot_thresholds,
        ),
        preprocess_logits_for_metrics=build_preprocess_logits_for_metrics(top_k),
        compute_loss_func=custom_loss,
    )

    object_dict = {
        "cfg": cfg,
        "data_wrapper": dataset_wrapper,
        "model": model,
        "trainer": trainer,
    }

    train_metrics = {}
    val_metrics = {}
    test_metrics = {}

    if training_args.do_train:
        log.info("Training start.")
        train_results = trainer.train()
        train_metrics = train_results.metrics
        log.info("Done training.")
        if training_args.do_eval:
            log.info("Evaluating on validation set.")
            val_metrics = trainer.evaluate(dataset_wrapper.dataset[dataset_wrapper.val_split_name], metric_key_prefix="val")
    else:
        log.info("Training skipped as per training arguments, set training_arguments.do_train=true to change this.")

    # Validation is gated by do_eval; the TEST split needs eval_test as well, and defaults to
    # off. See should_evaluate_test_split for why.
    if not training_args.do_eval:
        log.info("Evaluation skipped as per training arguments, set training_arguments.do_eval=true to change this.")
    elif should_evaluate_test_split(cfg, training_args):
        log.warning(
            "⚠️ Evaluating on the TEST set. Do this once, on a final configuration — tuning against "
            "these numbers is what makes them stop meaning anything. Use the validation split to iterate."
        )
        test_metrics = trainer.evaluate(dataset_wrapper.dataset[dataset_wrapper.test_split_name], metric_key_prefix="test")
    else:
        log.info("Test-set evaluation skipped (eval_test=false). Set eval_test=true for a final run.")

    if cfg.model_push_to_hub:
        log.info(f"Pushing trained model to HuggingFace hub as «{training_args.hub_model_id}».")
        url = trainer.push_to_hub(dataset=dataset_wrapper.name, license="mit")
        log.info(f"Pushed model is available at: {url}.")
    else:
        log.info("Model push to HuggingFace hub skipped, set model_push_to_hub=true to change this.")

    # merge train and test metrics
    metric_dict = {**train_metrics, **val_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """``pz_train`` entry point: run training/eval and return the optimized metric.

    Hydra composes ``cfg`` from ``configs/train.yaml`` (plus CLI overrides), then this delegates
    to :func:`train`. Returns the value of ``cfg.optimized_metric`` (or ``None`` if unset) so
    Hydra hyperparameter-optimization sweeps have a scalar objective to optimize.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        float | None: The selected optimized metric value, or ``None`` when no metric is configured.
    """
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    main()  # type: ignore
