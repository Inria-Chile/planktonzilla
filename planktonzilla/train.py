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

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import os
from functools import partial

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
from planktonzilla.utils.hydra import (
    get_metric_value,
    task_wrapper,
)
from planktonzilla.utils.logger import get_pylogger

log = get_pylogger(__name__)

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


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


def compute_metrics(eval_pred):
    """
    requires training_args.eval_do_concat_batches = True
    """

    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


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

    if cfg.get("peft"):
        log.info("Adding LoRA adapter(s).")
        for adapter_name in cfg.peft:
            adapter = hydra.utils.instantiate(cfg.peft[adapter_name])
            model.add_adapter(adapter, adapter_name=adapter_name)
            log.info(f"Added LoRA adapter «{adapter_name}»: {cfg.peft[adapter_name]}.")

    # freeze backbone
    if cfg.freeze_backbone:
        log.info("Model backbone will not be trained.")
        for name, param in model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    log.info("Instantiating training arguments.")
    training_args: TrainingArguments = hydra.utils.instantiate(cfg.training_arguments, _convert_="all")

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
        custom_loss = partial(loss_instance.forward)
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
    training_args.report_to = report_to if report_to else "none"
    training_args.run_name = model.name_or_path.replace("/", "_") + "__" + cfg.dataset.name.replace("/", "_")

    log.info("Instantiating trainer.")
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_wrapper.dataset["train"],
        eval_dataset=dataset_wrapper.dataset[dataset_wrapper.val_split_name],
        compute_metrics=compute_metrics,
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

    if training_args.do_eval:
        log.info("Evaluating on test set.")
        test_metrics = trainer.evaluate(dataset_wrapper.dataset[dataset_wrapper.test_split_name], metric_key_prefix="test")
    else:
        log.info("Evaluation skipped as per training arguments, set training_arguments.do_eval=true to change this.")

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
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    main()
