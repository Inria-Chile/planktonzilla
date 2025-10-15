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

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import os
from functools import partial

import hydra
import numpy as np
import torch
from evaluate import combine, load
from huggingface_hub import DatasetCard
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
from rich import print as rich_print
from rich.markdown import Markdown
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments, set_seed

from planktonzilla.dataset import DatasetWrapper
from planktonzilla.utils.hydra import (
    get_metric_value,
    task_wrapper,
)
from planktonzilla.utils.logger import get_pylogger
from planktonzilla.utils.rich_utils import print_docstr_as_markdown

log = get_pylogger(__name__)

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass

def compute_metrics(eval_pred):
    """requires training_args.eval_do_concat_batches = True"""
    metrics = combine([load("f1"), load("precision"), load("recall")])
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    res = metrics.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    acc = load("accuracy").compute(predictions=predictions, references=eval_pred.label_ids)
    return {**res, **acc}

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

    if cfg.get("seed"):
        set_seed(cfg.seed, cfg.get("deterministic", False))

    # set proper matmul precision
    # hydra.utils.instantiate(cfg.torch_matmul_precision)

    log.info(f"Instantiating wrapper for dataset «{cfg.dataset.name}».")
    dataset_wrapper: DatasetWrapper = hydra.utils.instantiate(cfg.dataset)

    augmentation = hydra.utils.instantiate(cfg.augmentation)

    log.info(f"Preparing datasets in «{cfg.dataset.name}».")
    dataset_wrapper.prepare_datasets(augmentation)

    card = DatasetCard.load(cfg.dataset.name)

    if cfg.get("extras") and cfg.extras.get("print_config"):
        rich_print(Markdown(card.text))

    # wiring num_classes to model
    # cfg.model.num_classes = dataset_wrapper.num_classes

    log.info(f"Instantiating base model «{cfg.model._args_[0]}».")

    model: AutoModelForImageClassification = hydra.utils.instantiate(
        cfg.model,
        id2label=dataset_wrapper.id2label,
        label2id=dataset_wrapper.label2id,
        num_labels=len(dataset_wrapper.label2id),
        _convert_="all",
    )

    # freeze backbone
    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # TODO: lora setup shoule be here

    if cfg.get("extras") and cfg.extras.get("print_config"):
        print_docstr_as_markdown(model)

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
        # We are going to use a custom loss.
        try:
            if not cfg.custom_loss.get("cls_num_list"):
                loss_instance = hydra.utils.instantiate(cfg.custom_loss, _convert_="all")
        except MissingMandatoryValue:
            cfg.custom_loss["cls_num_list"] = "dummy_value"
            loss_instance = hydra.utils.instantiate(cfg.custom_loss, cls_num_list=dataset_wrapper.cls_num_list, _convert_="all")
        custom_loss = partial(loss_instance.forward)

    log.info("Instantiating trainer.")
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_wrapper.dataset["train"],
        eval_dataset=dataset_wrapper.dataset[dataset_wrapper.val_split_name],
        # data_collator=collate_fn,
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

        log.info(f"Logging metrics and/or models to: {report_to}.")
        training_args.report_to = report_to if report_to else "none"
        training_args.run_name = model.name_or_path.replace("/", "_") + "__" + cfg.dataset.name.replace("/", "_")

        log.info("Starting training.")
        train_results = trainer.train()
        train_metrics = train_results.metrics
        log.info("Done training, evaluating on validation set.")
        val_metrics = trainer.evaluate(dataset_wrapper.dataset[dataset_wrapper.val_split_name], metric_key_prefix="val")

        log.info("Starting evaluation on test set.")
        test_metrics = trainer.evaluate(dataset_wrapper.dataset[dataset_wrapper.test_split_name], metric_key_prefix="test")

        if cfg.model_push_to_hub:
            log.info(f"Pushing trained model to HuggingFace hub as «{training_args.hub_model_id}».")
            url = trainer.push_to_hub()
            log.info(f"Pushed model is available at: {url}.")

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
