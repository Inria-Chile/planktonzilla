"""
(c) Inria

Dataset loading, splitting and transform attachment for the training pipeline.

Centers on `DatasetWrapper`, a stateful wrapper over a Hugging Face `DatasetDict`
that loads a dataset from the Hub, derives missing validation/test splits, and
attaches per-batch augmentation/transform pipelines used by the HF `Trainer`.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch
from datasets import Dataset, load_dataset

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def augment_and_transform_batch(examples, transform, augmentation, input_column_name, label_column_name):
    """Apply the base transform (and optional augmentation) to a batch of examples.

    Each input image is converted to RGB, passed through `transform`, then through
    `augmentation` when one is supplied (training) and skipped otherwise (eval/predict).
    The resulting per-image tensors are stacked into a single `pixel_values` tensor.

    Args:
        examples: Batch dict mapping column names to lists, as delivered by
            `datasets.Dataset.with_transform`.
        transform: Callable applied to each RGB PIL image (e.g. resize/normalize).
        augmentation: Optional callable applied after `transform`; pass `None` to skip
            (used for the validation/test pipelines).
        input_column_name: Column holding the PIL images.
        label_column_name: Column holding the integer labels.

    Returns:
        dict: `{"pixel_values": stacked tensor, label_column_name: list of labels}`.
    """

    images = []
    annotations = []
    for image, label in zip(examples[input_column_name], examples[label_column_name], strict=True):
        # res = transform(images=[np.array(image.convert("RGB"))], category=[label])
        # images += res["images"]
        # annotations += res["category"]
        res = transform(image.convert("RGB"))
        res = augmentation(res) if augmentation else res
        images += [res]
        annotations += [label]

    # Apply the image processor transformations: resizing, rescaling, normalization
    # results = image_processor(images=images, return_tensors="pt")
    # results["label"] = annotations

    images = torch.stack(images)
    results = {"pixel_values": images, label_column_name: annotations}
    return results


def compute_mean_and_std_dev(huggingface_dataset: Dataset, input_column_name: str = "image"):
    """Compute per-channel mean and standard deviation for a dataset.

    Iterates over a Hugging Face `Dataset` of images and returns the mean and
    standard deviation for each channel. Returns lists sized according to the
    image channels (3 for RGB, 1 for grayscale).

    Args:
        huggingface_dataset (Dataset): Iterable Hugging Face dataset yielding
            dicts with an `input_column_name` PIL object.
        input_column_name (str): Name of the column containing the images. Default is "image".

    Returns:
        tuple: (mean, std_dev) where each is a sequence of floats per channel.
    """
    sum_pixels = np.zeros(3)  # For R, G, B channels
    sum_squared_pixels = np.zeros(3)
    num_pixels = 0

    for item in huggingface_dataset:
        # Access the image (assuming it's a PIL Image object)
        image = item[input_column_name]

        # Convert image to NumPy array and normalize to [0, 1] if needed
        image_array = np.array(image).astype(np.float32) / 255.0

        # Reshape the image to (height * width, channels) to easily work with pixels
        if len(image_array.shape) == 3:
            # it is a color image with three channels
            reshaped_image = image_array.reshape(-1, 3)
        elif len(image_array.shape) == 2:
            # monochrome image with one channel
            reshaped_image = image_array.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported image_array shape: {image_array.shape}")

        # Accumulate sums
        sum_pixels += np.sum(reshaped_image, axis=0)
        sum_squared_pixels += np.sum(reshaped_image**2, axis=0)

        # Update total number of pixels
        num_pixels += reshaped_image.shape[0]

    mean = sum_pixels / num_pixels
    std_dev = np.sqrt((sum_squared_pixels / num_pixels) - (mean**2))

    if len(image_array.shape) == 3:
        # it is a color image with three channels
        return mean, std_dev
    elif len(image_array.shape) == 2:
        # monochrome image with one channel
        return [mean[0]], [std_dev[0]]


@dataclass
class DatasetWrapper:
    """Stateful wrapper around a Hugging Face dataset for the training pipeline.

    Owns the load → split → transform lifecycle: it loads the dataset named by
    `name` from the Hub, derives any missing validation/test splits, computes the
    label mappings and per-class counts, and attaches the augmentation/transform
    pipelines consumed by the HF `Trainer`.

    The dataclass fields are the configuration knobs (split names, ratios, seed,
    transform, etc.). State produced at runtime — `dataset`, `id2label`,
    `label2id`, `num_classes`, and `cls_num_list` — is initialized in
    `__post_init__` and populated by `prepare_datasets`; the `*_dataset`
    properties are only valid once `prepare_datasets` has run.
    """

    name: str

    input_column_name: str = "image"
    label_column_name: str = "label"

    streaming: bool = False

    split_seed: int = 42
    shuffle: bool = True

    val_split: float = None
    test_split: float = None

    val_split_name: str = None
    test_split_name: str = None

    transform: Callable = None

    @property
    def training_dataset(self):
        """The transform-attached `train` split (requires `prepare_datasets` to have run)."""
        return self.dataset["train"]

    @property
    def validation_dataset(self):
        """The transform-attached validation split named by `val_split_name`."""
        return self.dataset[self.val_split_name]

    @property
    def test_dataset(self):
        """The transform-attached test split named by `test_split_name`."""
        return self.dataset[self.test_split_name]

    def __post_init__(self):
        """Initialize runtime state to its empty/sentinel values.

        Sets `dataset` to `None`, `id2label`/`label2id` to `None`, and
        `num_classes` to `-1`; these are populated later by `prepare_datasets`.
        """
        super().__init__()
        self.dataset = None
        self.id2label = self.label2id = None
        self.num_classes = -1

    def prepare_datasets(self, augmentation) -> None:
        """Load the dataset, derive missing splits and attach transform pipelines.

        Loads the dataset identified by `self.name` via `datasets.load_dataset`,
        then, when the test and/or validation splits are absent, carves them out
        of `train` with a stratified `train_test_split` (seeded by `split_seed`,
        shuffled per `shuffle`). The training split receives a transform that
        applies `augmentation`; the validation/test splits receive an
        augmentation-free transform.

        This method mutates `self` in place and is the only place the runtime
        state is populated. Side effects:

        - `self.dataset`: the loaded `DatasetDict`, with derived splits added and
            `with_transform` callables attached to each split.
        - `self.id2label` / `self.label2id`: label-id ↔ name mappings.
        - `self.num_classes`: number of distinct labels.
        - `self.cls_num_list`: per-class example counts over the (post-split) train
            split, used by imbalance-aware losses.

        Args:
            augmentation: a callable (or hydra-instantiate result) applied to
                training examples after the base `transform`.
        """

        self.dataset = load_dataset(self.name, streaming=self.streaming)

        categories = self.dataset["train"].features["label"].names
        self.id2label = {index: x for index, x in enumerate(categories, start=0)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.num_classes = len(self.id2label)

        # sub-optimal simple code (might reflect correct split sizes)
        if self.test_split_name not in self.dataset:
            split = self.dataset["train"].train_test_split(
                self.test_split, shuffle=self.shuffle, seed=self.split_seed, stratify_by_column="label"
            )
            self.dataset["train"] = split["train"]
            self.dataset[self.test_split_name] = split["test"]

        if self.val_split_name not in self.dataset:
            split = self.dataset["train"].train_test_split(
                self.val_split, shuffle=self.shuffle, seed=self.split_seed, stratify_by_column="label"
            )
            self.dataset["train"] = split["train"]
            self.dataset[self.val_split_name] = split["test"]

        _, self.cls_num_list = np.unique(self.dataset["train"]["label"], return_counts=True)

        train_transform_batch = partial(
            augment_and_transform_batch,
            transform=self.transform,
            augmentation=augmentation,
            input_column_name=self.input_column_name,
            label_column_name=self.label_column_name,
        )

        predict_transform_batch = partial(
            augment_and_transform_batch,
            transform=self.transform,
            augmentation=None,
            input_column_name=self.input_column_name,
            label_column_name=self.label_column_name,
        )

        self.dataset["train"] = self.dataset["train"].with_transform(train_transform_batch)
        self.dataset[self.val_split_name] = self.dataset[self.val_split_name].with_transform(predict_transform_batch)
        self.dataset[self.test_split_name] = self.dataset[self.test_split_name].with_transform(predict_transform_batch)
