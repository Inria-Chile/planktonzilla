"""
(c) Inria
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
    """Apply augmentations and transformations"""

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
        input_column_name (str): Name of the column containing the images. Deafault is "image".

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
    """Lightweight wrapper around a Hugging Face Dataset. Provides utilities for
    preparing splits, applying transforms and maintaining mappings between label
    ids and names.
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
        return self.dataset["train"]

    @property
    def validation_dataset(self):
        return self.dataset[self.val_split_name]

    @property
    def test_dataset(self):
        return self.dataset[self.test_split_name]

    def __post_init__(self):
        super().__init__()
        self.dataset = None
        self.id2label = self.label2id = None
        self.num_classes = -1

    def prepare_datasets(self, augmentation) -> None:
        """Load dataset, create splits and attach transform pipelines.

        This will load the dataset identified by `self.name` using
        `datasets.load_dataset`, create validation/test splits if missing,
        compute class counts, and attach `with_transform` callables that apply
        augmentation and preprocessing to batches.

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
