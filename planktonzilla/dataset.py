"""
(c) Inria
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoImageProcessor

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def augment_and_transform_batch(examples, transform, augmentation, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image, label in zip(examples["image"], examples["label"], strict=True):
        # res = transform(images=[np.array(image.convert("RGB"))], category=[label])
        # images += res["images"]
        # annotations += res["category"]
        res = transform(image.convert("RGB"))
        res = augmentation(res) if augmentation else res
        images += [res]
        annotations += [label]

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, return_tensors="pt")
    result["label"] = annotations

    return result


def compute_mean_and_std_dev(huggingface_dataset: Dataset):
    sum_pixels = np.zeros(3)  # For R, G, B channels
    sum_squared_pixels = np.zeros(3)
    num_pixels = 0
    for item in huggingface_dataset:
        # Access the image (assuming it's a PIL Image object)
        image = item["image"]

        # Convert image to NumPy array and normalize to [0, 1] if needed
        image_array = np.array(image).astype(np.float32) / 255.0

        # Reshape the image to (height * width, channels) to easily work with pixels
        reshaped_image = image_array.reshape(-1, 3)

        # Accumulate sums
        sum_pixels += np.sum(reshaped_image, axis=0)
        sum_squared_pixels += np.sum(reshaped_image**2, axis=0)

        # Update total number of pixels
        num_pixels += reshaped_image.shape[0]

        # 3. Calculate mean and standard deviation
        mean = sum_pixels / num_pixels
        std_dev = np.sqrt((sum_squared_pixels / num_pixels) - (mean**2))

        return mean, std_dev


@dataclass
class DatasetWrapper:
    name: str
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

    def prepare_datasets(self, image_processor: AutoImageProcessor, augmentation) -> None:
        self.dataset = load_dataset(self.name, streaming=self.streaming)
        # self.test_data = load_dataset("vendimia50/ct_metadataset", streaming=self.streaming)["train"]

        categories = self.dataset["train"].features["label"].names
        self.id2label = {index: x for index, x in enumerate(categories, start=0)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.num_classes = len(self.id2label)

        # sub-optimal simple code (might reflect correct split sizes)
        if self.test_split_name not in self.dataset:
            split = self.dataset["train"].train_test_split(self.test_split, shuffle=self.shuffle, seed=self.split_seed)
            self.dataset["train"] = split["train"]
            self.dataset[self.test_split_name] = split["test"]

        if self.val_split_name not in self.dataset:
            split = self.dataset["train"].train_test_split(self.val_split, shuffle=self.shuffle, seed=self.split_seed)
            self.dataset["train"] = split["train"]
            self.dataset[self.val_split_name] = split["test"]

        _, self.cls_num_list = np.unique(self.dataset["train"]["label"], return_counts=True)

        train_transform_batch = partial(
            augment_and_transform_batch,
            transform=self.transform,
            image_processor=image_processor,
            augmentation=augmentation,
        )

        predict_transform_batch = partial(
            augment_and_transform_batch,
            transform=self.transform,
            image_processor=image_processor,
            augmentation=None,
        )

        self.dataset["train"] = self.dataset["train"].with_transform(train_transform_batch)
        self.dataset[self.val_split_name] = self.dataset[self.val_split_name].with_transform(predict_transform_batch)
        self.dataset[self.test_split_name] = self.dataset[self.test_split_name].with_transform(predict_transform_batch)
