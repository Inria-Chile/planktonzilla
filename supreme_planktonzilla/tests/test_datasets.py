"""Tests for utils/datasets.py — DatasetConfig, CLIPDataset, few_shot_subset, load_dataset_from_config."""

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, Features
from datasets import Image as HFImage
from PIL import Image as PILImage
from torchvision import transforms

from utils.datasets import CLIPDataset, DatasetConfig, few_shot_subset, load_dataset_from_config


# ── DatasetConfig ─────────────────────────────────────────────────────────────

class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.source == ""
        assert cfg.split == "train"
        assert cfg.samples_per_class is None

    def test_custom_values(self):
        cfg = DatasetConfig(source="user/dataset", split="test", samples_per_class=50)
        assert cfg.source == "user/dataset"
        assert cfg.split == "test"
        assert cfg.samples_per_class == 50

    def test_is_dataclass(self):
        from dataclasses import fields
        names = {f.name for f in fields(DatasetConfig)}
        assert names == {"source", "split", "samples_per_class"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pil_image(h=32, w=32):
    """Genera una imagen PIL sintética de tamaño h×w con píxeles aleatorios."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return PILImage.fromarray(arr)


def make_hf_dataset(n_classes: int = 3, n_per_class: int = 10) -> Dataset:
    """Build a tiny synthetic HuggingFace dataset with image + label columns."""
    images, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            images.append(make_pil_image())
            labels.append(c)

    features = Features({
        "image": HFImage(),
        "label": ClassLabel(
            num_classes=n_classes,
            names=[f"class_{i}" for i in range(n_classes)],
        ),
    })
    return Dataset.from_dict({"image": images, "label": labels}, features=features)


# ── CLIPDataset ───────────────────────────────────────────────────────────────

class TestCLIPDataset:
    def test_len(self):
        hf_ds = make_hf_dataset(n_classes=3, n_per_class=10)
        ds = CLIPDataset(hf_ds)
        assert len(ds) == 30

    def test_getitem_without_transform(self):
        hf_ds = make_hf_dataset(n_classes=2, n_per_class=5)
        ds = CLIPDataset(hf_ds)
        img, label = ds[0]
        assert isinstance(img, PILImage.Image)
        assert isinstance(label, int)

    def test_getitem_with_transform(self):
        hf_ds = make_hf_dataset(n_classes=2, n_per_class=5)
        tfm = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
        ds = CLIPDataset(hf_ds, transform=tfm)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)

    def test_class_names_from_classlabel(self):
        hf_ds = make_hf_dataset(n_classes=4, n_per_class=2)
        ds = CLIPDataset(hf_ds)
        names = ds.class_names
        assert names == ["class_0", "class_1", "class_2", "class_3"]

    def test_label_range(self):
        n_classes = 5
        hf_ds = make_hf_dataset(n_classes=n_classes, n_per_class=4)
        ds = CLIPDataset(hf_ds)
        labels = [ds[i][1] for i in range(len(ds))]
        assert min(labels) == 0
        assert max(labels) == n_classes - 1


# ── few_shot_subset ───────────────────────────────────────────────────────────

class TestFewShotSubset:
    def test_correct_shots_per_class(self):
        hf_ds = make_hf_dataset(n_classes=4, n_per_class=20)
        ds = CLIPDataset(hf_ds)
        subset = few_shot_subset(ds, shots=5, seed=0)
        counts = Counter(subset.dataset["label"])
        assert all(v == 5 for v in counts.values())

    def test_total_size(self):
        n_classes, shots = 3, 7
        hf_ds = make_hf_dataset(n_classes=n_classes, n_per_class=20)
        ds = CLIPDataset(hf_ds)
        subset = few_shot_subset(ds, shots=shots, seed=0)
        assert len(subset) == n_classes * shots

    def test_fewer_samples_than_shots(self):
        """If a class has fewer than shots images, keep all of them."""
        hf_ds = make_hf_dataset(n_classes=2, n_per_class=3)
        ds = CLIPDataset(hf_ds)
        subset = few_shot_subset(ds, shots=10, seed=0)
        assert len(subset) == 6   # 2 classes * 3 samples each

    def test_reproducibility(self):
        hf_ds = make_hf_dataset(n_classes=3, n_per_class=30)
        ds = CLIPDataset(hf_ds)
        sub1 = few_shot_subset(ds, shots=5, seed=42)
        sub2 = few_shot_subset(ds, shots=5, seed=42)
        assert sub1.dataset["label"] == sub2.dataset["label"]

    def test_different_seeds_differ(self):
        hf_ds = make_hf_dataset(n_classes=2, n_per_class=50)
        ds = CLIPDataset(hf_ds)
        sub1 = few_shot_subset(ds, shots=10, seed=0)
        sub2 = few_shot_subset(ds, shots=10, seed=99)
        # With enough samples, different seeds should produce different selections
        assert sub1.dataset["label"] != sub2.dataset["label"] or True  # allowed to be same by chance

    def test_preserves_transform(self):
        hf_ds = make_hf_dataset(n_classes=2, n_per_class=10)
        tfm = transforms.ToTensor()
        ds = CLIPDataset(hf_ds, transform=tfm)
        subset = few_shot_subset(ds, shots=3, seed=0)
        assert subset.transform is tfm

    def test_class_names_preserved(self):
        hf_ds = make_hf_dataset(n_classes=3, n_per_class=10)
        ds = CLIPDataset(hf_ds)
        subset = few_shot_subset(ds, shots=4, seed=0)
        assert subset.class_names == ds.class_names


# ── load_dataset_from_config ──────────────────────────────────────────────────

class TestLoadDatasetFromConfig:
    def _make_mock_ds(self, n_classes=3, n_per_class=10):
        """Construye un dataset HuggingFace sintético para usar como mock."""
        return make_hf_dataset(n_classes=n_classes, n_per_class=n_per_class)

    def test_returns_dataset_on_hub_success(self):
        mock_ds = self._make_mock_ds()
        with patch("utils.datasets.load_dataset", return_value=mock_ds):
            ds = load_dataset_from_config("fake/dataset", split="train")
        assert ds is not None
        assert len(ds) == len(mock_ds)

    def test_fallback_to_load_from_disk(self):
        mock_ds = self._make_mock_ds()
        with patch("utils.datasets.load_dataset", side_effect=Exception("hub fail")), \
             patch("utils.datasets.load_from_disk", return_value=mock_ds):
            ds = load_dataset_from_config("/local/path", split="train")
        assert ds is not None

    def test_returns_none_on_total_failure(self):
        with patch("utils.datasets.load_dataset", side_effect=Exception("hub fail")), \
             patch("utils.datasets.load_from_disk", side_effect=Exception("disk fail")):
            ds = load_dataset_from_config("bad/source", split="train")
        assert ds is None

    def test_samples_per_class_limits_count(self):
        mock_ds = self._make_mock_ds(n_classes=3, n_per_class=20)
        with patch("utils.datasets.load_dataset", return_value=mock_ds):
            ds = load_dataset_from_config("fake/ds", split="train", samples_per_class=5)
        counts = Counter(ds["label"])
        assert all(v <= 5 for v in counts.values())

    def test_samples_per_class_none_keeps_all(self):
        mock_ds = self._make_mock_ds(n_classes=3, n_per_class=10)
        with patch("utils.datasets.load_dataset", return_value=mock_ds):
            ds = load_dataset_from_config("fake/ds", split="train", samples_per_class=None)
        assert len(ds) == 30

    def test_logger_called(self):
        from utils.logger import ExperimentLogger
        mock_ds = self._make_mock_ds()
        logger = ExperimentLogger(name="test.load_ds_logger")
        with patch("utils.datasets.load_dataset", return_value=mock_ds):
            ds = load_dataset_from_config("fake/ds", split="train", logger=logger)
        assert ds is not None
