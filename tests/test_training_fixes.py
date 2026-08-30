"""
(c) Inria

Regression tests for the step-1 training-pipeline prerequisites.

Each test below pins one defect that made a training run silently measure something
other than what it claimed, so that any architecture experiment run on top of this
pipeline is comparing what it thinks it is comparing.
"""

import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from planktonzilla.clip_model import ClipClassifier
from planktonzilla.dataset import DatasetWrapper, augment_and_transform_batch
from planktonzilla.train import build_compute_loss_func, freeze_backbone_except_head

# --------------------------------------------------------------------------------------
# Augmentation ordering
# --------------------------------------------------------------------------------------

NORMALIZE_MEAN = [0.481, 0.458, 0.408]
NORMALIZE_STD = [0.269, 0.261, 0.276]


def _transform():
    """The shape every configs/dataset/*.yaml uses: ToTensor -> Resize -> Normalize."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(16, 16), antialias=True),
            v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def _examples(n=2):
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)) for _ in range(n)]
    return {"image": images, "label": list(range(n))}


def test_augmentation_receives_the_pil_image_not_the_normalized_tensor():
    """Augmentation must run before `transform`, i.e. on the PIL image.

    It previously ran after the whole transform Compose, so torchvision's photometric
    ops were handed an already-normalized float tensor.
    """
    seen = []

    def spy(img):
        seen.append(img)
        return img

    augment_and_transform_batch(
        _examples(),
        transform=_transform(),
        augmentation=spy,
        input_column_name="image",
        label_column_name="label",
    )

    assert seen, "augmentation was never called"
    for img in seen:
        assert isinstance(img, Image.Image), (
            f"augmentation received {type(img).__name__}; it must see the PIL image, before Normalize"
        )


@pytest.mark.parametrize(
    "augmentation",
    [
        v2.RandAugment(num_ops=4, magnitude=9),
        v2.TrivialAugmentWide(),
        v2.ColorJitter(brightness=0.32, contrast=0.32, saturation=0.32, hue=0.08),
    ],
    ids=["randaugment", "trivialaugment", "colorjitter"],
)
def test_augmentation_does_not_destroy_the_normalization(augmentation):
    """The batch reaching the model must still be normalized when augmentation is on.

    With the old ordering, RandAugment and ColorJitter clamped the normalized tensor back
    into [0, 1] — silently discarding `Normalize` — so a run with augmentation enabled fed
    the backbone a completely different input distribution from a run without it.
    """
    torch.manual_seed(0)
    batch = augment_and_transform_batch(
        _examples(4),
        transform=_transform(),
        augmentation=augmentation,
        input_column_name="image",
        label_column_name="label",
    )
    pixel_values = batch["pixel_values"]

    # A normalized tensor straddles zero; a [0, 1] tensor cannot.
    assert pixel_values.min() < 0.0, (
        f"pixel_values min is {pixel_values.min():.4f} >= 0, so Normalize was undone by the augmentation"
    )


def test_augmentation_is_skipped_for_the_eval_pipeline():
    """`augmentation=None` is how the validation/test transforms are built."""
    batch = augment_and_transform_batch(
        _examples(),
        transform=_transform(),
        augmentation=None,
        input_column_name="image",
        label_column_name="label",
    )
    assert batch["pixel_values"].shape == (2, 3, 16, 16)


# --------------------------------------------------------------------------------------
# Per-class counts
# --------------------------------------------------------------------------------------


def _wrapper(num_classes):
    wrapper = DatasetWrapper(name="unused/for-unit-test")
    wrapper.num_classes = num_classes
    wrapper.id2label = {i: f"class_{i}" for i in range(num_classes)}
    return wrapper


def test_cls_num_list_is_indexed_by_class_id():
    """Counts must line up with class ids over the full label space.

    `np.unique(..., return_counts=True)` returns one count per *observed* label, so a
    class missing from train shifted every later class's count down a slot and shortened
    the vector — and the imbalance losses index it positionally by class id.
    """
    labels = [0, 0, 0, 1, 1, 3, 3, 3, 3]  # class 2 absent
    counts = _wrapper(4)._count_per_class(labels)

    assert len(counts) == 4, f"cls_num_list must cover every class, got length {len(counts)}"
    assert counts[0] == 3
    assert counts[1] == 2
    assert counts[3] == 4, f"class 3's count landed at the wrong index: {counts}"

    # np.unique would have produced [3, 2, 4] — class 3's count sitting at index 2.
    _, unique_counts = np.unique(labels, return_counts=True)
    assert list(unique_counts) == [3, 2, 4]
    assert counts[2] != 4, "class 2 must not inherit class 3's count"


def test_cls_num_list_clamps_empty_classes_and_warns(caplog):
    """An empty class is clamped to 1, because 0 turns the whole LDAM margin vector to NaN."""
    with caplog.at_level(logging.WARNING):
        counts = _wrapper(4)._count_per_class([0, 0, 1, 3])

    assert counts[2] == 1, f"empty class must be clamped to 1, got {counts[2]}"
    assert any("no examples in the train split" in record.message for record in caplog.records), (
        "an empty class must be reported, not silently clamped"
    )

    # The clamp is what keeps the margins finite: LDAM divides by these counts.
    margins = 1.0 / np.sqrt(np.sqrt(counts))
    margins = margins * (0.5 / np.max(margins))
    assert np.all(np.isfinite(margins)), f"margins are not finite: {margins}"


def test_cls_num_list_is_all_finite_for_a_dense_label_space():
    counts = _wrapper(3)._count_per_class([0, 1, 1, 2, 2, 2])
    assert list(counts) == [1, 2, 3]


# --------------------------------------------------------------------------------------
# freeze_backbone
# --------------------------------------------------------------------------------------


class _SequentialHeadModel(nn.Module):
    """Mirrors ClipClassifier's open_clip path: params are named `model.1.weight` / `.bias`."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 3))

    @property
    def head(self):
        return self.model[1]


class _HFStyleModel(nn.Module):
    """Mirrors a Hugging Face image-classification model: the head is named `classifier`."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)


class _HeadlessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.output_layer = nn.Linear(8, 3)


def test_freeze_backbone_keeps_the_sequential_head_trainable():
    """The open_clip path's head is `model.1.*`, which no substring match catches."""
    model = _SequentialHeadModel()
    trainable = freeze_backbone_except_head(model)

    assert trainable == ["model.1.weight", "model.1.bias"], trainable
    assert all(param.requires_grad for param in model.head.parameters())
    assert not any(param.requires_grad for param in model.model[0].parameters())

    # The whole point: something must still be able to learn.
    assert any(param.requires_grad for param in model.parameters())


def test_freeze_backbone_keeps_the_hf_classifier_trainable():
    model = _HFStyleModel()
    trainable = freeze_backbone_except_head(model)

    assert set(trainable) == {"classifier.weight", "classifier.bias"}
    assert not any(param.requires_grad for param in model.encoder.parameters())


def test_freeze_backbone_refuses_to_freeze_everything():
    """Freezing the head too is never a usable run, and used to happen silently."""
    with pytest.raises(ValueError, match="no trainable parameters"):
        freeze_backbone_except_head(_HeadlessModel())


def test_clip_classifier_head_property_adds_no_state_dict_keys():
    """`head` is a property, so published checkpoints keep loading unchanged."""
    assert isinstance(ClipClassifier.head, property)

    model = object.__new__(ClipClassifier)
    nn.Module.__init__(model)
    model.model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 3))

    assert model.head is model.model[1]
    assert sorted(model.state_dict()) == ["model.0.bias", "model.0.weight", "model.1.bias", "model.1.weight"]


def test_clip_classifier_head_property_follows_the_timm_path():
    model = object.__new__(ClipClassifier)
    nn.Module.__init__(model)
    trunk = nn.Module()
    trunk.head = nn.Linear(8, 3)
    model.model = trunk

    assert model.head is trunk.head


# --------------------------------------------------------------------------------------
# Gradient-accumulation normalisation
# --------------------------------------------------------------------------------------


class _MeanLoss(nn.Module):
    """Stands in for the planktonzilla losses: mean-reduced over the micro-batch."""

    def forward(self, output, target, **kwargs):
        return output.logits.sum(dim=-1).mean()


def _out(n):
    return ImageClassifierOutputWithNoAttention(loss=None, logits=torch.ones(n, 3), hidden_states=None)


def test_custom_loss_is_unchanged_without_accumulation():
    """With one accumulation step, num_items_in_batch == batch size, so nothing moves."""
    loss_fn = build_compute_loss_func(_MeanLoss())
    labels = torch.zeros(8, dtype=torch.long)

    plain = _MeanLoss()(_out(8), labels)
    adapted = loss_fn(_out(8), labels, num_items_in_batch=8)

    torch.testing.assert_close(adapted, plain)


def test_custom_loss_normalizes_across_accumulation_steps():
    """Four micro-batches of 8 must sum to the mean over all 32 items, not 4x it.

    Trainer.training_step skips its own `loss / accum_steps` whenever compute_loss_func is
    set, so an un-normalized custom loss inflated the gradient — and the effective learning
    rate — by the accumulation factor.
    """
    loss_fn = build_compute_loss_func(_MeanLoss())
    labels = torch.zeros(8, dtype=torch.long)
    total_items = 32  # 4 accumulation steps x 8

    accumulated = sum(loss_fn(_out(8), labels, num_items_in_batch=total_items) for _ in range(4))
    single_pass = _MeanLoss()(_out(32), torch.zeros(32, dtype=torch.long))

    torch.testing.assert_close(accumulated, single_pass)


def test_custom_loss_accepts_a_tensor_item_count():
    """HF passes num_items_in_batch as a device tensor."""
    loss_fn = build_compute_loss_func(_MeanLoss())
    labels = torch.zeros(8, dtype=torch.long)

    as_tensor = loss_fn(_out(8), labels, num_items_in_batch=torch.tensor(16))
    as_int = loss_fn(_out(8), labels, num_items_in_batch=16)

    torch.testing.assert_close(as_tensor, as_int)


def test_custom_loss_tolerates_a_missing_item_count():
    """`num_items_in_batch` is None on paths that do not count items."""
    loss_fn = build_compute_loss_func(_MeanLoss())
    labels = torch.zeros(8, dtype=torch.long)

    torch.testing.assert_close(loss_fn(_out(8), labels, num_items_in_batch=None), _MeanLoss()(_out(8), labels))
