"""
(c) Inria

Tests for the cosine classification head, mixed-precision resolution, and label smoothing.

These are the three levers of the "cheap, low-risk, independently ablatable" tier of the
training-architecture work. Each defaults to the previous behaviour, so the tests pin both
that the new path works and that the old one is untouched.
"""

import logging

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from omegaconf import OmegaConf
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from planktonzilla.clip_model import ClipClassifier
from planktonzilla.heads import DEFAULT_SCALE, MAX_SCALE, CosineClassifier, replace_head_with_cosine
from planktonzilla.train import resolve_precision, warn_if_label_smoothing_is_ignored

# --------------------------------------------------------------------------------------
# Cosine head
# --------------------------------------------------------------------------------------


def test_logits_are_scaled_cosine_similarity():
    """The whole point: a logit is bounded by the temperature, not by the feature norm."""
    torch.manual_seed(0)
    head = CosineClassifier(16, 5)

    small = head(torch.randn(32, 16) * 0.01)
    huge = head(torch.randn(32, 16) * 1000.0)

    for logits in (small, huge):
        assert logits.abs().max() <= head.scale + 1e-4, (
            f"a cosine logit must be bounded by the scale ({head.scale.item():.3f}), got {logits.abs().max():.3f}"
        )


def test_logits_match_the_explicit_cosine_formula():
    torch.manual_seed(0)
    head = CosineClassifier(8, 4)
    features = torch.randn(6, 8)

    expected = head.scale * F.normalize(features, dim=-1) @ F.normalize(head.weight, dim=-1).T

    torch.testing.assert_close(head(features), expected)


def test_feature_magnitude_does_not_change_the_ranking():
    """Scaling a feature vector cannot change which class wins — that is the invariance."""
    torch.manual_seed(0)
    head = CosineClassifier(16, 7)
    features = torch.randn(10, 16)

    torch.testing.assert_close(head(features).argmax(-1), head(features * 37.0).argmax(-1))


def test_head_has_no_bias():
    """A per-class bias would reintroduce the frequency-dependent offset."""
    head = CosineClassifier(8, 3)
    assert not any(name == "bias" for name, _ in head.named_parameters()), dict(head.named_parameters()).keys()


def test_scale_defaults_to_clip_temperature_and_is_learnable():
    head = CosineClassifier(8, 3)
    assert head.scale.item() == pytest.approx(DEFAULT_SCALE, rel=1e-5)
    assert head.log_scale.requires_grad


def test_scale_can_be_frozen_and_set():
    head = CosineClassifier(8, 3, scale=20.0, learnable_scale=False)
    assert head.scale.item() == pytest.approx(20.0, rel=1e-5)
    assert not head.log_scale.requires_grad


def test_scale_is_clamped():
    """CLIP clamps its learned temperature so a runaway cannot saturate the softmax."""
    head = CosineClassifier(8, 3, scale=10.0)
    with torch.no_grad():
        head.log_scale.fill_(torch.tensor(1e4).log())
    assert head.scale.item() == pytest.approx(MAX_SCALE)


def test_a_non_positive_scale_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        CosineClassifier(8, 3, scale=0.0)


def _clip_model(sequential: bool):
    """A ClipClassifier in either dispatch shape, without downloading CLIP weights."""
    model = object.__new__(ClipClassifier)
    nn.Module.__init__(model)
    if sequential:
        model.model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 5))
    else:
        trunk = nn.Module()
        trunk.head = nn.Linear(16, 5)
        model.model = trunk
    return model


class _HFStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 16)
        self.classifier = nn.Linear(16, 5)


@pytest.mark.parametrize(
    ("build", "attribute"),
    [
        (lambda: _clip_model(sequential=True), "head"),
        (lambda: _clip_model(sequential=False), "head"),
        (_HFStyleModel, "classifier"),
    ],
    ids=["clip-open_clip-path", "clip-timm-path", "huggingface"],
)
def test_swap_installs_the_head_on_every_model_shape(build, attribute):
    model = build()
    head = replace_head_with_cosine(model)

    assert getattr(model, attribute) is head
    assert head.in_features == 16
    assert head.out_features == 5


def test_swap_does_not_leave_a_dead_submodule_behind():
    """`nn.Module.__setattr__` never reaches a property setter.

    Assigning to `ClipClassifier.head` would register a dead `head.*` entry in
    `state_dict()` while `.head` kept returning the original Linear — the real head
    silently left in place. `set_head` exists to avoid exactly that.
    """
    model = _clip_model(sequential=True)
    head = replace_head_with_cosine(model)

    keys = sorted(model.state_dict())
    assert not any(key.startswith("head.") for key in keys), f"stray head.* entries: {keys}"
    assert keys == ["model.0.bias", "model.0.weight", "model.1.log_scale", "model.1.weight"]
    assert model.model[1] is head, "the head inside the Sequential must be the new one"


def test_swap_keeps_the_model_runnable_and_the_label_space_intact():
    model = _clip_model(sequential=True)
    replace_head_with_cosine(model)
    assert model.model(torch.randn(4, 8)).shape == (4, 5)


def test_swap_refuses_a_model_with_no_linear_head():
    class Headless(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(8, 16)

    with pytest.raises(TypeError, match="head_type=linear"):
        replace_head_with_cosine(Headless())


def test_frozen_head_still_trains_its_weight():
    """`learnable_scale=False` freezes the temperature only, not the class weights."""
    head = CosineClassifier(8, 3, learnable_scale=False)
    head(torch.randn(4, 8)).sum().backward()

    assert head.weight.grad is not None
    assert head.log_scale.grad is None


# --------------------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------------------


class _Args:
    def __init__(self, fp16=True, bf16=False, label_smoothing_factor=0.0):
        self.fp16 = fp16
        self.bf16 = bf16
        self.label_smoothing_factor = label_smoothing_factor


@pytest.mark.parametrize(
    ("precision", "expect_bf16", "expect_fp16"),
    [("bf16", True, False), ("fp16", False, True), ("fp32", False, False)],
)
def test_explicit_precision_wins(precision, expect_bf16, expect_fp16):
    args = _Args()
    assert resolve_precision(precision, args) == precision
    assert args.bf16 is expect_bf16
    assert args.fp16 is expect_fp16


def test_null_precision_leaves_training_arguments_untouched():
    """Full manual control: whatever training_arguments set, stands."""
    args = _Args(fp16=True, bf16=False)
    assert resolve_precision(None, args) == "unchanged"
    assert args.fp16 is True and args.bf16 is False


def test_auto_falls_back_to_fp32_without_a_gpu(monkeypatch):
    """Mixed precision on CPU buys nothing, so `auto` must not request it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = _Args()

    assert resolve_precision("auto", args) == "fp32"
    assert args.fp16 is False and args.bf16 is False


def test_auto_picks_bf16_when_the_gpu_supports_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    args = _Args()

    assert resolve_precision("auto", args) == "bf16"
    assert args.bf16 is True and args.fp16 is False


def test_auto_stays_on_fp16_for_pre_ampere_hardware(monkeypatch):
    """V100/T4 have no bf16; flipping the default outright would have broken them."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    args = _Args()

    assert resolve_precision("auto", args) == "fp16"
    assert args.fp16 is True and args.bf16 is False


def test_an_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="auto/bf16/fp16/fp32"):
        resolve_precision("float8", _Args())


# --------------------------------------------------------------------------------------
# Label smoothing
# --------------------------------------------------------------------------------------


def _out(logits):
    return ImageClassifierOutputWithNoAttention(loss=None, logits=logits, hidden_states=None)


def _losses(cls_num_list, label_smoothing):
    from planktonzilla.loss import BalancedMetaSoftmaxLoss, CrossEntropyLossHF, LDAMLoss, MaximumMarginLoss

    return {
        "cross_entropy": CrossEntropyLossHF(label_smoothing=label_smoothing),
        "ldam": LDAMLoss(cls_num_list=cls_num_list, s=1, label_smoothing=label_smoothing),
        "balanced_meta_softmax": BalancedMetaSoftmaxLoss(cls_num_list=cls_num_list, label_smoothing=label_smoothing),
        "max_margin": MaximumMarginLoss(cls_num_list=cls_num_list, s=1, label_smoothing=label_smoothing),
    }


def _default_losses(cls_num_list):
    """The same four losses built the way every existing config builds them: no smoothing kwarg."""
    from planktonzilla.loss import BalancedMetaSoftmaxLoss, CrossEntropyLossHF, LDAMLoss, MaximumMarginLoss

    return {
        "cross_entropy": CrossEntropyLossHF(),
        "ldam": LDAMLoss(cls_num_list=cls_num_list, s=1),
        "balanced_meta_softmax": BalancedMetaSoftmaxLoss(cls_num_list=cls_num_list),
        "max_margin": MaximumMarginLoss(cls_num_list=cls_num_list, s=1),
    }


def test_label_smoothing_defaults_to_off_on_every_cross_entropy_loss():
    """Existing runs must be bit-for-bit unchanged: omitting the kwarg equals passing 0.0."""
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))
    cls_num_list = [50, 20, 5]

    explicit_zero = _losses(cls_num_list, label_smoothing=0.0)
    for name, default_ctor in _default_losses(cls_num_list).items():
        torch.testing.assert_close(explicit_zero[name](_out(logits), target), default_ctor(_out(logits), target), msg=name)


@pytest.mark.parametrize("name", ["cross_entropy", "ldam", "balanced_meta_softmax", "max_margin"])
def test_label_smoothing_changes_the_loss_when_enabled(name):
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))
    cls_num_list = [50, 20, 5]

    off = _losses(cls_num_list, 0.0)[name](_out(logits), target)
    on = _losses(cls_num_list, 0.1)[name](_out(logits), target)

    assert not torch.isclose(off, on), f"{name}: label_smoothing=0.1 had no effect ({off} vs {on})"
    assert torch.isfinite(on)


def test_cross_entropy_smoothing_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))

    from planktonzilla.loss import CrossEntropyLossHF

    actual = CrossEntropyLossHF(label_smoothing=0.15)(_out(logits), target)
    torch.testing.assert_close(actual, F.cross_entropy(logits, target, label_smoothing=0.15))


def test_training_arguments_label_smoothing_is_reported_as_ignored(caplog):
    """`Trainer` skips its label smoother whenever compute_loss_func is set — always, here.

    A silent no-op on a knob a user deliberately turned on is exactly the failure mode this
    whole tier of work is removing.
    """
    cfg = OmegaConf.create({"custom_loss": {"_target_": "planktonzilla.loss.CrossEntropyLossHF"}})

    with caplog.at_level(logging.WARNING):
        warned = warn_if_label_smoothing_is_ignored(cfg, _Args(label_smoothing_factor=0.1))

    assert warned
    assert any("custom_loss.label_smoothing" in record.message for record in caplog.records), (
        "the warning must name the knob that actually works"
    )


def test_no_warning_when_label_smoothing_is_left_off():
    cfg = OmegaConf.create({"custom_loss": {"_target_": "planktonzilla.loss.CrossEntropyLossHF"}})
    assert not warn_if_label_smoothing_is_ignored(cfg, _Args(label_smoothing_factor=0.0))
