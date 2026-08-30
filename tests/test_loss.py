"""
(c) Inria
"""

import inspect

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from planktonzilla import loss as loss_mod
from planktonzilla.loss import FocalLoss


def test_focal_loss_axis():
    """Pin FocalLoss to axis-correct softmax (dim=-1) on a (N, C) tensor.

    If log_softmax slips back to a default that takes softmax over dim=0,
    the hand-computed reference no longer matches and this test fails.
    Closes CONCERNS.md #5 / FOCUS-01.
    """
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.1],
            [0.5, 2.5, 0.3],
            [1.0, 0.0, 3.0],
            [0.2, 0.1, 0.8],
        ]
    )
    target = torch.tensor([0, 1, 2, 2])
    alpha = [1.0, 1.0, 1.0]  # per-class weights for 3-class problem (uniform = no reweighting)
    gamma = 2.0

    output = ImageClassifierOutputWithNoAttention(loss=None, logits=logits, hidden_states=None)
    actual = FocalLoss(alpha=alpha, gamma=gamma, size_average=True)(output, target)

    # Hand-computed reference using axis-correct softmax over the class axis.
    logpt_full = F.log_softmax(logits, dim=-1)
    gathered = logpt_full.gather(1, target.view(-1, 1)).view(-1)
    pt = gathered.exp()
    alpha_tensor = torch.tensor(alpha)
    at = alpha_tensor.gather(0, target)
    weighted_logpt = gathered * at
    per_sample = -1 * (1 - pt) ** gamma * weighted_logpt
    expected = per_sample.mean()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_focal_loss_no_variable_wrapper():
    """Pin the absence of deprecated torch.autograd.Variable in loss.py.

    Variable was deprecated in PyTorch 0.4 (2018) and is a no-op on Tensors.
    Keeping the wrappers obscured the autograd path. This test guards the
    cleanup so a regression flips the gate. Closes FOCUS-01 cleanup half.
    """
    source = inspect.getsource(loss_mod)
    offending = [line for line in source.splitlines() if "Variable" in line and not line.lstrip().startswith("#")]
    assert offending == [], f"Variable references remain in planktonzilla/loss.py: {offending}"


def _batch(logits):
    return ImageClassifierOutputWithNoAttention(loss=None, logits=logits, hidden_states=None)


def _all_losses(cls_num_list):
    """One instance of every concrete loss in planktonzilla.loss, ready to call."""
    from planktonzilla.loss import (
        AsymmetricLoss,
        BalancedMetaSoftmaxLoss,
        CrossEntropyLossHF,
        LDAMLoss,
        MaximumMarginLoss,
        RobustAsymmetricLoss,
    )

    return {
        "focal": FocalLoss(alpha=[1.0] * len(cls_num_list), gamma=2.0, size_average=True),
        "ldam": LDAMLoss(cls_num_list=cls_num_list, s=1),
        "max_margin": MaximumMarginLoss(cls_num_list=cls_num_list, s=1),
        "asymmetric": AsymmetricLoss(),
        "ral": RobustAsymmetricLoss(),
        "balanced_meta_softmax": BalancedMetaSoftmaxLoss(cls_num_list=cls_num_list),
        "cross_entropy": CrossEntropyLossHF(),
    }


def test_every_loss_is_permutation_invariant():
    """A mean-reduced per-sample loss must not depend on the order of rows in the batch.

    This is the property that catches the whole class of "indexed the batch with a
    scalar" bug. `MaximumMarginLoss.obj_margins` used
    `rm_obj_dists[:, labels.data.cpu().numpy()[0]]` — the label of whichever example the
    DataLoader shuffled to position 0 — as the positive column for *every* row, so the
    same batch in a different order produced a different loss (0.9783 / 1.0556 / 1.0220
    on the batch below). With `shuffle=True` that reference column changed every step.
    """
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [0.9, 0.4, 0.2],
            [0.3, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.5, 0.6, 0.55],
        ]
    )
    target = torch.tensor([0, 1, 2, 1])
    cls_num_list = [100, 10, 1]

    permutations = [[1, 0, 2, 3], [2, 1, 0, 3], [3, 2, 1, 0]]

    for name, loss_fn in _all_losses(cls_num_list).items():
        reference = loss_fn(_batch(logits), target)
        for perm in permutations:
            index = torch.tensor(perm)
            permuted = loss_fn(_batch(logits[index]), target[index])
            torch.testing.assert_close(
                permuted,
                reference,
                rtol=1e-5,
                atol=1e-6,
                msg=lambda got, exp, name=name, perm=perm: (
                    f"{name} is not permutation-invariant: row order {perm} gives {got} but the original order gives {exp}"
                ),
            )


def test_maximum_margin_uses_each_row_own_positive_logit():
    """Pin the per-row gather in `MaximumMarginLoss.obj_margins`.

    The positive score must come from each sample's own ground-truth column. The previous
    version indexed a single column with the scalar `labels[0]`, so for the batch below it
    read the true-class logits as [5, 0, 0, 1] instead of [5, 5, 5, 3].
    """
    from planktonzilla.loss import MaximumMarginLoss

    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [1.0, 2.0, 3.0]])
    target = torch.tensor([0, 1, 2, 2])

    index = torch.zeros_like(logits)
    index.scatter_(1, target.view(-1, 1), 1.0)

    loss_fn = MaximumMarginLoss(cls_num_list=[10, 20, 30], s=1)
    actual = loss_fn.obj_margins(logits, target, index, loss_fn.max_m)

    # Reference built from each row's OWN positive logit.
    positives = (logits * index).sum(1)
    torch.testing.assert_close(positives, torch.tensor([5.0, 5.0, 5.0, 3.0]))
    negatives = (logits * (1.0 - index)).max(1)[0]

    fg = torch.clamp(positives - negatives, min=0.0)[:, None]
    bg = torch.clamp(negatives - positives, min=0.0)[:, None]
    expected = (
        torch.exp(-bg - loss_fn.max_m) * (bg > 0).float() + torch.exp(-fg - loss_fn.max_m * loss_fn.gamma) * (fg > 0).float()
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "RobustAsymmetricLoss does not suppress easy negatives: neither robustness term is "
        "masked by its label indicator, so the focusing base does not reduce to p on a "
        "negative column. Masking them the way AsymmetricLoss does is NOT the fix — the "
        "negative term itself tends to 0 as p -> 0, so the base tends to 1 either way. "
        "Repairing it needs the published RAL formulation, not an analogy to ASL. This test "
        "is strict so it flips to a failure the moment someone lands the real fix."
    ),
)
def test_robust_asymmetric_loss_does_not_yet_suppress_easy_negatives():
    """Executable record of the open RAL defect, with the measurement that shows it.

    `gamma_neg` exists to down-weight negatives the model already rejects confidently.
    On a well-classified 1000-class batch RAL returns ~1870x what AsymmetricLoss returns,
    because those easy negatives keep ~full weight and their label-smoothing tail
    (`eps/C * -log p`, summed over 999 classes) dominates the loss.
    """
    from planktonzilla.loss import AsymmetricLoss, RobustAsymmetricLoss

    num_classes = 1000
    logits = torch.full((1, num_classes), -6.0)
    logits[0, 0] = 8.0
    target = torch.tensor([0])

    asl = AsymmetricLoss()(_batch(logits), target)
    ral = RobustAsymmetricLoss()(_batch(logits), target)

    assert torch.isfinite(ral), f"RAL returned {ral}"
    assert ral < 10 * asl, (
        f"RAL={ral.item():.6g} is far above ASL={asl.item():.6g} on a well-classified batch, "
        f"which means the easy negatives are not being suppressed"
    )


def test_robust_asymmetric_loss_is_finite_under_softmax_underflow():
    """Pin the clamped logs: p underflows to 0 for a confidently-rejected class."""
    from planktonzilla.loss import RobustAsymmetricLoss

    logits = torch.tensor([[60.0, -60.0, -60.0]])
    target = torch.tensor([0])
    value = RobustAsymmetricLoss()(_batch(logits), target)
    assert torch.isfinite(value), f"RAL produced {value} under softmax underflow"
