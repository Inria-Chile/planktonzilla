"""
(c) Inria
"""

import inspect

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
