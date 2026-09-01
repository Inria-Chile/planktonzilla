"""
(c) Inria

Classification heads for the plankton training pipeline.

The default head everywhere is a plain ``nn.Linear``, whose logit for a class is
``w_c . x + b_c`` — a quantity whose magnitude grows with how often that class was seen,
because frequent classes get more gradient pushing ``||w_c||`` up. On a corpus as
long-tailed as planktonzilla that bias sits underneath every imbalance-aware loss in
`planktonzilla.loss`, which can only reshape the margins it is given.

:class:`CosineClassifier` removes it by L2-normalising both the feature and every class
weight, so the logit is a pure cosine similarity in ``[-1, 1]`` scaled by one shared
temperature. Class frequency can no longer express itself as weight norm, which is why
this is the usual first move on a long-tailed benchmark and why it composes with, rather
than competes against, the margin losses.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# CLIP's initial temperature, 1 / 0.07. A cosine logit lives in [-1, 1], so it needs a
# scale of this order before softmax produces usable gradients.
DEFAULT_SCALE = 1 / 0.07

# CLIP clamps its learned logit scale to 100 to stop a runaway temperature from saturating
# the softmax; the same bound applies here for the same reason.
MAX_SCALE = 100.0


class CosineClassifier(nn.Module):
    """A cosine-similarity classification head with a shared, optionally learned temperature.

    ``logits = scale * normalize(x) @ normalize(W).T``

    There is deliberately no bias: a per-class offset would reintroduce exactly the
    frequency-dependent shift that normalising the weights removes.

    Args:
        in_features: Feature dimension arriving from the backbone.
        out_features: Number of classes.
        scale: Initial temperature. ``None`` uses CLIP's ``1 / 0.07``.
        learnable_scale: Whether the temperature is trained. When False it stays fixed at
            ``scale``, which is the right choice for an ablation that must hold it constant.
    """

    def __init__(self, in_features: int, out_features: int, scale: float | None = None, learnable_scale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.01)

        # Parameterised in log space so the temperature cannot be driven negative.
        initial = DEFAULT_SCALE if scale is None else float(scale)
        if initial <= 0:
            raise ValueError(f"scale must be positive, got {initial}")
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial)), requires_grad=learnable_scale)

    @property
    def scale(self) -> torch.Tensor:
        """The current temperature, clamped to `MAX_SCALE`."""
        return self.log_scale.exp().clamp(max=MAX_SCALE)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between the normalised features and normalised class weights."""
        normalised_features = F.normalize(features, dim=-1)
        normalised_weight = F.normalize(self.weight, dim=-1)
        return self.scale * F.linear(normalised_features, normalised_weight)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scale={self.scale.item():.3f}, learnable_scale={self.log_scale.requires_grad}"
        )


def _resolve_linear_head(model: nn.Module) -> tuple[str, nn.Linear]:
    """Find the model's linear classification head and the attribute naming it.

    Covers both dispatch paths without importing either: `ClipClassifier` exposes ``head``
    (a property with a setter), and Hugging Face image-classification models name theirs
    ``classifier``.
    """
    for attribute in ("head", "classifier"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, nn.Linear):
            return attribute, candidate

    found = {name: type(getattr(model, name)).__name__ for name in ("head", "classifier") if hasattr(model, name)}
    raise TypeError(
        f"Cannot swap in a cosine head on {type(model).__name__}: expected an `nn.Linear` at `.head` or "
        f"`.classifier`, found {found or 'neither attribute'}. Set head_type=linear for this model, or "
        f"expose its head as an nn.Linear."
    )


def replace_head_with_cosine(model: nn.Module, scale: float | None = None, learnable_scale: bool = True) -> CosineClassifier:
    """Swap the model's linear classification head for a :class:`CosineClassifier`.

    The replacement inherits the linear head's `in_features` / `out_features`, so the label
    space is unchanged and the backbone is untouched. Returns the new head so the caller can
    log or inspect it.
    """
    attribute, linear = _resolve_linear_head(model)
    head = CosineClassifier(
        in_features=linear.in_features,
        out_features=linear.out_features,
        scale=scale,
        learnable_scale=learnable_scale,
    )
    head.to(device=linear.weight.device, dtype=linear.weight.dtype)

    # `set_head` where the model exposes one — `ClipClassifier.head` is a read-only property
    # over a nested module, and a plain `setattr` would be silently wrong there: `nn.Module`
    # overrides `__setattr__`, so assigning a Module registers a dead `head.*` entry in
    # `state_dict()` instead of reaching the property, leaving the real head in place.
    setter = getattr(model, "set_head", None)
    if attribute == "head" and callable(setter):
        setter(head)
    else:
        setattr(model, attribute, head)

    installed = getattr(model, attribute)
    if installed is not head:
        raise RuntimeError(
            f"Failed to install the cosine head on {type(model).__name__}: `.{attribute}` still holds a "
            f"{type(installed).__name__}. The model needs a `set_head` method, or a plain attribute."
        )
    return head
