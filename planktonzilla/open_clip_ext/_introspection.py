"""
(c) Inria

Introspection helper for the open_clip visual-tower seam.

Overrides:
    The former bare-except ViT-vs-timm discriminator in
    planktonzilla/clip_model.py (replaced via FIX-01; the helper is now
    wired in — see clip_model.ClipClassifier.__init__).

Why:
    Centralizes ViT-vs-timm dispatch in one explicit isinstance check.
    Per FEATURES D-03 and docs/open_clip_audit.md row 14
    (project-specific-as-override). Raises TypeError on unknown types
    instead of silently routing into the wrong branch via bare except.

Remove when:
    never (replaces an anti-pattern; permanent infrastructure).
"""

from typing import Literal

import torch.nn as nn

from planktonzilla.open_clip_ext.visual import TimmModel, VisionTransformer


def visual_tower_kind(visual: nn.Module) -> Literal["vit", "timm"]:
    """Return which open_clip visual-tower variant `visual` is.

    Branches on isinstance against the public open_clip class hierarchy.
    Update this function (and only this function) if open_clip ever
    introduces a third visual-tower kind.

    Args:
        visual: An ``open_clip`` visual tower (typically obtained as
            ``clip_model.visual`` from ``create_model_and_transforms``).

    Returns:
        ``"vit"`` for ``open_clip.transformer.VisionTransformer``
        instances; ``"timm"`` for ``open_clip.timm_model.TimmModel``
        instances.

    Raises:
        TypeError: If ``visual`` is neither a ``VisionTransformer`` nor a
            ``TimmModel``. The former bare ``except:`` discriminator in
            ``clip_model.py`` silently misclassified this case; we raise
            explicitly so the failure is diagnosable.
    """
    if isinstance(visual, VisionTransformer):
        return "vit"
    if isinstance(visual, TimmModel):
        return "timm"
    raise TypeError(
        f"Unknown open_clip visual tower: {type(visual).__name__}. "
        f"Expected VisionTransformer or TimmModel. If a new upstream "
        f"visual-tower kind has been introduced, add an isinstance "
        f"branch above and re-export the class from "
        f"planktonzilla.open_clip_ext.visual."
    )
