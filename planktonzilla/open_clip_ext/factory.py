"""
Overrides:
    open_clip.create_model_and_transforms, create_model,
    create_model_from_pretrained, load_checkpoint (pass-through
    wrappers with shape assertions).

Why:
    Forward-compat seam. docs/open_clip_audit.md row 4 found
    factory.py byte-identical between vendored and upstream v3.3.0;
    today these wrappers are pure passthroughs with shape assertions
    under `__debug__`. The seam exists so the next post-upstream-bump
    audit can inject project-specific behavior here without touching
    call sites.

    Per audit Q2 (no project-local model JSONs exist), we deliberately
    do NOT call open_clip.add_model_config() here. The
    model_configs/README.md documents how to wire it up if a future
    contributor adds a JSON.

Remove when:
    if and when planktonzilla is ready to drop the seam entirely.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

import open_clip


def create_model_and_transforms(
    model_name: str,
    pretrained: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, Callable, Callable]:
    """Wrap open_clip.create_model_and_transforms with a return-shape assertion."""
    result = open_clip.create_model_and_transforms(model_name, pretrained, **kwargs)
    assert __debug__ and (isinstance(result, tuple) and len(result) == 3), (
        f"open_clip.create_model_and_transforms return shape changed; "
        f"expected (model, preprocess_train, preprocess_val), got "
        f"{type(result).__name__} of length "
        f"{len(result) if hasattr(result, '__len__') else 'N/A'}"
    )
    return result


def create_model(
    model_name: str,
    pretrained: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Wrap open_clip.create_model with an nn.Module return-type assertion."""
    result = open_clip.create_model(model_name, pretrained, **kwargs)
    assert __debug__ and isinstance(result, nn.Module), (
        f"open_clip.create_model return type changed; expected nn.Module, got {type(result).__name__}"
    )
    return result


def create_model_from_pretrained(
    model_name: str,
    pretrained: str | None = None,
    return_transform: bool = True,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, Callable]:
    """Wrap open_clip.create_model_from_pretrained with a shape assertion.

    Note: return type depends on `return_transform` — when True (the
    upstream default), returns ``(model, preprocess)``; when False,
    returns ``model`` only.
    """
    result = open_clip.create_model_from_pretrained(model_name, pretrained, return_transform=return_transform, **kwargs)
    if return_transform:
        assert __debug__ and (isinstance(result, tuple) and len(result) == 2), (
            f"open_clip.create_model_from_pretrained(return_transform=True) "
            f"return shape changed; expected (model, preprocess), got "
            f"{type(result).__name__}"
        )
    else:
        assert __debug__ and isinstance(result, nn.Module), (
            f"open_clip.create_model_from_pretrained(return_transform=False) "
            f"return type changed; expected nn.Module, got {type(result).__name__}"
        )
    return result


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    weights_only: bool = True,
    device: str | torch.device = "cpu",
) -> Any:
    """Wrap open_clip.load_checkpoint pass-through.

    No assertion on return value: upstream returns a heterogeneous
    "incompatible_keys" dict whose shape varies by checkpoint era. The
    weights_only=True default is preserved verbatim — the
    weights_only=False fallback for legacy checkpoints (per PITFALLS
    P4) is Phase 3 SMOKE-02's job, NOT Phase 2's.
    """
    return open_clip.load_checkpoint(model, checkpoint_path, strict=strict, weights_only=weights_only, device=device)
