"""
Overrides:
    open_clip.transform.image_transform and image_transform_v2
    (pass-through wrappers).

Why:
    Forward-compat seam. docs/open_clip_audit.md Q3 found the entire
    open_clip transform pipeline is unreachable from pz_train:
    planktonzilla/clip_model.py:19,22 discards the returned transforms
    via `_, _, _`, and training-time transforms come from
    configs/dataset/*.yaml (torchvision.transforms.v2.Compose). So
    these wrappers may never actually fire in production — but the
    seam exists per the chosen layout for forward-compat.

Remove when:
    if and when planktonzilla is ready to drop the seam entirely, OR
    if a future override here graduates from pass-through to a real
    transform override (in which case update this docstring's
    Overrides/Why sections).
"""

from collections.abc import Callable
from typing import Any

from open_clip.transform import AugmentationCfg, PreprocessCfg

import open_clip


def image_transform(
    image_size: int | tuple[int, int],
    is_train: bool,
    **kwargs: Any,
) -> Callable:
    """Wrap open_clip.transform.image_transform with a callable assertion."""
    result = open_clip.image_transform(image_size, is_train, **kwargs)
    assert __debug__ and callable(result), (
        f"open_clip.image_transform return type changed; expected callable, got {type(result).__name__}"
    )
    return result


def image_transform_v2(
    cfg: PreprocessCfg,
    is_train: bool,
    aug_cfg: dict[str, Any] | AugmentationCfg | None = None,
) -> Callable:
    """Wrap open_clip.transform.image_transform_v2 with a callable assertion."""
    # NOTE: image_transform_v2 is in the open_clip.transform submodule
    # but is NOT re-exported from open_clip/__init__.py at v3.3.0.
    # We import from the submodule explicitly.
    from open_clip.transform import image_transform_v2 as _upstream

    result = _upstream(cfg, is_train, aug_cfg=aug_cfg)
    assert __debug__ and callable(result), (
        f"open_clip.transform.image_transform_v2 return type changed; expected callable, got {type(result).__name__}"
    )
    return result
