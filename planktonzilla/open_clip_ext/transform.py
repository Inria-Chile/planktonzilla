"""
(c) Inria

Overrides:
    open_clip.transform.image_transform, image_transform_v2, AugmentationCfg.

Why:
    Extends upstream AugmentationCfg with trivial_augment field and injects
    TrivialAugmentWide into the training pipeline when enabled.
    clip_train/main.py patches open_clip.transform.image_transform with this
    version at startup so create_model_and_transforms picks it up automatically.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple, Union

from open_clip.transform import PreprocessCfg
from open_clip.transform import image_transform as _upstream_image_transform

# Captured at module import time, BEFORE clip_train/main.py::_patch_upstream
# rebinds open_clip.transform.image_transform to the wrapper below. Re-importing
# the name from open_clip.transform after the patch would return THIS wrapper and
# recurse infinitely, so we keep a direct handle to the genuine upstream callable.


@dataclass
class AugmentationCfg:
    """Augmentation config mirroring upstream ``open_clip.transform.AugmentationCfg``.

    Adds the project-specific ``trivial_augment`` flag (consumed by ``image_transform``
    below to inject ``torchvision.transforms.TrivialAugmentWide`` into the training
    pipeline) alongside the upstream fields. ``color_jitter_prob`` and
    ``gray_scale_prob`` are carried for upstream compatibility so that an instance can
    be round-tripped back into the genuine upstream dataclass without losing fields.
    """

    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False
    trivial_augment: bool = False
    # upstream compat fields
    color_jitter_prob: Optional[float] = None
    gray_scale_prob: Optional[float] = None


def image_transform(
    image_size: int | tuple[int, int],
    is_train: bool,
    **kwargs: Any,
) -> Callable:
    """Like open_clip.transform.image_transform but with trivial_augment support.

    Strips the project-specific ``trivial_augment`` key from ``aug_cfg`` (whether
    passed as a dict or an ``AugmentationCfg``) before delegating to the genuine
    upstream callable captured at import time, then — for training transforms with
    ``trivial_augment`` enabled — inserts ``TrivialAugmentWide`` into the returned
    ``Compose`` just before the tensor-conversion step so it operates on PIL images.

    Returns:
        A callable transform. Asserts the result is callable before returning.
    """
    trivial_augment = False
    aug_cfg = kwargs.get("aug_cfg")

    if isinstance(aug_cfg, dict) and "trivial_augment" in aug_cfg:
        # strip our custom key before passing to upstream
        kwargs = {**kwargs, "aug_cfg": {k: v for k, v in aug_cfg.items() if k != "trivial_augment"}}
        trivial_augment = aug_cfg["trivial_augment"]
    elif isinstance(aug_cfg, AugmentationCfg):
        trivial_augment = aug_cfg.trivial_augment
        from open_clip.transform import AugmentationCfg as _UpstreamCfg

        upstream_fields = set(_UpstreamCfg.__dataclass_fields__)
        kwargs = {**kwargs, "aug_cfg": _UpstreamCfg(**{k: v for k, v in asdict(aug_cfg).items() if k in upstream_fields})}

    # Use the handle captured at import time; re-importing here would resolve to
    # this wrapper (after _patch_upstream rebinds the name) and recurse.
    result = _upstream_image_transform(image_size, is_train, **kwargs)

    if is_train and trivial_augment:
        from torchvision.transforms import Compose, TrivialAugmentWide

        if hasattr(result, "transforms"):
            transforms_list = list(result.transforms)
            # insert before MaybeToTensor so it operates on PIL images
            insert_pos = next(
                (i for i, t in enumerate(transforms_list) if type(t).__name__ in ("MaybeToTensor", "ToTensor")),
                len(transforms_list),
            )
            transforms_list.insert(insert_pos, TrivialAugmentWide())
            result = Compose(transforms_list)

    assert callable(result), f"image_transform returned non-callable: {type(result).__name__}"
    return result


def image_transform_v2(
    cfg: PreprocessCfg,
    is_train: bool,
    aug_cfg: dict[str, Any] | AugmentationCfg | None = None,
) -> Callable:
    """Wrap open_clip.transform.image_transform_v2 with a callable assertion."""
    from open_clip.transform import image_transform_v2 as _upstream

    result = _upstream(cfg, is_train, aug_cfg=aug_cfg)
    assert callable(result), (
        f"open_clip.transform.image_transform_v2 return type changed; expected callable, got {type(result).__name__}"
    )
    return result
