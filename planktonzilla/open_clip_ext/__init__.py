"""
planktonzilla.open_clip_ext — wrap-and-delegate seam over upstream open-clip-torch.

Overrides:
    (re-export module — see submodule docstrings for per-symbol provenance)

Why:
    Single import surface for the override layer per EXT-02. Consumers
    import from `planktonzilla.open_clip_ext` rather than `open_clip`
    directly; this guarantees every call site is auditable from one
    directory.

Remove when:
    if and when planktonzilla is ready to drop the seam entirely
    (consumers willing to `import open_clip` directly).
"""

from planktonzilla.open_clip_ext._introspection import visual_tower_kind
from planktonzilla.open_clip_ext.factory import (
    create_model,
    create_model_and_transforms,
    create_model_from_pretrained,
    load_checkpoint,
)
from planktonzilla.open_clip_ext.transform import (
    image_transform,
    image_transform_v2,
)
from planktonzilla.open_clip_ext.visual import (
    TimmModel,
    VisionTransformer,
)

__all__ = [
    "TimmModel",
    "VisionTransformer",
    "create_model",
    "create_model_and_transforms",
    "create_model_from_pretrained",
    "image_transform",
    "image_transform_v2",
    "load_checkpoint",
    "visual_tower_kind",
]
