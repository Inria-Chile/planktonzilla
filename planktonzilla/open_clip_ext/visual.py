"""
(c) Inria

Overrides:
    (re-export of upstream open_clip.transformer.VisionTransformer and
    open_clip.timm_model.TimmModel — no behavioral override)

Why:
    Single import surface for the classes that
    _introspection.visual_tower_kind discriminates on. If a future
    upstream bump adds a third visual-tower kind, only this file
    changes.

Remove when:
    if and when planktonzilla is ready to drop the seam entirely.
"""

from open_clip.timm_model import TimmModel
from open_clip.transformer import VisionTransformer

__all__ = ["TimmModel", "VisionTransformer"]
