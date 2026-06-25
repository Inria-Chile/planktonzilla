"""
(c) Inria

SLURM contrastive CLIP pretraining path.

This subpackage is a thin seam over the upstream ``open_clip_train`` PyPI
package. The audit (``docs/open_clip_audit.md``) found that nothing in
``open_clip_train/*`` needs re-implementing, so the entry point in
``main.py`` delegates to the upstream argparse + training loop verbatim,
applying only planktonzilla-specific pre-launch env setup and a handful of
behavioral overrides (classification-metric ``evaluate``, our
``image_transform``). Invoked on the cluster via
``torchrun -m planktonzilla.clip_train.main`` (see ``scripts/train_clip.sh``).

Remove when upstream exposes the override hooks we currently patch in, making
the seam unnecessary.
"""
