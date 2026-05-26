"""
planktonzilla.clip_train — thin wrapper over upstream open_clip_train.

Overrides:
    Nothing today. This subpackage is a thin wrapper that delegates to
    `open_clip_train.main` from the installed `open-clip-torch` PyPI
    package, with planktonzilla-specific env setup applied before
    delegation.

Why:
    docs/open_clip_audit.md Q1 found all 3 vendored open_clip_train/*.py
    modifications are `discardable` (driver-loop convenience tweaks
    unreachable from the planktonzilla pz_train HF-Trainer pipeline).
    But scripts/train_clip.sh — the standalone SLURM-launched CLIP
    contrastive pretraining workflow — DID use the vendored
    `python -m open_clip_train.main`. After Phase 5 deletes the
    vendored tree, this wrapper preserves that capability by routing
    the same CLI through the upstream PyPI package, with planktonzilla
    env hooks (e.g., tf32 per audit Q6 / ABSORB-02) injected at startup.

    HF `Trainer` remains the loop for the main pz_train workflow. This
    subpackage exists solely for the orthogonal CLIP-pretraining
    workflow that exists outside HF Trainer's scope.

Remove when:
    If/when CLIP contrastive pretraining is removed from the project's
    scope, this subpackage can be deleted along with scripts/train_clip.sh.
"""

from planktonzilla.clip_train.main import main

__all__ = ["main"]
