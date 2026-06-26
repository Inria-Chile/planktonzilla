"""
(c) Inria

Overrides:
    open_clip_train.main entry point — thin pre-launch hook wrapper.

Why:
    See planktonzilla/clip_train/__init__.py module docstring.
    The audit found nothing in open_clip_train/* needs to be re-implemented;
    we delegate to the upstream PyPI package's open_clip_train.main verbatim.
    The wrapper exists to apply planktonzilla-specific env setup (currently:
    tf32 per audit Q6) before invoking the upstream argparse + train loop.

Remove when:
    Same condition as the parent subpackage docstring.
"""

from __future__ import annotations


def _setup_planktonzilla_env() -> None:
    """Apply planktonzilla-specific pre-launch env adjustments.

    Per docs/open_clip_audit.md Q6 (ABSORB-02), the vendored open_clip_train
    set ``torch.backends.cuda.matmul.allow_tf32 = True`` at startup. This
    is the only behavioral non-default planktonzilla's pz_train pipeline
    didn't reproduce. cudnn settings already match HF Trainer defaults.

    For the CLIP-pretraining workflow, opt-in via the PLANKTONZILLA_TF32
    env var (defaults to off for backward compat). For the pz_train
    workflow, the field is exposed via cfg.tf32 in planktonzilla.train.

    Importing torch at function scope keeps module import time minimal
    when this module is used purely as a passthrough.
    """
    import os

    if os.environ.get("PLANKTONZILLA_TF32", "").lower() in {"1", "true", "yes"}:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True


def _patch_upstream() -> None:
    """Inject planktonzilla overrides into the open_clip / open_clip_train namespaces.

    Must run before upstream_main is called so that:
    - open_clip.transform.image_transform → our version (trivial_augment support).
      image_transform_v2 calls image_transform by name in open_clip.transform's
      module scope, so patching there is enough to cover create_model_and_transforms.
    - open_clip_train.main.evaluate / open_clip_train.train.evaluate → our version
      (classification metrics instead of retrieval R@k).
    """
    import open_clip.transform as _oc_transform
    import open_clip_train.main as _ocm
    import open_clip_train.train as _oct

    from planktonzilla.clip_train.train import evaluate

    _oct.evaluate = evaluate
    _ocm.evaluate = evaluate  # already imported at module level in _ocm, patch in place

    from planktonzilla.open_clip_ext.transform import image_transform

    _oc_transform.image_transform = image_transform


def main(args: list[str] | None = None) -> None:
    """Thin wrapper around open_clip_train.main from the installed PyPI package.

    Sets up planktonzilla-specific env (per ``_setup_planktonzilla_env``) and
    injects planktonzilla overrides (per ``_patch_upstream``) before delegating
    to ``open_clip_train.main.main``. All open_clip_train CLI flags are unchanged.

    Args:
        args: Optional CLI args list. When None, upstream parses sys.argv
            (the normal CLI / SLURM invocation pattern used by
            scripts/train_clip.sh).
    """
    _setup_planktonzilla_env()
    _patch_upstream()
    # Import at call time (not at module load) so `python -m planktonzilla.clip_train.main --help`
    # doesn't pay the full open_clip_train import cost just to print usage.
    from open_clip_train.main import main as upstream_main

    upstream_main(args)


if __name__ == "__main__":
    main()
