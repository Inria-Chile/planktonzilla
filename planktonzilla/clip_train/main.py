"""
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


def main(args: list[str] | None = None) -> None:
    """Thin wrapper around open_clip_train.main from the installed PyPI package.

    Sets up planktonzilla-specific env (per ``_setup_planktonzilla_env``) and
    delegates to ``open_clip_train.main.main``. All open_clip_train CLI
    flags are unchanged — see ``python -m planktonzilla.clip_train.main --help``.

    Args:
        args: Optional CLI args list. When None, upstream parses sys.argv
            (the normal CLI / SLURM invocation pattern used by
            scripts/train_clip.sh).
    """
    _setup_planktonzilla_env()
    # Import at call time (not at module load) so `python -m planktonzilla.clip_train.main --help`
    # doesn't pay the full open_clip_train import cost just to print usage.
    from open_clip_train.main import main as upstream_main

    upstream_main(args)


if __name__ == "__main__":
    main()
