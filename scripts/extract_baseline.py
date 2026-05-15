"""
(c) Inria

scripts/extract_baseline.py — one-shot utility that reads a HuggingFace `Trainer`
`trainer_state.json` and writes the pre-refactor baseline metrics to `docs/baseline.json`.

This script is NOT part of the planktonzilla package. It mirrors the standalone-shell-script
pattern of `scripts/train_clip.sh` and is committed so the baseline-extraction step is
reproducible. Stdlib-only on purpose: it must run on a clean Python environment, before
Phase 2 declares any new project dependency (`open-clip-torch`).

Usage (after the canonical `pz_train` baseline run from Phase 1, Plan 02, Task 1):

    python3 scripts/extract_baseline.py

Or, if the run used the CPU hardware fallback:

    python3 scripts/extract_baseline.py --hardware-override "fp16=false (CPU run)"

The defaults are locked to the values in `.planning/phases/01-audit-baseline/01-CONTEXT.md`
(model `vit-base-clip-224-openai`, dataset `project-oceania/lensless`, `seed=42`,
`step_K=100`). Override via CLI flags only if you intentionally re-ran the baseline with
a different config.
"""

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pre-refactor baseline metrics from HF Trainer's trainer_state.json.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("/tmp/pz_baseline_run/checkpoint-100/trainer_state.json"),
        help="Path to HF Trainer's trainer_state.json (output of save_strategy=steps save_steps=K).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/baseline.json"),
        help="Path to write the baseline JSON.",
    )
    parser.add_argument(
        "--model",
        default="vit-base-clip-224-openai",
        help="Model config name (CONTEXT.md locked).",
    )
    parser.add_argument(
        "--dataset",
        default="project-oceania/lensless",
        help="Dataset name (CONTEXT.md locked).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the run (CONTEXT.md locked).",
    )
    parser.add_argument(
        "--step-k",
        type=int,
        default=100,
        help="Step K at which metrics are captured (CONTEXT.md locked).",
    )
    parser.add_argument(
        "--open-clip-version",
        default="vendored 4.0.0.dev0 (pre-refactor)",
        help="open_clip version label recorded in baseline.json.",
    )
    parser.add_argument(
        "--extraction-method",
        default="trainer_state.json via save_strategy=steps save_steps=100",
        help=(
            "Documents the deviation from CONTEXT.md's ideal save_strategy=no — see Plan 01-02 "
            "Task 1's note. The single ~330MB checkpoint write to /tmp is the cheapest path to a "
            "machine-readable log_history."
        ),
    )
    parser.add_argument(
        "--hardware-override",
        default="",
        help='Optional. e.g. "fp16=false (CPU run)" if Task 1 used the GPU-fallback path.',
    )
    return parser.parse_args()


def _fail(message: str) -> None:
    """Print a clear error to stderr and exit 1 without writing baseline.json."""
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = _parse_args()

    if not args.state_path.exists():
        _fail(
            f"trainer_state.json not found at {args.state_path}. "
            "Run the canonical baseline first (see Plan 01-02 Task 1)."
        )

    try:
        state = json.loads(args.state_path.read_text())
    except json.JSONDecodeError as e:
        _fail(f"trainer_state.json at {args.state_path} is not valid JSON: {e}")

    log_history = state.get("log_history")
    if not isinstance(log_history, list):
        _fail("trainer_state.json has no log_history list (expected from HF Trainer).")

    eval_entries = [e for e in log_history if "eval_loss" in e]
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]

    if len(eval_entries) < 1:
        _fail(
            "no eval entries in log_history; eval_strategy=steps eval_steps=K must be set on "
            "the run, got 0 eval entries."
        )
    if len(train_entries) < 1:
        _fail("no train entries in log_history; logging_strategy=steps logging_steps=K must be set on the run.")

    train_loss = train_entries[-1]["loss"]
    val_acc = eval_entries[-1].get("eval_accuracy")
    val_f1 = eval_entries[-1].get("eval_f1")

    if val_acc is None:
        _fail("last eval entry is missing eval_accuracy (compute_metrics returned no 'accuracy' key?).")
    if val_f1 is None:
        _fail("last eval entry is missing eval_f1 (compute_metrics returned no 'f1' key?).")

    # Sanity checks (RESEARCH Pitfall 5: do not write baseline.json if any check fails).
    if not (train_loss == train_loss):  # NaN check
        _fail(f"train_loss is NaN: {train_loss!r}")
    if not (0.0 < train_loss < 100.0):
        _fail(f"train_loss out of plausible range (0, 100): {train_loss!r}")
    if not (0.0 <= val_acc <= 1.0):
        _fail(f"val_acc out of [0.0, 1.0]: {val_acc!r}")
    if not (0.0 <= val_f1 <= 1.0):
        _fail(f"val_f1 out of [0.0, 1.0]: {val_f1!r}")

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "step_K": args.step_k,
        "open_clip_version": args.open_clip_version,
        "extraction_method": args.extraction_method,
        "hardware_override": args.hardware_override,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "tolerance_band": {
            "val_acc": "±5 absolute points",
            "val_f1": "±5 absolute points",
            "train_loss": "±10% relative",
        },
        "tolerance_band_authority": (
            "docs/open_clip_audit.md (BASELINE-02; gate for SMOKE-01 in Phase 3)"
        ),
    }

    # ensure_ascii=False preserves the literal ± (U+00B1) in the tolerance band so the
    # verification regex `±5` / `±10%` matches in baseline.json (the alternative would be
    # `±5` which fails the plan's grep-style acceptance check).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
