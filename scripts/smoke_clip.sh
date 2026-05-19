#!/usr/bin/env bash
# scripts/smoke_clip.sh — Phase 3 SMOKE gate.
#
# Reruns the canonical baseline through the post-refactor override layer (no PYTHONPATH
# injection — by design; the override layer routes through PyPI open-clip-torch==3.3.0
# via planktonzilla.open_clip_ext). Compares post-refactor metrics to docs/baseline.json
# against the BASELINE-02 tolerance band (val_acc ±0.05 abs, val_f1 ±0.05 abs,
# train_loss ±0.10 relative). Exits 0 iff all three bands met. Exits 1 with a structured
# deviation report otherwise.
#
# What this script does:
#   SMOKE-01: full canonical baseline rerun (~33 min CPU on macOS) → metric extraction →
#             tolerance-band comparison vs docs/baseline.json.
#   SMOKE-02: load pre-refactor HF safetensors checkpoint and run one forward pass.
#   SMOKE-05: exercise both ClipClassifier-path configs (ViT-B-16 + EVA02 timm-trunk)
#             for one training step each.
#
# No vendored-path injection — Phase 1's baseline command prepended the vendored
# src dir to the Python module search path so the in-tree copy of the library was
# resolved first. The Phase 3 smoke MUST NOT do that — the cutover's whole purpose
# is to route through the PyPI package the override layer wraps. The pre-flight
# check below asserts open_clip.__file__ contains '.venv'.
#
# Runnable from a clean checkout:
#   bash scripts/smoke_clip.sh
#
# Wall time: ~40 minutes total (~33 min SMOKE-01 + ~5-7 min SMOKE-02/05) on macOS CPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASELINE_JSON="docs/baseline.json"
POST_JSON="docs/baseline_post.json"
RUN_DIR="/tmp/pz_smoke_run"

echo "==> Pre-flight: verifying open_clip resolves to PyPI .venv (not vendored)"
uv run python -c "
import open_clip
assert '.venv' in open_clip.__file__, (
    f'open_clip resolves to {open_clip.__file__}; smoke would test the wrong code path. '
    f'Did PYTHONPATH leak in?'
)
print(f'open_clip {open_clip.__version__} from {open_clip.__file__}')
"

# Clean previous run output (idempotent)
rm -rf "$RUN_DIR"

echo "==> SMOKE-01: rerunning canonical baseline through the post-refactor override layer"
echo "    (this takes ~33 min on CPU; no PYTHONPATH injection — routes through PyPI open-clip-torch)"

# NOTE: invocation MIRRORS the Phase 1 baseline EXCEPT no vendored-tree prefix on the
# Python path (see header). (See .planning/phases/01-audit-baseline/01-RESEARCH.md
# Pattern 2 for provenance of each override.)
# save_strategy=steps save_steps=100 differs from CONTEXT.md's ideal save_strategy=no
# because the smoke needs trainer_state.json written to disk for extract_baseline.py
# to read (per 03-RESEARCH.md Pattern 4 note 1).
uv run pz_train \
  model=vit-base-clip-224-openai \
  dataset=lensless \
  training_arguments=test_minirun \
  ++training_arguments.max_steps=100 \
  ++training_arguments.seed=42 \
  ++training_arguments.data_seed=42 \
  ++training_arguments.eval_strategy=steps \
  ++training_arguments.eval_steps=100 \
  ++training_arguments.logging_strategy=steps \
  ++training_arguments.logging_steps=10 \
  ++training_arguments.save_strategy=steps \
  ++training_arguments.save_steps=100 \
  ++training_arguments.report_to=none \
  ++training_arguments.do_train=true \
  ++training_arguments.do_eval=true \
  ++training_arguments.per_device_train_batch_size=16 \
  ++training_arguments.per_device_eval_batch_size=16 \
  ++seed=42 \
  ++model_push_to_hub=false \
  ++tracking.use_wandb=false \
  ++tracking.use_mlflow=false \
  ++tracking.use_trackio=false \
  ++extras.print_config=false \
  ++extras.enforce_tags=false \
  hydra.run.dir="$RUN_DIR"

echo "==> SMOKE-01: extracting post-refactor metrics"
# Always pass --state-path explicitly (per 03-RESEARCH.md Common Pitfall 5);
# the default in extract_baseline.py points at Phase 1's /tmp/pz_baseline_run path.
uv run python scripts/extract_baseline.py \
  --state-path "$RUN_DIR/checkpoint-100/trainer_state.json" \
  --output "$POST_JSON" \
  --open-clip-version "PyPI open-clip-torch==3.3.0 (post-refactor)" \
  --hardware-override "fp16=false (CPU run, macOS, no CUDA)"

echo "==> SMOKE-01: comparing post-refactor vs baseline against tolerance band"
uv run python -c "
import json, sys

base = json.loads(open('$BASELINE_JSON').read())
post = json.loads(open('$POST_JSON').read())

val_acc_diff = abs(post['val_acc'] - base['val_acc'])
val_f1_diff = abs(post['val_f1'] - base['val_f1'])
train_loss_rel = abs(post['train_loss'] - base['train_loss']) / abs(base['train_loss'])

# Tolerance band (BASELINE-02, locked in docs/open_clip_audit.md):
#   val_acc:    ±0.05 (5 absolute points on the 0..1 scale)
#   val_f1:     ±0.05 (5 absolute points)
#   train_loss: ±0.10 (10% relative)
band_val_acc, band_val_f1, band_loss = 0.05, 0.05, 0.10

deviations = []
if val_acc_diff > band_val_acc:
    deviations.append(f'val_acc deviation {val_acc_diff:.4f} exceeds band ±{band_val_acc}')
if val_f1_diff > band_val_f1:
    deviations.append(f'val_f1 deviation {val_f1_diff:.4f} exceeds band ±{band_val_f1}')
if train_loss_rel > band_loss:
    deviations.append(f'train_loss relative deviation {train_loss_rel:.4f} exceeds band ±{band_loss}')

# Print diagnostic surface BEFORE potential sys.exit(1) so the executor's captured
# stdout contains the deviation breakdown even on failure (Common Pitfall 6).
print('baseline:  ', json.dumps({k: base[k] for k in ('val_acc','val_f1','train_loss')}, indent=2))
print('post:      ', json.dumps({k: post[k] for k in ('val_acc','val_f1','train_loss')}, indent=2))
print(f'val_acc diff:       {val_acc_diff:.4f} (band ±{band_val_acc})')
print(f'val_f1 diff:        {val_f1_diff:.4f} (band ±{band_val_f1})')
print(f'train_loss rel:     {train_loss_rel:.4f} (band ±{band_loss})')

if deviations:
    print('SMOKE-01 FAIL:')
    for d in deviations:
        print(f'  - {d}')
    sys.exit(1)
print('SMOKE-01 PASS: all three bands met.')
"

echo "==> SMOKE-02: loading pre-refactor HF checkpoint and running one forward pass"
uv run python -c "
import torch
from transformers import AutoModelForImageClassification

REPO = 'project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt'
print(f'Loading {REPO} via AutoModelForImageClassification.from_pretrained...')
model = AutoModelForImageClassification.from_pretrained(REPO, trust_remote_code=False)
model.eval()

# Run one forward pass on a zero-tensor of the right shape (sanity, not metric).
# Image size: 224 from configs/model/vit-base-clip-224-openai.yaml.
pixel_values = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out = model(pixel_values=pixel_values)

assert hasattr(out, 'logits'), f'Output missing .logits attribute: {type(out)}'
print(f'SMOKE-02 PASS: forward pass returned logits of shape {tuple(out.logits.shape)}')
"

echo "==> SMOKE-05: exercising all configs/model/*clip*.yaml for >= 1 training step"
# Configs that go through ClipClassifier (per docs/open_clip_audit.md Q4):
#   - vit-base-clip-224-openai            (pure-ViT path)
#   - eva02-large-clip-224-2b-s4b-b131k   (timm-trunk path)
# SKIPPED on purpose:
#   - default_clip.yaml  (defaults file with _args_=[????], never instantiated standalone)
#   - timm-vit-base-16-clip-openai.yaml  (inherits default.yaml -> AutoModelForImageClassification
#     path, NOT ClipClassifier; does not exercise the open_clip externalization)
for cfg_name in vit-base-clip-224-openai eva02-large-clip-224-2b-s4b-b131k; do
    echo "    -- $cfg_name --"
    SMOKE_RUN_DIR="/tmp/pz_smoke05_${cfg_name//[^a-zA-Z0-9]/_}"
    rm -rf "$SMOKE_RUN_DIR"
    uv run pz_train \
        model="$cfg_name" \
        dataset=lensless \
        training_arguments=test_minirun \
        ++training_arguments.max_steps=1 \
        ++training_arguments.eval_strategy=no \
        ++training_arguments.save_strategy=no \
        ++training_arguments.report_to=none \
        ++training_arguments.per_device_train_batch_size=2 \
        ++training_arguments.do_eval=false \
        ++seed=42 \
        ++model_push_to_hub=false \
        ++tracking.use_wandb=false \
        ++tracking.use_mlflow=false \
        ++tracking.use_trackio=false \
        ++extras.print_config=false \
        ++extras.enforce_tags=false \
        hydra.run.dir="$SMOKE_RUN_DIR"
    echo "    PASS: $cfg_name ran one step without raising"
done

echo "==> ALL SMOKE GATES PASSED."
