#!/bin/bash
# scripts/smoke_clip_ci.sh — CPU-fast CI subset of scripts/smoke_clip.sh.
#
# Covers two regression surfaces:
#   - SMOKE-02: load pre-refactor HF Hub checkpoint via open_clip_ext and run one
#     forward pass. Exercises the open_clip externalization (v1.0 milestone).
#   - SMOKE-05 (eva02 only): one-step pz_train with the EVA02-L-14 CLIP config.
#     Exercises WIRE-02 (Phase 06, v1.1) — eva02 ClipClassifier instantiation
#     and kw-only repo_path signature.
#
# Skipped vs full scripts/smoke_clip.sh:
#   - SMOKE-01 (baseline-band comparison run, ~33 min CPU) — too slow for CI minutes.
#   - SMOKE-03 / SMOKE-04 (vendored-vs-override fixtures) — already skipped post-DEL-01.
#   - SMOKE-05 for vit-base-clip-224-openai — eva02 is the wiring gate; vit-base is
#     redundant for the CI cost budget.
#
# Target runtime: ≤ 8 min on a standard GitHub Actions runner.

set -euo pipefail

echo "==> [CI] SMOKE-02: loading pre-refactor HF checkpoint and running one forward pass"
uv run python -c "
import torch
from planktonzilla import open_clip_ext

REPO = 'hf-hub:project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt'
print(f'Loading {REPO} via open_clip_ext.create_model_from_pretrained (override-layer path)...')
model, _ = open_clip_ext.create_model_from_pretrained(REPO)
model.eval()

pixel_values = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    image_features = model.encode_image(pixel_values)

assert image_features.shape[-1] > 0, f'Got empty image_features: shape={image_features.shape}'
assert not image_features.isnan().any(), 'image_features contains NaN'
print(f'SMOKE-02 PASS: forward pass returned image_features of shape {tuple(image_features.shape)}')
"

echo "==> [CI] SMOKE-05 (eva02): one-step pz_train through ClipClassifier (timm-trunk path, WIRE-02 gate)"
SMOKE_RUN_DIR="/tmp/pz_ci_smoke05_eva02"
rm -rf "$SMOKE_RUN_DIR"
uv run pz_train \
    model=eva02-large-clip-224-2b-s4b-b131k \
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
echo "    PASS: eva02 ran one step without raising"

echo "==> [CI] ALL FAST SMOKE GATES PASSED."
