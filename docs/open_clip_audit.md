# open_clip Externalization — Vendored vs Upstream Audit

**Audit baseline:** `open-clip-torch v3.3.0` (released 2026-02-27, first stable release with explicit `transformers>=5` support; per `.planning/research/STACK.md`).

**Vendored copy:** `open_clip/` directory at branch `luis/open-clip-port`, identifies as `__version__ = "4.0.0.dev0"` (floating dev marker — NOT a tagged release; see `.planning/phases/01-audit-baseline/01-CONTEXT.md` for rationale).

**Diff command (reproducible):**

```bash
git clone --depth 1 --branch v3.3.0 \
  https://github.com/mlfoundations/open_clip.git /tmp/open_clip_upstream

diff -ruN \
  /tmp/open_clip_upstream/src/open_clip/ \
  open_clip/src/open_clip/ \
  > /tmp/vendored_vs_v3.3.0.open_clip.diff

diff -ruN \
  /tmp/open_clip_upstream/src/open_clip_train/ \
  open_clip/src/open_clip_train/ \
  > /tmp/vendored_vs_v3.3.0.open_clip_train.diff

diff -q \
  /tmp/open_clip_upstream/src/open_clip/model_configs/ \
  open_clip/src/open_clip/model_configs/ \
  > /tmp/vendored_vs_v3.3.0.model_configs.diff
```

**Diff sizes** (recorded for posterity — `wc -l` on the three diff files at audit time):

| Diff file | Lines |
|-----------|-------|
| `/tmp/vendored_vs_v3.3.0.open_clip.diff` | 49 |
| `/tmp/vendored_vs_v3.3.0.open_clip_train.diff` | 144 |
| `/tmp/vendored_vs_v3.3.0.model_configs.diff` | 0 |
| **Total** | **193** |

**Audit date:** 2026-05-14
**Auditor:** Claude (planktonzilla open_clip externalization milestone, Phase 1, plan 01-01).

> **Headline finding:** Against the chosen tagged baseline `open-clip-torch v3.3.0`, the vendored fork is **dramatically smaller than CONCERNS.md #16 anticipated**. CONCERNS.md cites `33 files changed, +3,419 / −2,652` — but that delta was measured against upstream `main` (which identifies as `4.0.0.dev0`, the same floating dev marker the vendored copy uses). Against the **tagged release v3.3.0**, the actual delta is **6 files / 193 lines / 3 substantive behavioral changes** (vendored `transform.py`'s opt-in `TrivialAugmentWide`; vendored `transformer.py`'s `forward_intermediates` `output_fmt` default flip; vendored `open_clip_train/train.py`'s classification-style `evaluate()` rewrite). Two of those three are unreachable from the planktonzilla `pz_train` pipeline. The externalization milestone is therefore much closer to "drop the vendored tree and pin upstream v3.3.0" than to "carefully port a forked codebase." See the audit table below.

---

## Baseline Run Configuration (BASELINE-01 cross-link)

The pre-refactor metrics in `docs/baseline.json` (produced by plan `01-02-PLAN.md`) were captured from this exact `pz_train` invocation, using the *vendored* `open_clip/` tree:

| Field | Value |
|-------|-------|
| Model config | `configs/model/vit-base-clip-224-openai.yaml` (pure-ViT, `_args_: [ViT-B-16, openai, null]`, `num_features: 768`, `img_size: 224`) |
| Dataset | `project-oceania/lensless` (via `configs/dataset/lensless.yaml`) |
| Seed | `42` (both `cfg.seed` top-level and `++training_arguments.seed=42`) |
| Step K | `100` (`max_steps=100` — overrides `test_minirun.yaml:7` default of 2) |
| Per-device batch size | `16` train + `16` eval |
| Eval strategy | `steps`, `eval_steps=100` (force eval at exactly step 100) |
| Logging strategy | `steps`, `logging_steps=10` (capture intermediate `train_loss`) |
| Save strategy | `steps`, `save_steps=100` (writes `trainer_state.json` for machine-readable metrics extraction; checkpoint goes to `/tmp` and is NOT committed) |
| open_clip version | vendored `4.0.0.dev0` (pre-refactor) |
| Hardware | CPU run on macOS (no NVIDIA GPU); `fp16=false`. Recorded in `docs/baseline.json` `hardware_override` field. |
| Wall time | ~33 min for 100 steps on CPU (vs CONTEXT.md's "~minutes on a single GPU" estimate). Phase 3 SMOKE-01 should re-baseline if comparison hardware differs from CPU. |

**Full reproducibility command** (verbatim from `.planning/phases/01-audit-baseline/01-RESEARCH.md` Pattern 2):

```bash
# From repo root, with vendored open_clip on PYTHONPATH (since pyproject.toml does NOT
# declare open-clip-torch yet — the vendored tree is the only open_clip available).
PYTHONPATH=$PWD/open_clip/src:$PYTHONPATH \
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
  ++training_arguments.save_strategy=no \
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
  hydra.run.dir=/tmp/pz_baseline_run
```

> **Note:** Hardware and wall time are filled in by plan `01-02-PLAN.md` (Baseline), which actually runs this command. Plan 01-01 (this audit) records only the canonical command.

---

## Tolerance Band (BASELINE-02)

**Authoritative gate for SMOKE-01 in Phase 3.** SMOKE-01 passes only when the post-refactor metrics fall within all three bands compared to `docs/baseline.json`:

| Metric | Band |
|--------|------|
| `val_acc` | within ±5 absolute points |
| `val_f1` (macro) | within ±5 absolute points |
| `train_loss` | within ±10% relative |

**Rationale:** Step K=100 is intentionally early so that the baseline run completes in minutes (the milestone's bottleneck is review time, not training compute). Numbers at step 100 will be noisy — the band is generous to absorb that noise while still flagging order-of-magnitude regressions. The band may be tightened in a follow-up release-hardening milestone if SMOKE-01 produces too many false positives. **Do not** tighten this band in Phase 1 or Phase 3 without rerunning the baseline at the new step K and re-checking variance.

---

## Pre-refactor Checkpoint Provenance

**SMOKE-02 source** (Phase 3, NOT this phase): `huggingface.co/project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt`.

The checkpoint was pushed via `trainer.push_to_hub` (`planktonzilla/train.py:194-206` wires the `TrainingArguments.hub_model_id` and `:290-293` invokes the post-training push). Therefore it is in **HuggingFace `safetensors` + `config.json` format**, NOT a raw `open_clip` `state_dict.bin`.

SMOKE-02 must use one of:

```python
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id="project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt")
```

or

```python
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained(
    "project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt",
    trust_remote_code=False,  # default; just being explicit
)
```

Phase 1 does NOT fetch this checkpoint — provenance is taken on the user's authority per `.planning/phases/01-audit-baseline/01-CONTEXT.md`. Phase 3 implements the load + forward-pass smoke check.

---

## Eight Open Questions (AUDIT-03)

### Q1: Which `open_clip_train/*` modifications affect training behavior (must absorb) vs. are mere convenience tweaks (drop)?

**Investigation:** Walked `/tmp/vendored_vs_v3.3.0.open_clip_train.diff` (144 lines). Only **3 of 8** `open_clip_train/*.py` files differ from upstream v3.3.0:

- `main.py` — 5-line change in checkpoint-deletion logic (rotates by `save_frequency` instead of always deleting epoch-1).
- `params.py` — 1-line change to default `--logs` argument: `./logs/` → `../../logs/train_clip/` (relative-path tweak for the SLURM script's working directory).
- `train.py` — substantial rewrite of the `evaluate()` function (lines 264-308): replaces the upstream `get_clip_metrics(image_features, text_features, logit_scale)` retrieval-style metric with a **classification-style** `precision/recall/f1` against `txt_unique_tokens`. Adds `from sklearn.metrics import precision_score, recall_score, f1_score`.

`{data.py, params.py, profiler.py, scheduler.py, zero_shot.py}` (other than the 1-line `params.py` `--logs` change) and `__init__.py`, `distributed.py`, `file_utils.py`, `logger.py`, `precision.py` are **byte-identical to upstream v3.3.0**.

**Finding:** All three changes are **driver-loop convenience tweaks**, NOT loss/data/scheduler behavior the production `pz_train` pipeline depends on:

- `main.py` checkpoint rotation: only relevant if running `python -m open_clip_train.main ...` (the standalone CLI). `pz_train` uses HF Trainer's `save_total_limit`/`save_strategy` for checkpoint rotation — entirely independent code path.
- `params.py` `--logs` default: only affects the `python -m open_clip_train.main` CLI's default log dir. Hydra-composed `pz_train` derives output dirs from `configs/hydra/default.yaml`, never reads `args.logs`.
- `train.py` `evaluate()` rewrite: only invoked from `open_clip_train.main.main()` → `evaluate(model, data, ...)` for **CLIP contrastive pretraining**. `pz_train` uses HF Trainer's eval loop with `compute_metrics()` (`planktonzilla/train.py:99-112`) which already computes accuracy/f1/precision/recall via sklearn — the same metric semantics, computed via a totally different code path. The vendored eval rewrite is dead code in the planktonzilla pipeline.

The three modifications were made when `open_clip_train.main` was being used directly via `scripts/train_clip.sh`. Per CLAUDE.md and ABSORB-01, that pretraining workflow is **out of scope** for the externalization milestone — `pz_train` (HF Trainer-based) is the only training path that matters.

**Action:** Categorize all 3 file-level changes as `discardable` in the audit table (rows 5, 6, 7). If/when CLIP contrastive pretraining is re-introduced via `planktonzilla/clip_train/`, the `train.py` evaluate rewrite is the only piece worth porting (and it should be re-implemented inside an HF-Trainer-compatible callback rather than carried verbatim — per PITFALLS P5: never carry the upstream training loop forward).

---

### Q2: Does `add_model_config(path)` cover all project-local CLIP architectures, or do any require a custom model class?

**Investigation:** Ran `diff -q /tmp/open_clip_upstream/src/open_clip/model_configs/ open_clip/src/open_clip/model_configs/` and inspected the result (`/tmp/vendored_vs_v3.3.0.model_configs.diff`).

Output: **0 bytes — the model_configs directory is byte-identical to upstream v3.3.0.** No project-local JSON configs were added; no upstream JSON configs were modified or removed.

Cross-checked: no project-specific Python model class exists in vendored `open_clip/src/open_clip/factory.py` or `model.py` (those files are byte-identical to upstream — they do not appear in `/tmp/vendored_vs_v3.3.0.open_clip.diff`).

**Finding:** All currently-used CLIP architectures (`ViT-B-16/openai`, `EVA02-L-14/merged2b_s4b_b131k`) are stock upstream model definitions. The override layer in Phase 2 will therefore **not need a custom factory** for model-config registration. `open_clip.add_model_config(path)` is unneeded for now (no project-local JSON configs to register), but the seam should still exist in `planktonzilla/open_clip_ext/factory.py` for forward-compatibility per `.planning/research/FEATURES.md` TS-07.

**Action:** Captured as audit row 9 (`already-merged-upstream`). Phase 2 still creates a wrap-and-delegate `factory.py` per FEATURES TS-06, but it does not need to call `add_model_config()` at this time.

---

### Q3: What does the vendored `transform.py` actually change? (antialias default, interpolation mode, v1 vs v2 path)

**Investigation:** Per RESEARCH.md command:

```bash
grep -nE "(antialias|interpolation|Resize|Normalize|image_transform)" \
  open_clip/src/open_clip/transform.py
grep -nE "(antialias|interpolation|Resize|Normalize|image_transform)" \
  /tmp/open_clip_upstream/src/open_clip/transform.py
```

Both greps return identical kwarg surfaces: `interpolation='bicubic'` default, `InterpolationMode.BICUBIC` for the resize path, `Normalize(mean=mean, std=std)` for normalization. Neither vendored nor upstream sets the `antialias` kwarg explicitly anywhere — so on `torchvision>=0.17` both default to `True` for tensor inputs (PITFALLS P7 risk does NOT apply here — neither side has a stale antialias default).

The actual diff in vendored `transform.py` (per `/tmp/vendored_vs_v3.3.0.open_clip.diff` lines 1-30) is:

1. Adds `TrivialAugmentWide` to the `torchvision.transforms` import line.
2. Adds a new `trivial_augment: bool = False` field to the `AugmentationCfg` dataclass.
3. Inserts a 4-line block that appends `TrivialAugmentWide()` to `train_transform` if `aug_cfg.trivial_augment` is set.

Critically: this is **opt-in via the `trivial_augment` field** which defaults to `False`. With the field unset (its default), the vendored `transform.py` is behaviorally identical to upstream.

**Cross-reference to the planktonzilla call site:** `planktonzilla/clip_model.py:19,22` calls `open_clip.create_model_and_transforms(...)` and assigns the returned transforms to `_, _` — **the open_clip transform pipeline is discarded entirely**. Training-time transforms come from `configs/dataset/*.yaml` (e.g., `lensless.yaml` instantiates `torchvision.transforms.v2.Compose([...])` directly). The vendored `TrivialAugmentWide` addition is **completely unreachable** from `pz_train`.

**Finding:** The vendored `transform.py` change is opt-in, defaults to off, and the entire `open_clip` transform pipeline is unreachable from the planktonzilla training path because `ClipClassifier` discards the returned transforms.

**Action:** Categorized as audit row 2 (`discardable`). The vendored `TrivialAugmentWide` addition can be safely dropped because (a) it is unused by `ClipClassifier` and (b) it would not be a hard-to-restore feature anyway — `torchvision.transforms.v2.TrivialAugmentWide` can be added to `configs/augmentation/*.yaml` directly if ever needed, with no override-layer code.

---

### Q4: Are current CLIP configs pure-ViT or do any use timm trunks?

**Investigation:** Ran:

```bash
for f in configs/model/*clip*.yaml; do
  echo "=== $f ==="
  grep -A 4 "_args_:" "$f"
done
```

Results (verbatim from the four `*clip*.yaml` files):

- `configs/model/default_clip.yaml` — `_args_: [????]` (placeholder; `_target_: planktonzilla.clip_model.ClipClassifier`). Defaults file, never instantiated as-is.
- `configs/model/vit-base-clip-224-openai.yaml` — `_args_: [ViT-B-16, openai, null]`. **Pure-ViT path** through `ClipClassifier` → `open_clip.transformer.VisionTransformer`. Has `num_features: 768`, `img_size: 224`. Inherits `_target_: planktonzilla.clip_model.ClipClassifier` from `default_clip.yaml`.
- `configs/model/eva02-large-clip-224-2b-s4b-b131k.yaml` — `_args_: [EVA02-L-14, merged2b_s4b_b131k]`. **Timm-trunk path** through `ClipClassifier` → `open_clip.timm_model.TimmModel`. Inherits `_target_: planktonzilla.clip_model.ClipClassifier` from `default_clip.yaml`. **Note:** `num_features` is NOT set here — `default_clip.yaml` requires it (`num_features: ???`); this is the CONCERNS.md #11 config breakage.
- `configs/model/timm-vit-base-16-clip-openai.yaml` — `_args_: [timm/vit_base_patch16_clip_224.openai]`. Inherits from `default.yaml` (NOT `default_clip.yaml`), so `_target_` is `planktonzilla.dataset.HuggingFaceModelLoader` / `transformers.AutoModelForImageClassification`. **Does NOT go through `ClipClassifier` at all** — this is the HF/timm path that uses `transformers.AutoModelForImageClassification.from_pretrained("timm/vit_base_patch16_clip_224.openai")`.

Inventory matches the RESEARCH.md "Eight Open Questions" Q4 expected inventory verbatim — no corrections needed.

**Finding:** Through `ClipClassifier` we have exactly 2 distinct code paths: (a) pure-ViT via `vit-base-clip-224-openai.yaml`, and (b) timm-trunk via `eva02-large-clip-224-2b-s4b-b131k.yaml`. The `timm-vit-base-16-clip-openai.yaml` config does NOT go through `ClipClassifier` and therefore does NOT exercise the override layer at all — it's a pure HF/transformers path.

**Action:** SMOKE-01 in Phase 3 must cover both `ClipClassifier` branches (pure-ViT AND timm-trunk). The `timm-vit-base-16-clip-openai.yaml` config can be tested separately (and was per RESEARCH SMOKE-05) but it's NOT a regression target for the open_clip externalization. The override layer's `factory.py` and `visual.py` (Phase 2) must dispatch correctly on the `isinstance(visual, VisionTransformer)` vs `isinstance(visual, TimmModel)` discriminator that today's bare `except:` at `clip_model.py:38` performs implicitly. See audit row 14 (`project-specific-as-override`).

---

### Q5: Does any tokenizer override affect the image-only path (no text tower)?

**Investigation:** Per RESEARCH.md command:

```bash
grep -rn "tokeniz" planktonzilla/
```

Output: **completely empty.** No file under `planktonzilla/` references any tokenizer — neither `open_clip.tokenizer`, `transformers.AutoTokenizer`, nor any tokenizer construction or method call.

Additionally: `planktonzilla/clip_model.py:31` does `self.model = clip_model.visual` and `self.name_or_path = name + pretrained` — the text tower (`clip_model.text` / `clip_model.transformer`) is **immediately garbage-collected** after `__init__` returns because no reference is kept to `clip_model`. The text path is provably not exercisable from the planktonzilla `ClipClassifier` post-construction.

Cross-check: `/tmp/vendored_vs_v3.3.0.open_clip.diff` does NOT include `tokenizer.py` in its file list (the file is byte-identical to upstream v3.3.0). So even if there were a hypothetical impact, there is no tokenizer divergence to categorize.

**Finding:** The vendored `tokenizer.py` is byte-identical to upstream v3.3.0 (no divergence at all). The image-only training path through `ClipClassifier` does not touch the tokenizer in any way. Tokenizer overrides in the override layer are categorically **not needed** — there is nothing to override.

**Action:** Captured as audit row 12 (`already-merged-upstream`). Phase 2 does NOT create `planktonzilla/open_clip_ext/tokenizer.py`. If a future milestone re-introduces text-tower training (CLIP contrastive pretraining), the tokenizer story can be revisited then with the upstream v3.3.0 baseline as a clean starting point.

---

### Q6: What does `open_clip_train/main.py` set in `os.environ` / `torch.backends`?

**Investigation:** Per RESEARCH.md command:

```bash
grep -nE "(os\.environ|torch\.backends|cudnn|manual_seed|set_seed|random\.seed|np\.random\.seed)" \
  open_clip/src/open_clip_train/main.py
```

Output (3 distinct effects):

```
45: torch.manual_seed(seed + rank)
46: np.random.seed(seed + rank)
47: random.seed(seed + rank)
78: torch.backends.cuda.matmul.allow_tf32 = True
79: torch.backends.cudnn.benchmark = True
80: torch.backends.cudnn.deterministic = False
```

Cross-reference vs `planktonzilla/train.py`:

- **`set_seed` (lines 45-47):** `planktonzilla/train.py:134-135` calls `transformers.set_seed(cfg.seed, cfg.get("deterministic", False))` which seeds Python `random`, `numpy`, `torch`, and (per HF impl) `torch.cuda.manual_seed_all`. **HF's `set_seed` covers the same ground** as the open_clip per-rank seeding. Per-rank offset (`seed + rank`) is not done — but HF's data loaders use `data_seed`/`dataloader_drop_last` semantics that differ from open_clip's contrastive-pretraining sampling needs. Not portable verbatim and not required for HF-Trainer-based runs.
- **`torch.backends.cuda.matmul.allow_tf32 = True` (line 78):** NOT set anywhere in `planktonzilla/`. HF Trainer respects `tf32: bool` field in `TrainingArguments` — `configs/training_arguments/default.yaml:14-15` specifies `bf16: false / fp16: true`, no `tf32`. **Setting `tf32=True` would speed up matmuls on Ampere+ GPUs but change numerics slightly.** This is the only side effect that, if dropped, could cause measurable but small regression in throughput (not in correctness).
- **`torch.backends.cudnn.benchmark = True` and `cudnn.deterministic = False` (lines 79-80):** NOT set anywhere in `planktonzilla/`. HF Trainer's defaults — `benchmark=True` is implicit when `set_seed(... deterministic=False)`; `deterministic=False` is the default torch state. So these are effectively already in effect via HF Trainer defaults.

The `validate_environment()` function in `planktonzilla/train.py:50-87` only inspects HF Hub / W&B / MLflow env vars — it does NOT touch any `torch.backends` or seeds.

**Finding:** Of the three side-effect classes in vendored `main.py`, only `tf32` is a meaningful non-default that planktonzilla doesn't reproduce. The other two (`cudnn.benchmark=True`, `cudnn.deterministic=False`) are already the effective behavior under HF Trainer defaults. None of these were specifically modified by the vendored fork — they are upstream-stock side effects that simply don't run because `pz_train` doesn't call `open_clip_train.main()`.

**Action:** Captured as audit row 11 (`already-merged-upstream`, since the lines themselves are byte-identical to upstream — there is no fork modification to categorize). If Phase 4 ABSORB-02 chooses to add `tf32=True` for parity with the open_clip default, it should set it explicitly in `planktonzilla/train.py` (e.g., guarded by `cfg.training_arguments.tf32`) — not by porting `main.py` boilerplate.

---

### Q7: Pre-refactor checkpoint availability (for SMOKE-02 in Phase 3).

**Investigation:** Already resolved by CONTEXT.md and RESEARCH.md. Phase 1 documents the checkpoint provenance only; it does NOT fetch the checkpoint.

**Finding:** A pre-refactor CLIP-ViT-B-16 + planktonzilla classifier-head checkpoint exists at `huggingface.co/project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt`. It was pushed via `trainer.push_to_hub` (`planktonzilla/train.py:194-206` and `:290-293` regions wire up `push_to_hub` on the `TrainingArguments` and the post-training push call), so it is in **HuggingFace `safetensors` + `config.json` format**, NOT a raw `open_clip` `state_dict.bin`. SMOKE-02 must use either:

```python
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id="project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt")
```

or

```python
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained(
    "project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt",
    trust_remote_code=False,  # default; just being explicit
)
```

**Action:** Phase 1 records the provenance here (this section). Phase 3 SMOKE-02 implements the load + forward-pass check. Phase 1 does NOT need to fetch or verify the checkpoint runtime — its existence is taken on the user's authority per CONTEXT.md.

---

### Q8: Precision default — does the vendored `open_clip_train/*` set AMP precision explicitly or rely on upstream default?

**Investigation:** Per RESEARCH.md command:

```bash
grep -nE "(precision|amp|bf16|fp16)" open_clip/src/open_clip_train/*.py
grep -A 6 '"--precision"' open_clip/src/open_clip_train/params.py
grep -A 6 '"--precision"' /tmp/open_clip_upstream/src/open_clip_train/params.py
```

Vendored `params.py`:

```python
"--precision",
choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
default="amp",
```

Upstream v3.3.0 `params.py`: **byte-identical** — `default="amp"`, same choice set. `params.py` is NOT in the diff for any field except the unrelated `--logs` default.

**Critically:** PITFALLS P9 / Q8 expected upstream to have changed the default to `amp_bf16` in v3.0.0. This expectation is **incorrect for v3.3.0**. As of v3.3.0, upstream still defaults to `amp` (which is `amp_fp16`-equivalent). Vendored matches upstream exactly.

Cross-reference vs `configs/training_arguments/default.yaml:14-15`: `bf16: false / fp16: true`. The HF Trainer pipeline used by `pz_train` independently selects `fp16` for AMP precision — it does NOT consult any `open_clip_train` precision field. So the convergence is automatic.

**Finding:** No precision-default divergence exists between vendored and upstream v3.3.0. The vendored fork did NOT modify `params.py`'s `--precision` default. The expectation in PITFALLS that upstream changed to `amp_bf16` in v3.0.0 was either wrong or refers to a different release. Either way, it does not affect this audit.

**Action:** Captured as audit row 10 (`already-merged-upstream`). Document the convergence: "Vendored and upstream v3.3.0 share `--precision=amp` default. The planktonzilla pipeline overrides this anyway via HF Trainer's `fp16: true` in `configs/training_arguments/default.yaml:14-15`."

---

## Audit Table (AUDIT-01)

**Behavioral-unit granularity. One row per coherent behavioral change. Multi-file fixes get one row with the file list as a sub-field.**

| # | Behavior / Change | Files Touched | Category | Evidence | Override Target |
|---|-------------------|---------------|----------|----------|-----------------|
| 1 | Vendored `open_clip` identifies as floating dev marker `4.0.0.dev0` instead of the upstream tagged release `3.3.0`. Pure metadata; no behavior delta. | `open_clip/src/open_clip/version.py` | discardable | `/tmp/vendored_vs_v3.3.0.open_clip.diff` lines 44-49 — single-line `__version__` string change. | — |
| 2 | Adds opt-in `TrivialAugmentWide` augmentation to `image_transform`'s training-side branch via a new `aug_cfg.trivial_augment` boolean field (defaults to `False`). When the flag is False the entire `image_transform` is byte-equivalent to upstream v3.3.0. | `open_clip/src/open_clip/transform.py` | discardable | `/tmp/vendored_vs_v3.3.0.open_clip.diff` lines 4-31 (3 hunks); the `trivial_augment: bool = False` default makes the addition behaviorally inert. Cross-reference: `planktonzilla/clip_model.py:19,22` discards the returned transforms (`_, _, _`) and `configs/dataset/lensless.yaml:6-19` instantiates `torchvision.transforms.v2.Compose([...])` directly — the `open_clip` transform pipeline is unreachable from `pz_train`. | — |
| 3 | Changes the default `output_fmt` of `VisionTransformer.forward_intermediates` from `'NCHW'` to `'NLC'`. Affects callers that request intermediate-feature extraction without explicitly passing `output_fmt`. | `open_clip/src/open_clip/transformer.py` | discardable | `/tmp/vendored_vs_v3.3.0.open_clip.diff` lines 32-43 — single-line default-kwarg change inside `forward_intermediates`. Cross-reference: `planktonzilla/clip_model.py:42-58` (`ClipClassifier.forward`) calls `self.model[0](pixel_values)` (ViT path) or `self.model.trunk(pixel_values)` (timm path) — neither path invokes `forward_intermediates`, so the default-kwarg flip is unreachable. Also unreachable in `planktonzilla/train.py` and `planktonzilla/dataset.py` (no callers). | — |
| 4 | Vendored `open_clip/src/open_clip/{__init__.py, coca_model.py, constants.py, convert.py, factory.py, hf_configs.py, hf_model.py, loss.py, model.py, modified_resnet.py, openai.py, pos_embed.py, pretrained.py, push_to_hf_hub.py, timm_model.py, tokenizer.py, utils.py, zero_shot_classifier.py, zero_shot_metadata.py, model_configs/*.json}` are byte-identical to upstream v3.3.0 — zero divergence to categorize. | `open_clip/src/open_clip/__init__.py`, `open_clip/src/open_clip/coca_model.py`, `open_clip/src/open_clip/constants.py`, `open_clip/src/open_clip/convert.py`, `open_clip/src/open_clip/factory.py`, `open_clip/src/open_clip/hf_configs.py`, `open_clip/src/open_clip/hf_model.py`, `open_clip/src/open_clip/loss.py`, `open_clip/src/open_clip/model.py`, `open_clip/src/open_clip/modified_resnet.py`, `open_clip/src/open_clip/openai.py`, `open_clip/src/open_clip/pos_embed.py`, `open_clip/src/open_clip/pretrained.py`, `open_clip/src/open_clip/push_to_hf_hub.py`, `open_clip/src/open_clip/timm_model.py`, `open_clip/src/open_clip/tokenizer.py`, `open_clip/src/open_clip/utils.py`, `open_clip/src/open_clip/zero_shot_classifier.py`, `open_clip/src/open_clip/zero_shot_metadata.py`, `open_clip/src/open_clip/model_configs/*.json` | already-merged-upstream | upstream `open-clip-torch` v3.3.0 source tarball (cloned via `git clone --depth 1 --branch v3.3.0 https://github.com/mlfoundations/open_clip.git`) — `diff -ruN` produces no diff hunks for these files. The recursive directory diff in `/tmp/vendored_vs_v3.3.0.open_clip.diff` only mentions `transform.py`, `transformer.py`, and `version.py`. | — |
| 5 | Vendored `open_clip_train/main.py` rewires the `--delete-previous-checkpoint` logic to rotate by `args.save_frequency` rather than always deleting `epoch-1`, so that strided-save runs (`save_frequency > 1`) don't delete checkpoints they meant to keep. | `open_clip/src/open_clip_train/main.py` | discardable | `/tmp/vendored_vs_v3.3.0.open_clip_train.diff` lines 1-18 (single 5-line hunk). Justification for `discardable`: `pz_train` uses HF Trainer's `save_total_limit` + `save_strategy` for checkpoint rotation; `open_clip_train.main` is not invoked from the planktonzilla pipeline (only from the standalone `scripts/train_clip.sh` SLURM driver, which is out-of-scope per CLAUDE.md / ABSORB-01). | — |
| 6 | Vendored `open_clip_train/params.py` changes the `--logs` argparse default from `./logs/` to `../../logs/train_clip/` so the standalone `python -m open_clip_train.main` CLI writes logs into the project's top-level `logs/` directory when invoked from `open_clip/src/open_clip_train/`. | `open_clip/src/open_clip_train/params.py` | discardable | `/tmp/vendored_vs_v3.3.0.open_clip_train.diff` lines 19-30 (single 1-line hunk). Justification for `discardable`: `pz_train` derives output dirs from `configs/hydra/default.yaml` (Hydra runtime); `args.logs` is unreferenced in any planktonzilla module. | — |
| 7 | Vendored `open_clip_train/train.py` rewrites the `evaluate()` function for **classification-style** metrics (`precision_score`, `recall_score`, `f1_score` from sklearn against `txt_unique_tokens`) instead of the upstream **retrieval-style** `get_clip_metrics(image_features, text_features, logit_scale)`. Adds `from sklearn.metrics import precision_score, recall_score, f1_score` import. Removes the upstream "unwrap DDP for single process eval" branch and the `cumulative_gen_loss` / `maybe_compute_generative_loss` CoCa path. | `open_clip/src/open_clip_train/train.py` | discardable | `/tmp/vendored_vs_v3.3.0.open_clip_train.diff` lines 31-145 (two large hunks rewriting `evaluate()`). Justification for `discardable`: `evaluate()` is invoked only from `open_clip_train.main.main()` (CLIP contrastive pretraining); `pz_train` uses HF Trainer's eval loop with `compute_metrics()` defined at `planktonzilla/train.py:99-112` — which already returns `{accuracy, f1, precision, recall}` via sklearn, the same metric semantics. The vendored eval rewrite is dead code in the planktonzilla pipeline. If CLIP contrastive pretraining is ever re-introduced (separate milestone), the rewrite should be re-implemented as an HF-Trainer-compatible `compute_metrics`/callback rather than carried verbatim — per PITFALLS P5: never carry the upstream training loop forward. | — |
| 8 | Vendored `open_clip_train/{__init__.py, data.py, distributed.py, file_utils.py, logger.py, precision.py, profiler.py, scheduler.py, zero_shot.py}` are byte-identical to upstream v3.3.0 — no divergence to categorize. | `open_clip/src/open_clip_train/__init__.py`, `open_clip/src/open_clip_train/data.py`, `open_clip/src/open_clip_train/distributed.py`, `open_clip/src/open_clip_train/file_utils.py`, `open_clip/src/open_clip_train/logger.py`, `open_clip/src/open_clip_train/precision.py`, `open_clip/src/open_clip_train/profiler.py`, `open_clip/src/open_clip_train/scheduler.py`, `open_clip/src/open_clip_train/zero_shot.py` | already-merged-upstream | upstream `open-clip-torch` v3.3.0 source tarball — `diff -ruN /tmp/open_clip_upstream/src/open_clip_train/ open_clip/src/open_clip_train/` only mentions `main.py`, `params.py`, `train.py`. All other `open_clip_train/*.py` files produce no diff hunks. | — |
| 9 | No project-local CLIP model JSON configs were added to `open_clip/src/open_clip/model_configs/` — every JSON file is byte-identical to upstream v3.3.0. Implication: Phase 2 does NOT need to call `open_clip.add_model_config(__path__[0])` at this time (no project configs to register). | `open_clip/src/open_clip/model_configs/*.json` (144 files) | already-merged-upstream | `diff -q /tmp/open_clip_upstream/src/open_clip/model_configs/ open_clip/src/open_clip/model_configs/` produces zero output (file `/tmp/vendored_vs_v3.3.0.model_configs.diff` is 0 bytes). Confirmed: 144 JSONs in vendored == 144 JSONs in upstream v3.3.0, byte-for-byte. | — |
| 10 | Vendored `open_clip` does NOT modify upstream `params.py`'s `--precision` argparse default (`default="amp"`). The PITFALLS expectation that upstream changed the default to `amp_bf16` in v3.0.0 is not supported by v3.3.0's source — both vendored and upstream v3.3.0 share `default="amp"`. The planktonzilla pipeline overrides this anyway via `configs/training_arguments/default.yaml:14-15` (`bf16: false / fp16: true`) read by HF Trainer. | `open_clip/src/open_clip_train/params.py` (precision argparse block) | already-merged-upstream | `grep -A 6 '"--precision"' open_clip/src/open_clip_train/params.py` and the same against `/tmp/open_clip_upstream/src/open_clip_train/params.py` produce identical output. The `params.py` diff hunk in `/tmp/vendored_vs_v3.3.0.open_clip_train.diff` lines 19-30 only touches `--logs` (row 6), NOT `--precision`. | — |
| 11 | Vendored `open_clip_train/main.py` per-rank seeding (`torch.manual_seed(seed + rank)`, `np.random.seed(seed + rank)`, `random.seed(seed + rank)`) and `torch.backends.cuda.matmul.allow_tf32 = True` / `torch.backends.cudnn.benchmark = True` / `torch.backends.cudnn.deterministic = False` are upstream-stock side effects, NOT vendored modifications. Already-equivalent behavior is provided by `transformers.set_seed(cfg.seed, ...)` (`planktonzilla/train.py:134-135`) for seeds, and HF Trainer defaults for cuDNN benchmark/deterministic. The TF32 toggle is the only non-default the planktonzilla pipeline doesn't reproduce — but it's also not part of the vendored fork's modifications. | `open_clip/src/open_clip_train/main.py` (lines 45-47, 78-80 — both byte-identical to upstream) | already-merged-upstream | `grep -nE "(os\\.environ|torch\\.backends|cudnn|manual_seed|set_seed|random\\.seed|np\\.random\\.seed)" open_clip/src/open_clip_train/main.py` returns 6 matches (lines 45-47, 78-80); these lines are NOT in the diff for `main.py` — only the checkpoint-rotation block (line ~523) is modified. Upstream v3.3.0 contains the same lines. | — |
| 12 | Tokenizer code path (`open_clip/src/open_clip/tokenizer.py`) is byte-identical to upstream v3.3.0 AND is unreachable from the planktonzilla `ClipClassifier` (text tower garbage-collected at construction, no `tokenizer` references anywhere under `planktonzilla/`). No override needed; no divergence to categorize. | `open_clip/src/open_clip/tokenizer.py` | already-merged-upstream | `tokenizer.py` does not appear in `/tmp/vendored_vs_v3.3.0.open_clip.diff`. Cross-check: `grep -rn "tokeniz" planktonzilla/` returns empty (no usage). `planktonzilla/clip_model.py:31` retains only `clip_model.visual`. | — |
| 13 | `planktonzilla/clip_model.py:19,22` calls `open_clip.create_model_and_transforms(...)` and **discards the returned transforms** (`_, _, _`). All training-time transforms come from `configs/dataset/*.yaml` (e.g., `lensless.yaml` instantiates `torchvision.transforms.v2.Compose([...])` directly). Implication: the entire `open_clip` transform pipeline is unreachable from `pz_train`, regardless of vendored mods. | `planktonzilla/clip_model.py`, `configs/dataset/lensless.yaml` (and all other `configs/dataset/*.yaml`) | discardable | `grep -nE "(image_transform\|transform\|preprocess\|create_model_and_transforms)" planktonzilla/clip_model.py` shows lines 19, 22 receive into `_, _`; head of `configs/dataset/lensless.yaml` shows `transform: _target_: torchvision.transforms.v2.Compose`. Documenting this here so future maintainers don't try to "fix" the unused-transforms-return path. | — |
| 14 | `ClipClassifier` uses a **bare `except:`** at `planktonzilla/clip_model.py:38` to discriminate between `open_clip.transformer.VisionTransformer` (has `.proj`) and `open_clip.timm_model.TimmModel` (uses `.trunk`). This is the visual-tower-kind discriminator referenced by FEATURES TS-04 and AP-05. Not a vendored-vs-upstream divergence per se — but the override layer (Phase 2) needs an explicit `isinstance(visual, ...)` dispatch (per FEATURES D-03 `visual_tower_kind` helper) to replace this anti-pattern, and FIX-01 (Phase 3) replaces the bare `except` itself. | `planktonzilla/clip_model.py:33-40` | project-specific-as-override | Direct read of `planktonzilla/clip_model.py:33-40`. The override layer needs a single `visual_tower_kind(visual)` helper (per FEATURES D-03) that uses `isinstance` against the upstream classes (`open_clip.transformer.VisionTransformer`, `open_clip.timm_model.TimmModel`). | `planktonzilla/open_clip_ext/_introspection.py` (helper) + `planktonzilla/clip_model.py` post-FIX-01 |
| 15 | Vendored `open_clip/` is committed as a single import (commit `7827fb9 add open_clip to planktonzilla`) — no fork-specific bug fixes have been layered on top against `main`. The 33-files / +3,419/−2,652 delta cited in CONCERNS.md #16 was measured against upstream `main` (`4.0.0.dev0`). Against the **tagged release v3.3.0** that this milestone pins to, the vendored copy is byte-identical except for the 6 file-level changes documented in rows 1-7 above (transform.py, transformer.py, version.py, main.py, params.py, train.py). | (entire `open_clip/` tree at branch `luis/open-clip-port`) | discardable | `git log --oneline HEAD -- open_clip/` returns a single commit `7827fb9`; `git log --oneline HEAD -- open_clip/src/open_clip/transform.py` returns the same single commit. Implication: there are no separately-attributable fork patches to reverse-engineer. The audit-decision matrix collapses to "the v3.3.0 source IS the externalization target." This drastically simplifies Phase 2: the override layer needs to handle only the bare-except discriminator (row 14) and the `num_features` config breakage (CONCERNS.md #11, scope of FIX-01 in Phase 3). | — |

**Citation discipline (AUDIT-02):** every `already-merged-upstream` row above has a non-empty Evidence column citing a reproducible diff command, an upstream tagged-release source location, or a `grep` invocation that anyone can re-run.

---

## Closing Notes

- **`docs/baseline.json`** (produced by plan 01-02) holds the captured pre-refactor metrics. Cross-link target: `baseline.json` lives in this same `docs/` directory.
- **`.planning/REQUIREMENTS.md`** has the full text of AUDIT-01/02/03 and BASELINE-01/02. Section "Phase 1 — Audit & Baseline" lists the requirement-to-plan mapping.
- **`.planning/research/PITFALLS.md`** P1 (shallow audit), P3 (no baseline), P6 (rabbit hole) are the three pitfalls Phase 1 is responsible for preventing. This audit document is the prevention artifact for P1 and P6 (P3 is prevented by `docs/baseline.json` from plan 01-02).
- **Implication for Phase 2 planning:** The override layer is significantly smaller than originally feared. The audit table has zero rows in the `still-needed-as-override` category and only one row (#14) in `project-specific-as-override`. Phase 2's actual work surface is: (a) the `visual_tower_kind` discriminator helper that replaces the bare `except:` at `clip_model.py:38`, (b) the wrap-and-delegate `factory.py` for forward-compatibility per FEATURES TS-06, and (c) the `num_features` config breakage handling per CONCERNS.md #11 (which is technically Phase 3 FIX-01 territory, not Phase 2). Phase 2 may therefore complete faster than the roadmap budget suggests.
