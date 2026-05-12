# Stack Research — Publishing Stack

**Domain:** Publishing fine-tuned image classifiers + Gradio demo to the Hugging Face Hub / Spaces (research-grade artifact for biologists, ML researchers, students, reviewers).
**Researched:** 2026-05-12
**Confidence:** HIGH

> **Scope reminder.** This file covers the **publishing / release** stack only — the libraries needed to (1) push curated checkpoints + model cards to HF Hub, (2) ship a Gradio demo on HF Spaces, (3) make a "load this published model" snippet work in a clean Python env without cloning `planktonzilla`. The training stack is locked and documented in `.planning/codebase/STACK.md` (Python 3.11+, Poetry, Hydra 1.3, `transformers` `4.57.3` locked / `^5.3.0` declared, vendored `open_clip`, six imbalance-aware losses, HF `Trainer`).

---

## TL;DR (what to install on top of the existing stack)

```toml
# Add to pyproject.toml [project.dependencies] / [tool.poetry.dependencies]
huggingface_hub  = ">=1.0,<2"      # 1.14.0 latest (2026-05-06)
gradio           = ">=6.0,<7"      # 6.14.0 latest (2026-04-30); only needed by the Spaces demo
transformers     = ">=5.0,<6"      # 5.8.0 latest (2026-05-05); fixes the `^5.3.0` line that the lock currently downgrades to 4.57.3
safetensors      = ">=0.5,<1"      # already pulled transitively; pin so push_to_hub uses safetensors not pickle
```

That's it for "load + demo". No `evaluate`, no `lighteval`, no model-conversion tooling, no S3 or model registry.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `huggingface_hub` | `1.14.0` (latest, 2026-05-06; `>=1.0,<2`) | Authoritative client for Hub model/dataset/space repos. Provides `HfApi`, `ModelCard`, `ModelCardData`, `EvalResult`, `metadata_update`, `create_repo`, `upload_folder`, `hf_hub_download`, `snapshot_download`, `PyTorchModelHubMixin`. | The single official Python entry point to the Hub. The current locked version (`0.36.0`) is pre-1.0 and will need a major bump anyway — the `transformers>=5` upgrade requires `huggingface_hub>=1`. **CONFIDENCE: HIGH** (verified via Context7 `/huggingface/huggingface_hub` and PyPI). |
| `transformers` | `5.8.0` (latest, 2026-05-05; `>=5.0,<6`) | Hosts `AutoModelForImageClassification`, `AutoImageProcessor`, `PreTrainedModel`, `PreTrainedConfig`, `Trainer.push_to_hub`. The "load published model in clean env" path goes through `AutoModelForImageClassification.from_pretrained(...)`. | The training code already pins `^5.3.0` in `pyproject.toml` (see `.planning/codebase/STACK.md` — currently overridden by lock to `4.57.3`, which is the drift the milestone has to resolve as part of HARD-01). v5+ is required to use `huggingface_hub>=1.0`. **CONFIDENCE: HIGH** (PyPI + Context7 `/huggingface/transformers`, version `v5.0.0` and `v4.57.3` both indexed). |
| `gradio` | `6.14.0` (latest, 2026-04-30; `>=6.0,<7`) | The Spaces demo runtime. `gr.Interface(fn, gr.Image, gr.Label)` is the canonical 5-line image-classifier demo; `gr.Blocks` + `gr.Dropdown` is the canonical multi-model picker. | Gradio 6 is the supported major on HF Spaces in 2026 (Spaces uses `sdk_version` from the README to pin a runtime, and the Spaces docs default to `gradio` SDK with the latest version). v4 is hard-deprecated by Spaces; v5 still runs but no new features. **CONFIDENCE: HIGH** (Context7 `/gradio-app/gradio` indexed at `gradio_6.0.1` and PyPI shows 6.14.0). |
| `safetensors` | `>=0.5,<1` (pin transitively) | Binary weight container that `push_to_hub` writes to (`model.safetensors`). | Safetensors is the **only** acceptable serialization for a public model in 2026 — `pickle`/`torch.save` is a security red flag on the Hub and triggers warnings on the model page. The current lock has `0.7.0` and `transformers` writes safetensors by default. Just don't override it. **CONFIDENCE: HIGH**. |
| `python` | `>=3.11,<3.14` | Already pinned by the training stack and unchanged by this milestone. | Gradio 6 supports Python 3.10–3.13; `transformers` 5.8 supports 3.10–3.14; `huggingface_hub` 1.14 supports 3.9+. The training stack's `>=3.11,<3.14` is the binding constraint. **CONFIDENCE: HIGH**. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `Pillow` | `>=11.0` (already locked at `11.3.0`) | PIL image I/O for the Gradio `gr.Image(type="pil")` flow and end-user inference. | Always — the demo and the "load model and predict" snippet both take a `PIL.Image`. |
| `torch` | `>=2.4,<3` (locked at `2.8.0`) | PyTorch runtime for the published checkpoints. | Always — published models are torch state-dicts in safetensors. The Space's `requirements.txt` pulls a CPU build (`torch --index-url https://download.pytorch.org/whl/cpu`) on `cpu-basic` hardware to keep image size manageable. |
| `timm` | `>=1.0,<2` (locked at `1.0.24`) | Backbone for ViT/CLIP/EVA visual towers; downloaded transitively when loading any of the existing `timm/*` model configs. | Required at load-time for any of the seven curated models that wrap a `timm`-built backbone (most of them). Keep it in the published model repo's implicit deps via the `pip` snippet in the README, not in `auto_map`. |
| `open_clip_torch` | `>=2.30,<3` (PyPI; **NOT** the vendored copy) | Reference implementation for CLIP architectures used by `ClipClassifier`. | Required only for the CLIP-based published model(s). See [§ ClipClassifier packaging](#the-clipclassifier-question-canonical-2026-pattern) — the recommended pattern declares this as a Hub-side dependency (in the model repo's README install snippet) so end users don't need the vendored copy. |
| `numpy` | `>=2.0` (locked at `2.4.1`) | Tensor interop for the demo and inference snippets. | Transitively required by `torch`, `gradio`, `transformers`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `hf` CLI (ships with `huggingface_hub>=0.34`) | Authentication (`hf auth login`), repo creation (`hf repo create`), folder upload (`hf upload`), Spaces deploy. | Replaces the deprecated `huggingface-cli` entry point. Use this for one-off repo creation in shell scripts. |
| `gradio deploy` | One-shot push of the local `app.py` + `requirements.txt` to a Space. | Easiest path for v1 — bundles the local directory, gathers metadata, pushes to `huggingface.co/spaces/project-oceania/<name>`. Alternative: write `app.py` + `README.md` + `requirements.txt` and `git push` to the Space repo. |
| Existing `Trainer.push_to_hub` (from training pipeline) | Already used at `planktonzilla/train.py:294` for raw checkpoint dumps. | **Insufficient** for v1 — autogenerates a stub `README.md`. The milestone explicitly requires real model cards (REL-02), so `push_to_hub` is the *transport* but the model card has to be authored separately via `ModelCard.from_template(...)` and pushed with `card.push_to_hub(repo_id)`. |

---

## The `ClipClassifier` question — canonical 2026 pattern

**Problem stated by the milestone:** the repo defines a custom `ClipClassifier(nn.Module)` (`planktonzilla/clip_model.py`) that wraps a vendored `open_clip` visual tower with a fresh `nn.Linear` head and exposes HF's `ImageClassifierOutput`. For `from_pretrained` to work for the CLIP-based published model **without cloning `planktonzilla`**, what's the right pattern?

**Four options, ranked.**

### Option A — Wrap as `PreTrainedModel` + `register_for_auto_class` + `trust_remote_code=True` ✅ **RECOMMENDED**

This is the official 2026 Hugging Face pattern for "custom architecture, AutoModel-loadable, no external pip install of the source repo". Verified directly from `huggingface.co/docs/transformers/main/en/custom_models` (fetched 2026-05-12).

**What you do:**

1. Create a tiny `clip_classifier_hub/` folder with three files (no other deps on `planktonzilla`):
   ```
   clip_classifier_hub/
     __init__.py            # empty
     configuration_clip_classifier.py   # ClipClassifierConfig(PreTrainedConfig)
     modeling_clip_classifier.py        # ClipClassifierModel(PreTrainedModel)
   ```

2. **Configuration** subclasses `PreTrainedConfig`, sets `model_type = "planktonzilla_clip_classifier"`, and stores everything needed to rebuild the model: `open_clip_model_name`, `open_clip_pretrained_tag`, `num_features`, `num_labels`, `id2label`, `label2id`. **Do not store the full open_clip weights here** — store the open_clip identifiers and let `open_clip` resolve them.

3. **Model** subclasses `PreTrainedModel`, sets `config_class = ClipClassifierConfig`, in `__init__` calls `open_clip.create_model_and_transforms(config.open_clip_model_name, pretrained=config.open_clip_pretrained_tag)` to build the visual tower, then attaches `nn.Linear(config.num_features, config.num_labels)`. `forward(pixel_values, labels=None)` returns a `dict` with `logits` and (if labels) `loss` — same contract as `planktonzilla/clip_model.py:ClipClassifier` today.

4. Register with auto classes **before** `push_to_hub` so the `auto_map` lands in `config.json`:
   ```python
   ClipClassifierConfig.register_for_auto_class()
   ClipClassifierModel.register_for_auto_class("AutoModelForImageClassification")
   ```

5. `model.push_to_hub("project-oceania/pz_clip_eva02_isiisnet")` — this uploads weights (`model.safetensors`), `config.json` (with `auto_map` pointing at the two `.py` files), `configuration_clip_classifier.py`, and `modeling_clip_classifier.py`.

**End-user load snippet (works on a clean `pip install transformers torch open_clip_torch huggingface_hub` env, no `planktonzilla` clone):**

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image

model = AutoModelForImageClassification.from_pretrained(
    "project-oceania/pz_clip_eva02_isiisnet",
    trust_remote_code=True,                       # required for custom code
    revision="<commit-sha>",                      # pin for reproducibility (recommended)
)
processor = AutoImageProcessor.from_pretrained("project-oceania/pz_clip_eva02_isiisnet")
inputs = processor(images=Image.open("plankton.png"), return_tensors="pt")
logits = model(**inputs).logits
print(model.config.id2label[logits.argmax(-1).item()])
```

**Why this wins for this project:**
- Zero pip-install steps for the model code — `transformers` resolves it from the Hub repo itself.
- Same `AutoModelForImageClassification.from_pretrained` snippet works for **all** seven published models (the six standard HF backbones + the CLIP one) — no per-model branch in DOC-02.
- The `open_clip_torch` PyPI package is a single, stable, pip-installable dep that the model card can declare in its `## Requirements` section. It's the public, versioned upstream of the vendored `open_clip/src/open_clip/` directory — internal version `4.0.0.dev0` per `.planning/codebase/STACK.md` matches `open_clip-torch>=2.30` upstream.
- Survives the eventual unvendoring (HARD-01 deferred to a hardening milestone). The published Hub artifact is decoupled from whether `planktonzilla` keeps `open_clip/` in-tree.

**Caveats:**
- `trust_remote_code=True` is mandatory for end users. The model card MUST disclose this prominently and recommend pinning `revision=` to a commit SHA, per the official Transformers security guidance (`transformers/docs/source/en/models.md`: "load from a specific commit hash rather than the latest version to prevent loading potentially modified code").
- The two `.py` files uploaded to the Hub repo cannot import from `planktonzilla.*` — they have to be self-contained. Today's `planktonzilla/clip_model.py:ClipClassifier` is already only ~80 lines and only imports `torch`, `torch.nn`, `open_clip`, and `transformers.modeling_outputs.ImageClassifierOutput`. The port is mechanical: copy the class, replace the bare `nn.Module` base with `PreTrainedModel`, and move the `(model_name, num_features, num_labels)` constructor args into a `Config`.
- The `ImageProcessor` for the CLIP model needs special handling: `open_clip.create_model_and_transforms` returns its own `transform` (torchvision `Compose`). For a Hub `AutoImageProcessor` to work, either (a) shim it into a `transformers.image_processing_utils.BaseImageProcessor` subclass and `register_for_auto_class("AutoImageProcessor")`, or (b) document the CLIP transform inline in the model card and skip `AutoImageProcessor` for that one model. Option (b) is fine for v1 — DOC-02 just shows a per-model snippet for the CLIP case if needed.

### Option B — `PyTorchModelHubMixin` from `huggingface_hub`

Lighter than Option A. Inherit from `nn.Module, PyTorchModelHubMixin` and the mixin gives you `save_pretrained` / `push_to_hub` / `from_pretrained` for free. Verified pattern from `huggingface_hub/docs/source/en/guides/integrations.md`:

```python
class ClipClassifier(nn.Module, PyTorchModelHubMixin, library_name="planktonzilla", tags=["plankton","clip"]):
    def __init__(self, open_clip_name="EVA02-L-14", pretrained="merged2b_s4b_b131k", num_labels=32, num_features=768): ...
```

**Trade-off vs Option A:** Loses `AutoModelForImageClassification.from_pretrained` compatibility — users have to import `ClipClassifier` directly (`from clip_classifier_hub import ClipClassifier; ClipClassifier.from_pretrained(...)`). That breaks DOC-02's "one snippet works for all 7 models" goal. **Choose only if** the project decides to ship a tiny pip package (`pip install planktonzilla-models`) — but PyPI release is explicitly out of scope per `PROJECT.md`.

### Option C — Convert to a standard HF architecture (e.g., ViT)

Re-train or surgically transplant the CLIP visual tower weights into a `ViTForImageClassification` instance. Avoids `trust_remote_code` entirely.

**Why not:** EVA02-L-14 is not bit-compatible with `transformers.ViTModel` (rotary position embeddings, sub-LN, GLU MLP, custom patch embed) — would either lose accuracy or require a multi-day weight-conversion effort. The milestone explicitly forbids new training campaigns. Skip.

### Option D — Ship `ClipClassifier` as a separate tiny pip package

`pip install planktonzilla-clip` provides `ClipClassifier`; published model uses Option B but users always `pip install` first.

**Why not:** PyPI release is explicitly out of scope (`PROJECT.md` Out of Scope #1). Adds a publishing target this milestone doesn't have appetite for.

### Decision

**Use Option A.** It's the only option that (a) requires no PyPI release, (b) keeps `AutoModelForImageClassification.from_pretrained(...)` as the *single* universal snippet across all 7 published models, and (c) survives unvendoring/replacement of `open_clip` in a later hardening milestone. The cost is `trust_remote_code=True` in user code and ~150 lines of new code in a `clip_classifier_hub/` directory inside the model repo (not the planktonzilla source repo).

**Confidence: HIGH** (verified against `huggingface.co/docs/transformers/main/en/custom_models` fetched 2026-05-12; matches `huggingface_hub` integrations guide; matches Transformers' own `register_for_auto_class` API at v5.8).

---

## Model Card schema (REL-02)

The milestone requires "real model card[s] (eval metrics on a held-out split, dataset card link, training recipe / config snapshot, license, citation BibTeX, intended-use + limitations sections — not autogenerated stubs)". Use `huggingface_hub.ModelCard` + `ModelCardData` + `EvalResult` rather than hand-rolled YAML.

### Authoring API (verified pattern, Context7 `/huggingface/huggingface_hub`)

```python
from huggingface_hub import ModelCard, ModelCardData, EvalResult, create_repo

card_data = ModelCardData(
    language="en",
    license="cc-by-nc-4.0",                             # match the source dataset's license
    library_name="transformers",
    pipeline_tag="image-classification",
    tags=["plankton", "marine-biology", "image-classification", "oceania", "isiisnet"],
    datasets=["project-oceania/isiisnet"],              # link to the data card
    base_model="microsoft/resnet-18",                   # original backbone, link to its card
    model_name="pz_resnet18_isiisnet",
    eval_results=[
        EvalResult(
            task_type="image-classification",
            dataset_type="image-classification",
            dataset_name="ISIISNet (project-oceania)",
            metric_type="accuracy",
            metric_value=0.872,
        ),
        EvalResult(
            task_type="image-classification",
            dataset_type="image-classification",
            dataset_name="ISIISNet (project-oceania)",
            metric_type="f1",
            metric_value=0.864,
            metric_name="macro-F1",
        ),
    ],
)

card = ModelCard.from_template(
    card_data,
    model_id="pz_resnet18_isiisnet",
    model_description="ResNet-18 fine-tuned on the ISIISNet plankton dataset (32 classes) using the planktonzilla framework.",
    developers="Inria Chile — OcéanIA project",
    repo="https://github.com/Inria-Chile/deep_plankton",
    # plus longer-form sections rendered as Markdown body:
)
# Append project-specific sections to card.text directly (Markdown):
card.text += "\n\n## Intended use\n...\n## Limitations\n...\n## Training recipe\n```yaml\n<config snapshot>\n```\n## Citation\n```bibtex\n@misc{...}\n```\n"

create_repo("project-oceania/pz_resnet18_isiisnet", exist_ok=True)
card.push_to_hub("project-oceania/pz_resnet18_isiisnet")
```

### Required YAML frontmatter fields (per `huggingface/hub-docs/docs/hub/model-cards.md` and `model-release-checklist.md`)

| Field | Required | Source / value for planktonzilla |
|-------|----------|----------------------------------|
| `pipeline_tag` | yes (powers Hub search & API widget) | `image-classification` |
| `library_name` | yes | `transformers` (uniform across all 7 models, including the CLIP one — Option A keeps it `transformers`) |
| `language` | optional | `en` for the card prose |
| `license` | yes | per dataset (see `.planning/codebase/INTEGRATIONS.md` — ISIISNet `cc-by-nc-4.0`, WHOI-Plankton `mit`, ZooLake/Lensless `cc-by-4.0`, others TBD) |
| `datasets` | yes | `[project-oceania/<dataset>]` — clickable link on Hub model page |
| `base_model` | yes | the upstream backbone (`microsoft/resnet-18`, `google/vit-base-patch16-224`, etc.); for CLIP, the `open_clip` model identifier in tags |
| `tags` | recommended | `["plankton", "marine-biology", "<dataset>", "oceania"]` for searchability |
| `model-index` | recommended for eval results | auto-generated from `EvalResult` entries by `ModelCardData` |

### Body sections (Markdown after the YAML frontmatter — REL-02 explicit list)

1. **Model description** — what backbone, what dataset, one-paragraph framing.
2. **Intended use** — "research-grade plankton-class identification on imagery from <instrument family>" + "not for biomass quantification, biodiversity surveys without expert validation, or biosecurity decisions".
3. **How to use** — copy-pasteable inference snippet (DOC-02 reuses this verbatim).
4. **Training data** — link the data card; one paragraph on splits, class balance.
5. **Training recipe** — embed the relevant `configs/<group>/*.yaml` snapshot (or a digest) so the run is reproducible.
6. **Evaluation results** — table mirroring `eval_results=` for human reading + the YAML powers the Hub's auto-eval widget.
7. **Limitations** — "trained on <dataset> only, OOD performance unknown", "class imbalance addressed via <loss>", "imaging-modality-specific".
8. **Citation** — BibTeX for the planktonzilla project + the source dataset's citation.
9. **License** — restate the YAML field with rationale ("inherits from source dataset's CC-BY-NC-4.0").

**Confidence: HIGH** (schema verified against `huggingface/hub-docs` Context7 dump, 2026-05-12).

---

## Gradio Spaces demo (DEMO-01)

**Goal recap (`PROJECT.md` DEMO-01):** single-image upload, model picker across the 7 published models, top-K predictions with probabilities. No saliency.

### Recommended app shape

`gr.Blocks` (not `gr.Interface`) because of the model picker. `gr.Interface` is great for the "1 model, 1 image, 1 label" demo but doesn't compose well with a `gr.Dropdown` that swaps models in-flight. The Blocks pattern is documented as the canonical multi-component demo (Context7 `/gradio-app/gradio` "Blocks: Speech-to-Text and Sentiment Analysis").

```python
# app.py — the only Python file the Space needs
import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from functools import lru_cache

MODELS = {
    "ISIISNet (ResNet-18)":     "project-oceania/pz_resnet18_isiisnet",
    "FlowCamNet (ViT-base)":    "project-oceania/pz_vit_flowcamnet",
    "Lensless (BEiT-base)":     "project-oceania/pz_beit_lensless",
    "UVP6Net (ViT-base)":       "project-oceania/pz_vit_uvp6net",
    "WHOI-Plankton (ResNet-18)":"project-oceania/pz_resnet18_whoi",
    "ZooLake (ConvNeXtV2-huge)":"project-oceania/pz_convnextv2_zoolake",
    "JEDI-Oceans (CLIP-EVA02)": "project-oceania/pz_clip_eva02_jedi",   # uses Option A trust_remote_code
}

@lru_cache(maxsize=2)
def load(repo_id: str):
    model = AutoModelForImageClassification.from_pretrained(repo_id, trust_remote_code=True).eval()
    processor = AutoImageProcessor.from_pretrained(repo_id)
    return model, processor

def predict(image: Image.Image, model_label: str, top_k: int = 5):
    if image is None:
        return {}
    model, processor = load(MODELS[model_label])
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        probs = torch.softmax(model(**inputs).logits[0], dim=-1)
    return {model.config.id2label[i.item()]: float(p) for p, i in zip(*probs.topk(top_k))}

with gr.Blocks(title="Planktonzilla — pre-trained plankton classifiers") as demo:
    gr.Markdown("# Planktonzilla\nPre-trained plankton image classifiers from the OcéanIA project.")
    with gr.Row():
        image_in = gr.Image(type="pil", label="Plankton image")
        with gr.Column():
            model_in = gr.Dropdown(choices=list(MODELS), value=list(MODELS)[0], label="Model")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            label_out = gr.Label(num_top_classes=10, label="Predictions")
    image_in.change(predict, inputs=[image_in, model_in, top_k], outputs=label_out)
    model_in.change(predict, inputs=[image_in, model_in, top_k], outputs=label_out)
    gr.Examples(examples=[["examples/isiis_acantharea.jpg"], ["examples/uvp6_chaetognatha.jpg"]], inputs=image_in)

demo.launch()
```

### Spaces config (`README.md` frontmatter — verified against `huggingface/hub-docs/docs/hub/spaces-config-reference.md`)

```yaml
---
title: Planktonzilla
emoji: 🐚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.14.0           # pin to current; HF Spaces respects this
python_version: "3.11"
app_file: app.py
suggested_hardware: cpu-upgrade   # see hardware section below
short_description: Pre-trained plankton image classifiers from the OcéanIA project.
models:
  - project-oceania/pz_resnet18_isiisnet
  - project-oceania/pz_vit_flowcamnet
  - project-oceania/pz_beit_lensless
  - project-oceania/pz_vit_uvp6net
  - project-oceania/pz_resnet18_whoi
  - project-oceania/pz_convnextv2_zoolake
  - project-oceania/pz_clip_eva02_jedi
datasets:
  - project-oceania/isiisnet
  - project-oceania/flowcamnet
  - project-oceania/lensless
  - project-oceania/uvp6net
  - project-oceania/whoi-plankton
  - project-oceania/zoolake
  - project-oceania/jedi_oceans
tags:
  - plankton
  - marine-biology
  - image-classification
  - oceania
pinned: true
---
```

### Spaces `requirements.txt`

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.8.0
torchvision==0.23.0
transformers>=5.0,<6
huggingface_hub>=1.0,<2
gradio>=6.0,<7
open_clip_torch>=2.30,<3      # only needed if a CLIP-based model is in the picker
Pillow>=11.0
safetensors>=0.5
```

CPU `torch` keeps the Space image small (no CUDA libraries). No need for the full Poetry-managed env.

### Hardware tier choice

Per `huggingface/hub-docs/docs/hub/spaces-config-reference.md` valid `suggested_hardware` values include `cpu-basic`, `cpu-upgrade`, `t4-small`, `t4-medium`, `l4x1`, `a10g-small`, `a10g-large`, `a100-large`, etc.

| Tier | Cost | Choose if | Verdict |
|------|------|-----------|---------|
| `cpu-basic` (free) | 2 vCPU, 16 GB | Demo only loads small backbones (ResNet-18, ViT-base) | **Avoid** — `convnextv2_huge` and CLIP-EVA02-L will OOM or time out (>60s/inference). |
| `cpu-upgrade` (paid, ~$0.03/hr) | 8 vCPU, 32 GB | Need bigger backbones at acceptable latency on CPU | **Recommended for v1.** All 7 models load (use `lru_cache(maxsize=2)` to avoid simultaneous loading), inference under 10s for the heaviest. |
| `t4-small` (paid, ~$0.40/hr) | 1× T4 GPU | Sub-second inference for biologists actually using the demo | **Recommended if budget allows** — the demo's primary audience (biologists) will perceive >5s latency as broken. |
| `a10g-small` and up | $$$ | Production traffic, large CLIP demos | Overkill for v1. |

**Recommendation: `suggested_hardware: cpu-upgrade`** as the README hint, then manually toggle to `t4-small` in the Space settings UI if the project has the budget. (`suggested_hardware` is advisory only — actual hardware is set via the Space settings page; the YAML key only helps users who duplicate the Space.)

**Confidence: HIGH** for config schema (verified against the Spaces config reference). **MEDIUM** for the hardware-tier latency estimates (extrapolated from CPU inference times for similar backbones; project should benchmark on the actual heaviest model before launch).

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pickle` / `torch.save(model, ...)` for published weights | Hub flags pickle as a security risk (arbitrary code execution on `torch.load`). User-facing red banner on the model page. | `safetensors` — the `transformers` `save_pretrained` / `push_to_hub` path writes `.safetensors` by default in v5+. Don't override. |
| Hand-rolled `README.md` YAML frontmatter | Easy to typo `pipeline_tag` / `model-index` / `eval_results`; loses the Hub's auto-eval widget; loses dataset/model linking. | `huggingface_hub.ModelCard` + `ModelCardData` + `EvalResult` (validated by the Hub's CI on push). |
| Gradio `< 4` (`gr.inputs.Image`, `gr.outputs.Label`) | API removed in v4; HF Spaces refuses to build with `sdk_version: 3.x`. Many old tutorials still show this. | Gradio 6: `gr.Image(type="pil")`, `gr.Label(num_top_classes=K)`. The `gr.inputs.*` / `gr.outputs.*` aliases are gone. |
| `gr.Interface.load("huggingface/<model>")` for the demo | Backed by HF Inference API, which has rate limits and no guarantee for custom-code models (`trust_remote_code` required) — the CLIP model will not work. | Local model loading via `AutoModelForImageClassification.from_pretrained` inside `predict()`, cached via `functools.lru_cache`. |
| `evaluate` (Hugging Face) | The library has been in maintenance mode since 2025; the Hub team explicitly recommends `LightEval` for new evaluation workflows (per PyPI page note 2026-05). The training code already imports `evaluate` at `planktonzilla/train.py:32` but doesn't use it — metrics computed via `sklearn.metrics`. | Keep `sklearn.metrics` (already in use); manually format `EvalResult` entries. Don't add `evaluate` as a publishing dependency. |
| `huggingface_hub < 1.0` | Pre-1.0 API surface; `transformers>=5` requires `huggingface_hub>=1`; some `ModelCardData` fields (notably `eval_results`) gained validation in 1.x. | `huggingface_hub>=1.0,<2`. |
| Vendored `open_clip` (`open_clip/src/open_clip/`) declared on the Hub model | Cannot be installed by an end user without cloning `planktonzilla`. The vendored copy's internal version `4.0.0.dev0` is opaque to pip and reproducibility tools. | `open_clip_torch>=2.30,<3` from PyPI, declared in the published model card's `## Requirements` section and in the Space's `requirements.txt`. |
| `Trainer.push_to_hub(...)` as the *only* publish step | Generates a placeholder `README.md` ("This model is a fine-tuned version of...") that will not satisfy REL-02. | Use `Trainer.push_to_hub` to dump checkpoints + tokenizer/processor, then **overwrite** `README.md` with the `ModelCard.from_template(...)` output via `card.push_to_hub(repo_id)`. |
| `requirements.txt` for the planktonzilla source repo | Poetry is the canonical lockfile; mixing creates drift. | Keep `pyproject.toml` + `poetry.lock` for the source repo; only the **Space** uses a flat `requirements.txt` (Spaces requires it). |
| `gradio.Interface` for the multi-model demo | `Interface` is one-fn-in, one-fn-out; the model picker requires `Blocks`. | `gr.Blocks` + `gr.Dropdown` + explicit `.change()` event wiring. |
| `streamlit` SDK for the Space | Larger image, slower cold start, heavier deps; HF tooling is deeper for `gradio` SDK; project has no Streamlit experience (no Streamlit code in the repo). | `sdk: gradio`. |
| Persistent storage for the Space | Per the `huggingface/hub-docs` Spaces config reference (2026-05): "The persistent storage feature is no longer available so this setting will be ignored." | Don't request it. Stateless `predict()` + Hub-cached model downloads is enough. |

---

## Stack Patterns by Variant

**Standard published model (6 of 7 — ResNet, ViT, BEiT, ConvNeXtV2, EVA02 variants):**
- `Trainer.push_to_hub` writes `model.safetensors` + `config.json` + `preprocessor_config.json`.
- Author `ModelCard` + `card.push_to_hub` over the autogenerated stub.
- End-user load: `AutoModelForImageClassification.from_pretrained("project-oceania/<repo>")` — no `trust_remote_code`.
- Confidence: **HIGH**.

**CLIP-based published model (1 of 7 — JEDI-Oceans on EVA02-L-14 + merged2b):**
- Use Option A (`PreTrainedModel` + `register_for_auto_class` + `trust_remote_code=True`).
- Add an `## Installation` section to the model card: `pip install transformers>=5 torch>=2.4 open_clip_torch>=2.30 huggingface_hub>=1`.
- Recommend pinned `revision=` in DOC-02's load snippet for the CLIP model.
- Confidence: **HIGH** (matches Transformers official docs verbatim).

**Spaces demo (single deploy):**
- One `app.py` (Blocks + Dropdown + Image + Label).
- One `requirements.txt` (CPU torch).
- One `README.md` with Spaces YAML frontmatter pinning `sdk_version: 6.14.0`.
- Initial deploy via `gradio deploy` from the project's local working dir; thereafter `git push` to the Space repo.
- Confidence: **HIGH**.

**README "load published model" snippet (DOC-02):**
- One canonical 6-line snippet using `AutoModelForImageClassification.from_pretrained` + `AutoImageProcessor.from_pretrained` + `Image.open` + `processor` + `model` + `softmax + topk`.
- For the CLIP model only: add `trust_remote_code=True` and the recommended `revision=`.
- Test by running it in a fresh `python:3.11` Docker image with **only** `pip install transformers>=5 torch>=2.4 huggingface_hub>=1 Pillow open_clip_torch` — no `git clone`. This validates the entire publishing pipeline end-to-end.
- Confidence: **HIGH**.

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `transformers>=5.0` | `huggingface_hub>=1.0` | Hard requirement; older `huggingface_hub` (e.g. the locked 0.36.0) will fail import on `transformers>=5`. **Action**: when HARD-01 resolves the `^5.3.0` constraint, the lock will pull `huggingface_hub>=1.0` automatically. |
| `gradio>=6.0` | `python>=3.10`, `huggingface_hub>=0.28` | No conflict with the project's `python>=3.11,<3.14`. |
| `gradio` on HF Spaces | `sdk_version` must be a Gradio version Spaces actually supports | All Gradio 4+ versions are listed as supported; Gradio 3 builds will be rejected. Pin `sdk_version: 6.14.0` exactly to avoid surprise updates breaking the demo. |
| `open_clip_torch>=2.30` | `torch>=2.0`, `timm>=0.9` | No conflict with the locked `torch==2.8.0` and `timm==1.0.24`. The vendored `open_clip` (internal `4.0.0.dev0`) corresponds to a recent upstream — verify by diffing `open_clip/src/open_clip/version.py` against PyPI before swapping. |
| `safetensors>=0.5` | `torch>=2.0` | Standard pairing. |
| `transformers>=5` | `python>=3.10` | Project pins `>=3.11`. Compatible. |

---

## Installation

```bash
# Source repo (planktonzilla) — add publishing deps to pyproject.toml [project.dependencies]:
poetry add 'huggingface_hub>=1.0,<2' 'transformers>=5.0,<6'
# Note: gradio is NOT a source-repo dep — it lives only in the Space's requirements.txt.

# To resolve the existing lock-vs-pyproject drift (HARD-01):
poetry lock   # regenerates lock against the corrected pyproject

# For the HF Space (in the Space's git repo, not in planktonzilla):
cat > requirements.txt <<'EOF'
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.8.0
torchvision==0.23.0
transformers>=5.0,<6
huggingface_hub>=1.0,<2
gradio>=6.0,<7
open_clip_torch>=2.30,<3
Pillow>=11.0
safetensors>=0.5
EOF

# End-user reproduction of DOC-02 (verifies the publish path works):
docker run --rm -it python:3.11 bash -lc "
  pip install 'transformers>=5' 'huggingface_hub>=1' torch open_clip_torch Pillow &&
  python -c \"
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
m = AutoModelForImageClassification.from_pretrained('project-oceania/pz_resnet18_isiisnet')
p = AutoImageProcessor.from_pretrained('project-oceania/pz_resnet18_isiisnet')
print(m.config.id2label[0])
\"
"
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `huggingface_hub.ModelCard` + `ModelCardData` | Hand-rolled `README.md` with manual YAML | Never for v1 — loses validation, eval widget, dataset/model linking. Acceptable only as a last resort if the `ModelCard` API doesn't support a custom field (it does). |
| Option A (`PreTrainedModel` + `trust_remote_code`) for `ClipClassifier` | Option B (`PyTorchModelHubMixin`) | If the project decides to ship a PyPI package later and accepts a per-model code branch in DOC-02. |
| Option A for `ClipClassifier` | Option C (convert to standard HF arch) | If accuracy degradation from weight conversion is acceptable AND a multi-day weight-port effort is funded. Not for v1. |
| `gr.Blocks` with `gr.Dropdown` model picker | `gr.TabbedInterface` (one tab per model) | If the project later wants to show per-model README content side-by-side; for v1, dropdown is simpler. |
| `cpu-upgrade` Spaces hardware | `t4-small` GPU | If demo latency on the heaviest backbones (>5s on CPU) is judged unacceptable for biologist users — and budget allows. |
| `gradio deploy` CLI | `git push` to `https://huggingface.co/spaces/project-oceania/<name>` | If the project wants Spaces CI on git push (more durable, plays nicer with code review); CLI is faster for the first deploy. |
| `safetensors` weights | `pytorch_model.bin` (legacy pickle) | Never. Hub will warn on pickle; safetensors is mandatory in 2026. |
| `transformers>=5` `Trainer.push_to_hub` | Manual `HfApi.upload_folder` | If a custom-architecture model needs files the Trainer doesn't know about (e.g., the `modeling_clip_classifier.py` for Option A — `Trainer.push_to_hub` won't include it; need a follow-up `api.upload_file` for the modeling/configuration `.py` files). **In practice you'll do both.** |

---

## Sources

- **Context7** `/huggingface/huggingface_hub` — `ModelCard`, `ModelCardData`, `EvalResult`, `metadata_update`, `PyTorchModelHubMixin`, `hf_hub_download`, `HfApi.upload_folder`. Fetched 2026-05-12. Confidence: **HIGH**.
- **Context7** `/huggingface/transformers` — `register_for_auto_class`, `PreTrainedModel`, `PreTrainedConfig`, `AutoImageProcessor`, `AutoModelForImageClassification.from_pretrained(..., trust_remote_code=True)`. Indexed at v4.51.3, v5.0.0, v4.57.3 (training stack already on v5 path). Fetched 2026-05-12. Confidence: **HIGH**.
- **Context7** `/gradio-app/gradio` — `gr.Interface`, `gr.Blocks`, `gr.Image(type="pil")`, `gr.Label(num_top_classes=K)`, `gr.Dropdown`, `gradio deploy`. Indexed at `gradio_6.0.1`. Fetched 2026-05-12. Confidence: **HIGH**.
- **Context7** `/huggingface/hub-docs` — Spaces YAML frontmatter (`sdk`, `sdk_version`, `python_version`, `suggested_hardware`, `models`, `datasets`, `tags`, `app_file`), model-card metadata schema, eval-results YAML schema. Fetched 2026-05-12. Confidence: **HIGH**.
- **Official docs** `https://huggingface.co/docs/transformers/main/en/custom_models` (full doc fetched 2026-05-12) — definitive 2026 pattern for `PreTrainedConfig` + `PreTrainedModel` + `register_for_auto_class` + Hub upload + `trust_remote_code=True`. **Single most important source for the `ClipClassifier` decision.** Confidence: **HIGH**.
- **PyPI** version verification (fetched 2026-05-12):
  - `gradio` 6.14.0 (released 2026-04-30), supports Python 3.10–3.13.
  - `huggingface_hub` 1.14.0 (released 2026-05-06).
  - `transformers` 5.8.0 (released 2026-05-05), supports Python 3.10–3.14.
  - `evaluate` 0.4.6 (last release 2025-09-18, in maintenance — not recommended for new work). Confidence: **HIGH**.
- **Local context** `.planning/PROJECT.md`, `.planning/codebase/STACK.md`, `.planning/codebase/INTEGRATIONS.md`, `.planning/codebase/ARCHITECTURE.md` — established constraints, current locked versions, dataset licenses, vendored `open_clip` notes. Confidence: **HIGH** (read directly).

---

*Stack research for: planktonzilla v1 publishing milestone (HF Hub model release + Gradio Spaces demo)*
*Researched: 2026-05-12*
*Confidence overall: HIGH*
