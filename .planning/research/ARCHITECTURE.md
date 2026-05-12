# Architecture Research

**Domain:** Public release of a fine-tuned image-classifier project across GitHub + HuggingFace Hub + HF Spaces (brownfield, training pipeline already exists)
**Researched:** 2026-05-12
**Confidence:** HIGH (Context7 + official HF docs verified for every load-bearing claim; observed-pattern claims about real-world classifier Spaces marked MEDIUM where appropriate)

> Scope reminder: this document covers **release-time architecture** — what artifacts exist, where they live, how they link to each other, and in what order they get built. It does NOT re-research the training pipeline, which is locked in `.planning/codebase/ARCHITECTURE.md`.

## Standard Architecture

### System Overview — release-time topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  PUBLIC SURFACES                                        │
│                                                                                         │
│  ┌──────────────────────────┐   ┌─────────────────────────┐   ┌───────────────────┐   │
│  │  GitHub                  │   │  HF Spaces              │   │  HF Hub: Org page │   │
│  │  Inria-Chile/            │   │  project-oceania/       │   │  /project-oceania │   │
│  │   deep_plankton          │   │   plankton-classifier   │   │  + (optional)     │   │
│  │                          │   │   (Gradio SDK)          │   │  Collection       │   │
│  │  - README.md (entry)     │   │  - app.py               │   │                   │   │
│  │  - planktonzilla/ pkg    │   │  - requirements.txt     │   │  Read-only        │   │
│  │  - configs/, scripts/    │   │  - README.md (YAML FM)  │   │  aggregation +    │   │
│  │  - tests/                │   │                         │   │  discovery view   │   │
│  │  - LICENSE (MIT)         │   │  Runs the demo UI       │   │                   │   │
│  └──────────┬───────────────┘   └────────┬────────────────┘   └─────────▲─────────┘   │
│             │ links to ↓                  │ loads ↓                       │           │
└─────────────┼─────────────────────────────┼───────────────────────────────┼───────────┘
              │                             │                               │
              ▼                             ▼                               │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          HUGGING FACE HUB — project-oceania org                         │
│                                                                                         │
│  ┌─────────────── 7 MODEL REPOS ───────────────┐  ┌───── 7 DATASET REPOS ──────────┐  │
│  │ project-oceania/                              │  │ project-oceania/                │  │
│  │   pz-isiisnet              ← winning ckpt    │  │   isiisnet                      │  │
│  │   pz-flowcamnet                              │  │   flowcamnet                    │  │
│  │   pz-lensless                                │  │   lensless                      │  │
│  │   pz-uvp6net                                 │  │   uvp6net                       │  │
│  │   pz-whoi-plankton                           │  │   whoi-plankton                 │  │
│  │   pz-zoolake                                 │  │   zoolake                       │  │
│  │   pz-jedi-oceans                             │  │   jedi_oceans (or sim.)         │  │
│  │                                               │  │                                  │  │
│  │ Each repo:                                    │  │ ALREADY EXIST (per              │  │
│  │   config.json                                 │  │ INTEGRATIONS.md)                │  │
│  │   model.safetensors                           │  │                                  │  │
│  │   preprocessor_config.json                    │  │ Cards updated by                │  │
│  │   modeling_planktonzilla.py  (if custom)      │  │ DatasetImporter.update_         │  │
│  │   configuration_planktonzilla.py (if custom)  │  │  metadata                       │  │
│  │   README.md  ← model card (YAML + body)       │  │                                  │  │
│  │     - model-index w/ eval_results             │  │                                  │  │
│  │     - datasets: project-oceania/<dataset>     │  │                                  │  │
│  │     - links back to GitHub                    │  │                                  │  │
│  └───────────────────────────────────────────────┘  └─────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| **GitHub repo (`Inria-Chile/deep_plankton`)** | Source of truth for code, configs, tests, training scripts; landing surface for issues/PRs/citation; canonical README that drives users to HF Hub | Stays where it is. README rewritten to lead with HF artifacts; existing `planktonzilla/` package, `configs/`, `scripts/`, `tests/` unchanged. Git-LFS NOT used (weights live on HF, not in git). |
| **HF model repo (×7)** | One git repo per dataset on HF Hub. Hosts trained weights + config + processor config + model card + (optionally) custom modeling code so users can `AutoModelForImageClassification.from_pretrained(repo_id)` from a clean env | `huggingface.co/project-oceania/pz-<dataset>` — created with `huggingface_hub.create_repo`; weights uploaded via `model.push_to_hub()` or `HfApi.upload_folder()`; model card via `ModelCard(...).push_to_hub()` |
| **HF dataset repos (×7)** | Already exist — hosted training data + dataset cards. Referenced from each model card's `datasets:` metadata so the "Datasets used to train" widget appears | Owned by existing `DatasetImporter` pipeline (`pz_import_dataset`). No new code in this milestone unless dataset cards need a content edit. |
| **HF Space (`project-oceania/plankton-classifier`)** | Gradio demo. Single image upload → model dropdown (7 options) → top-K predictions with probabilities. The "look at it work" surface. | Separate git repo on HF Hub (`spaces/project-oceania/plankton-classifier`) with `app.py` + `requirements.txt` + `README.md` (YAML frontmatter declaring `sdk: gradio`, `app_file: app.py`, `models: [...]` listing all 7 model IDs, `datasets: [...]` listing all 7 dataset IDs, `preload_from_hub` for warm cache). Hardware: `cpu-basic` is plausible for a 7-model image classifier demo if loading is lazy; bump to `t4-small` if EVA02-L-14 inference is too slow. |
| **HF Collection (optional)** | A curated grouping page on HF that bundles the 7 models + 7 datasets + the Space under one URL for discovery and for the launch announcement | Created via the Hub UI or `huggingface_hub.add_collection_item`. Defer to last; not on the critical path. |
| **Custom modeling code (`modeling_planktonzilla.py`)** | If the published model is the `ClipClassifier` variant (EVA02-L-14 + linear head), the model can't be loaded from a clean env without the wrapper class. This file ships the class + config inside each affected model repo so `trust_remote_code=True` makes load work without cloning planktonzilla. | One pair of files per model repo that uses `ClipClassifier`: `modeling_planktonzilla.py` and `configuration_planktonzilla.py`, plus `auto_map` entries in `config.json`. The classes are derived from `planktonzilla/clip_model.py` but rewritten to depend only on `pip install open-clip-torch transformers torch` — NOT on the planktonzilla package. (See "Custom ClipClassifier packaging" below.) |
| **READMEs / cards** | The user-visible documentation surface for each artifact. Each card answers: what this is, who it's for, how to load it, what its eval numbers are, what its license is, how to cite it, where the source code lives. | GitHub README hand-written; model cards generated programmatically (via `ModelCard.from_template` or a project-local Jinja template) so the 7 cards stay consistent; Space README written by hand. |

## Recommended Project Structure

### What stays in the GitHub repo (no change)

```
Inria-Chile/deep_plankton/                  ← unchanged top-level layout
├── planktonzilla/                          # training package (unchanged)
├── configs/                                # Hydra configs (unchanged)
├── scripts/                                # SLURM launchers (unchanged)
├── tests/                                  # pytest (unchanged)
├── notebooks/                              # exploratory (unchanged)
├── open_clip/                              # vendored (unchanged — see "open_clip at release time" below)
├── pyproject.toml, poetry.lock             # unchanged except HARD-01 transformers fix
├── README.md                               # ← REWRITTEN for public-use entry
└── LICENSE                                 # MIT (unchanged)
```

### What is ADDED to the GitHub repo for release

```
Inria-Chile/deep_plankton/
├── release/                                # NEW — artifact-publishing scripts
│   ├── __init__.py
│   ├── modeling_planktonzilla.py           # standalone ClipClassifier (no planktonzilla/ deps)
│   ├── configuration_planktonzilla.py      # PreTrainedConfig subclass
│   ├── card_template.md.j2                 # Jinja model-card template
│   ├── publish_model.py                    # CLI: take a checkpoint + dataset → HF model repo
│   ├── eval_model.py                       # CLI: run held-out eval → emit eval_results.json
│   └── manifest.yaml                       # the 7 (dataset, checkpoint_uri, model_arch) tuples
├── space/                                  # NEW — mirrored to HF Space repo as its own git
│   ├── app.py
│   ├── requirements.txt
│   └── README.md                           # YAML frontmatter for Spaces config
└── docs/release/                           # NEW — supporting docs
    ├── model-card-template.md              # human-readable canonical version of the Jinja
    └── publishing-runbook.md               # how to (re)publish a model end-to-end
```

### What lives in each HF model repo (`project-oceania/pz-<dataset>`)

```
pz-<dataset>/                                 ← per-dataset HF model repo
├── README.md                                # model card (YAML frontmatter + body)
├── config.json                              # HF model config (incl. id2label, label2id, auto_map for custom)
├── model.safetensors                        # weights (preferred over pytorch_model.bin)
├── preprocessor_config.json                 # image processor / transform metadata (mean, std, size)
├── modeling_planktonzilla.py                # ONLY for ClipClassifier-based models
├── configuration_planktonzilla.py           # ONLY for ClipClassifier-based models
└── (optional) examples/                     # 1-3 sample images for the inference widget
```

For pure-HF-arch models (anything where `cfg.model._target_: transformers.AutoModelForImageClassification.from_pretrained` works directly — `microsoft/resnet-18`, `microsoft/beit-base-patch16-224`, `google/vit-base-patch16-224`, the `timm-*` models loaded via Transformers' timm bridge), `modeling_planktonzilla.py` and `configuration_planktonzilla.py` are NOT needed and `auto_map` is omitted.

### What lives in the HF Space repo (`spaces/project-oceania/plankton-classifier`)

```
plankton-classifier/                          ← HF Space repo (Gradio SDK)
├── app.py                                  # Gradio Blocks app (REQUIRED)
├── requirements.txt                        # transformers, gradio, torch, pillow, open-clip-torch
└── README.md                               # YAML frontmatter (REQUIRED), human description below
```

Example README YAML for the Space (verified against [HF Spaces config reference](https://huggingface.co/docs/hub/en/spaces-config-reference)):

```yaml
---
title: Plankton Classifier
emoji: "🦠"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.x"          # pin the latest stable Gradio at release
app_file: app.py
pinned: true
short_description: "7 fine-tuned plankton image classifiers across major datasets."
tags:
  - image-classification
  - plankton
  - marine-biology
  - ecology
models:
  - project-oceania/pz-isiisnet
  - project-oceania/pz-flowcamnet
  - project-oceania/pz-lensless
  - project-oceania/pz-uvp6net
  - project-oceania/pz-whoi-plankton
  - project-oceania/pz-zoolake
  - project-oceania/pz-jedi-oceans
datasets:
  - project-oceania/isiisnet
  - project-oceania/flowcamnet
  - project-oceania/lensless
  - project-oceania/uvp6net
  - project-oceania/whoi-plankton
  - project-oceania/zoolake
  - project-oceania/jedi-oceans
preload_from_hub:
  - project-oceania/pz-isiisnet
  - project-oceania/pz-flowcamnet
  - project-oceania/pz-lensless
  - project-oceania/pz-uvp6net
  - project-oceania/pz-whoi-plankton
  - project-oceania/pz-zoolake
  - project-oceania/pz-jedi-oceans
---
```

`models:` and `datasets:` create the cross-link badges on the Space's Hub page back to the model and dataset repos. `preload_from_hub` warms the Space's HF cache at build time so first-request latency for any model isn't a 600 MB download (per the Spaces config reference, files land in `~/.cache/huggingface/hub`). Note that there is **no Space-side `models` parameter that "auto-loads" the models** — the YAML field is purely metadata + linkage; `app.py` still has to call `from_pretrained` itself.

### Structure Rationale

- **`release/` is in the GitHub repo, not its own repo.** The publishing scripts are short, only run by maintainers, and need to evolve in lockstep with the training-side code. Keeping them in-tree avoids a second source-of-truth repo.
- **`space/` is in the GitHub repo AND mirrored to its own Space repo.** HF Spaces requires its own git repo on `huggingface.co/spaces/...`. The canonical version lives under `space/` in GitHub for review/diff/PR; a sync step (e.g. `huggingface_hub.upload_folder` or a thin push script) replicates it to the Space repo. This pattern ("source of truth in GitHub, mirror to HF") is the standard approach used by HF organizations themselves.
- **`modeling_planktonzilla.py` is duplicated into each affected model repo.** The HF "custom code" mechanism does not support a shared dependency between model repos — each `trust_remote_code=True` repo must be self-contained ([Customizing models docs](https://huggingface.co/docs/transformers/en/custom_models)). Our `release/` directory holds the canonical copy; `publish_model.py` copies it into each repo at publish time.
- **No `pip install planktonzilla` requirement.** PROJECT.md is explicit that PyPI is out of scope for v1, and a goal is "load published model from a clean env." That forces the standalone-`modeling_planktonzilla.py` approach (the alternative — importing the planktonzilla package — would force PyPI on v1).

## Architectural Patterns

### Pattern 1: One model repo per (dataset, winning checkpoint)

**What:** Each of the 7 datasets gets exactly one HF model repo holding exactly one checkpoint — the winning architecture for that dataset. Repo naming: `project-oceania/pz-<dataset>` (short, dataset-anchored, easily picked from a dropdown).

**When to use:** Standard for fine-tuned-classifier releases. One repo per checkpoint maximizes per-model metadata richness (each card has dataset-specific eval, dataset-specific intended use, dataset-specific limitations) and keeps the discovery URL stable per dataset.

**Trade-offs:**
- Pro: Each model is independently citable, versionable, deletable; widgets and "Inference API" work per-repo; the Hub's task/dataset filters surface each model individually.
- Pro: Aligns with PROJECT.md REL-01 ("Publish one curated trained model per dataset (7 models)").
- Con: 7 model cards to maintain — mitigated by the Jinja template in `release/card_template.md.j2`.
- Alternative considered: One big `pz-all` repo with 7 `.safetensors` files. Rejected — the Hub UI assumes one model per repo (widget, eval results, downloads counter, Inference API), so a multi-checkpoint repo loses all of those.

### Pattern 2: Custom code shipped on the Hub via `trust_remote_code=True`

**What:** For the `ClipClassifier`-based models, ship a self-contained `modeling_planktonzilla.py` + `configuration_planktonzilla.py` inside the model repo, register the auto-class mapping in `config.json` via `auto_map`, and require users to pass `trust_remote_code=True` on `from_pretrained`.

**When to use:** When the model architecture is custom (i.e., not a built-in Transformers class) AND a clean-env load is required AND a pip package isn't viable. This is exactly our `ClipClassifier` situation.

**Trade-offs:**
- Pro: Zero install steps for the user beyond `pip install transformers torch open-clip-torch`. Verified by [transformers custom_models docs](https://huggingface.co/docs/transformers/en/custom_models).
- Pro: Keeps planktonzilla package off PyPI in v1.
- Con: `trust_remote_code=True` is a security signal users have to opt into. Mitigation: model card prominently documents *why* it's needed, what code is being loaded, and recommends pinning a specific commit `revision=` for production use. (Per the docs: "For additional security, you can load a custom model from a specific revision using a commit hash to avoid loading model code that may have changed.")
- Con: The custom code must be standalone — no relative imports outside the model repo's directory. This is enforced by the upload mechanism (relative imports work *within* the same directory only).

**Reference snippet (canonical pattern from `transformers` docs):**
```python
# Inside release/configuration_planktonzilla.py
from transformers import PreTrainedConfig

class PlanktonzillaClipConfig(PreTrainedConfig):
    model_type = "planktonzilla-clip"

    def __init__(self, open_clip_arch="EVA02-L-14",
                 open_clip_pretrained="merged2b_s4b_b131k",
                 num_features=768, num_labels=32, id2label=None,
                 label2id=None, **kwargs):
        self.open_clip_arch = open_clip_arch
        self.open_clip_pretrained = open_clip_pretrained
        self.num_features = num_features
        self.num_labels = num_labels
        self.id2label = id2label or {}
        self.label2id = label2id or {}
        super().__init__(**kwargs)

# Inside release/modeling_planktonzilla.py
import torch
import open_clip
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from .configuration_planktonzilla import PlanktonzillaClipConfig

class PlanktonzillaClipForImageClassification(PreTrainedModel):
    config_class = PlanktonzillaClipConfig

    def __init__(self, config):
        super().__init__(config)
        backbone, _, _ = open_clip.create_model_and_transforms(
            config.open_clip_arch, pretrained=config.open_clip_pretrained
        )
        # mirror the ClipClassifier head construction from planktonzilla/clip_model.py
        backbone.visual.proj = None
        self.encoder = backbone.visual
        self.classifier = torch.nn.Linear(config.num_features, config.num_labels)

    def forward(self, pixel_values, labels=None, **kwargs):
        feats = self.encoder(pixel_values)
        logits = self.classifier(feats)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=loss, logits=logits)
```

At publish time (`release/publish_model.py`):
```python
PlanktonzillaClipConfig.register_for_auto_class()
PlanktonzillaClipForImageClassification.register_for_auto_class("AutoModelForImageClassification")
model.push_to_hub(f"project-oceania/pz-{dataset}", private=False)
```

End-user load:
```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
model = AutoModelForImageClassification.from_pretrained(
    "project-oceania/pz-isiisnet",
    trust_remote_code=True,
    revision="v1.0.0",  # pin the published commit/tag for security + reproducibility
)
processor = AutoImageProcessor.from_pretrained("project-oceania/pz-isiisnet")
```

### Pattern 3: Model card as a templated artifact, not a hand-written one

**What:** All 7 model cards are generated by `release/publish_model.py` using `huggingface_hub.ModelCard.from_template(...)` (or a project-local Jinja template) populated from a single per-model manifest entry. Hand editing happens only on the *template*, not on individual cards.

**When to use:** Whenever you have ≥3 cards that share structure. Below 3, hand-writing wins; above 3, drift between cards becomes inevitable.

**Trade-offs:**
- Pro: 7 cards stay structurally identical — same sections in same order, same metadata schema, same tone. Reviewers don't have to read 7 different stories.
- Pro: Eval results flow from `release/eval_model.py` → `eval_results.json` → template substitution → `model-index` YAML, with no transcription step.
- Pro: Re-publish is mechanically a re-run of `publish_model.py`, which makes "fix a typo across all 7 cards" trivial.
- Con: Some per-dataset nuance (e.g., dataset-specific limitations, known failure modes) needs explicit per-model fields in the manifest. The template has to be designed to accommodate this rather than fight it.

The canonical pattern (verified via Context7 / `huggingface_hub` docs):

```python
from huggingface_hub import ModelCard, ModelCardData, EvalResult

card_data = ModelCardData(
    language="en",
    license="mit",  # or per-dataset; some sources are CC-BY-NC-4.0
    library_name="transformers",
    pipeline_tag="image-classification",
    tags=["plankton", "marine-biology", "image-classification", "fine-tuned"],
    datasets=["project-oceania/isiisnet"],
    base_model="timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    model_name="pz-isiisnet",
    eval_results=[
        EvalResult(task_type="image-classification",
                   dataset_type="image-folder",
                   dataset_name="ISIISNet (held-out test split)",
                   metric_type="accuracy", metric_value=0.91),
        EvalResult(task_type="image-classification",
                   dataset_type="image-folder",
                   dataset_name="ISIISNet (held-out test split)",
                   metric_type="f1", metric_value=0.88),
    ],
)
card = ModelCard.from_template(card_data, template_path="release/card_template.md.j2",
                               # any extra Jinja variables here
                               github_repo="https://github.com/Inria-Chile/deep_plankton",
                               source_dataset_url="https://huggingface.co/datasets/project-oceania/isiisnet",
                               num_classes=32,
                               training_recipe_link="...",
                               citation_bibtex="...")
card.push_to_hub("project-oceania/pz-isiisnet")
```

The Hub parses the `model-index` block from the YAML frontmatter and renders an "Evaluation results" widget on the model page (verified at [Model Cards docs](https://huggingface.co/docs/hub/en/model-cards)).

### Pattern 4: Space loads models lazily and caches them in process memory

**What:** The Space's `app.py` does NOT load all 7 models at startup. It loads each model on first selection, caches the loaded model in a module-level dict, and reuses on subsequent requests. Combined with `preload_from_hub` in the Space README (which only warms the disk cache, not GPU/RAM), this gives reasonable cold-start while keeping memory bounded.

**When to use:** Whenever a Space hosts >2 large models AND only one is needed per request. (Always, for our case — the user picks one model per inference.)

**Trade-offs:**
- Pro: First page load is fast (~seconds, not minutes). Memory usage scales with the number of distinct models the user *actually* picks, not with 7×model_size.
- Pro: On `cpu-basic` (16 GB RAM) hardware, loading all 7 EVA02-L-14-class checkpoints upfront would OOM. Lazy loading is what makes `cpu-basic` viable.
- Con: First request to each new model has model-load latency (read from disk + materialize). Acceptable for a demo.
- Mitigation: Optional eviction policy if RAM gets tight (`functools.lru_cache(maxsize=2)`), or move heavy backbones to a single shared encoder. v1 should NOT bother with eviction — measure first.

**Reference shape:**
```python
# space/app.py
from functools import lru_cache
import gradio as gr
from transformers import AutoModelForImageClassification, AutoImageProcessor

MODEL_OPTIONS = {
    "ISIISNet": "project-oceania/pz-isiisnet",
    "FlowCAMNet": "project-oceania/pz-flowcamnet",
    # ... 7 entries
}

@lru_cache(maxsize=None)  # one slot per model picked, never evicts (size with hardware)
def get_pipeline(repo_id: str):
    model = AutoModelForImageClassification.from_pretrained(
        repo_id, trust_remote_code=True
    )
    processor = AutoImageProcessor.from_pretrained(repo_id)
    return model.eval(), processor

def predict(image, model_choice: str, top_k: int = 5):
    model, processor = get_pipeline(MODEL_OPTIONS[model_choice])
    inputs = processor(images=image, return_tensors="pt")
    with __import__("torch").inference_mode():
        logits = model(**inputs).logits
    probs = logits.softmax(-1)[0]
    topk = probs.topk(top_k)
    return {model.config.id2label[int(i)]: float(p)
            for p, i in zip(topk.values, topk.indices)}

with gr.Blocks(title="Plankton Classifier") as demo:
    with gr.Row():
        img = gr.Image(type="pil")
        with gr.Column():
            model_choice = gr.Dropdown(choices=list(MODEL_OPTIONS), label="Model")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            out = gr.Label(num_top_classes=10)
    btn = gr.Button("Classify")
    btn.click(predict, inputs=[img, model_choice, top_k], outputs=out)

if __name__ == "__main__":
    demo.launch()
```

(MEDIUM confidence on the exact `lru_cache` pattern — Gradio's own docs and the [Diffusers/Gradio memory-leak discussion](https://github.com/huggingface/diffusers/discussions/10936) note that "Pipeline / model needs to be created in a function called *after* the Gradio UI is defined" to avoid context issues with `app.load()`. The pattern above puts model loading inside `predict()`, which respects that constraint.)

## Data Flow

### Where the trained checkpoints come from today

Per `.planning/codebase/ARCHITECTURE.md` and `.planning/codebase/INTEGRATIONS.md`, the current state is:

```
SLURM run on Jean Zay
    ↓ (training writes)
logs/train/runs/<timestamp>_<slurm>/checkpoint-<step>/    ← Hugging Face Trainer checkpoint
    ↓ (W&B mirrors)
wandb/run-<timestamp>-<id>/files/                         ← W&B run dir (offline mode)
    ↓ (sometimes)
trainer.push_to_hub(...)  → project-oceania/pz_<model>_<dataset>  ← may already exist as PRIVATE repos
```

So checkpoints live in three places that we know of: SLURM scratch / Lustre paths, the local `wandb/` mirror, and possibly already on the Hub under the long-form `pz_<model>_<dataset>` naming if `model_push_to_hub: true` was set during training. No central artifact registry beyond that.

### Release-time data flow (target state)

```
                                   release/manifest.yaml
                              (curated 7-tuple: dataset, ckpt_uri,
                               model_arch, expected_metrics, ...)
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│             release/eval_model.py   (one invocation per dataset)      │
│   Inputs:  ckpt_uri (Lustre / Hub-private / W&B artifact URI)         │
│            project-oceania/<dataset>                                  │
│   Loads:   model from ckpt + held-out test split from HF dataset      │
│   Outputs: release/eval_results/<dataset>.json                        │
│            { accuracy, f1, precision, recall, per_class_metrics }     │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│           release/publish_model.py  (one invocation per dataset)      │
│   Inputs:  ckpt_uri, eval_results/<dataset>.json,                     │
│            release/card_template.md.j2,                               │
│            release/modeling_planktonzilla.py + configuration_*.py     │
│   Steps:                                                              │
│     1. Load checkpoint (HF Trainer dir or W&B artifact)               │
│     2. If ClipClassifier: re-instantiate as                           │
│        PlanktonzillaClipForImageClassification(config) and copy       │
│        weights; otherwise use as-is.                                  │
│     3. register_for_auto_class() (custom-code path only)              │
│     4. create_repo("project-oceania/pz-<dataset>", exist_ok=True)     │
│     5. model.push_to_hub(...) — pushes config.json, model.safetensors,│
│        and (if registered) modeling/configuration .py + auto_map      │
│     6. AutoImageProcessor.save_pretrained / push_to_hub              │
│        (preprocessor_config.json with mean, std, size — sourced       │
│        from the dataset card's mean/std written by the importer       │
│        and from the model's input size)                               │
│     7. Build ModelCardData with EvalResult list from the JSON         │
│     8. ModelCard.from_template(...).push_to_hub(...)                  │
│     9. Tag the commit with a release version (e.g. v1.0.0)            │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                project-oceania/pz-<dataset>  on the Hub
                                   │
                                   │ (linked from)
                                   ▼
              ┌────────────────────────────────────────────┐
              │  HF Space: plankton-classifier             │
              │  loads via AutoModelForImageClassification │
              │  .from_pretrained("project-oceania/...")   │
              └────────────────────────────────────────────┘
                                   │
                                   │ (linked from)
                                   ▼
              ┌────────────────────────────────────────────┐
              │  GitHub README.md                          │
              │  links to all 7 model repos + Space        │
              │  + provides the copy-pasteable load snippet│
              └────────────────────────────────────────────┘
```

**Key points about this flow:**

- **Eval is a fresh pass, not transcribed from W&B.** The downstream consumer asked: is the eval-results table generated, or transcribed from W&B? Answer: it MUST be generated by `release/eval_model.py` against a frozen held-out split. W&B numbers are training-time evaluations on whatever split was hot at the time and may use slightly different preprocessing; transcribing them risks "different number on the card vs. what reviewers reproduce" — the worst possible failure mode for a citable artifact. PROJECT.md REL-02 requires "eval metrics on a held-out split" which implies a real eval pass.

- **`release/manifest.yaml` is the curation surface.** A maintainer-edited file listing the 7 winning checkpoints with provenance. Format suggestion:
  ```yaml
  - dataset: isiisnet
    hf_dataset_id: project-oceania/isiisnet
    arch: timm-eva02-large-m38m
    ckpt_source: lustre://...checkpoint-12000/   # or wandb://..., or hub://project-oceania/pz_old_name
    is_clip_classifier: false                     # toggles custom-code path
    license: cc-by-nc-4.0
    notes: |
      Won over BEiT and ConvNeXt-V2 on val F1.
  - dataset: lensless
    ...
  ```
  This is the single source of truth for "what gets published." Updating it and re-running `publish_model.py` is the re-publish workflow.

- **`open_clip` at release time:** PROJECT.md is explicit that re-vendoring/un-vendoring `open_clip` cleanly is deferred. The release-time question is narrower: **what does a clean-env user need installed?** Answer: `pip install open-clip-torch` from PyPI. The standalone `release/modeling_planktonzilla.py` does `import open_clip` — that resolves to the PyPI package in a clean env. The vendored `open_clip/` source tree in the GitHub repo stays put for the training side (where SLURM scripts add it to `PYTHONPATH`), but does NOT need to ship to HF Hub. The vendored copy is `4.0.0.dev0` (per INTEGRATIONS.md); pinning the public release to a known-compatible PyPI version of `open-clip-torch` (e.g. `>=2.24,<3.0` — verify against vendored API surface) is a small spike inside HARD-01.

## Suggested Build Order

This is the part that directly drives the roadmap. Phases are named by what they unblock, not by their internal complexity.

### Phase A — Foundation: pick the winning checkpoints, freeze the eval methodology

**Why first:** Everything downstream needs concrete answers to: which checkpoint per dataset, where does it physically live, how do we evaluate it. Without this, scripts have nothing to point at and cards have nothing to publish.

**Includes:**
- Survey existing W&B runs and Hub repos to identify the winning checkpoint per dataset.
- Resolve where each winning checkpoint physically resides (Lustre? local? Hub-private already?).
- Write `release/manifest.yaml`.
- Decide on the eval split convention (a single deterministic test split per dataset, derived how — from the dataset's `test` split if it has one, else from a seeded `train_test_split`).
- Write `release/eval_model.py` and run it against all 7 — produces `release/eval_results/*.json`.
- Decide which models are ClipClassifier-based vs. pure HF (sets the per-model branch in `publish_model.py`).

**Unblocks:** Phase B (publishing has inputs), Phase C (cards have eval numbers), Phase D (Space knows what to load).

**Parallelizable with:** None. This is the gate.

### Phase B — Publish one model end-to-end (the spike)

**Why second:** Get one model up, with a real card, custom code working, `from_pretrained(..., trust_remote_code=True)` verified from a clean Python env, BEFORE scaling to 7. Front-load the unknowns.

**Includes:**
- Write `release/configuration_planktonzilla.py` and `release/modeling_planktonzilla.py` (the standalone CLIP wrapper).
- Write `release/card_template.md.j2`.
- Write `release/publish_model.py`.
- Pick one dataset (suggest: ISIISNet — first one alphabetically, real codebase support, real eval data) and run it end-to-end.
- Verify clean-env load: `python -c "from transformers import AutoModelForImageClassification; m = AutoModelForImageClassification.from_pretrained('project-oceania/pz-isiisnet', trust_remote_code=True); print(m)"` in a venv with only `transformers torch open-clip-torch pillow` installed.

**Unblocks:** Phase C (template proven), Phase D (Space load path proven).

**Parallelizable with:** Initial drafting of Phase D `app.py` skeleton — but the skeleton can't be tested end-to-end until Phase B's one model exists.

### Phase C — Publish the remaining 6 models

**Why third:** Mechanical scale-up of Phase B with the spike's lessons applied. This is when REL-02 ("real model card per model") gets fully satisfied.

**Includes:**
- Loop `publish_model.py` over the remaining 6 manifest entries.
- Manual review of each generated card (formatting, dataset-specific limitations, citation accuracy).
- Add the optional Collection on the HF org page bundling the 7 models + 7 datasets.

**Unblocks:** Phase D's full functionality (all 7 dropdown options actually work), Phase E (announcement can credibly link to all 7).

**Parallelizable with:** Phase D's `app.py` and Spaces deployment (D can use the partial set for development).

### Phase D — Build and deploy the Space

**Why fourth (or in parallel with C):** The Space is the headline demo (PROJECT.md DEMO-01) but it depends on at least one published model existing. It can begin as soon as Phase B is done (using one model in dev) and becomes fully populated as Phase C completes.

**Includes:**
- Write `space/app.py`, `space/requirements.txt`, `space/README.md`.
- Push to `huggingface.co/spaces/project-oceania/plankton-classifier`.
- Verify cold-start time is acceptable on chosen hardware.
- Verify each of the 7 models actually loads + classifies correctly via the dropdown.
- Decide hardware tier (`cpu-basic` first; bump if any model takes >30 s for inference).

**Unblocks:** Phase E (announcement has a working URL).

**Parallelizable with:** Phase C — yes, fully. App skeleton + deploy can happen against a single-model Space; expand the dropdown as new model repos go live.

### Phase E — GitHub README rewrite + launch announcement

**Why last:** Needs working URLs for everything downstream consumers will click. Writing a "load this model" snippet in the README that points at a not-yet-existent repo is a great way to ship a broken README.

**Includes:**
- Rewrite `README.md` to lead with HF artifacts (model list with one-liner each, Space link, copy-pasteable load snippet).
- Update DOC-01's four headline use cases (install, load published model, retrain on your own data, import a new dataset).
- Verify DOC-02: the load snippet works in a clean env.
- HARD-01: fix only the bugs that block the load path (transformers `^5.3.0` constraint resolved by `poetry.lock` sanity check; `open_clip` PYTHONPATH issue if it touches the load path).
- Write the launch blurb and link list (REL-03).
- Optional v1.0.0 GitHub release tag.

**Unblocks:** Public announcement.

**Parallelizable with:** Phase D's polish work.

### Build order summary

```
A (foundation) ──► B (one-model spike) ──► C (six more models) ──┐
                                                                   ├──► E (README + launch)
                                          ──► D (Space)  ─────────┘
                                          (D parallel with C, gated on B)
```

Phase boundaries are natural here because each phase produces a verifiable artifact and unblocks specific downstream phases.

## Cross-References Between Artifacts

| From | To | Mechanism |
|------|-----|-----------|
| GitHub README | All 7 model repos | Markdown links + a copy-pasteable load snippet |
| GitHub README | HF Space | Markdown link + (optional) embedded iframe |
| GitHub README | Source datasets on HF | Markdown links to `huggingface.co/datasets/project-oceania/<dataset>` |
| Each model card → GitHub | Card body has a "Source code & training recipe" section linking to `Inria-Chile/deep_plankton` and (if appropriate) the specific config file | Markdown link + (optional) `base_model:` YAML field if a HF backbone exists |
| Each model card → source dataset | YAML `datasets: [project-oceania/<dataset>]` triggers HF's "Datasets used to train" widget on the model page | YAML metadata, parsed by Hub |
| Each model card → backbone | YAML `base_model: <hf_repo_id>` triggers "Finetuned from" widget | YAML metadata |
| Each model card → eval split | `model-index` YAML with `EvalResult` entries renders as the "Evaluation results" widget | YAML metadata |
| Each model card → Space | Body has a "Try it interactively" link to the Space URL | Markdown link |
| Space README → 7 model repos | YAML `models: [...]` field renders model badges on the Space's Hub page | YAML metadata |
| Space README → 7 dataset repos | YAML `datasets: [...]` field renders dataset badges | YAML metadata |
| Space README → GitHub | Body has a "Source code" link | Markdown link |
| Each dataset card → models trained on it | Auto-populated by HF reverse lookup once any model card declares the dataset in its `datasets:` field | Automatic |

The key insight: **YAML frontmatter does the heavy lifting for cross-linking.** As long as the metadata is correct, HF's web UI auto-generates badges, widgets, and reverse links between artifacts. The only manual cross-linking is in the README/card *body* (where prose links are still needed for context).

## Anti-Patterns

### Anti-Pattern 1: Requiring `pip install -e git+https://github.com/...` to load a model

**What people do:** Skip the standalone `modeling_planktonzilla.py` work and tell users in the README to `pip install git+https://github.com/Inria-Chile/deep_plankton` before they can `from_pretrained` the model.

**Why it's wrong:** Defeats the entire point of a HF Hub release for the marine-biologist audience. They want to paste 3 lines into a notebook, not navigate Poetry / vendored `open_clip` / Hydra. PROJECT.md DOC-02 explicitly requires "no clone of the planktonzilla repo required."

**Do this instead:** Either (a) ship the standalone modeling code on Hub via `trust_remote_code=True` (recommended, see Pattern 2), or (b) defer ClipClassifier-based models to v1.1 and only publish the pure-HF-arch winners in v1. (b) is a fallback if the custom-code path turns out to have unexpected issues during Phase B.

### Anti-Pattern 2: Transcribing W&B numbers into model cards

**What people do:** Copy validation accuracy from the W&B dashboard into the `model-index` YAML.

**Why it's wrong:** W&B logs validation metrics computed during training, which may use a different preprocessing pipeline, a different split, or a different epoch's weights than the published checkpoint. When a reviewer runs eval and gets a different number, the discrepancy looks like fraud or a bug. For a citable artifact this is the worst possible failure.

**Do this instead:** Run a fresh evaluation pass via `release/eval_model.py` against the *exact* published checkpoint and the *exact* held-out split that the model card describes. Persist the JSON output. Have the publish script consume that JSON. Numbers in cards must be reproducible by anyone with the model + the dataset + the eval script.

### Anti-Pattern 3: One mega-repo holding all 7 checkpoints

**What people do:** Push all 7 `.safetensors` files + 7 `config.json` files + 7 sets of weights to a single `project-oceania/planktonzilla-models` repo to "keep them together."

**Why it's wrong:** The Hub is built around one-model-per-repo. Widgets, the Inference API, the model-index parser, the downloads counter, the dataset-link badges — all assume one model per `from_pretrained` call. A multi-checkpoint repo gets none of those, and users can't tell which weights are the right ones to load.

**Do this instead:** One repo per dataset (Pattern 1). If grouping is desired, use a HF Collection — that gives a discovery view without breaking single-repo semantics.

### Anti-Pattern 4: Loading all 7 models in the Space at startup

**What people do:** At the top of `app.py`, eagerly call `from_pretrained` for all 7 models so they're "warm" when the user picks one.

**Why it's wrong:** `cpu-basic` Spaces have 16 GB RAM. Seven EVA02-L-14-class models won't fit. Even if they did, cold start would be many minutes — Spaces have a startup-timeout (default 30 min, configurable) and an unhealthy Space gets flagged. Most users will only try 1-2 models, so 5-6 of the 7 loads are wasted.

**Do this instead:** Lazy load on first selection (Pattern 4). Pair with `preload_from_hub` in the README YAML to warm the disk cache so the first per-model load is from local disk, not network.

### Anti-Pattern 5: `trust_remote_code=True` without `revision=`

**What people do:** README example reads `AutoModelForImageClassification.from_pretrained(repo_id, trust_remote_code=True)` with no `revision=` argument.

**Why it's wrong:** `trust_remote_code=True` executes arbitrary Python from the Hub repo. Without pinning to a specific commit, the user implicitly trusts whatever the latest commit is — which could change without notice. The HF docs explicitly recommend pinning revisions for security.

**Do this instead:** Tag the published commit (e.g. `v1.0.0`) and put `revision="v1.0.0"` in every code snippet in the README and the model card. Document this as a security best practice in the model card body.

### Anti-Pattern 6: Editing model cards by hand after generation

**What people do:** Run `publish_model.py`, then go to each model repo on the web UI and manually fix typos / add per-model nuance.

**Why it's wrong:** Drift. Six months later the template updates and someone re-publishes; the manual edits silently disappear. Or the manual edits never propagate to a regenerated card and reviewers see two different stories on different models.

**Do this instead:** All edits go in `release/card_template.md.j2` or in the per-model fields of `release/manifest.yaml`. Re-running `publish_model.py` is always idempotent. If there's a piece of per-model content that doesn't fit cleanly, add a new field to the manifest.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| HF Hub (models, org `project-oceania`) | `huggingface_hub.create_repo`, `model.push_to_hub`, `ModelCard.push_to_hub`, `metadata_update` | Auth via `HF_TOKEN` env var (already wired per INTEGRATIONS.md). Org token must have write access. |
| HF Hub (datasets) | Already integrated via existing `DatasetImporter._push_to_hub` pipeline | No new code in this milestone unless dataset cards need content edits. |
| HF Hub (Spaces) | Either `huggingface_hub.create_repo(repo_type="space", space_sdk="gradio")` + `upload_folder`, or `git push` to the Space repo URL | Recommend `upload_folder` from `release/sync_space.py` so it stays scriptable. |
| HF Inference Widget | Activated automatically by `pipeline_tag: image-classification` in model card YAML | No code, just metadata. Provides a per-model browser-side widget for quick checks alongside the dedicated Space. |
| W&B / Weights & Biases | Read-only at release time — used only to look up which run owned the winning checkpoint | No new W&B writes from `release/`. |
| Lustre / cluster filesystem | Read-only at release time — `release/eval_model.py` and `publish_model.py` need read access to `/lustre/.../checkpoints/...` to load the winning weights | Easiest if these are run from a cluster login node. Alternative: mirror selected checkpoints to a private Hub repo first, then `publish_model.py` reads from Hub. |
| PyPI (`open-clip-torch`) | A clean-env dependency for users loading ClipClassifier-based models | Pin to a known-compatible version range in the model card's "Requirements" section, NOT in any HF-side file (HF model repos don't have requirements.txt — Spaces do, but model repos don't). |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `release/publish_model.py` ↔ `planktonzilla/` package | One-way: `publish_model.py` may import from `planktonzilla` to load a checkpoint, but the *output* (the standalone `modeling_planktonzilla.py` shipped to Hub) MUST NOT import anything from `planktonzilla`. | This is the load-bearing constraint that lets DOC-02 work. Easy to violate by accident. Mitigation: write a CI-style check in `publish_model.py` that greps the staged `modeling_planktonzilla.py` for `import planktonzilla` and fails if found. |
| GitHub repo `space/` ↔ HF Space repo | Sync via `huggingface_hub.upload_folder(repo_type="space", folder_path="space/")` from a `release/sync_space.py` script | Keeps the "code lives in GitHub" invariant. Optional: a GitHub Action that triggers the sync on push to `main` — but PROJECT.md says CI hardening is deferred, so this is a v1.1 concern. |
| `release/modeling_planktonzilla.py` ↔ `planktonzilla/clip_model.py` | Source-level duplication. The release-side file is a hand-maintained standalone version of the training-side `ClipClassifier`. | Drift risk. Mitigation: add a comment header in `clip_model.py` ("if you change this, also update release/modeling_planktonzilla.py") and a smoke test that loads the published model and verifies its forward output matches the training-side one given the same weights and same input. |
| Each model repo ↔ each dataset repo | Pure metadata link via `datasets:` YAML in the model card | Hub auto-renders the back-link on the dataset card too. |
| Space ↔ all 7 model repos | Runtime `from_pretrained` calls from `app.py`; metadata link via `models:` YAML in Space README | Both paths are needed: YAML for the visible badges on the Hub page, runtime calls for actual functionality. |

## Sources

**HIGH confidence (Context7 + official HF docs, verified):**
- [Hugging Face Hub — Customizing models](https://huggingface.co/docs/transformers/en/custom_models) — definitive `register_for_auto_class` / `auto_map` / `trust_remote_code` workflow, including the `modeling.py` + `configuration.py` repo layout requirement.
- [Hugging Face Hub — Loading models](https://huggingface.co/docs/transformers/models) — `trust_remote_code=True` security guidance + `revision=` pinning recommendation.
- [Hugging Face Hub — Model Cards](https://huggingface.co/docs/hub/en/model-cards) — model-index / EvalResult YAML schema, recognized YAML fields (`pipeline_tag`, `library_name`, `datasets`, `base_model`, `tags`, `language`, `license`).
- [Hugging Face Hub — Annotated Model Card Template](https://huggingface.co/docs/hub/en/model-card-annotated) — section structure for "intended use / limitations / training procedure / eval".
- [Create and share Model Cards (huggingface_hub guide)](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards) — `ModelCard`, `ModelCardData`, `EvalResult`, `from_template`, `push_to_hub`, `metadata_update` APIs.
- [Hugging Face Spaces config reference](https://huggingface.co/docs/hub/en/spaces-config-reference) — every YAML field in a Space README, including `sdk`, `app_file`, `models`, `datasets`, `preload_from_hub`, hardware flavors.
- [Gradio Spaces docs](https://huggingface.co/docs/hub/en/spaces-sdks-gradio) — required files (`app.py`, `requirements.txt`, README YAML), Gradio version pinning via `sdk_version`.
- [HF Hub Repositories — uploading models](https://huggingface.co/docs/hub/models-uploading) — recommended files (`config.json`, weights as `.safetensors`, `README.md`, processor configs).
- Context7 `/huggingface/huggingface_hub` — verified `ModelCard.from_template` + `EvalResult` + `push_to_hub` + `metadata_update` + `upload_folder` snippets.
- Context7 `/huggingface/transformers` — verified `register_for_auto_class("AutoModelForImageClassification")` + `push_to_hub` + `trust_remote_code=True` snippets.

**MEDIUM confidence (single official source or pattern observed in HF blog/forums):**
- [Using & Mixing Hugging Face Models with Gradio 2.0 (HF blog)](https://huggingface.co/blog/gradio) — pattern of multi-model Gradio Spaces.
- [Showcase Your Projects in Spaces using Gradio (HF blog)](https://huggingface.co/blog/gradio-spaces) — Space-deployment workflow.
- [Using Hugging Face Integrations (Gradio docs)](https://www.gradio.app/guides/using-hugging-face-integrations) — model-loading patterns inside `app.py`.
- [Diffusers/Gradio memory-leak discussion](https://github.com/huggingface/diffusers/discussions/10936) — guidance that model loading should happen in event-handler functions, not at module top-level, to avoid context issues.
- [Gradio Examples docs](https://www.gradio.app/main/docs/gradio/examples) — `cache_examples="lazy"` default behaviour on HF Spaces.

**LOW confidence (general patterns, not protocol-defined):**
- The "GitHub `space/` directory mirrored to HF Space repo" pattern is widely used by HF org repos but not documented as a Hub feature — it's a project-side convention.
- The exact PyPI `open-clip-torch` version range needed by `release/modeling_planktonzilla.py` requires a small spike against the vendored `4.0.0.dev0` API surface — flagged as a Phase B sub-task, not asserted here.

---
*Architecture research for: public release of fine-tuned plankton image classifiers across GitHub + HF Hub + HF Spaces*
*Researched: 2026-05-12*
