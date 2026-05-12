# Feature Research

**Domain:** Public release of a curated, citable scientific image-classification model collection on Hugging Face Hub (plankton / marine microscopy)
**Researched:** 2026-05-12
**Confidence:** HIGH for HF Hub conventions and table stakes (sourced from official HF docs + multiple in-domain exemplars). MEDIUM for plankton/marine specifics (small sample of public exemplars; one is auto-generated, one is encoder-only — leadership in the niche is open).

## Scope Note

This research is for the `planktonzilla` v1 release milestone — preparing an *existing* trained-model collection for public consumption. It explicitly does NOT re-research training pipelines, dataset import, or losses (that work is `Validated` in `.planning/PROJECT.md`).

The driving question: **what does a public scientific ML release on HF Hub look like such that it's credible to ML researchers, usable by marine biologists, citable by paper reviewers, and forkable by students?**

Reference exemplars used throughout (all examined directly):
- General CV baselines: [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50), [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224), [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base)
- Scientific foundation models: [`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [`imageomics/bioclip-2`](https://huggingface.co/imageomics/bioclip-2)
- Plant / ecology adjacent: [`gerald29/plantclef2024`](https://huggingface.co/gerald29/plantclef2024), [`vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights`](https://huggingface.co/vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights)
- Plankton (in-niche): [`Binou/vit-base-plankton`](https://huggingface.co/Binou/vit-base-plankton), [`Jookare/plankton_vit_large_patch16_224.mae`](https://huggingface.co/Jookare/plankton_vit_large_patch16_224.mae)
- Org-level pattern: [`imageomics`](https://huggingface.co/imageomics) organization profile, 18 models / 41 datasets / 10 Spaces / 7 collections
- Authoritative refs: [HF Model Release Checklist](https://huggingface.co/docs/hub/en/model-release-checklist), [Model Cards](https://huggingface.co/docs/hub/model-cards), [Eval Results spec](https://huggingface.co/docs/hub/eval-results), [Spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference), [Collections](https://huggingface.co/docs/hub/collections)

## Feature Landscape

### Table Stakes (Public Scientific ML Release Without These Looks Unprofessional)

These are the things a paper reviewer or marine biologist would penalize the project for missing. Confirmed across **all** general-CV exemplars, **all** scientific-foundation exemplars, and the [HF Model Release Checklist](https://huggingface.co/docs/hub/en/model-release-checklist).

#### Per-Model Card Content (the README.md inside each `project-oceania/*` repo)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **YAML frontmatter: `license`, `library_name`, `pipeline_tag`, `tags`, `datasets`, `base_model`, `language`** | Drives Hub search, the auto-generated code-snippet widget, the dataset cross-link, and the model-tree visualization. Missing → model is invisible in filtered search. | LOW | Use `library_name: transformers` for the HF-classifier models; `library_name: open_clip` for the `ClipClassifier` models (BiomedCLIP and BioCLIP-2 both use this). Tag `image-classification` for pipeline. Cross-link the `project-oceania/{dataset}` and the upstream backbone (e.g. `microsoft/resnet-18`) via `base_model:` + `base_model_relation: finetune`. |
| **YAML frontmatter: `model-index` with eval results** | This is the single biggest signal of a "real" release vs an experiment dump. Auto-displays an evaluation widget on the model page. ResNet-50, ViT, BioCLIP, PlantCLEF2024 all carry it. | MEDIUM | Format: per-task `task.type=image-classification`, `dataset.name=project-oceania/{dataset}`, `dataset.type=image-classification`, then one `metrics:` entry per metric (top-1 accuracy at minimum; macro-F1 strongly preferred for class-imbalanced plankton). Schema: [HF docs](https://huggingface.co/docs/hub/model-cards#evaluation-results). The newer simpler `.eval_results/` YAML format ([eval-results docs](https://huggingface.co/docs/hub/eval-results)) is for benchmark-leaderboard datasets — the inline `model-index` is the right choice for project-internal evals. |
| **Held-out test split with reproducible numbers** | Without a test-split number, the model card is unfalsifiable. Reviewers will not cite. | MEDIUM | The `Trainer.push_to_hub()` autogen card already records `eval_loss` / `eval_accuracy` (see [`Binou/vit-base-plankton`](https://huggingface.co/Binou/vit-base-plankton)). v1 must promote this from the autogen blob into structured `model-index` and add macro-F1 + per-class accuracy table. Plankton-specific: report top-1, top-5, and macro-F1 — top-1 alone is misleading on long-tailed plankton distributions. |
| **"Model description" section** | Architecture, pretraining backbone, fine-tuning recipe in 3-5 sentences. Standard across all exemplars. | LOW | One paragraph: backbone (e.g. "ResNet-18 fine-tuned from `microsoft/resnet-18`"), training data (link to dataset card), key training choices (loss family, image resolution, epochs). |
| **"Intended uses & limitations" section** | Required by the [Mitchell 2018 model-card framework](https://arxiv.org/abs/1810.03993) HF references, present in 100% of exemplars. For scientific models this is *the* trust-building section. | LOW-MEDIUM | Plankton-specific: state imaging modality (FlowCam vs ISIIS vs UVP6 vs IFCB are visually different sensors); state taxonomic scope (which classes, what taxonomic level — species/genus/morphotype); state biogeographic scope (which oceans / regions the training data covers). |
| **"Bias, Risks, and Limitations" section** | Scientific exemplars (BiomedCLIP, BioCLIP-2) treat this as load-bearing. BiomedCLIP explicitly forbids deployed/commercial use; BioCLIP-2 names long-tail bias and conservation risk. | MEDIUM | Plankton-specific call-outs: (a) long-tailed class distribution → low-frequency taxa under-served (the imbalance loss work is the *response* to this, not the *resolution*); (b) trained on labeled-data biogeography may not transfer to other oceans or seasons; (c) species-level vs morphotype-level boundaries vary by source dataset; (d) **explicitly state "not a regulatory or ecological-monitoring tool" if not validated for that** — this is the BiomedCLIP pattern. |
| **"How to use" / "How to Get Started" copy-pasteable inference snippet** | Universal across exemplars. Must work in a clean Python env from a single `pip install` line. This is verbatim DOC-02 in `.planning/PROJECT.md`. | MEDIUM | Two snippets: (a) `transformers.AutoModelForImageClassification.from_pretrained("project-oceania/...")` + `AutoImageProcessor` for the HF-classifier models; (b) `open_clip.create_model_and_transforms('hf-hub:project-oceania/...')` for the `ClipClassifier` models — this is exactly the BioCLIP-2 pattern. The latter implies the `ClipClassifier` either needs to be loadable without the `planktonzilla` repo, or the v1 CLIP releases are the open_clip-native checkpoints (decision needed — flag for roadmap). |
| **Citation block (BibTeX)** | Universal across general CV, mandatory for scientific exemplars. Reviewers and downstream papers need it. | LOW | Cite the model itself + the source dataset paper. BiomedCLIP and BioCLIP-2 both bundle multiple BibTeX entries (model paper + tooling paper + dataset paper). For planktonzilla: cite the planktonzilla repo (v1.0.0 GitHub release tag), the source dataset paper for that specific model (e.g. WHOI-Plankton arxiv:1510.00745 for the WHOI model), and optionally the OcéanIA project. |
| **License declaration (model + dataset, distinct)** | A model derived from a CC-BY-NC dataset cannot itself be MIT-licensed without addressing the upstream restriction. | MEDIUM (legal, not technical) | Plankton-specific concern: `INTEGRATIONS.md` shows ISIISNet is **CC-BY-NC-4.0**, ZooLake/Lensless are CC-BY-4.0, WHOI-Plankton is MIT, and several others (FlowCamNet, UVP6Net, JEDI-Oceans) have undocumented licenses in the importer configs. Each model card must declare both its own license AND the source-dataset license, and the model license must be compatible with the dataset's. The ISIISNet-derived model cannot be released as MIT — likely needs CC-BY-NC-4.0 or `license: other` with a clear `license_name` / `license_link`. **Audit the licenses of the 7 target datasets BEFORE deciding what to publish on each model card.** |
| **`safetensors` weight format** | Explicitly recommended by HF over `pickle`/`.bin` for safety + speed. Most current scientific releases ship `.safetensors`. | LOW | `transformers.Trainer.push_to_hub` already emits `.safetensors` by default in modern versions. CLIP/`open_clip` checkpoints also support safetensors via `open_clip.create_model_and_transforms(..., load_weights_only=True)`. Verify each pushed checkpoint, convert if not. |
| **Reproducibility metadata: training config / framework versions** | The Trainer-autogen card already includes hyperparameters + framework versions block. Keep it; don't strip it when humanizing the card. | LOW | Promote-and-keep, don't replace. The hand-written narrative goes *above* the autogen training-procedure block. |

#### Per-Dataset Card Content (`project-oceania/{dataset}` README)

The 9 dataset repos already exist on the Hub (per `INTEGRATIONS.md`); their cards are the responsibility of this milestone if the model cards link to them.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **YAML: `task_categories`, `language`, `license`, `size_categories`, `tags`, `source_datasets`** | Drives dataset-page widget and search filters. The existing `project-oceania/whoi-plankton` card has this — confirmed adequate baseline. | LOW | Add `tags: [biology, marine-biology, plankton, microscopy]`, `task_categories: [image-classification]`. |
| **Source attribution + DOI/URL** | Dataset cards must credit the original publisher (WHOI, Eawag/ZooLake, IBM/Lensless, etc.) and link to the canonical source. | LOW | The importer configs already carry the source URLs. Render them into the cards. |
| **License (potentially different from model license)** | See above — heterogeneous across the 7 datasets. | LOW per card (the audit is the work). | |
| **Class distribution** | The existing `project-oceania/whoi-plankton` card already includes this — and it's load-bearing for a reviewer assessing class imbalance. Confirmed table-stakes via this exemplar. | MEDIUM | Auto-generatable from the parquet split. |
| **Citation BibTeX** | Confirmed in existing card. Must include the original dataset paper, not the planktonzilla wrapper. | LOW | |

#### Repo-level (the `Inria-Chile/deep_plankton` GitHub repo)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **README "load a published model" snippet that works on a clean env** | This is `DOC-02` in PROJECT.md — verbatim user-facing requirement. Without it the model cards' inference snippets are unverifiable claims. | MEDIUM | The unresolvable `transformers ^5.3.0` constraint in `pyproject.toml` (HARD-01 in PROJECT.md) blocks this directly — the snippet is `pip install transformers torch huggingface_hub` followed by `from_pretrained(...)`, and it must work without cloning planktonzilla. For the ClipClassifier models, this requires either un-vendoring `open_clip` or releasing those checkpoints in `open_clip`-native format. |
| **README four-use-case structure** | This is `DOC-01` in PROJECT.md. | MEDIUM | (a) install, (b) load published model + infer, (c) retrain on your own data, (d) import a new dataset. |
| **`CITATION.cff`** | Standard for citable scientific software — GitHub renders a "Cite this repository" widget. Companion to per-model BibTeX. | LOW | One file at repo root, points at the v1.0.0 release. |
| **GitHub Release with version tag** | Anchors the citation. PROJECT.md REL-03 already lists "optional v0.2.0 / v1.0.0 GitHub release"; reframe as table-stakes for the citation chain. | LOW | Tag matching the version baked into model cards. |
| **MIT LICENSE file** | Already in place (validated). | DONE | |

### Differentiators (Competitive Advantage in the HF "Model Garden")

These features set the planktonzilla release apart from the existing plankton models on the Hub (`Binou/vit-base-plankton` is autogenerated and minimal; `Jookare/plankton_vit_large_patch16_224.mae` is encoder-only with `More information needed` placeholders) and align with the high-quality scientific exemplars (BioCLIP-2, BiomedCLIP).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **HF Collection grouping the 7 models** | The single highest-leverage discoverability win. Per [HF Collections docs](https://huggingface.co/docs/hub/collections) and [Release Checklist](https://huggingface.co/docs/hub/en/model-release-checklist): "Collections help users discover related models and understand relationships across versions." `imageomics` does this with 7 collections. A landing-page URL like `huggingface.co/collections/project-oceania/planktonzilla-v1` is what gets shared in talks, papers, and Slack. | LOW | Web UI → "+ New" on org page. Group: 7 model repos + 7 dataset repos + 1 Space + 1 collection note. Add a per-item note explaining which sensor / which dataset. |
| **HF Spaces Gradio demo (`DEMO-01`)** | Already in v1 scope. Differentiator vs the autogenerated `Binou/vit-base-plankton` (no demo). Per the Release Checklist: "Create a Hugging Face Space with an interactive demo. This lets users try your model without writing code." | MEDIUM | Spec already constrained: single-image upload, model picker (7 options), top-K with probabilities, no Grad-CAM. `sdk: gradio`, `suggested_hardware: t4-small` (CPU is too slow for ViT/EVA-class models, t4-small is the cheapest GPU tier — a10g or l4x1 if needed for multi-model warm-load), use `models:` YAML field to declare the 7 model deps so they cross-link, use `preload_from_hub:` to avoid cold-start downloads. Cache examples (`cache_examples=True`) so the gallery loads instantly. |
| **Org-level README on `huggingface.co/project-oceania`** | The org page is what a reviewer or potential user lands on when they Google "OcéanIA plankton". `imageomics` uses this for mission, funding (NSF #2118240 is named explicitly), team, and discoverability. | LOW | One markdown page, the org's "elevator pitch": OcéanIA mission, Inria Chile attribution, collection links, paper links, contact. **Inria-specific value:** acknowledge funding, link to the OcéanIA project page. |
| **Per-class example images on each model card** | The dataset cards already have class lists (existing WHOI card has class-distribution table). Adding 1-2 representative images per class on the model card is the "visual examples" pattern from the Release Checklist (`<Gallery>` component) and is highly differentiating for a domain experts who aren't ML people. | MEDIUM | The 32-class `data/isiisnetdatasetimporter_imagefolder/` already exists locally — extracting one example per class is mechanical. Risk: 100+ classes for WHOI-Plankton/ZooLake makes a full gallery unwieldy. Mitigation: top-5 most-common + 5 hardest classes (lowest per-class accuracy from the eval). |
| **Notebook in repo: end-to-end load + infer for each model** | Release Checklist explicitly recommends "well-structured `notebook.ipynb` in the repo showing inference or fine-tuning, so users can open it in Google Colab and Kaggle Notebooks directly." This converts the README claim into runnable evidence. | MEDIUM | One `notebooks/load_planktonzilla_models.ipynb` that walks through all 7. The repo already has a `notebooks/` directory and Jupyter tooling installed, so no new infra. Add a Colab badge to the README. |
| **Inference dependency hygiene: works without cloning planktonzilla** | This is `DOC-02` material; calling it out as a differentiator because **most plankton/scientific models on the Hub fail this** (e.g. `Jookare/plankton_vit_large_patch16_224.mae` ships only the encoder). A complete, classifier-included, runs-from-pip release is genuinely rare in this niche. | MEDIUM-HIGH | Forces a decision on the `ClipClassifier` packaging: option A ship CLIP releases as native `open_clip` checkpoints (BioCLIP-2 pattern, no custom class needed); option B publish a tiny `planktonzilla-inference` PyPI sidecar containing only `ClipClassifier`. Option A is preferred for v1 since PyPI is out of scope. |
| **DOI minted on HF (Hub-issued DOI)** | HF mints DOIs for models — `imageomics/bioclip-2` carries `DOI: 10.57967/hf/5765`. Makes the model formally citable in academic publications and gets indexed by Crossref. | LOW | Click in HF UI: "Settings → Generate DOI" per model. Free. Big credibility signal for paper reviewers. Most general CV models DON'T have this, BiomedCLIP/BioCLIP-2 do. |
| **Bilingual (FR/EN) model cards or org page** | The Inria audience is partly French-speaking; the dataset providers (Sorbonne, IFREMER for ISIIS) are French. Differentiates from the US-anchored norm. | MEDIUM | Optional per model card via the language YAML list (`language: [en, fr]`); more impactful as a single bilingual org-level README. |
| **Linked Paper Page** | If/when there's an arxiv preprint, the Release Checklist auto-cross-links via `arxiv:NNNN.NNNNN` tag. Even a tech report on arxiv as a single anchor for all 7 models would multiply citation discoverability. | LOW (the linking) / OUT-OF-SCOPE (writing the paper) | Out of scope for v1 per PROJECT.md ("Per-dataset benchmark paper / leaderboard — out of scope"). Note: the moment any paper exists, add `arxiv:` to every model card's tags. |
| **CO2 emissions metadata** | YAML `co2_eq_emissions:` block — recommended by Release Checklist. Differentiator for a sustainability-conscious / EU-funded project. | LOW (if logged) / MEDIUM (if not) | If the original training runs logged compute (W&B does), backfill via [CodeCarbon](https://codecarbon.io). If not logged at training time, skip rather than fabricate. |

### Anti-Features (What Scientific ML Releases Deliberately Don't Ship at v1)

| Feature | Why Tempting | Why Problematic | Alternative |
|---------|--------------|-----------------|-------------|
| **"SOTA" badges or claims in the model card** | Easy way to grab attention; many community cards do it. | Without a peer-reviewed benchmark or a public leaderboard you can prove it on, "SOTA" is overclaim and is the first thing a paper reviewer notices. None of the high-quality exemplars (ResNet-50, ViT, BiomedCLIP, BioCLIP-2) use the term. BioCLIP-2 carefully says "improves on" with numbers. | Report numbers honestly, on named splits, with confidence intervals where feasible. Let the reader conclude. |
| **"Production-ready" / "for clinical use" / "for regulatory monitoring" framing** | Marine-biology audience may want to use these for real surveys. | Implies a level of validation, support, and uptime planktonzilla cannot sustain. BiomedCLIP's model card explicitly excludes deployed use ("Any deployed use case ... is currently out of scope"). | Adopt the BiomedCLIP pattern: "Primary intended use: research only — supporting researchers building on plankton classification." Name out-of-scope uses (regulatory monitoring, biodiversity reports, automated species inventories) that would require additional validation. |
| **Saliency / Grad-CAM in the demo** | Aids interpretability for biologists; widely requested. | Already explicitly out of scope per PROJECT.md (CNN vs ViT/CLIP saliency techniques diverge; non-uniform UX across 7 models). | Defer to v1.1. Document as future-work in Limitations section. |
| **A dedicated docs site (MkDocs / Sphinx / GitHub Pages)** | Looks polished. | Already explicitly out of scope per PROJECT.md. The HF model cards + GitHub README are the documentation surface. | One README, four use cases. Ship it. |
| **PyPI package** | Lowers install friction. | Out of scope per PROJECT.md. The clone-and-load and the from-Hub-load paths cover the audiences. | Mention "PyPI release planned" in README. The clean-env load snippet (DOC-02) already validates the use case without PyPI. |
| **A unified leaderboard page comparing the 7 models** | Tempting because the 7 models all do "plankton classification." | The 7 models classify *different things over different label spaces from different sensors*. Comparing top-1 accuracy across them is meaningless. A naive leaderboard misleads users into picking "the best one" when the right answer is "the one matching your sensor." Per PROJECT.md: "Per-dataset benchmark paper / leaderboard — out of scope." | Each model card carries its own numbers. The Collection page groups them with per-item notes describing the sensor / dataset. |
| **Auto-uploaded `runs/` / wandb traces / TensorBoard logs in the model repo** | Free reproducibility theatre — the Trainer can do it automatically. | Bloats repo size, exposes training-internal failed runs, and the wandb runs are already tracked at `oceania-plankton/planktonzilla-turbo`. Per `INTEGRATIONS.md`, the local `wandb/` dir already has 20 historical runs — none of which belong on Hub. | Link to the wandb project (or the Trackio HF dataset `project-oceania/pz_experiments`) from the model card; don't dump raw traces. |
| **Eval results submitted to HF Benchmark leaderboards (`.eval_results/` format)** | New shiny feature ([eval-results docs](https://huggingface.co/docs/hub/eval-results)). | The `.eval_results/` format requires the dataset to be a registered "Benchmark" with an `eval.yaml`. The plankton datasets aren't registered benchmarks. Registering them is its own multi-week project. | Use the legacy inline `model-index` YAML on model cards — that's universally supported and what every exemplar uses today. Revisit `.eval_results/` if/when planktonzilla datasets become formally benchmarked (post-v1). |
| **Gated access on the models** | Sounds responsible. | The MIT-licensed datasets and the publish-them-openly framing in PROJECT.md don't motivate gating. Gating raises friction for the marine-biology audience without conferring research-ethics benefit. | Public access. Use the Bias/Limitations section for responsible-use guidance instead. The CC-BY-NC-derived models *might* warrant attribution-required language but not gating. |
| **Quantized variants (GGUF / int8)** | Recommended by Release Checklist for large LLMs. | Listed Release Checklist item, but image classifiers in the 86M-100M parameter range don't benefit much from quantization for inference and the `open_clip` ecosystem doesn't standardly distribute quantized image-CLIP. | Out for v1. Reconsider for the largest checkpoints (`timm/eva_giant_patch14_336`, `timm/convnextv2_huge`) only if a downstream user actually requests it. |

## Plankton / Marine Microscopy Domain Specifics

These are the places where scientific-image-classification norms in this niche differ from generic computer vision:

| Topic | Generic CV norm | Plankton/marine norm | Implication for planktonzilla |
|-------|-----------------|----------------------|-------------------------------|
| **Imaging modality** | Often single-modality (RGB photos). | Highly heterogeneous: FlowCam (lens-based), ISIIS (in-situ shadowgraph), UVP6 (underwater vision profiler), IFCB (Imaging FlowCytobot), ZooScan (flatbed scanner), Lensless (no-lens microscopy). Each has different visual statistics, resolution, contrast, color (or grayscale). | Each model card MUST name the imaging instrument. Mixing instruments at inference time is a cross-domain task; documenting this prevents silent misuse. |
| **Class taxonomy** | Flat 1000-class ImageNet-style. | Hierarchical biological taxonomy (kingdom → phylum → class → … → species), with most plankton datasets labeled at *morphotype* level (an operationally identifiable form), not species. Boundaries differ between datasets even for "the same" organism. | Document taxonomic level per dataset (the existing `data/isiisnetdatasetimporter_imagefolder/` shows 32 classes mixing phylum-level (`Annelida`, `Cnidaria`) and other ranks). Don't claim "species classification" if labels are morphotypes. |
| **Class imbalance** | Often modest; ImageNet is roughly balanced. | Extreme: a single dominant taxon often >50% of frames; rare-class long tail with single-digit examples. The whole point of the imbalance loss family in `planktonzilla/loss.py` (Focal, LDAM, Asymmetric, RAL) is to address this. | Always report **macro-F1** alongside top-1; per-class metrics are essential. The default `eval_accuracy` from `Trainer.push_to_hub` is dangerously misleading on these datasets. |
| **Biogeographic transfer** | Less of a story — ImageNet doesn't have a "where" dimension. | Models trained on Mediterranean ISIIS data may not work on North Atlantic UVP6 data — different communities, different morphological selection. | Limitations section: name the geography of the training data. SeanOE/ISIIS has provenance metadata; surface it. |
| **Citation chain** | Cite the model paper (1 BibTeX). | Cite the model paper, the source dataset paper(s), the imaging instrument paper, sometimes the cruise/expedition. BioCLIP-2 stacks 4 BibTeX entries; this is normal in biology. | Each model card BibTeX section: planktonzilla repo + dataset paper + (optional) instrument paper. |
| **Demo audience** | ML engineers testing capabilities. | Marine biologists who need to know "is this thing I'm looking at *Chaetoceros*?" — different question, different UX. | Demo design: top-K labels are good, but **also** show 1-2 reference images of the predicted class so the biologist can visually confirm. This is achievable in v1 by serving from the per-class examples already extracted for the model cards (see "Per-class example images" differentiator). |
| **Existing competition** | Saturated. | Sparse. The two plankton models on HF are (a) auto-generated `Binou/vit-base-plankton` and (b) encoder-only `Jookare/plankton_vit_large_patch16_224.mae`. The ecology-foundation model (BioCLIP-2) covers TreeOfLife but is not plankton-specialized. **There is no "plankton classification" landing page on HF**. The Collection becomes that landing page. | Lean into being the canonical reference. "The 7 official planktonzilla models" framing on the Collection page. |

## Feature Dependencies

```
Per-model card content (text)
    └──depends on──> safetensors weights pushed to org repo (already done per Validated)
    └──depends on──> Eval numbers from a held-out test split (re-run if not on disk)
    └──depends on──> License audit per source dataset (SOURCE OF UNCERTAINTY)
                                      │
                                      └──determines──> Model license per repo
                                                            │
                                                            └──may force──> "license: other" + license_name + license_link

Per-model card content
    └──cross-references──> Dataset card on project-oceania/{dataset}
                                      │
                                      └──depends on──> Dataset card content (largely already in place per WHOI exemplar)

"Load a published model" inference snippet (DOC-02)
    └──depends on──> HARD-01 (transformers ^5.3.0 constraint fix)
    └──depends on──> Decision about ClipClassifier packaging (option A: open_clip-native checkpoints; option B: defer CLIP releases; option C: PyPI sidecar — out of scope)

Spaces demo (DEMO-01)
    └──depends on──> All 7 model cards published (so models: YAML field can cross-link)
    └──depends on──> The same load-snippet-works-without-clone constraint as DOC-02

HF Collection
    └──depends on──> All 7 model cards published + 7 dataset cards present + Spaces demo live
    └──enhances──> Discoverability of every individual artifact

Org-level README
    └──depends on──> Collection URL existing (links into it)
    └──independent of──> Individual model-card readiness (can be drafted in parallel)

CITATION.cff + GitHub release
    └──depends on──> Version number agreed (e.g. v1.0.0)
    └──referenced by──> Every model card's BibTeX entry
```

### Key Dependency Notes

- **License audit is a blocking prerequisite for the model cards.** It cannot be skipped, and several `INTEGRATIONS.md` entries have undocumented dataset licenses (FlowCamNet, UVP6Net, JEDI-Oceans). Resolution: contact upstream maintainers OR drop those models from v1 OR publish them privately first. Recommend: surface this as the first work item in REL-02.
- **HARD-01 (transformers constraint + open_clip path) is a true blocker for DOC-02 and the demo, not a "nice to fix."** The `transformers ^5.3.0` constraint resolves only via `poetry.lock` — a clean-env user without the lockfile cannot install. For the CLIP models, the vendored open_clip + hardcoded `/home/acontreras/...` PYTHONPATH means the model card's load snippet will be a lie for those checkpoints unless we (a) un-vendor, (b) ship as native open_clip checkpoints, or (c) defer the 1-2 CLIP-based releases to v1.1.
- **Collection must be created last** — it links to artifacts that need to exist first. Good Phase-N closer.
- **Per-class example images depend on class-level eval results** (to choose "hardest classes"), not just on the dataset.

## MVP Definition

### Launch With (v1 — maps to PROJECT.md REL-01..REL-03, DOC-01..02, HARD-01, DEMO-01)

The minimum that makes the release credible and falsifiable:

- [ ] **License audit** of all 7 source datasets, with per-model license decision recorded (potentially demotes some models from v1 if the upstream license is incompatible or undocumented and the upstream author is unreachable)
- [ ] **HARD-01 fix** — `pyproject.toml` transformers constraint resolved on a clean env; open_clip path resolved for CLIP-derived models (or those models deferred)
- [ ] **7 model cards** with: full YAML frontmatter (license, library_name, pipeline_tag, tags, datasets, base_model), `model-index` with top-1 + macro-F1 on a held-out test split, Model description, Intended uses & limitations, Bias/Risks/Limitations, How to use (verified copy-pasteable), BibTeX
- [ ] **7 dataset cards** completed to the level of the existing `project-oceania/whoi-plankton` card (source attribution, class distribution, license, citation, usage snippet)
- [ ] **GitHub README** covering 4 use cases (DOC-01); CITATION.cff at root; v1.0.0 GitHub release tagged
- [ ] **Spaces Gradio demo** (DEMO-01) — single-image upload, model picker, top-K probabilities, examples gallery with cached examples, `models:` YAML cross-linking all 7 model repos
- [ ] **HF Collection** at `huggingface.co/collections/project-oceania/planktonzilla-v1` grouping the 7 models + 7 datasets + the Space, with per-item notes
- [ ] **Org-level README** at `huggingface.co/project-oceania` (mission, funding, collection link, contact)

### Add After Validation (v1.x)

Triggered by user feedback after v1 ships:

- [ ] **Saliency / explainability in demo** — when the load-snippet UX is solid and biologists ask "why did it predict X?"
- [ ] **PyPI package** — when external users hit the "I don't want to clone the repo" wall (already on the followup list per PROJECT.md)
- [ ] **DOI per model** — quick win, do early in v1.x (or even in v1 — the cost is one click per model)
- [ ] **Per-class example galleries** at scale (top-N most-common + top-N hardest per model)
- [ ] **CO2 emissions backfill** if the wandb runs have GPU-hours data
- [ ] **Bilingual model cards (FR/EN)** if Inria audience asks
- [ ] **Notebook (`load_planktonzilla_models.ipynb`)** with Colab badge if v1 README load snippet is insufficient for new users

### Future Consideration (v2+)

- [ ] **Registered HF Benchmark** with `eval.yaml` for one or more planktonzilla datasets, enabling the new `.eval_results/` format and a public leaderboard
- [ ] **Tech-report / arxiv preprint** to cross-link via `arxiv:` tag
- [ ] **Quantized variants** of the largest models (only on demand)
- [ ] **Multi-modality demo** (CLIP zero-shot on user-supplied class names) for the CLIP-based releases
- [ ] **Per-dataset benchmark paper** (currently out of scope)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| License audit (per dataset) | HIGH | LOW-MEDIUM | P1 (blocker) |
| HARD-01 transformers constraint fix | HIGH | LOW-MEDIUM | P1 (blocker for DOC-02) |
| Per-model YAML frontmatter (full) | HIGH | LOW | P1 |
| `model-index` with macro-F1 + top-1 | HIGH | MEDIUM | P1 |
| Hand-written Model description / Intended use / Limitations | HIGH | MEDIUM | P1 |
| Bias/Risks/Limitations section (plankton-specific) | HIGH | MEDIUM | P1 |
| Verified copy-pasteable inference snippet | HIGH | MEDIUM | P1 |
| BibTeX citation block | HIGH | LOW | P1 |
| Source-dataset cross-link in YAML | MEDIUM | LOW | P1 |
| Dataset cards completed to WHOI exemplar level | HIGH | MEDIUM | P1 |
| README 4-use-case structure (DOC-01) | HIGH | MEDIUM | P1 |
| CITATION.cff | MEDIUM | LOW | P1 |
| GitHub v1.0.0 release tag | MEDIUM | LOW | P1 |
| Spaces Gradio demo (DEMO-01) | HIGH | MEDIUM | P1 |
| HF Collection grouping all artifacts | HIGH | LOW | P1 |
| Org-level README | MEDIUM | LOW | P1 |
| HF DOI per model | MEDIUM | LOW | P1 (or P2) |
| Per-class example images on model cards | MEDIUM | MEDIUM | P2 |
| Notebook with Colab badge | MEDIUM | MEDIUM | P2 |
| CO2 emissions metadata | LOW-MEDIUM | LOW (if logged) | P2 |
| Bilingual model cards | LOW-MEDIUM | MEDIUM | P3 |
| Saliency/explainability in demo | HIGH | HIGH | v1.1 (deferred) |
| PyPI package | MEDIUM | MEDIUM | v1.x (deferred) |
| Registered HF Benchmark | LOW | HIGH | v2+ |
| Quantized variants | LOW | MEDIUM | v2+ on demand |

**Priority key:**
- P1: Required for v1 launch, in PROJECT.md scope
- P2: Cheap, high-leverage; include in v1 if time permits
- P3: Nice to have, defer cleanly
- v1.1 / v1.x / v2+: Explicitly out of v1, named here for the deferred-list

## Competitor / Exemplar Feature Analysis

| Feature | `microsoft/resnet-50` (general CV baseline) | `imageomics/bioclip-2` (gold-std scientific) | `Binou/vit-base-plankton` (in-niche, autogen) | `Jookare/plankton_vit_large_patch16_224.mae` (in-niche, encoder-only) | **planktonzilla v1 plan** |
|---------|--------------|----------------|--------------------------|--------------------------|--------------------------|
| `model-index` YAML eval results | No (just description) | Yes (across 10+ tasks) | Eval numbers in markdown table only, not structured | No | **YES — top-1 + macro-F1** |
| Bias/Risks/Limitations section | No | YES (long, specific to ecology) | No | No | **YES (plankton-specific)** |
| BibTeX citation | YES | YES (4 entries) | No | YES | **YES (planktonzilla + dataset)** |
| Inference code snippet | YES | YES (open_clip) | No | YES (timm + transformers) | **YES (verified clean-env)** |
| HF Collection membership | n/a (single model) | YES (multiple) | No | No | **YES — grouped collection** |
| Spaces demo cross-link | n/a | YES | No | No | **YES (DEMO-01)** |
| HF DOI | No | YES (10.57967/hf/5765) | No | No | **YES (P1 or P2)** |
| Source dataset cross-link in YAML | YES (imagenet-1k) | YES (TreeOfLife-200M) | YES (plankton_fairscope) | No | **YES (project-oceania/{dataset})** |
| Per-class example images | No | No (in demo, not card) | No | No | **YES (P2)** |
| Org-level README | n/a (Microsoft has one) | YES (Imageomics has one) | n/a (personal account) | n/a | **YES** |
| Plankton-specific (sensor, taxonomic level) | n/a | n/a | No | No | **YES — clear differentiator** |

The two-axis takeaway: **on the YAML / structured-metadata axis, planktonzilla v1 needs to match `imageomics/bioclip-2` (the scientific gold standard).** **On the plankton-specifics axis (sensor disclosure, taxonomic level, class imbalance, biogeographic scope), planktonzilla v1 has no competitors to match — it can define the norm.** Both axes are within the v1 scope as defined in PROJECT.md.

## Sources

### Authoritative HF Documentation
- [Model Cards documentation](https://huggingface.co/docs/hub/model-cards) — YAML metadata schema, `model-index`, `library_name`, `base_model`, license, paper linking [HIGH confidence]
- [Model Release Checklist](https://huggingface.co/docs/hub/en/model-release-checklist) — recommended sections, `safetensors`, Collections, Spaces, demos, CO2 emissions, post-release [HIGH confidence]
- [Eval Results spec](https://huggingface.co/docs/hub/eval-results) — newer `.eval_results/` benchmark-leaderboard format [HIGH confidence]
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference) — Space YAML, `sdk`, `models`, `suggested_hardware`, `preload_from_hub` [HIGH confidence]
- [Collections documentation](https://huggingface.co/docs/hub/collections) — grouping models / datasets / Spaces / Papers [HIGH confidence]

### Exemplar Model Cards (Examined Directly)
- [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50) — general CV baseline pattern [HIGH]
- [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) — ViT exemplar with full Training procedure / Evaluation results sections [HIGH]
- [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) — feature-extraction model card [HIGH]
- [`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — scientific-domain card with intended-use disclaimer pattern [HIGH]
- [`imageomics/bioclip-2`](https://huggingface.co/imageomics/bioclip-2) — biology-specific model card with risk/conservation framing [HIGH]
- [`gerald29/plantclef2024`](https://huggingface.co/gerald29/plantclef2024) — plant-classification community model [MEDIUM]
- [`vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights`](https://huggingface.co/vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights) — PlantCLEF 2024 entry [MEDIUM]
- [`Binou/vit-base-plankton`](https://huggingface.co/Binou/vit-base-plankton) — autogenerated Trainer card (anti-pattern reference) [HIGH]
- [`Jookare/plankton_vit_large_patch16_224.mae`](https://huggingface.co/Jookare/plankton_vit_large_patch16_224.mae) — encoder-only plankton model (incompleteness reference) [HIGH]

### Org-Level Patterns
- [`imageomics`](https://huggingface.co/imageomics) — exemplar scientific-org page (mission, funding, collections, papers, 95-member team) [HIGH]

### Dataset Card Patterns
- [`project-oceania/whoi-plankton`](https://huggingface.co/datasets/project-oceania/whoi-plankton) — own existing dataset card, used as completeness baseline [HIGH]

### Project Internal
- `.planning/PROJECT.md` — v1 active requirements, audiences, constraints
- `.planning/codebase/INTEGRATIONS.md` — current HF integration surface, dataset license inventory, vendored open_clip status, env-var dependencies

---
*Feature research for: planktonzilla v1 public release on HuggingFace Hub*
*Researched: 2026-05-12*
