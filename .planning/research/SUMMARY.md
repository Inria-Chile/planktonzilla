# Project Research Summary

**Project:** planktonzilla — public release of pre-trained plankton image classifiers
**Domain:** Scientific ML model release on Hugging Face Hub + Spaces (marine microscopy)
**Researched:** 2026-05-12
**Confidence:** HIGH

## Executive Summary

planktonzilla is a mature internal research codebase (Hydra + HuggingFace Trainer + custom imbalance losses + vendored open_clip) that needs to be packaged as a credible, citable public artifact for marine biologists, ML researchers, and paper reviewers. The release is not a software product launch — it is an academic artifact release on the HF Hub, benchmarked against the BioCLIP-2/BiomedCLIP quality bar. The recommended approach is: fix only what blocks the load path, publish one model per dataset with full structured model cards, ship a Gradio Space as the interactive face, and close with the GitHub README rewrite. Everything else is explicitly deferred.

Two findings from the research materially change the scope described in PROJECT.md. First, license audit for three datasets (FlowCamNet, UVP6Net, JEDI-Oceans) is currently undocumented and is a hard publication blocker — the PROJECT.md commitment to "all 7 datasets" may shrink to 4 if upstream maintainers cannot be reached before launch. Second, the CLIP-backed models require materially more engineering than the standard HF-architecture models (ResNet, ViT, BEiT): each CLIP-backed model needs a self-contained `modeling_clip_classifier.py` + `configuration_clip_classifier.py` shipped into its Hub repo via the `trust_remote_code=True` pattern, plus a verified `open_clip_torch` PyPI version range that maps to the vendored `open_clip 4.0.0.dev0`. Standard models need only `Trainer.push_to_hub` + an authored model card.

The critical risks are: (1) eval numbers transcribed from W&B rather than reproduced from the published checkpoint — the only safe pattern is a fresh `release/eval_model.py` pass against a pinned held-out split; (2) silent preprocessing drift if `preprocessor_config.json` does not match training-time mean/std; (3) `id2label` ordering inconsistency leading to scientifically wrong species labels; and (4) Gradio Space OOM on the free CPU tier if models are loaded lazily. All four are preventable with explicit gates at MODEL_RELEASE and DEMO phases.

---

## Key Findings

### Recommended Stack

The publishing stack adds four packages to the existing training environment: `huggingface_hub>=1.0,<2` (Hub client), `transformers>=5.0,<6` (resolves the existing `^5.3.0` lock drift that is HARD-01), `gradio>=6.0,<7` (Spaces demo only — lives in `space/requirements.txt`, not in `pyproject.toml`), and `safetensors>=0.5` (already transitive; explicit pin prevents pickle fallback). The `open_clip_torch>=2.30,<3` PyPI package replaces the vendored copy for published Hub artifacts — end users install it from PyPI; the vendored copy stays in-tree for the training side.

The `ClipClassifier` packaging decision is settled: use the `PreTrainedModel` + `register_for_auto_class` + `trust_remote_code=True` pattern (STACK Option A). This is the only option that requires no PyPI release, keeps `AutoModelForImageClassification.from_pretrained(...)` as the single universal load snippet for all 7 models, and survives later open_clip unvendoring. The cost is ~150 lines of new standalone code in `release/modeling_clip_classifier.py` and `release/configuration_clip_classifier.py`.

**Core technologies:**
- `huggingface_hub 1.14.0`: Hub client for repo creation, card push, metadata — required by `transformers>=5`
- `transformers 5.8.0`: `AutoModelForImageClassification`, `Trainer.push_to_hub`, `register_for_auto_class` — resolves HARD-01 lock drift
- `gradio 6.14.0`: `gr.Blocks` + `gr.Dropdown` multi-model Space — Space-only dep, cpu-upgrade tier recommended
- `safetensors>=0.5`: mandatory weight format; `push_to_hub` writes it by default in transformers 5.x
- `open_clip_torch>=2.30,<3`: PyPI replacement for vendored open_clip in published Hub artifacts

**Do not use:** `pickle`/`torch.save` for weights; Gradio <4 API (`gr.inputs.*`); `gr.Interface.load("huggingface/...")` for custom-code models; `evaluate` library (maintenance mode); `huggingface_hub<1.0`; hand-rolled README YAML (use `ModelCard` + `ModelCardData` + `EvalResult`).

### Expected Features

**Must have (table stakes — a paper reviewer will penalize missing any of these):**
- License audit per dataset before any model card is authored — P1 blocker; FlowCamNet/UVP6Net/JEDI-Oceans undocumented
- Per-model YAML frontmatter: `license`, `library_name`, `pipeline_tag`, `tags`, `datasets`, `base_model`
- `model-index` with structured `EvalResult` entries (top-1 + macro-F1 minimum; per-class metrics strongly preferred given extreme plankton class imbalance)
- Nine model card body sections: Model description, Intended use, Out-of-scope uses, Bias/Risks/Limitations, How to use (verified clean-env snippet), Training data, Training recipe, Evaluation, Citation BibTeX + License
- `safetensors` weights; `preprocessor_config.json` with exact training-time mean/std/size
- Eval numbers from a fresh pass against the published checkpoint, not transcribed from W&B
- Dataset cards completed for all published datasets (existing `project-oceania/whoi-plankton` card is the baseline)
- GitHub README covering 4 use cases (DOC-01); `CITATION.cff`; v1.0.0 release tag
- Gradio Space: single-image upload, model dropdown, top-K probabilities (DEMO-01)
- HF Collection grouping all published models + datasets + Space

**Should have (differentiators — no plankton competitors do these):**
- HF DOI per model (one click per model, free, major credibility signal for reviewers)
- Org-level README at `huggingface.co/project-oceania` (mission, funding, collection link)
- Instrument name prominently in Space dropdown (e.g. "ISIIS — In-Situ Shadowgraph")
- OOD example in Space defaults (bubble/debris image demonstrating the confident-wrong limitation)
- Per-class metrics in model cards (confusion matrix or per-class F1 + support count)

**Defer to v1.x or v2+:**
- Saliency/Grad-CAM (already explicitly OOS in PROJECT.md)
- PyPI package (explicitly OOS)
- Per-class example image galleries at scale
- Notebook with Colab badge
- Registered HF Benchmark with `.eval_results/` leaderboard format
- WoRMS AphiaID taxonomy mapping

### Architecture Approach

The release topology is: one GitHub source repo (`Inria-Chile/deep_plankton`) as source of truth → N HF model repos (`project-oceania/pz-<dataset>`) holding weights + cards → 7 existing HF dataset repos → 1 HF Space (`project-oceania/plankton-classifier`) as the interactive demo → 1 HF Collection grouping all artifacts. The GitHub repo gains a `release/` directory (publish scripts, Jinja card template, `manifest.yaml`, eval script, standalone CLIP modeling files) and a `space/` directory (mirrored to the HF Space repo). YAML frontmatter cross-links everything automatically — model cards declare `datasets:` and `base_model:`, Space README declares `models:` and `datasets:`, and HF renders reverse-link badges on every page.

**Major components:**
1. `release/manifest.yaml` — single source of truth for the N (dataset, checkpoint_uri, arch, license) tuples; the curation surface
2. `release/eval_model.py` — fresh held-out eval per checkpoint → `eval_results/<dataset>.json`; gates MODEL_RELEASE
3. `release/publish_model.py` — loads checkpoint, registers auto class (CLIP path only), pushes weights + card; idempotent re-run
4. `release/card_template.md.j2` — Jinja template ensuring all N cards are structurally identical
5. `space/app.py` — `gr.Blocks` + `gr.Dropdown` + lazy-load via `lru_cache`; mirrored to HF Space repo
6. Per-model HF repos — each a standalone `from_pretrained`-loadable artifact; CLIP repos include `modeling_clip_classifier.py`

### Critical Pitfalls

1. **CLIP models unloadable in clean env (C1)** — Ship `modeling_clip_classifier.py` + `configuration_clip_classifier.py` into each CLIP-backed model repo; set `auto_map` in `config.json`; require `trust_remote_code=True`; smoke-test every checkpoint from a fresh venv before MODEL_RELEASE.

2. **License conflict: model declared MIT but derived from CC-BY-NC dataset (C5)** — License audit is a hard gate before any card is authored. ISIISNet requires `cc-by-nc-4.0`. FlowCamNet/UVP6Net/JEDI-Oceans: block publication until upstream license is documented. Model license = most restrictive of (code, data, backbone) licenses.

3. **Eval numbers not reproducible from published checkpoint (C4)** — Never transcribe W&B numbers. Run `release/eval_model.py` against the exact published checkpoint and pinned dataset revision. Gate MODEL_RELEASE on `pz_verify_release` passing within ±0.5pp.

4. **`id2label` wrong or out of order → scientifically incorrect species labels (C2)** — During publish, read `dataset.features["label"].names` (canonical order) and set `model.config.id2label/label2id` from it. Verify index-by-index. A marine biologist will spot a swap in 30 seconds.

5. **Preprocessing drift: `preprocessor_config.json` does not match training-time transforms (C3)** — For CLIP models, write `preprocessor_config.json` explicitly from `open_clip.create_model_and_transforms` output. Re-run eval from published artifacts and assert within 0.5pp of training-time numbers.

---

## Implications for Roadmap

The architecture research 5-phase ladder (A→B→{C‖D}→E) maps cleanly onto the pitfalls phase tags. The canonical build order is:

### Phase 1: Foundation — Checkpoint Curation, License Audit, Eval Methodology
**Rationale:** Nothing downstream can start without knowing which checkpoint per dataset is being published, where it lives, whether its source license permits publication, and what its reproducible eval numbers are. License audit belongs here — it may reduce the v1 set from 7 to fewer and changes every downstream artifact.
**Delivers:** `release/manifest.yaml`; license decision matrix per dataset; CLIP-vs-standard classification per model; `release/eval_results/*.json` (fresh held-out eval per cleared checkpoint)
**Addresses:** C4 (eval reproducibility), C5 (license conflict), C2 (id2label alignment starts here)
**Avoids:** Discovering mid-publish that 3 datasets have no usable license, or that W&B numbers do not match the published checkpoint
**Research flag:** No additional research needed. License resolution for FlowCamNet/UVP6Net/JEDI-Oceans requires contacting upstream maintainers; set a 2-week deadline; defer if no response.

### Phase 2: One-Model Spike — End-to-End Publish + Clean-Env Verify
**Rationale:** Front-load unknowns. Get one model (recommend ISIISNet — clearest license, strongest existing dataset card) fully published and verified from a clean Python env before scaling to 7. This proves the CLIP custom-code path works, validates the card template, and de-risks the remaining models. HARD-01 is resolved here.
**Delivers:** `release/modeling_clip_classifier.py` + `release/configuration_clip_classifier.py`; `release/card_template.md.j2`; `release/publish_model.py`; one live `project-oceania/pz-<dataset>` repo; smoke-test passing in a fresh venv; `transformers ^5.3.0` lock drift fixed; `open_clip_torch` PyPI version range verified
**Avoids:** C1 (CLIP unloadable), C3 (preprocessing drift), C2 (id2label)
**Research flag:** The exact `open_clip_torch` version range compatible with vendored `open_clip 4.0.0.dev0` is a Phase 2 spike (LOW confidence in research; one afternoon of API surface comparison).

### Phase 3: Publish Remaining Models (parallel with Phase 4)
**Rationale:** Mechanical scale-up of Phase 2 with lessons applied. Each model gets the full MODEL_RELEASE gate checklist. Collection created last once all model repos exist.
**Delivers:** Remaining `project-oceania/pz-<dataset>` repos with full model cards; `pz_verify_release` passing for each; HF Collection
**Implements:** Card template (templated, not hand-written), one repo per dataset
**Avoids:** M1 (missing card sections), M2 (broken eval frontmatter), M3 (private dataset links), M4 (per-class metrics missing), D1–D4 (all domain-specific disclaimers)
**Research flag:** No additional research needed. Standard patterns fully documented.

### Phase 4: Gradio Space (parallel with Phase 3)
**Rationale:** Depends on at least one published model (Phase 2 output). Can be developed with one model and expanded as Phase 3 models go live. Hardware tier decision should be made after benchmarking the heaviest models on cpu-upgrade.
**Delivers:** Live `project-oceania/plankton-classifier` Space; all published models loadable via dropdown; cold start <30s; `requirements.txt` with exact pinned versions
**Uses:** `gradio 6.14.0`, `gr.Blocks` + `gr.Dropdown`, lazy load via `lru_cache`, `preload_from_hub` YAML field
**Avoids:** C6 (cold start / OOM), C7 (deps pinning), M5 (cherry-picked examples), M7 (Gradio security), D1 (instrument name in dropdown), D3 (OOD example in defaults)
**Research flag:** Hardware tier (cpu-upgrade vs t4-small) — benchmark EVA02-L-14 and ConvNeXtV2-Huge inference latency on cpu-upgrade before deciding. MEDIUM confidence on current latency estimates.

### Phase 5: GitHub README Rewrite + Launch
**Rationale:** Last because it requires working URLs for everything it links to. DOC-02 tested in a clean venv is the final integration test for the entire pipeline.
**Delivers:** Rewritten `README.md` with 4 use cases (DOC-01); `CITATION.cff`; verified DOC-02 clean-env snippet; v1.0.0 GitHub release tag; launch blurb (REL-03); org-level `huggingface.co/project-oceania` README
**Avoids:** M3 (dead links), M6 (unsustainable support promises), C5 (no commercial-use claim for CC-BY-NC models in announcement), D2 (no "trained on plankton" claim without dataset-specific caveat)
**Research flag:** No additional research needed.

### Phase Ordering Rationale

- **Foundation first** because license and checkpoint provenance gate every other artifact. A license problem discovered in Phase 3 forces re-work of every card already published.
- **One-model spike second** because the CLIP custom-code path is the highest-risk unknown. Proving it for one model before writing 7 card templates prevents wasted effort.
- **Phases 3 and 4 parallel** because they share no dependencies after Phase 2. The Space can run against 1 model while the model scale-up is independent.
- **README last** because it cannot be verified until the artifacts it links to exist.

### Research Flags

Needs additional spikes during planning:
- **Phase 2:** `open_clip_torch` PyPI version range vs vendored `open_clip 4.0.0.dev0` — one-afternoon API surface spike.
- **Phase 4:** Hardware tier — benchmark EVA02-L-14 + ConvNeXtV2-Huge on cpu-upgrade before committing to tier.

Standard patterns (skip research phase):
- **Phase 1:** Checkpoint survey, eval script, license matrix — well-understood scope, no new tooling.
- **Phase 3:** Card templating with `ModelCard.from_template` — HIGH confidence, fully documented.
- **Phase 5:** README structure — already constrained by PROJECT.md DOC-01 use cases.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified against PyPI and official HF docs 2026-05-12. `open_clip_torch` version range is MEDIUM pending Phase 2 spike. |
| Features | HIGH (HF conventions) / MEDIUM (plankton domain) | HF model card requirements from official docs + BioCLIP-2/BiomedCLIP examination. Plankton-specific norms from 2024-2025 peer-reviewed benchmarks; recommend marine biologist review before LAUNCH. |
| Architecture | HIGH | Topology and patterns verified against HF Hub docs, Transformers custom_models docs, Spaces config reference. Lazy-load `lru_cache` pattern is MEDIUM (observed, not protocol-defined). |
| Pitfalls | MEDIUM-HIGH | HF/Gradio pitfalls are HIGH (official docs + current GitHub issues). Plankton domain pitfalls are MEDIUM (two recent peer-reviewed benchmark papers). |

**Overall confidence:** HIGH for the release approach; MEDIUM for the exact v1 scope (license unknowns for 3 datasets may reduce the model set).

### Gaps to Address

- **License unknowns for FlowCamNet, UVP6Net, JEDI-Oceans** — contact upstream maintainers immediately with a 2-week deadline. No response = defer to v1.1. This is the highest-priority gap because it gates Phase 1 and may change the PROJECT.md "all 7 datasets" commitment.

- **CLIP vs non-CLIP classification of winning checkpoints** — Phase 1 must survey the actual winning checkpoint per dataset. This determines what proportion of the publish pipeline needs the trust_remote_code path and how much Phase 2 de-risks.

- **Eval split strategy for datasets with cruise/station metadata** — WHOI and ISIIS carry sampling-event metadata. Phase 1 should decide: grouped split (more honest, more work) or random split with explicit limitation disclosure. Changing the split after publishing invalidates the published numbers.

- **Existing Hub-private checkpoint audit** — if `Trainer.push_to_hub` was called with `model_push_as_private: true` during training, there may be private repos with stale or missing `preprocessor_config.json`. Phase 1 should audit these before treating them as starting points.

---

## Sources

### Primary (HIGH confidence)
- `huggingface.co/docs/transformers/main/en/custom_models` — definitive `trust_remote_code` + `register_for_auto_class` + `auto_map` pattern
- `huggingface.co/docs/hub/model-cards`, `model-card-annotated`, `model-release-checklist` — required sections, YAML schema, eval-results format
- `huggingface.co/docs/hub/spaces-config-reference` — Spaces YAML fields, `sdk_version`, `preload_from_hub`, hardware tiers
- Context7 `/huggingface/huggingface_hub` — `ModelCard`, `ModelCardData`, `EvalResult`, `push_to_hub`, `upload_folder`
- Context7 `/huggingface/transformers` — `AutoModelForImageClassification`, `register_for_auto_class`, `PreTrainedModel`, `PreTrainedConfig`
- Context7 `/gradio-app/gradio` — `gr.Blocks`, `gr.Image`, `gr.Label`, `gr.Dropdown`, lazy-load patterns
- PyPI version verification (2026-05-12): `gradio 6.14.0`, `huggingface_hub 1.14.0`, `transformers 5.8.0`
- `imageomics/bioclip-2`, `microsoft/BiomedCLIP-*` — scientific-domain model card quality bar
- `Binou/vit-base-plankton`, `Jookare/plankton_vit_large_patch16_224.mae` — in-niche anti-pattern references

### Secondary (MEDIUM confidence)
- arxiv 2401.14256 — plankton cross-instrument domain shift (10-30pp OOD degradation)
- arxiv 2510.17179 — OOD detection benchmark for plankton (Far-OOD bubbles/particles scenario)
- arxiv 2402.05160 — analysis of 32K HF model cards (missing sections are modal failure mode)
- arxiv 2502.04484 — HF licensing challenges empirical analysis
- Princeton reproducibility survey — data leakage in ML-based science
- GitHub diffusers/discussions #10936 — Gradio + model loading memory patterns
- Springer 2024 plankton survey — taxonomy non-standardisation across datasets

### Tertiary (LOW confidence)
- `open_clip_torch` PyPI version range compatible with vendored `open_clip 4.0.0.dev0` — requires Phase 2 API surface spike
- Hardware-tier latency estimates for EVA02-L-14 and ConvNeXtV2-Huge on cpu-upgrade — extrapolated, needs benchmarking

---
*Research completed: 2026-05-12*
*Ready for roadmap: yes, with the note that Phase 1 license audit may reduce the v1 scope from 7 models to 4*
