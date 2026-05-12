# Requirements: planktonzilla

**Defined:** 2026-05-12
**Core Value:** Curated, citable pre-trained plankton-classification models on Hugging Face Hub, each with a real model card, backed by a repo a stranger can credibly load a model from.

These requirements supersede the initial Active list in `.planning/PROJECT.md` (REL-01..03, DOC-01..02, HARD-01, DEMO-01) by incorporating four scope decisions surfaced by research and user direction:

1. **Anchor v1 on "documentation and code", not on dataset-side work** (user direction). Datasets are inherited as-is — no license audit, no dataset card edits, no taxonomy work, no negotiation with upstream dataset authors. Each shipped model card declares the dataset's existing license verbatim; if a dataset's license is unknown, the card declares `license: other` with a pointer note and ships anyway. Dataset-side work is explicitly out of scope (see Out of Scope table).
2. **CLIP-first sequencing** — 4 of 7 winning checkpoints are CLIP-backed (the headline models). They're harder to package (`trust_remote_code=True` shim + `open_clip` strategy) but they go first because they are the most compelling artifacts.
3. **Eval reproduction is a separate requirement** — REL-04 below: every model-card metric must be regenerable from the published checkpoint, not transcribed from W&B.
4. **`open_clip` un-vendoring strategy is a code requirement, not a dataset requirement** — covered by HARD-02 below (pin `open-clip-torch` from PyPI, kill the hardcoded `/home/acontreras/...` PYTHONPATH path).

## v1 Requirements

### Foundation (FOUND)

Curate the inputs to the release: which checkpoints, which label mappings.

- [ ] **FOUND-01**: Identify the winning checkpoint per dataset (file path / W&B run / HF private repo) and verify each is physically retrievable (not on cleared SLURM scratch). Output: `.planning/release-manifest.yaml` with one entry per dataset listing checkpoint location, architecture (`pure-hf` | `clip`), source dataset, the dataset's existing license verbatim (or `other` if unknown), training config snapshot.
- [ ] **FOUND-02**: Class taxonomy + label-mapping freeze. For each checkpoint, lock `id2label` / `label2id` mapping by reading from the trained model's classification head and the dataset's existing class list. Cross-instrument label-name normalization stays out of v1 (per-dataset taxonomies are kept as-is).

### Hardening (HARD)

Minimum-viable hardening — only what blocks the load-and-use path.

- [ ] **HARD-01**: Bump `transformers` to `>=5.x` and `huggingface_hub` to `>=1.x` in `pyproject.toml`; resolve via `poetry lock`. Verify imports of `Trainer`, `AutoModelForImageClassification`, `ModelCard`, `ModelCardData`, `EvalResult`, and `register_for_auto_class` still work.
- [ ] **HARD-02**: Resolve the `open_clip` packaging strategy: pin a compatible version of `open-clip-torch` from PyPI as a runtime dependency, AND keep the vendored copy for training-side use behind an opt-in path. Output: `pyproject.toml` declares `open-clip-torch = "<resolved-pin>"`; `scripts/train_clip.sh` no longer hardcodes `/home/acontreras/...`.
- [ ] **HARD-03**: Build the standalone `clip_classifier_hub/` shim — a `PreTrainedConfig` + `PreTrainedModel` pair (~150 LOC) that depends only on `transformers`, `torch`, `open-clip-torch`, with NO import of the `planktonzilla` package. Lives in the repo for review but the file pair gets pushed inside each CLIP model repo on the Hub.
- [ ] **HARD-04**: Smoke-test the load path on a clean Python env (fresh `venv` in a Docker container, no `planktonzilla` clone): `pip install transformers huggingface_hub open-clip-torch torch` then run a 6-line `AutoModelForImageClassification.from_pretrained(repo_id, trust_remote_code=True, revision=<sha>)` snippet. Both pure-HF and CLIP code paths must succeed.

### Model Release (REL)

Push the curated checkpoints to HF Hub with real, defensible model cards.

- [ ] **REL-01**: Per-dataset HF model repo created at `huggingface.co/project-oceania/<repo-name>` for each of the 7 shipped datasets. Each repo contains: model weights (safetensors), `config.json`, `preprocessor_config.json`, `README.md` (model card), `model_card.md` source if generated programmatically, and (for CLIP repos) `modeling_clip_classifier.py` + `configuration_clip_classifier.py`.
- [ ] **REL-02**: Each model card built from `huggingface_hub.ModelCard` + `ModelCardData` + `EvalResult` (NOT hand-rolled YAML; NOT the `Trainer.push_to_hub` autostub) and contains: dataset card cross-link (existing dataset card, no edits), base-model link, **license declared verbatim from the source dataset** (no audit; if the dataset's license is unknown, `license: other` with a one-line pointer to the dataset card), training-config snapshot reference (commit SHA + W&B run URL), citation BibTeX (model + dataset + base model), intended-use, limitations, bias/risks, plankton-domain caveats from `PITFALLS.md` (cross-instrument transfer, taxonomy non-standardization, OOD/background absence, cruise leakage).
- [ ] **REL-03**: Per-class evaluation metrics in each model card include macro-F1 (mandatory for class-imbalanced plankton) alongside top-1 accuracy. Numbers come from REL-04, not from W&B transcripts.
- [ ] **REL-04**: Reproducible eval pass — `release/eval_model.py` script loads each published checkpoint via `AutoModelForImageClassification.from_pretrained(repo_id, trust_remote_code=True, revision=<sha>)`, runs eval on the dataset's held-out split, writes `eval_results.json` per repo. Numbers in each model card's `model-index` come from this script, not from W&B. The script itself is committed and citable.
- [ ] **REL-05**: HF "Collection" feature groups all shipped models under a single OcéanIA collection page (cheap, high-leverage discoverability win — no existing plankton collection on HF Hub).
- [ ] **REL-06**: CLIP-backed models ship FIRST (Phase 2 spike + Phase 3 publish). Pure-HF models follow in a parallel/subsequent wave. (Decision per scope-change Q2.)

### Demo (DEMO)

Single-image classifier on Hugging Face Spaces.

- [ ] **DEMO-01**: HF Space at `huggingface.co/spaces/project-oceania/planktonzilla-demo` running Gradio 6.x with `gr.Blocks` + `gr.Dropdown` model picker across the shipped models. UX: drag/drop one image → pick which model → see top-K predictions with probabilities. NO saliency/Grad-CAM in v1.
- [ ] **DEMO-02**: Space lazy-loads models with `@lru_cache` per `repo_id`; `preload_from_hub` in Space `README.md` YAML warms disk cache at build time. Hardware tier decided after benchmarking the heaviest backbone on `cpu-upgrade` vs `t4-small`.
- [ ] **DEMO-03**: Space `requirements.txt` exactly pins all versions (matches HARD-01's `transformers`/`huggingface_hub` pins + the `open-clip-torch` pin from HARD-02). Examples gallery seeded with 3–5 representative images (with disclaimer the demo is for exploration, not for production classification).
- [ ] **DEMO-04**: Space includes a license + intended-use disclaimer in the UI footer (especially relevant for CC-BY-NC-derived models that cannot be used commercially).

### Documentation & Launch (DOC)

Public-facing surface that makes the artifacts discoverable and usable.

- [ ] **DOC-01**: README.md rewritten to cover the four headline use cases end-to-end (in this order): (1) load a published planktonzilla model in 6 lines from a clean env, (2) try the live Spaces demo, (3) retrain a published model on your own data, (4) import a new dataset into the planktonzilla pipeline. Existing README content (training framework features, Hydra config groups) demoted to an "Advanced Usage" section.
- [ ] **DOC-02**: The "load a published model" snippet in DOC-01 is verified by HARD-04's clean-env smoke test, and the smoke test is committed as `tests/test_clean_env_load.sh` (run-on-demand, not in default CI per the "minimum-viable hardening" boundary).
- [ ] **DOC-03**: HF org-level README at `huggingface.co/project-oceania` updated with: project description, link to the OcéanIA collection (REL-05), link back to the GitHub repo, citation, contact. (Org README only — does NOT touch existing dataset cards.)
- [ ] **DOC-04**: GitHub release tag (`v1.0.0` or `v0.2.0` — version bump TBD) with release notes that summarize: which models shipped, link to the collection, link to the demo, what's deferred to v1.1.

### Launch Coordination (LAUNCH)

The discipline gate before going public.

- [ ] **LAUNCH-01**: Pre-launch checklist completed (from `.planning/research/PITFALLS.md` "must-pass" checklist): every model-card YAML validates; every model-card inference snippet runs in a clean env; every model-card `license:` field matches the source dataset's existing license verbatim (no audit, just a verbatim check); no SOTA/"production-ready"/"best in class" claims anywhere; intended-use + limitations are not autogenerated stubs.
- [ ] **LAUNCH-02**: Announcement copy ready (one-paragraph blurb + key links) — includes plankton-domain disclaimers from `PITFALLS.md` (cross-instrument transfer, taxonomy specifics). Inria/OcéanIA channels for distribution identified.

## v1.1 Requirements

Deferred to fast-follow milestone. Tracked but not in v1 roadmap.

### Demo enhancements

- **DEMO-1.1-01**: Saliency/explanation overlay (Grad-CAM for CNN models, attention-rollout for ViT/CLIP models). Demo picks the right method per loaded model.
- **DEMO-1.1-02**: Batch upload + CSV-of-predictions export.

### Distribution

- **DIST-1.1-01**: Pip-installable PyPI package (`pip install planktonzilla`). Not required for v1's "load a published model" snippet, which works without cloning the source repo.

### Hardening (separate milestone)

- **HARD-1.1-01..N**: Full training-side hardening — kill bare `except:` in `planktonzilla/train.py:167` and `planktonzilla/clip_model.py:39`; fix deprecated `F.log_softmax(logits)` (no `dim=`) and `torch.autograd.Variable` in `planktonzilla/loss.py:63,66,72`; un-skip `tests/shared.py:19-25` `skip_in_github_ci`; complete vendoring policy decision for `open_clip` (full removal vs. submodule vs. soft-fork).

## Out of Scope (v1)

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| **Dataset-side work (license audit, dataset card edits, taxonomy normalization, contacting upstream dataset authors)** | User direction: "leave datasets as they are." Each model card declares the dataset's existing license verbatim; unknowns become `license: other` with a pointer note, and the model ships anyway. |
| Pip-installable PyPI release | Deferred to v1.1; "load a published model" snippet works without source clone |
| Saliency / Grad-CAM in demo | Deferred to v1.1; CNN vs ViT/CLIP requires different methods, inconsistent v1 UX |
| Full install + CI hardening | Separate hardening milestone; v1 covers only what blocks the load-and-use path |
| Re-vendoring or full un-vendoring of `open_clip` | Deferred; v1 pins `open-clip-torch` from PyPI for the load path, leaves vendored copy in place for training |
| New training campaign | Out of scope; v1 is curation, not new research; winners already identified |
| New datasets / new model architectures / new losses | Ship what exists, then iterate |
| Pretty docs site (MkDocs / Sphinx / GitHub Pages) | README is the documentation surface for v1 |
| Per-dataset benchmark paper / leaderboard write-up | Separate research artifact; not blocking the model release |
| Cross-dataset taxonomic harmonization | Deferred; per-dataset taxonomies kept as-is; mentioned in card limitations |
| Inference API / "Use this model" widget on Hub | Auto-disabled for `trust_remote_code=True` models — not a v1 lever; demo Space replaces it |
| HF "Benchmark" registration (the new `.eval_results/` format) | Multi-week project per research; legacy inline `model-index` is sufficient and standard |
| Inria legal / institutional sign-off gates | User confirmed full authority — no institutional review step needed |
| Multi-language model cards (FR/EN) | Differentiator deferred to v1.1+; English-only for v1 |

## Traceability

Empty until roadmap creation. Updated by `gsd-roadmapper` agent.

| Requirement | Phase | Status  |
| ----------- | ----- | ------- |
| FOUND-01    | TBD   | Pending |
| FOUND-02    | TBD   | Pending |
| HARD-01     | TBD   | Pending |
| HARD-02     | TBD   | Pending |
| HARD-03     | TBD   | Pending |
| HARD-04     | TBD   | Pending |
| REL-01      | TBD   | Pending |
| REL-02      | TBD   | Pending |
| REL-03      | TBD   | Pending |
| REL-04      | TBD   | Pending |
| REL-05      | TBD   | Pending |
| REL-06      | TBD   | Pending |
| DEMO-01     | TBD   | Pending |
| DEMO-02     | TBD   | Pending |
| DEMO-03     | TBD   | Pending |
| DEMO-04     | TBD   | Pending |
| DOC-01      | TBD   | Pending |
| DOC-02      | TBD   | Pending |
| DOC-03      | TBD   | Pending |
| DOC-04      | TBD   | Pending |
| LAUNCH-01   | TBD   | Pending |
| LAUNCH-02   | TBD   | Pending |

**Coverage:**

- v1 requirements: 22 total
- Mapped to phases: 0 (roadmapper will fill)
- Unmapped: 22 ⚠️ (expected; resolved during roadmap creation)

---
*Requirements defined: 2026-05-12*
*Last updated: 2026-05-12 after user redirect: "leave datasets as they are, focus on documentation and code"*
