# Release Pitfalls — `planktonzilla` Public Use Milestone

**Domain:** public release of fine-tuned image classifiers (HF Hub + Gradio Space) in scientific / marine-microscopy CV
**Researched:** 2026-05-12
**Confidence:** MEDIUM-HIGH (HF docs and current HF/Gradio bug threads are HIGH; plankton-domain claims grounded in two recent benchmark papers, MEDIUM)
**Out of scope (already documented in `.planning/codebase/CONCERNS.md` and explicitly OOS for this milestone):** training-side bare `except:`, deprecated `F.log_softmax`, `skip_in_github_ci`, full vendoring policy for `open_clip`, `peft`/`accelerate` version churn. Those are not re-listed here.

## Phase Mapping Legend

- **HARDENING** — minimum-viable bug fixes that block the load/demo path (corresponds to `HARD-01`)
- **MODEL_RELEASE** — packaging + push of the 7 curated checkpoints to HF Hub (`REL-01`, `REL-02`)
- **DEMO** — Gradio Space (`DEMO-01`)
- **DOCS** — README + load snippet (`DOC-01`, `DOC-02`)
- **LAUNCH** — announcement, GitHub release, public-facing surface (`REL-03`)

---

## Critical Pitfalls

Mistakes that cause a public release to fail loudly (the snippet doesn't run, the Space won't start) or silently (predictions look plausible but are wrong). These are blockers for shipping with a straight face.

### Pitfall C1: `ClipClassifier` cannot be loaded by a stranger because the modelling code is not on the Hub

**Phase:** HARDENING + MODEL_RELEASE (loading-path blocker)

**What goes wrong:** `planktonzilla.clip_model.ClipClassifier` is a custom `nn.Module` that wraps the vendored `open_clip` (also not on PyPI). When the user runs the snippet from `DOC-02` against any of the EVA-CLIP / ViT-CLIP checkpoints, `AutoModelForImageClassification.from_pretrained("project-oceania/pz_eva02-large-clip_...")` will either (a) raise `KeyError`/`ValueError` because there is no registered architecture for it in `transformers`, or (b) load a generic `AutoModel` that ignores the custom head and silently returns features instead of class probabilities.

**Why it happens:** HF `from_pretrained` only resolves three things: registered `transformers` architectures, `timm` models (if `library_name: timm` is in the card), or a `modeling_*.py` file that lives **in the model repo itself** and is loaded with `trust_remote_code=True`. None of the three is set up today.

**Consequences:** The marketing-headline use case ("a stranger can load a published model") fails for 4 of the 7 datasets (every dataset whose winning checkpoint is a CLIP variant). The Space crashes on cold start when `model_picker == "EVA02-L-14"`.

**Warning signs:**
- The snippet works for ResNet / BEiT / ViT-base checkpoints but errors for any CLIP variant
- `AutoConfig.from_pretrained` returns a `PretrainedConfig` with `architectures=[]` or `architectures=["ClipClassifier"]` but `transformers` doesn't know `ClipClassifier`
- `from_pretrained` succeeds but `model.config.num_labels == 0` and there is no classification head

**Prevention:**
- For each CLIP-backed checkpoint repo on the Hub, push a self-contained `modeling_clip_classifier.py` + `configuration_clip_classifier.py` alongside `pytorch_model.bin`
- In `config.json`, set `"auto_map": {"AutoModelForImageClassification": "modeling_clip_classifier.ClipClassifier"}` so `trust_remote_code=True` resolves it
- The modelling file must NOT `import open_clip` (which isn't on the user's machine) — inline the minimum subset of `open_clip` needed (vision tower load + transform), or pin `open_clip-torch` from PyPI as a hard dependency in the model card's "Use with library" snippet
- Add a smoke test (separate fresh `venv`, no `planktonzilla` clone) that runs the snippet for every published checkpoint before declaring `MODEL_RELEASE` done
- Document `trust_remote_code=True` in the snippet and in the model card's "How to use" section, including the security disclaimer ("only enable for repos you trust")

**Detection:** The HARDENING phase must include a smoke test that loads each of the 7 checkpoints from a clean Python env. If the snippet fails for any, that checkpoint is not shippable in v1 — either the modelling file goes on the Hub, or the model is dropped from the v1 set (the PROJECT.md "no triage" decision means we have to fix it, not drop it).

**Sources:** [HF Customizing models docs](https://huggingface.co/docs/transformers/custom_models), [trust_remote_code GitHub issue #22260](https://github.com/huggingface/transformers/issues/22260), [trust_remote_code module not found #29251](https://github.com/huggingface/transformers/issues/29251)

---

### Pitfall C2: `id2label` / `label2id` are missing, wrong, or unstably ordered → predictions decode to the wrong species

**Phase:** MODEL_RELEASE

**What goes wrong:** When a `transformers.Trainer`-trained checkpoint is pushed without explicitly setting `id2label`/`label2id` on the `config`, downstream `pipeline("image-classification")` returns either `LABEL_0`, `LABEL_1`, … or — worse — labels in the wrong order because `os.listdir()` of an `ImageFolder` is not sorted lexicographically on every OS, and `datasets.ClassLabel.names` reflects whatever order the importer happened to discover the class folders in.

For plankton, this is not "looks ugly" — it means the model card claims `top-1 = Acantharea` and the user reads `Acantharea` while the actual prediction was index 0 which on the user's reload is `Annelida`. A marine biologist will lose trust the first time this happens.

**Why it happens:** Three independent failure points:
1. `Trainer.push_to_hub` doesn't enforce `id2label` consistency check
2. `planktonzilla/dataset.py:171` derives the label set from `np.unique(self.dataset["train"]["label"])` — order depends on dataset shuffle / streaming
3. The HF `imagefolder` builder in `planktonzilla/dataset_import/` discovers classes by walking the filesystem, which on macOS vs Linux vs Lustre can return different orders

**Consequences:** Public model returns scientifically wrong labels. Worse than no label, because users trust the strings. A marine biologist running it on a reference image will spot the swap immediately and the project loses credibility.

**Warning signs:**
- `model.config.id2label == {0: "LABEL_0", 1: "LABEL_1", ...}`
- `model.config.id2label[0]` differs between two runs of the same pipeline on different machines
- `len(model.config.id2label) != model.config.num_labels`
- The labels in the README example output don't match `dataset.features["label"].names` of the dataset card it links to

**Prevention:**
- During the model-publish step, explicitly read `dataset["train"].features["label"].names` (which IS the canonical, dataset-card-aligned order) and set `model.config.id2label = dict(enumerate(class_names))` and `model.config.label2id = {n: i for i, n in enumerate(class_names)}` before `model.push_to_hub`
- Add a unit test that for every (model, dataset) pair, asserts `set(model.config.id2label.values()) == set(dataset_card.features.label.names)` AND that the orderings match index-by-index
- In the model card "How to use" snippet, demonstrate `pipeline(image)[0]["label"]` returning a real species name (e.g. `Copepoda`), so reviewers can spot generic `LABEL_X` instantly
- Pin the `datasets` revision in the model card so a future relabelling of the dataset (different label order) doesn't silently re-index without bumping a model version

**Detection checklist (one item per shipped model):**
- [ ] `model.config.id2label[0]` is a real species name, not `LABEL_0`
- [ ] All values in `id2label` are present in the linked dataset card's `class_label.names`
- [ ] Index alignment verified by running 1 image of a known class through the snippet

**Sources:** [HF forum: id2label/label2id confusion](https://discuss.huggingface.co/t/errors-with-label2id-id2label-with-muticlass-classification/160188), [transformers issue #28589 — id2label assignment bug in run_classification.py](https://github.com/huggingface/transformers/issues/28589), [HF forum: setting id2label with Trainer](https://discuss.huggingface.co/t/how-to-set-useful-id2label-and-label2id-in-config-json-using-trainer/43545)

---

### Pitfall C3: Image preprocessing drifts between training and inference → model loads but predicts garbage

**Phase:** HARDENING + MODEL_RELEASE + DEMO

**What goes wrong:** The training pipeline uses one set of transforms (mean / std / image size / interpolation / RGB-vs-BGR / center-crop-vs-resize) applied through `planktonzilla/dataset.py` augmentations + the upstream `image_processor`. The published checkpoint either (a) doesn't ship its `preprocessor_config.json` at all, (b) ships the *upstream* preprocessor (e.g. `microsoft/resnet-18`'s default ImageNet mean/std) which is fine, or (c) ships a hand-edited preprocessor whose values drift from what `planktonzilla` actually used. Inference in the snippet and in the Space then re-derives "reasonable" defaults that silently disagree with what was used at training time. Predictions are 5-15 pp worse than the model card claims, with no error message.

This is *the* canonical "ML demo embarrassment" — top of every list of mistakes shipping fine-tuned vision models.

**Why it happens:**
- HF `Trainer` does not auto-save the augmentation pipeline; `save_pretrained` only persists the `image_processor` config
- The vendored `open_clip` path uses `open_clip.create_model_and_transforms`, returning transforms that are NOT a HF `image_processor` and so are NOT saved by `Trainer` at all
- `timm`-backed models (5 of `planktonzilla`'s 8 model configs) get their preprocessing from `timm.data.resolve_data_config(model.pretrained_cfg)`, which depends on the timm version installed at inference time

**Consequences:** Silent accuracy drop. Predictions on the Space look "kind of right" because the model is robust to mild preprocessing drift, but rare classes degrade catastrophically and a marine biologist will reasonably conclude the model "doesn't know what a Phaeodaria is."

**Warning signs:**
- Inference in a fresh env gives different top-1 probabilities than the same image processed in the training repo
- `preprocessor_config.json` is absent from the model repo, or contains the upstream backbone's ImageNet mean/std even though `planktonzilla` used dataset-specific mean/std (`compute_mean_and_std_dev` in `planktonzilla/dataset.py:36-93`)
- Model card eval numbers are X%, but re-running the eval split with the published preprocessor gives X-5%

**Prevention:**
- During `MODEL_RELEASE`, for every published checkpoint, recompute eval metrics from the published artefacts (no access to the training repo) and assert they match the training-run W&B numbers within ±0.5pp
- Persist the training-time transform: for HF-Auto models, ensure `image_processor.save_pretrained()` is called with the actual mean/std/size used; for `ClipClassifier`, write a `preprocessor_config.json` with `"image_processor_type": "CLIPImageProcessor"` and explicit values pulled from `open_clip.create_model_and_transforms(...)` at training time
- Pin `timm`, `open_clip-torch`, `transformers`, `torch`, `Pillow` in the model card's snippet (specific patch versions, not `^`) — preprocessing pipelines have changed between Pillow 9 and 10 (resampling enum), torchvision v1 → v2 transforms, and timm 0.9 → 1.0
- In the Space (`DEMO-01`), use the model's own `image_processor` (`AutoImageProcessor.from_pretrained(repo_id)`) — never re-implement preprocessing inline

**Detection checklist (gate for MODEL_RELEASE on each checkpoint):**
- [ ] `preprocessor_config.json` exists in the repo
- [ ] Its `image_mean`, `image_std`, `size`, `do_center_crop`, `interpolation` match what training actually used
- [ ] Re-running eval from the published repo only matches training-time eval within 0.5pp
- [ ] Model card snippet's first sample image returns the same top-1 in the Space, in the README, and in the verification harness

**Sources:** [HF preprocess docs (use image_processor.image_mean/std)](https://huggingface.co/docs/transformers/main/preprocessing), [timm quickstart — use resolve_data_config](https://huggingface.co/docs/timm/quickstart), [HF image classification task — exact preprocess match](https://huggingface.co/docs/transformers/tasks/image_classification)

---

### Pitfall C4: Eval numbers in the model card cannot be regenerated from the published checkpoint

**Phase:** MODEL_RELEASE

**What goes wrong:** Model card publishes "Top-1 = 87.4% on ISIIS held-out test", but a reviewer who downloads the published checkpoint, runs the published snippet on the published dataset's `test` split, gets 81.3%. The 6pp gap is a combination of C2 (label re-ordering), C3 (preprocessing drift), the use of in-training EMA / SWA weights that weren't pushed, or evaluation done on a different split than the snippet would use (e.g. `dataset.py:152-169` recomputes splits each run with `stratify_by_column="label", seed=split_seed`, and the published dataset on the Hub may have changed under the model — `CONCERNS.md` "Train/val/test splits are recomputed from scratch on every run" is precisely this).

**Why it happens:** Three modes:
1. The eval was done in a notebook with the training-time codepath (in-memory split, training-time preprocessor, full Trainer state)
2. Splits aren't pinned — `load_dataset(name)` without `revision=` pulls latest, and the dataset on the Hub may have grown new classes since training
3. The W&B/Trackio run that produced the number is not linked to the checkpoint commit on the Hub

**Consequences:** First serious reviewer (paper reviewer, grant reviewer, marine biologist) opens an issue: "I cannot reproduce your numbers." This is the #1 trust-destroyer for a research artefact. The PROJECT.md "Quality bar: defensible to a paper reviewer" goal collapses.

**Warning signs:**
- The model card cites a number but the model card itself doesn't include the exact eval command
- The eval split is not pinned by HF dataset revision SHA
- The published `config.json` lacks `"_planktonzilla_train_run"` / `"_planktonzilla_git_sha"` provenance fields

**Prevention:**
- Build a `pz_verify_release` script that, given a model repo ID, downloads the model + linked dataset @ pinned revision, runs a single deterministic eval, and asserts agreement with the model card number. Run it as the gate for `MODEL_RELEASE`.
- Pin the dataset revision in the model card's `datasets:` metadata AND in the snippet (`load_dataset("project-oceania/isiisnet", revision="abc123")`). This is the same fix recommended in `CONCERNS.md` for split drift — turn it into a release-gate requirement.
- Stamp every published checkpoint with provenance: `config.json` gains `_planktonzilla_train_run` (W&B run URL), `_planktonzilla_git_sha` (training-repo commit), `_planktonzilla_config_yaml_sha` (Hydra resolved-config hash). The eval recompute compares to this.
- Embed the exact eval command in the model card under "## Evaluation" — not pseudocode, copy-pasteable

**Detection checklist:**
- [ ] `config.json` contains git SHA + W&B run URL
- [ ] `datasets:` frontmatter lists dataset @ pinned revision SHA
- [ ] `pz_verify_release` exits 0 for every published checkpoint
- [ ] Re-eval matches model card number within 0.5pp

**Sources:** [Reproducibility crisis in ML — Princeton survey](https://reproducible.cs.princeton.edu/), [Reproducibility standards for ML in life sciences (PMC9131851)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9131851/), [How to solve reproducibility in ML (Neptune)](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml)

---

### Pitfall C5: License conflict — model declared MIT but trained on a CC-BY-NC dataset

**Phase:** MODEL_RELEASE + LAUNCH

**What goes wrong:** `planktonzilla` repo is MIT (correctly). The model card auto-templates inherit `license: mit`. But ISIISNet is **CC-BY-NC-4.0** (per `INTEGRATIONS.md`). A trained model is widely (and increasingly judicially) considered a derivative work of its training data. Publishing an MIT model derived from a NonCommercial dataset is a license violation — the dataset license restricts commercial use, the model license claims unrestricted use. A subset of marine-biology research orgs and any commercial user will refuse to use the artefact; in the worst case the dataset owner files a takedown.

For the v1 set (per `INTEGRATIONS.md` "Dataset Sources" table):
- **ISIISNet** → CC-BY-NC-4.0 (NonCommercial)
- **WHOI-Plankton** → MIT (compatible)
- **ZooLake** → CC-BY-4.0 (compatible, attribution required)
- **Lensless** → CC-BY-4.0 (compatible, attribution required)
- **FlowCamNet, UVP6Net, JEDI Oceans** → license currently UNDOCUMENTED in the importer configs (`license: —`). Cannot ship a model derived from these without resolving this.

**Why it happens:** Defaults. `huggingface_hub`'s `ModelCard` template defaults to inheriting the calling repo's license. The HF Hub UI doesn't validate cross-license compatibility between `datasets:` and `license:`.

**Consequences:** Takedown risk; immediate credibility loss with anyone who reads licenses; for academic consumers, blocks downstream use in commercially-funded projects. ISIISNet on its own is the long pole.

**Warning signs:**
- `license: mit` in a model card whose `datasets:` frontmatter points to a CC-BY-NC dataset
- "License" section of the model card contains generic boilerplate, not a per-dataset attribution chain
- No `license_link` or attribution to the upstream paper / dataset owner

**Prevention:**
- **Per-checkpoint license decision matrix:** model license = the most restrictive of (code license, data license, base-model weights license). For ISIISNet → `license: cc-by-nc-4.0`. For WHOI/ZooLake/Lensless → `license: cc-by-4.0` (the more permissive of CC-BY-4.0 and MIT, with the BY-attribution requirement carried through). Document the matrix.
- For FlowCamNet, UVP6Net, JEDI Oceans, MedPlanktonSet, etc. — block `MODEL_RELEASE` for that dataset until license is documented in the importer config. This is a hard gate, not a "fix later".
- Model card "License" section must list both the model license AND the training-data license, with attribution to the dataset paper. Cite both.
- Add the same license metadata to the corresponding *dataset* card on `project-oceania/*` so the chain is consistent (if the dataset card on HF is wrong, fix it first).
- Backbone licenses also flow through: `microsoft/resnet-18` is Apache-2.0, BEiT is MIT, EVA-02 weights ship under Apache-2.0 / MIT — generally fine but check each.

**Detection checklist (per shipped model):**
- [ ] Dataset license documented (not `—`)
- [ ] Model license is the ∩ of (dataset license, base-model license, code license)
- [ ] BibTeX citation for the dataset paper present in the model card
- [ ] Attribution requirements (CC-BY) satisfied in the README rendered output

**Sources:** [HF Licenses doc](https://huggingface.co/docs/hub/en/repositories-licenses), [License conflict example: vicgalle/xlm-roberta-large-xnli-anli MIT vs CC BY-NC 4.0](https://huggingface.co/vicgalle/xlm-roberta-large-xnli-anli/discussions/3), [Empirical analysis of HF licensing challenges (arxiv 2502.04484)](https://arxiv.org/html/2502.04484v2), [Quick guide to popular AI licenses (Mend)](https://www.mend.io/blog/quick-guide-to-popular-ai-licenses/)

---

### Pitfall C6: Gradio Space crashes on cold start because models load lazily / OOM / hit memory cap

**Phase:** DEMO

**What goes wrong:** First user visits the Space. The handler tries to load all 7 models on demand → cold-start latency >60s → user hits Space timeout. Or: the handler loads a single model lazily inside the click handler each time the model picker changes, so the first prediction OOMs the free CPU tier (16 GB) because two big checkpoints (EVA02-L and ConvNeXtV2-Huge) coexist transiently. Or: free Space goes to sleep, wakes up on next visit, and the Gradio `Examples` component tries to pre-cache examples through every model, exceeding memory or timing out.

EVA-02 Large (~300M params) and ConvNeXtV2-Huge (~660M params) at fp32 are ~1.2 GB and ~2.6 GB on disk respectively — loading more than one fully resident on a 16 GB CPU box is fine, but if the user repeatedly switches and the previous model isn't garbage-collected (Gradio + PyTorch have a documented hidden-reference issue, see Diffusers GH discussion #10936), memory walks up.

**Why it happens:**
- The "right" pattern (load at module top-level) is non-obvious
- Gradio holds references in event handler closures
- ZeroGPU semantics (model must be moved to CUDA at module top-level under the emulation layer) differ from CPU-only Spaces
- Hardware tier mismatch: free Space is CPU 16 GB; CLIP-L inference on CPU is ~10s/image; user assumes broken

**Consequences:** Demo is the most public-facing artefact short of the README. A crashing or 30-second-cold-start Space is read as "this project doesn't work."

**Warning signs:**
- Space build succeeds but shows "Runtime Error" on first visit
- Logs contain `Memory limit exceeded` or `Killed`
- Inference time per prediction grows with each model switch (memory leak)
- Cold start >45s

**Prevention:**
- **Load all 7 models at module top-level**, not in handlers. CPU memory math: 7 × (avg ~300 MB int8 / fp16) ≈ 2-4 GB total → fits in 16 GB. If too tight, ship in fp16 only (no fp32 weights pushed). Quantize the two huge models if needed.
- Use the `@spaces.GPU` decorator only for the actual forward pass, not the load
- Don't pre-cache examples through every model on Space build — set `cache_examples=False` for the model-picker dropdown, OR pre-cache only a single model's examples
- Set `max_file_size="10mb"` on `gr.Image` and reject non-image MIME types (not all uploaded files are valid microscopy images — see Pitfall C7)
- Pick the hardware tier explicitly: CPU `cpu-upgrade` (32 GB, ~$0/mo for free, but slower) is enough for the v1 single-image-only demo. ZeroGPU is overkill; ZeroGPU also adds the "ahead-of-time compilation" surface from the recent HF blog.
- Gate cold start with a synthetic warmup at Space build (one dummy forward per model) so the first user doesn't pay the JIT cost
- Add `gc.collect()` + `torch.cuda.empty_cache()` between model switches even though models live at module level (defends against the Gradio leak referenced in Diffusers #10936)

**Detection checklist:**
- [ ] Cold start <30s end-to-end (page load → first prediction)
- [ ] Switching all 7 models in sequence does NOT grow RSS more than 100 MB
- [ ] Loading a 50 MB image upload returns a useful error, not a crash
- [ ] Space rebuilds cleanly from git — `factory reboot` works (pre-requirements pinning correct)

**Sources:** [Gradio app runtime error (HF forum)](https://discuss.huggingface.co/t/gradio-app-runtime-error/111441), [Memory leak with Gradio + Diffusers (#10936)](https://github.com/huggingface/diffusers/discussions/10936), [HF Spaces ZeroGPU docs (load at module level)](https://huggingface.co/docs/hub/spaces-zerogpu), [HF blog: ZeroGPU AOT compilation](https://huggingface.co/blog/zerogpu-aoti), [HF Spaces dependencies docs](https://huggingface.co/docs/hub/spaces-dependencies)

---

### Pitfall C7: Space requirements.txt drifts from training environment → silent prediction differences

**Phase:** DEMO + DOCS

**What goes wrong:** `Space/requirements.txt` is hand-written with loose constraints (`transformers`, `torch`, `gradio`). On Space rebuild months later, `transformers` resolves to a newer version, `torch` to a newer CUDA build, `Pillow` to v11 (resampling enum changed), `timm` from 1.0 → 1.1. Same uploaded image, same checkpoint, different top-K. The user notices because they screenshotted last month's prediction.

This is the same `C3` failure (preprocessing drift) but caused at the dependency-pin layer instead of the preprocessor-config layer.

**Why it happens:** HF Spaces docs explicitly warn that `requirements.txt` without pins risks future breakage; the pin is the developer's responsibility. There is no equivalent of `poetry.lock` for Spaces unless the developer uses `uv` or pip's hash mode.

**Consequences:** Hard-to-reproduce "the demo gave a different answer last week" issues. Particularly bad for scientific demos — researchers screenshot outputs into reports.

**Prevention:**
- Pin `requirements.txt` exactly: `transformers==4.57.3`, `torch==2.x.y+cu121`, `Pillow==10.4.0`, `timm==1.0.x`, `open_clip_torch==Z`, `gradio==5.x.y`. Match patch versions to what training used (per `poetry.lock`).
- Pin Python via `python_version: "3.11"` in the Space's README YAML frontmatter (per HF Spaces config reference)
- Pin the Gradio SDK version with `sdk_version: 5.x.y` in README YAML
- Use a `pre-requirements.txt` that pins `pip` itself if you rely on resolver behaviour (the Space changelog has had multiple pip-resolver-related regressions)
- Add a CI job in the GitHub repo that, on every push to the Space's source dir, asserts the pinned versions in `Space/requirements.txt` match a known-good set

**Detection checklist:**
- [ ] All packages pinned to exact versions (`==`, not `^` or `>=`)
- [ ] `python_version` and `sdk_version` set in README YAML
- [ ] A "factory rebuild" of the Space produces byte-identical predictions on a fixed test image (reproduce-after-rebuild test)

**Sources:** [HF Spaces dependencies docs](https://huggingface.co/docs/hub/spaces-dependencies), [HF forum: pinning Spaces package versions](https://discuss.huggingface.co/t/how-to-pin-version-of-spaces-package/56702), [HF forum: Spaces not updating packages from requirements.txt](https://discuss.huggingface.co/t/huggingface-spaces-not-updating-packages-from-requirements-txt/92865), [HF Spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference)

---

## Moderate Pitfalls

Mistakes that don't break the release but materially erode trust or limit usefulness.

### Pitfall M1: Model card lacks "Intended use" / "Limitations" / "Bias" sections — fails review even if it loads

**Phase:** MODEL_RELEASE

**What goes wrong:** Card has eval numbers and a snippet but no "Intended Use", no "Out of Scope Uses", no "Limitations", no "Bias / Ethical Considerations" — so it reads as a model dump rather than a research artefact. A 2024 systematic analysis of 32K HF model cards found this is the modal failure mode: high-quality cards are dominated by major orgs, community cards range from thorough to empty README files. For a scientific release with a defensible-to-reviewers bar, missing these sections is disqualifying.

**Prevention:**
- Use the [HF annotated model card template](https://huggingface.co/docs/hub/model-card-annotated) as a hard checklist
- Required sections per shipped model: `Model Description`, `Intended Use` (research, ID assistance — explicitly NOT operational ecology decisions), `Out-of-Scope` (commercial monitoring, regulatory reporting, real-time taxonomy without expert review), `Training Data` (with dataset card link + revision SHA), `Training Procedure` (Hydra config link/snapshot), `Evaluation` (metrics + the eval command from C4), `Limitations`, `Bias` (class imbalance — `cls_num_list` from `dataset.py:171`), `Citation` (BibTeX for both the model + the dataset paper), `License`
- Include a "Limitations" disclaimer explicitly stating the model was trained on a single instrument's image distribution and is not validated on other instruments (the C8 marine pitfall)
- Run `huggingface_hub.ModelCard.load(repo_id).validate()` before pushing each card — it catches missing-required-fields

**Detection checklist (gate for MODEL_RELEASE on each model):**
- [ ] All 9 sections present and non-empty
- [ ] `ModelCard.validate()` passes
- [ ] BibTeX entries for the model AND the source dataset paper

**Sources:** [HF Model Cards docs](https://huggingface.co/docs/hub/model-cards), [HF Annotated Model Card](https://huggingface.co/docs/hub/model-card-annotated), [What's documented in AI? Analysis of 32K model cards (arxiv 2402.05160)](https://arxiv.org/html/2402.05160v1)

---

### Pitfall M2: `model-index` evaluation frontmatter doesn't render → eval numbers hidden from leaderboards

**Phase:** MODEL_RELEASE

**What goes wrong:** Eval numbers are in the prose of the model card but not in the YAML `model-index` block, OR the `model-index` block has a typo (wrong `task.type` enum, wrong `dataset.type`, missing `metrics.type`) so the Hub silently doesn't render the eval widget and Papers-with-Code-style aggregators don't pick it up. The model is invisible to anyone discovering models via leaderboard / metric filters.

**Prevention:**
- Use the new simpler eval-results format (`eval-results` doc page) rather than the legacy `model-index` whenever possible
- Push the card via `huggingface_hub.ModelCard` Python API with an explicit `EvalResult` list rather than hand-writing YAML — the API rejects malformed entries
- After push, fetch the rendered card and verify the eval widget appears
- Include `source.name` and `source.url` linking the W&B run URL so the number is auditable

**Detection:** Open the rendered model page in a browser and confirm an "Evaluation Results" table is visible. If absent, the YAML didn't validate.

**Sources:** [HF Model Cards — Evaluation Results section](https://huggingface.co/docs/hub/model-cards#evaluation-results), [HF Hub — eval-results doc](https://huggingface.co/docs/hub/leaderboard-data-guide), [HF Model Card spec (modelcard.md)](https://github.com/huggingface/hub-docs/blob/main/docs/hub/model-card-annotated.md)

---

### Pitfall M3: Dataset card link in the model card points to a private / nonexistent repo

**Phase:** MODEL_RELEASE + DOCS

**What goes wrong:** `INTEGRATIONS.md` notes `model_push_as_private: true` is the default (`planktonzilla/train.py` push pattern). The model card YAML lists `datasets: [project-oceania/isiisnet]` but that dataset repo is still private (or, for some datasets, has never been pushed yet). The model page then displays a broken "Datasets used to train: 🔒 (you don't have access)" badge. Worse, the model card snippet `load_dataset("project-oceania/isiisnet")` errors with 401.

**Prevention:**
- Audit each linked dataset repo: confirm public, confirm `README.md` exists, confirm class labels match the model's `id2label` (cross-card consistency check from C2)
- Set `model_push_as_private: false` for the v1 release; flip it as part of the publish step, not an afterthought
- Add a `pz_check_release_links` script that opens every URL in the rendered model card with an unauthenticated client and asserts 200 OK
- Document required dataset-card fields: provenance / collection methodology / instrument / sampling location and time / license / class definitions (genus vs species vs morphotype) / how OOD samples were excluded

**Detection checklist:** Anonymous `curl` of every model card URL returns 200 — for the model itself, the dataset link, and any paper / W&B / GitHub URL the card references.

---

### Pitfall M4: Class imbalance / collection bias not reported → users misuse the model on rare classes

**Phase:** MODEL_RELEASE

**What goes wrong:** Plankton datasets are wildly imbalanced — WHOI has classes from 50k to 50 images, ISIIS similarly. The trained model gets ~80% top-1 by predicting the dominant 5 classes well and the long tail badly. The model card reports a single "Top-1 = 87%" macro-or-micro accuracy that doesn't surface this. A user runs the model on a rare-class image, gets a wrong answer, and concludes the model is broken — when in fact it was never trained on enough samples to learn that class.

**Prevention:**
- Report per-class metrics in the model card (top-1 per class, F1 per class, support count). At minimum: confusion matrix image + macro-F1 alongside top-1.
- Expose the `cls_num_list` distribution in a "Bias" section of the card with the absolute support per class
- If a class has fewer than N samples in train, add it to the "Limitations" section by name ("Predictions for `Phaeodaria` should be treated as low-confidence; only 47 training samples")
- For models trained with imbalance-aware losses (Focal, LDAM, RAL — already in `planktonzilla/loss.py`), say so in the "Training Procedure" section so users know what was tried

**Sources:** [Survey of automatic plankton image recognition (Springer 10.1007/s10462-024-10745-y)](https://link.springer.com/article/10.1007/s10462-024-10745-y), [Operational phytoplankton recognition (Frontiers Marine Science)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.867695/full)

---

### Pitfall M5: Demo's default examples are cherry-picked → over-promises model quality

**Phase:** DEMO

**What goes wrong:** Default `examples` in the Gradio Space are 3 photogenic, well-lit, centred plankton images of the 3 dominant classes. Predictions are all 99%+. A marine biologist tries their own field image — blurry, partial, off-centre, different instrument — and gets a meaningless answer. The contrast is read as "the demo is rigged".

**Prevention:**
- Default examples MUST include a representative cross-section: one easy (high-confidence common class), one hard (rare class), one ambiguous (commonly-confused pair, e.g. Copepod vs Ostracod), and one explicit out-of-distribution (a particle / bubble / blank — to demonstrate the model has no "I don't know" output, which is itself a documented limitation)
- Cite the source of each example image in its caption (which dataset, which split, which file) so reviewers can audit
- Add a banner to the Space explaining the model was trained on `<dataset>` images from `<instrument>`; predictions on images from other instruments may not transfer (this is the C8 limitation, made user-visible)

---

### Pitfall M6: Promising support / contributions / a roadmap that the team can't sustain

**Phase:** LAUNCH

**What goes wrong:** README ships with `CONTRIBUTING.md` saying "PRs welcome, we triage weekly", an issue template requesting reproducer scripts, a roadmap promising new datasets quarterly. Six months later the issue tracker has 30 open issues, 12 stale PRs, and the project looks abandoned. For a research project from a single institution, this is the modal outcome — recent ML-maintenance research finds funding-for-maintenance is the structural bottleneck.

**Prevention:**
- Use the BSI / "honest README" pattern: explicit "this is research code, support is best-effort, response time may be weeks"
- Don't ship issue templates that demand more than the team can read — a single "Bug report" template asking for repro + env is enough; skip "Feature request" until/unless someone is committed to triaging
- Put the support model on the README: maintained-by-X, response-target-Y-business-days, sponsored-by-Z, expected-EOL-Q4-2027 if applicable
- Contribution guide: only document the workflow the team actually uses — if you don't run pre-commit, don't require it of contributors

**Sources:** [Scalability and maintainability challenges in ML (arxiv 2504.11079)](https://arxiv.org/html/2504.11079v1), [Predicting maintenance cessation of OSS repos (arxiv 2507.21678)](https://arxiv.org/html/2507.21678v1/)

---

### Pitfall M7: Gradio Space exposes file-system-read vulnerabilities or lacks file-size caps

**Phase:** DEMO

**What goes wrong:** Gradio has had a documented history of file-read vulnerabilities (see Horizon3.ai 2024 disclosure on stealing secrets from Spaces). Default Gradio doesn't cap upload size — a single user uploading a 4K microscopy TIFF (50-200 MB) can saturate the Space's disk. Corrupted / non-image uploads cause Pillow exceptions that aren't user-facing.

**Prevention:**
- Set `max_file_size="20mb"` on `gr.Image` (covers realistic microscopy + leaves headroom)
- Set `delete_cache=(3600, 7200)` on `Blocks(...)` to evict cached uploads on a schedule
- Wrap the inference handler in a try/except that converts `PIL.UnidentifiedImageError` → user-visible "Could not read this image format. Supported: JPG, PNG, TIFF, WebP."
- Use Gradio 5.x — the 5.x security review (HF blog) added a security pass and Semgrep CI; older versions have known CVEs
- Pin gradio to a known-good 5.x.y patch (per `C7`)

**Sources:** [Gradio Security and File Access docs](https://www.gradio.app/guides/file-access), [HF blog: Security Review of Gradio 5](https://huggingface.co/blog/gradio-5-security), [Horizon3.ai: Exploiting File Read in Gradio to Steal HF Spaces Secrets](https://horizon3.ai/attack-research/disclosures/exploiting-file-read-vulnerabilities-in-gradio-to-steal-secrets-from-hugging-face-spaces/), [Gradio CVE feed](https://feedly.com/cve/vendors/gradio_project)

---

## Plankton / Marine-Microscopy Domain-Specific Pitfalls

These are the failure modes a marine biologist will spot first. They are not generic CV mistakes; they are how the public release fails when domain experts read it.

### Pitfall D1: Cross-instrument domain shift — model trained on one imager is silently invalid on another

**Phase:** MODEL_RELEASE + DEMO

**What goes wrong:** The 7 datasets in `INTEGRATIONS.md` come from completely different imaging instruments — ISIIS (in-situ shadowgraph), FlowCam (flow-through brightfield), UVP6 (in-situ underwater vision profiler), WHOI Imaging FlowCytobot (IFCB), ZooScan (flatbed scanner of preserved samples), Lensless (holographic), JEDI/CPICS. The image *appearance* (colour, background, magnification, illumination, presence of preservation artifacts) is wildly different across instruments — even for the same species.

A model trained on, say, WHOI IFCB images and applied to a FlowCam image of the same Copepod will degrade catastrophically. Recent benchmarking work (Bureš et al. 2024 "Producing plankton classifiers robust to dataset shift", and the 2025 OOD plankton benchmark) puts the typical drop at 10-30 pp top-1 across instruments — far worse than the few-percent drop typical in natural-image OOD benchmarks. A demo that lets users upload "any plankton image" and run any model risks producing systematically wrong answers when the user's instrument doesn't match the model's training instrument.

**Why it happens:** Different instruments produce fundamentally different image distributions. Image-only DL models with no domain conditioning have no way to know "this image is from instrument X, route to model trained on X."

**Consequences:** Marine biologist uploads a typical flow-cytometer image to a model trained on shadowgraph data. Model returns a confident species name. Biologist accepts it. Downstream ecological interpretation is wrong. Project is held responsible.

**Prevention:**
- Each model card MUST explicitly name the source instrument in "Intended Use" and explicitly call out cross-instrument transfer in "Out of Scope": "This model was trained on images from <Instrument X> and is not validated on images from other instruments. Cross-instrument application requires retraining or domain adaptation."
- The Space's model picker dropdown must show the instrument prominently next to the dataset name (e.g. "ISIIS — In-Situ Shadowgraph", not just "ISIIS")
- The Space's banner says "Choose the model that matches your instrument."
- (Stretch / future) Compute pairwise cross-instrument eval matrix (model from X applied to test set from Y) and link it from each model card. Even a small table of "this model on the OTHER 6 datasets' test sets" makes the limitation undeniable.
- The "Limitations" section cites the OOD benchmark paper(s) so users know this is a known field issue, not a bug we're hiding

**Detection:** Domain-expert review pass — a marine biologist looks at each model card and the Space, attempts to predict on an image from a non-matching instrument. If the model gives a confident wrong answer with no warning, the disclaimer is insufficient.

**Sources:** [Producing Plankton Classifiers Robust to Dataset Shift (arxiv 2401.14256)](https://arxiv.org/abs/2401.14256), [Benchmarking OOD Detection for Plankton Recognition (arxiv 2510.17179)](https://arxiv.org/html/2510.17179), [In-domain vs out-of-domain transfer learning in plankton (Nature Sci Rep s41598-023-37627-7)](https://www.nature.com/articles/s41598-023-37627-7)

---

### Pitfall D2: Class taxonomy drift — same species has different label names across the 7 datasets

**Phase:** MODEL_RELEASE + DOCS

**What goes wrong:** Plankton class definitions are not standardised across datasets. The same organism may appear as `Copepoda` (order-level) in one dataset, `Calanoida` (suborder) in another, `Calanus_finmarchicus` (species) in a third, or `Copepods_general` (morphotype/free-form). Some datasets use morphological categories (`oblong_dark_object`) rather than taxonomic ones. WHOI uses imaging-flow-cytometer-specific functional groups; ZooScan uses flatbed-scanner morphotypes.

A user who reads "this model classifies plankton" and naïvely compares predictions across the 7 models will find that the SAME image of a copepod returns 7 different labels. They will conclude the project is inconsistent.

**Why it happens:** Each dataset was annotated by its source lab using the taxonomy convention of that lab / instrument. There is no community-standard plankton taxonomy mapping (efforts exist — WoRMS, EcoTaxa — but adoption is partial).

**Consequences:** Users compare models' predictions and get confused. Cross-model comparisons in the announcement / paper risk being apples-to-oranges. A marine biologist will spot this in 30 seconds and lose trust.

**Prevention:**
- Each model card's "Limitations" section explicitly states the labels are *the source dataset's labels*, NOT a unified taxonomy: "Class names follow the <SourceDataset> annotation convention. The same organism may have a different label in `<OtherDataset>`. We do not provide a unified cross-dataset taxonomy in v1."
- The Space displays the FULL class label as it came from the dataset, including any prefix that disambiguates (e.g. `whoi/Copepoda` vs `zooscan/copepoda_general`)
- README / launch announcement explicitly does NOT make the claim "trained on plankton" without adding "label spaces are dataset-specific"
- Consider (out of scope for v1, but flag for v1.1) a `taxonomy.csv` in each model repo mapping that model's labels to WoRMS AphiaIDs
- The `notebooks/planktonzilla_taxo.csv` file referenced in `notebooks/gen_planktonzilla.py:466` is exactly this artefact — promote it to a public, citable file in the repo

**Sources:** [Survey of automatic plankton image recognition (Springer)](https://link.springer.com/article/10.1007/s10462-024-10745-y) ("the division in classes of existing databases is quite varied and existing approaches are tuned to perform well in specific datasets"), [Computer Vision and Deep Learning Meet Plankton (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0262885624000374)

---

### Pitfall D3: No "I don't know" output — model confidently mis-classifies bubbles, particles, debris, fragments

**Phase:** DEMO + MODEL_RELEASE

**What goes wrong:** Plankton imagers produce vast quantities of non-plankton images: bubbles, marine snow, body fragments, suspended sediment, instrument noise. The 2025 OOD-plankton benchmark identifies these as the dominant Far-OOD class and shows that all-classes-trained classifiers will confidently assign them to whatever plankton class is morphologically nearest. None of `planktonzilla`'s losses or models, as inventoried, include an explicit "Other / Background / Detritus" class or an OOD-detector.

A marine biologist (or the demo's own example set, per M5) uploads a bubble. The model returns "92% Acantharea." Trust evaporates.

**Why it happens:** The training pipeline (`planktonzilla/dataset.py` → `imagefolder`) uses whatever classes the source dataset chose. Most plankton datasets exclude non-biological objects from training (they were filtered out during annotation). The model has never seen a bubble at training time, so its outputs on a bubble are arbitrary.

**Consequences:** Field deployment trust collapses. The model card cannot honestly claim usefulness on raw imager output without flagging this.

**Prevention:**
- Add a "Out of Scope" bullet in every model card: "This model was trained only on annotated plankton classes. Non-biological objects (bubbles, particles, debris, fragments) were excluded from training and the model will assign them with high confidence to the nearest plankton class. Apply a foreground / quality filter before classification."
- In the Space, include at least one OOD example (bubble image, blank image, non-plankton image) with a caption demonstrating the model's confident-wrong behaviour, so users see the limitation rather than read about it
- Show top-K (e.g. K=5) probabilities, not just top-1, so users can see when the prediction is "evenly spread" — partial proxy for uncertainty even without a real OOD detector
- Flag in PROJECT.md "Active" as a v1.1 candidate: ship a foreground-filter / OOD-rejection model alongside the classifiers (this is exactly what the 2025 OOD benchmark studies)

**Sources:** [Benchmarking OOD Detection for Plankton (arxiv 2510.17179)](https://arxiv.org/html/2510.17179) (explicit Bubbles & Particles Far-OOD scenario), [Operational phytoplankton recognition (Frontiers)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.867695/full)

---

### Pitfall D4: Eval split is not stratified by sampling-event / cruise / station → optimistic numbers from data leakage

**Phase:** MODEL_RELEASE

**What goes wrong:** Plankton images from a single cruise / sampling station are highly correlated (same water mass, same hour, same instrument settings, often near-duplicate images of the same organism cluster). A random train/test split (which is what `dataset.py:152-169` does — `train_test_split(stratify_by_column="label", seed=split_seed)`) will leak near-duplicates into the test set. The reported "Top-1 = 92%" is then 5-15pp inflated relative to held-out-cruise performance, which is what users care about (they're going to apply the model to NEW samples from NEW cruises).

The reproducibility-crisis literature (Princeton survey on data leakage in ML-based science) flags this as the #1 cause of optimistic numbers in scientific ML.

**Why it happens:** Random split is the framework default. Cruise / station / haul metadata is in the source dataset but typically not propagated through the importer.

**Consequences:** Eval numbers are inflated. First field deployment shows much lower accuracy. Reviewers who know this domain (marine biology paper reviewers) will deduct credibility on sight.

**Prevention:**
- Where source datasets carry cruise / station / sampling-event metadata (WHOI does — IFCB images carry sample IDs; ISIIS does — in-situ time/depth), use a *grouped* split (`sklearn.model_selection.GroupShuffleSplit` or `GroupKFold` with group=sampling-event) instead of `train_test_split`
- Where they don't, add a "Limitations" caveat to the model card: "Train/test split is random. Held-out cruise performance is likely lower; we do not report it because cruise metadata is not available in the source dataset."
- Surface the split methodology in the model card under "Evaluation" — exactly which split, what stratification, what was held out
- For v1.1, add a `dataset.split_strategy: random | grouped_by_cruise | held_out_year` enum and re-eval where data permits

**Sources:** [Leakage and the Reproducibility Crisis in ML-based Science (Princeton)](https://reproducible.cs.princeton.edu/), [Reproducibility standards for ML in life sciences (PMC9131851)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9131851/)

---

## Phase-Specific Warning Matrix

Use this in the roadmap to attach the right pitfalls to each phase as checklist items.

| Phase | Critical pitfalls to gate on | Moderate pitfalls to address | Notes |
|-------|------------------------------|------------------------------|-------|
| HARDENING | C1 (CLIP loading), C3 (preprocessing) | — | Smallest possible scope: just enough to make the snippet work for all 7 checkpoints |
| MODEL_RELEASE | C2 (id2label), C3 (preprocessor pin), C4 (eval reproducible), C5 (license matrix) | M1 (card sections), M2 (eval frontmatter), M3 (links), M4 (per-class metrics) | One full pre-publish gate per checkpoint |
| DEMO | C6 (cold start / OOM), C7 (deps pinning) | M5 (examples), M7 (Gradio security) | Plus D3 (OOD example in defaults) |
| DOCS | C1 (snippet works clean-env) | M3 (links), M6 (honest support model) | DOC-02 must be tested from a fresh `venv` |
| LAUNCH | C5 (license claims) | M6 (sustainability), M2 (eval visibility) | Plus D1, D2 disclaimers in announcement |
| **All MODEL_RELEASE phases** | — | — | **D1 (cross-instrument), D2 (taxonomy), D3 (no OOD class), D4 (split leakage)** belong in every model card |

---

## Pre-Release "Must-Pass" Checklist (extracted)

A single linear gate that consolidates all `[ ]` items above. The roadmap can paste this directly.

**Per-checkpoint gate (run for each of the 7 models):**
- [ ] Snippet from `DOC-02` runs in a fresh `python -m venv` with the model card's pinned versions, returns a non-`LABEL_X` species name (C1, C2)
- [ ] `model.config.id2label[0]` is a real species name AND matches `dataset.features.label.names[0]` index-by-index (C2)
- [ ] `preprocessor_config.json` exists and matches the training-time mean/std/size (C3)
- [ ] `pz_verify_release` recomputes the model card's headline metric within ±0.5pp (C4)
- [ ] `model.config` contains `_planktonzilla_git_sha` and `_planktonzilla_train_run` (C4)
- [ ] License matrix decision logged; model `license:` matches the most-restrictive parent (C5)
- [ ] All 9 model card sections present; `ModelCard.validate()` passes (M1)
- [ ] Model card YAML `model-index` renders an evaluation widget on the rendered Hub page (M2)
- [ ] All URLs in the rendered card return 200 anonymously (M3)
- [ ] Per-class metrics (confusion matrix or per-class F1 + support count) included (M4)
- [ ] "Out of Scope" section names the instrument and forbids cross-instrument use (D1)
- [ ] "Limitations" section calls out taxonomy non-standardisation (D2) and absence of OOD class (D3)
- [ ] "Evaluation" section discloses the split methodology (D4)

**Demo gate (single Space):**
- [ ] Cold start <30s, all 7 models loaded at module top-level (C6)
- [ ] Switching all 7 models in sequence does not grow RSS >100 MB (C6)
- [ ] `requirements.txt` pins exact patch versions; `python_version` and `sdk_version` set (C7)
- [ ] Default examples include 1 easy + 1 hard + 1 ambiguous + 1 OOD (M5, D3)
- [ ] `max_file_size` set; non-image upload returns user-visible error (M7)
- [ ] Gradio pinned to a 5.x.y reviewed version (M7)
- [ ] Banner names the instrument-per-model (D1) and the dataset-specific labels (D2)

**Launch gate:**
- [ ] README disclaims the support model honestly (M6)
- [ ] Announcement does not claim "trained on plankton" without the dataset-specific labels caveat (D2)
- [ ] No commercial-use claim made for any CC-BY-NC-derived model (C5)

---

## Sources Referenced (consolidated)

### HuggingFace Hub & model cards
- [HF Model Cards docs](https://huggingface.co/docs/hub/model-cards) — HIGH (official)
- [HF Annotated Model Card template](https://huggingface.co/docs/hub/model-card-annotated) — HIGH (official)
- [HF Customizing models / trust_remote_code](https://huggingface.co/docs/transformers/custom_models) — HIGH (official)
- [HF Licenses doc](https://huggingface.co/docs/hub/en/repositories-licenses) — HIGH (official)
- [HF Spaces dependencies docs](https://huggingface.co/docs/hub/spaces-dependencies) — HIGH (official)
- [HF Spaces ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) — HIGH (official)
- [HF Spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference) — HIGH (official)
- [HF preprocess docs (image_processor)](https://huggingface.co/docs/transformers/main/preprocessing) — HIGH (official)
- [HF image classification task](https://huggingface.co/docs/transformers/tasks/image_classification) — HIGH (official)
- [timm quickstart (resolve_data_config)](https://huggingface.co/docs/timm/quickstart) — HIGH (official)
- [HF blog: Security review of Gradio 5](https://huggingface.co/blog/gradio-5-security) — HIGH (official)
- [HF blog: ZeroGPU AOT compilation](https://huggingface.co/blog/zerogpu-aoti) — HIGH (official)

### Bug threads / forum incidents
- [HF forum: id2label/label2id confusion](https://discuss.huggingface.co/t/errors-with-label2id-id2label-with-muticlass-classification/160188) — MEDIUM (community)
- [transformers issue #28589 — id2label assignment bug](https://github.com/huggingface/transformers/issues/28589) — MEDIUM (GitHub issue)
- [transformers issue #22260 — trust_remote_code local code](https://github.com/huggingface/transformers/issues/22260) — MEDIUM
- [transformers issue #29251 — ModuleNotFoundError with trust_remote_code](https://github.com/huggingface/transformers/issues/29251) — MEDIUM
- [HF forum: Gradio app runtime error](https://discuss.huggingface.co/t/gradio-app-runtime-error/111441) — MEDIUM
- [diffusers discussion #10936 — Gradio + memory leak](https://github.com/huggingface/diffusers/discussions/10936) — MEDIUM
- [HF forum: pinning Spaces packages](https://discuss.huggingface.co/t/how-to-pin-version-of-spaces-package/56702) — MEDIUM
- [HF forum: Spaces not updating from requirements.txt](https://discuss.huggingface.co/t/huggingface-spaces-not-updating-packages-from-requirements-txt/92865) — MEDIUM
- [vicgalle/xlm-roberta-large-xnli-anli license conflict discussion](https://huggingface.co/vicgalle/xlm-roberta-large-xnli-anli/discussions/3) — MEDIUM (real-world example)
- [Horizon3.ai: Gradio file-read vulnerability disclosure](https://horizon3.ai/attack-research/disclosures/exploiting-file-read-vulnerabilities-in-gradio-to-steal-secrets-from-hugging-face-spaces/) — HIGH (security advisory)
- [Gradio CVE feed](https://feedly.com/cve/vendors/gradio_project) — HIGH

### Reproducibility / sustainability research
- [Princeton: Leakage and Reproducibility Crisis in ML-based Science](https://reproducible.cs.princeton.edu/) — HIGH
- [Reproducibility standards for ML in life sciences (PMC9131851)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9131851/) — HIGH
- [Neptune: How to solve reproducibility in ML](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml) — MEDIUM
- [What's documented in AI? Analysis of 32K HF model cards (arxiv 2402.05160)](https://arxiv.org/html/2402.05160v1) — HIGH (recent peer-reviewed)
- [Empirical analysis of HF licensing challenges (arxiv 2502.04484)](https://arxiv.org/html/2502.04484v2) — HIGH
- [Scalability and maintainability challenges in ML (arxiv 2504.11079)](https://arxiv.org/html/2504.11079v1) — MEDIUM
- [Predicting maintenance cessation of OSS repos (arxiv 2507.21678)](https://arxiv.org/html/2507.21678v1/) — MEDIUM

### Plankton / marine domain
- [Producing Plankton Classifiers Robust to Dataset Shift (arxiv 2401.14256)](https://arxiv.org/abs/2401.14256) — HIGH (peer-reviewed, recent)
- [Benchmarking OOD Detection for Plankton Recognition (arxiv 2510.17179)](https://arxiv.org/html/2510.17179) — HIGH (recent benchmark paper)
- [In-domain vs out-of-domain transfer learning in plankton (Nature Sci Rep 2023)](https://www.nature.com/articles/s41598-023-37627-7) — HIGH
- [Survey of automatic plankton image recognition (Springer 2024)](https://link.springer.com/article/10.1007/s10462-024-10745-y) — HIGH (review article)
- [Computer Vision and DL Meet Plankton (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0262885624000374) — HIGH
- [Operational phytoplankton recognition (Frontiers Marine Science)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.867695/full) — MEDIUM
- [Annotation-free learning of plankton (Nature Sci Rep s41598-020-68662-3)](https://www.nature.com/articles/s41598-020-68662-3) — MEDIUM

---

*PITFALLS audit: 2026-05-12. Confidence: MEDIUM-HIGH overall. HF / Gradio claims are HIGH (verified against official docs and current GitHub/forum threads). Plankton-domain claims are MEDIUM (grounded in two recent peer-reviewed benchmark papers from 2024 and 2025; should be reviewed by a marine biologist before LAUNCH).*
