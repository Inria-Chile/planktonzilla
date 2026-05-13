# State: planktonzilla

**Last updated:** 2026-05-13
**Project initialized:** 2026-05-12

## Project Reference

**Core Value:** Curated, citable pre-trained plankton-classification models on Hugging Face Hub, each with a real model card, backed by a repo a stranger can credibly load a model from.

**If everything else slips, the artifact that must ship:** The HF Hub model release on `huggingface.co/project-oceania` (4 CLIP-backed models from Phase 3 are the headline; pure-HF wave + Gradio Space + OcéanIA Collection complete the v1 surface in Phase 4).

**Current focus:** Phase 1 — Foundation (checkpoint manifest + label-mapping freeze). Nothing downstream can credibly start without this.

## Current Position

**Milestone:** v1 — public release of pre-trained plankton classifiers on HF Hub
**Phase:** 1 — Foundation
**Plan:** Not yet planned (next: `/gsd-plan-phase 1`)
**Status:** Roadmap complete, ready for plan-phase
**Progress:** [░░░░░░░░░░] 0/5 phases complete

### Phase 1 progress

- [ ] FOUND-01 — release-manifest.yaml created with 7 entries, each verified physically retrievable
- [ ] FOUND-02 — id2label / label2id frozen per checkpoint against source dataset's class list

## Performance Metrics

**Phases:** 5 (coarse granularity, at upper edge — justified by clean per-phase verifiable units)
**Requirements coverage:** 22/22 (100%)
**Orphaned requirements:** 0
**Duplicates:** 0
**Plans created:** 0
**Plans completed:** 0
**Phases completed:** 0

## Accumulated Context

### Key Decisions (carried from PROJECT.md + REQUIREMENTS.md)

| Decision | Rationale | Where |
|----------|-----------|-------|
| Anchor v1 on HF Hub model release, not codebase rebuild | Models reach all four audiences; codebase rebuild reaches none directly | PROJECT.md |
| Ship one model per dataset for all 7 datasets (no triage) | README already advertises all 7; cutting any of them creates a credibility gap | PROJECT.md |
| CLIP models ship FIRST (REL-06) | 4 of 7 winners are CLIP-backed; harder to package (`trust_remote_code`) but they are the most compelling artifacts — front-load risk | REQUIREMENTS.md |
| Use Option A (`PreTrainedModel` + `register_for_auto_class` + `trust_remote_code=True`) for ClipClassifier packaging | Single universal `AutoModelForImageClassification.from_pretrained` snippet for all 7 models; no PyPI release needed; survives later un-vendoring | research/STACK.md |
| Eval numbers come from `release/eval_model.py` running against the published checkpoint, NEVER transcribed from W&B | Reproducibility for paper reviewers; the worst possible failure mode is "I can't reproduce your numbers" | research/PITFALLS.md (C4) |
| Leave datasets as they are (no license audit, no card edits, no taxonomy work) | User direction 2026-05-12 — focus the milestone on documentation and code, not dataset-side cleanup | REQUIREMENTS.md preamble |
| Pip / saliency / full hardening explicitly deferred to v1.1 | Scope discipline — quality of headline artifact > breadth of artifacts | PROJECT.md |
| Demo is single-image + top-K only, no Grad-CAM | Saliency varies by architecture (CNN vs ViT/CLIP); not worth inconsistent UX in v1 | PROJECT.md |
| 5-phase roadmap (at upper edge of coarse) rather than 4 | Each phase has a clean, verifiable unit of completion; collapsing Phase 4 (pure-HF + demo) into Phase 3 would mix two logically independent concerns and obscure the parallelism | ROADMAP.md |

### Active Todos

None — roadmap just completed; next action is `/gsd-plan-phase 1`.

### Blockers

None.

### Open Questions (for plan-phase to resolve)

These are flagged in research as needing a small spike during planning, NOT blockers for the roadmap itself:

- **Phase 2:** Exact `open_clip_torch` PyPI version range compatible with the vendored `open_clip 4.0.0.dev0` — one-afternoon API surface comparison spike (research/SUMMARY.md, research/STACK.md). Resolution belongs in Phase 2's plan.
- **Phase 4:** Spaces hardware tier — benchmark EVA02-L-14 + ConvNeXtV2-Huge inference latency on `cpu-upgrade` before deciding whether to bump to `t4-small` (research/SUMMARY.md, research/STACK.md). Resolution belongs in Phase 4's plan.
- **Phase 1:** Eval split strategy for datasets with cruise/station metadata (WHOI, ISIIS) — grouped split (more honest, more work) vs random split with explicit limitation disclosure. Decision belongs in Phase 1's plan because changing the split after publishing invalidates the published numbers (research/PITFALLS.md D4).
- **Phase 1:** Audit existing private Hub checkpoints — if `Trainer.push_to_hub` was called with `model_push_as_private: true` during training, there may be private repos with stale or missing `preprocessor_config.json` to clean up (research/SUMMARY.md). Belongs in Phase 1's plan.

## Quick Tasks Completed

| ID         | Title                                  | Date       | Commit    | Summary                                                                                  |
|------------|----------------------------------------|------------|-----------|------------------------------------------------------------------------------------------|
| 260512-u20 | Restructure README.md for v1 milestone | 2026-05-13 | `1d6d293` | [SUMMARY](quick/260512-u20-restructure-readme-md-for-v1-milestone/260512-u20-SUMMARY.md) |

## Session Continuity

**Last session:** 2026-05-12 (initialization)

**What happened this session:**
1. Project initialized with `/gsd-new-project` orchestrator
2. PROJECT.md drafted (Core Value framed around HF Hub model release)
3. Codebase mapped (`.planning/codebase/*` — STACK, ARCHITECTURE, STRUCTURE, CONVENTIONS, TESTING, INTEGRATIONS, CONCERNS)
4. Research conducted (`.planning/research/*` — STACK, FEATURES, ARCHITECTURE, PITFALLS, SUMMARY)
5. User redirect 2026-05-12: "leave datasets as they are, focus on documentation and code" → REQUIREMENTS.md authored with 22 v1 requirements across FOUND/HARD/REL/DEMO/DOC/LAUNCH categories, dataset-side work explicitly OOS
6. ROADMAP.md created — 5 phases, 100% coverage, REL-06 CLIP-first sequencing honored
7. STATE.md (this file) initialized

**Next session should:**
- Run `/gsd-plan-phase 1` to decompose Phase 1 (Foundation) into concrete plans
- Phase 1 plan should resolve the eval-split-strategy and private-checkpoint-audit open questions

---
*State initialized: 2026-05-12*
