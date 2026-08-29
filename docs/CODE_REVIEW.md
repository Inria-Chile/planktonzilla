# `planktonzilla` — in-depth code review

**Reviewed:** `0df8004` (tip of `main`, 2026-08-28) · whole tree, ~17.2k lines of source + ~11k lines of tests
**Method:** 18 subsystem/cross-cutting reviewers → per-finding adversarial verification (2 independent
refuters each, majority-refute kills the finding) → completeness sweep → manual re-verification of every
HIGH finding by the reviewer of record.

132 raw findings → 123 after dedup → **49 refuted outright**, 51 confirmed unanimously, 28 contested.
A completeness sweep added 10 new candidates, 5 of which survived. **79 findings reported below.**

---

## Baseline health

Everything the project gates on is green, and that is worth stating before the findings:

| Gate | Result |
| --- | --- |
| `ruff check planktonzilla/ tests/ scripts/` | **All checks passed** |
| `ruff format --check` | **89 files already formatted** |
| `pytest` (the CI set) | **675 passed, 2 skipped** in 89 s |
| `pytest tests/test_train.py tests/test_datasets.py` (excluded from CI) | **9 passed** in 6 m 47 s |

The security surface is genuinely clean: no `subprocess`/`os.system`/`eval`/`exec` anywhere in
`planktonzilla/`, no `extractall`, no `yaml.load`, no `verify=False`, no committed secret, and
`open_clip_ext/factory.py:101-126` already handles `torch.load(weights_only=True)` with a documented
fallback. The `KNOWN_ISSUES.md` / `RESOLVED_ISSUES.md` ledger is unusually disciplined — deliberate
deferrals are recorded with risk labels and exit conditions, and reviewers were instructed not to
re-report them.

**The findings below are therefore not "the codebase is bad".** They are the residue left after a clean
lint, a green suite, and a curated known-issues ledger — which is exactly where the interesting bugs live.

---

## Tier 1 — verified by hand, reproduced

I reproduced each of these myself against the working tree. These are not model claims.

### 1.1 `build_frepj_taxonomy.write_csv` destroys 644 taxonomy rows on a no-op re-run

`planktonzilla/planktonzilla_dataset/utils/build_frepj_taxonomy.py:434-454`

`write_csv` copies the file prefix up to the **first** `frepj` row and then writes `prefix + frepj_block`
— and nothing else. Everything after the frepj block is discarded. That was safe when frepj was the last
block; it no longer is. The CSV has since grown `daplankton` (line 1716) and four `tara_pacific_*` blocks
(lines 1760-2358) **after** frepj (line 1487).

Reproduced by feeding the file's own existing frepj rows back in — a perfect no-op re-run:

```
frepj rows fed back in: 229
data rows: 2358 -> 1714   (delta -644)
datasets entirely GONE: ['daplankton', 'tara_pacific_bongo', 'tara_pacific_decknet',
                         'tara_pacific_hsn', 'tara_pacific_manta']
```

The process exits 0 and logs `"Appended 229 rows … (idempotent, append-only)"`. The docstring at
lines 442-444 still asserts "a re-run … leaves the file byte-identical". Because `build_taxonomy_lookup`
is a LEFT join, the next build then emits ~2.35 M Tara Pacific images and all of DAPlankton with null
taxonomy rather than failing.

The sibling engine already does this correctly — `build_tara_pacific_taxonomy.append_to_master:743`
filters only its own `Dataset` lines and passes every other line through. Mirror it, and add a regression
test that runs `write_csv` against a CSV with a post-frepj block and asserts the row count is unchanged.

> Same file, second defect (finding #22): `Parsed.as_csv_row` hardcodes the four external-ID columns to
> `""` (lines 332-335), so a re-run also blanks the `wikidata_ID`/`aphia_ID`/`NCBI_ID`/`BOLD_ID` values
> that `resolve_frepj_ids.backfill_csv` wrote into 208 of the 229 frepj rows.

### 1.2 `atomic_replace` deletes the whole data directory on the migration command the tool itself prints

`planktonzilla/planktonzilla_dataset/make_planktonzilla.py:425-455`

For an existing target, `atomic_replace` does `output_dir.rename(previous)` → `staged.rename(output_dir)`
→ `shutil.rmtree(previous)`. It assumes `output_dir` contains nothing but the saved dataset. Nothing
enforces that, and `update_planktonzilla.py:234-239` — the deprecation notice printed on **every** run of
the old command — tells users the equivalent invocation is literally:

```
pz_planktonzilla base=hub sources=[] output_dir='${data_dir}'
```

Reproduced with `output_dir == data_dir`:

```
before: ['manual_downloads', 'whoi_imagefolder', 'zooscan_raw_download']
after : ['data-00000-of-00001.arrow', 'dataset_info.json', 'state.json']
```

Every imagefolder, every raw archive, and `manual_downloads/` — the hand-fetched archives that by
definition cannot be re-downloaded automatically — are gone, and the run logs success.

**Fix:** refuse a target that is not a saved dataset (`state.json`/`dataset_info.json` absent) before
staging; reject `output_dir` resolving to `data_dir`; and correct the migration line to point at
`${data_dir}/planktonzilla-17M`.

### 1.3 `apply_version` writes a version shape the project's own reader cannot parse

`make_planktonzilla.py:380-386` and `:604`

`apply_version` does `ds.info.version = version` with a plain `str`. `DatasetInfo` only coerces to
`datasets.utils.Version` in `__post_init__`, so a post-hoc assignment bypasses it and `save_to_disk`
serialises a bare string. `check_base_on_disk` then does `(info.get("version") or {}).get("version_str")`
— `.get` on a `str`. Reproduced end to end:

```
serialized version field -> '1.4.0'
check_base_on_disk RAISED: AttributeError: 'str' object has no attribute 'get'
```

That call site is the **unconditional** pre-build guard (line 1263, and again inside `run_preflight`), so
after any `pz_planktonzilla version=… ` release run, *every* subsequent incremental run against that
artifact dies before building anything — with an exception naming neither the file nor the reason.

The suite hides it from both sides: `test_make_planktonzilla_splice.py:480` asserts
`str(ds.info.version) == "1.4.0"` (which passes because `DatasetInfo.from_dict` re-parses the string), and
`test_make_planktonzilla_hydra.py:1173` hand-writes `info["version"] = {"version_str": …}` — the dict shape
the real code path never produces.

**Fix:** `ds.info.version = Version(version)`; harden line 604 against both shapes; build the fixture
through `apply_version` + `save_to_disk` instead of hand-writing the info dict.

### 1.4 `MaximumMarginLoss` uses the first sample's label as the positive class for the entire batch

`planktonzilla/loss.py:238`

```python
min_pos_prob = rm_obj_dists[:, labels.data.cpu().numpy()[0]].data
```

`labels.data.cpu().numpy()[0]` is a **scalar** — the ground truth of whichever example the DataLoader
shuffled to position 0 — so one logit column is used as the positive score for every row. The correct
per-row one-hot (`index_float`) is already built by the caller and passed into this very function, where
it is used only for the negative mask.

Reproduced (`cls_num_list=[100,10,1]`, `s=1`, one fixed batch, four row orderings):

```
original order      : 0.978300
permuted [1,0,2,3]  : 1.055593
permuted [2,1,0,3]  : 1.022042
permuted [3,2,1,0]  : 1.055593

true-class logits (correct): [0.90, 0.80, 0.70, 0.60]
what the code uses         : [0.90, 0.30, 0.20, 0.50]   <- column 0 for every row
```

A mean-reduced per-sample loss **must** be permutation-invariant. With `shuffle=True` the reference column
changes every step, so `custom_loss=max_margin` trains against noise plus a per-batch bias toward one
arbitrary class, reporting normal-looking losses throughout. `tests/test_loss.py` covers only `FocalLoss`.

**Fix:** `min_pos_prob = (rm_obj_dists * index_float).sum(1).data` (also removes a GPU→CPU sync). Add
`L(x, y) == L(x[perm], y[perm])` for every loss in the module — that one property test catches this class
of bug outright.

### 1.5 `RobustAsymmetricLoss`'s focusing weight is inverted

`planktonzilla/loss.py:407-428` — *flagged CONTESTED by the panel; upheld by my own reproduction.*

The sibling `AsymmetricLoss` masks both probability terms by the label indicators before forming the
focusing weight (lines 332-333: `xs_pos = xs_pos * targets`, `xs_neg = xs_neg * anti_targets`). `RAL`
computes `targets`/`anti_targets` but uses them only in the exponent — the `xs_pos`/`xs_neg`
reassignments are applied to every class unconditionally. Line 421's `(1 - xs_pos)` additionally reads the
*already-reassigned* `xs_pos` from line 411, not the probability.

Measured with the shipped defaults:

```
p=1e-06   focusing base= 1.0000  base^gamma_neg= 0.9999   (intended ~p^4=1e-24)
p=0.001   focusing base= 0.9828  base^gamma_neg= 0.9331   (intended ~p^4=1e-12)
p=0.01    focusing base= 0.8843  base^gamma_neg= 0.6115   (intended ~p^4=1e-08)
p=0.1     focusing base= 0.4212  base^gamma_neg= 0.03147  (intended ~p^4=1e-04)
p=0.5     focusing base= 0.6818  base^gamma_neg= 0.2161   (intended ~p^4=0.0625)
```

The weight is non-monotonic and **inverted at the most common operating point**: a confidently-correct
negative (p=1e-6) keeps full weight 0.9999, which is precisely the case `gamma_neg=4` exists to suppress.
On one well-classified sample with C=1000, `AsymmetricLoss` returns `7.5e-4` and `RobustAsymmetricLoss`
returns `1.399` — 1870× larger, dominated by the label-smoothing tail of unsuppressed easy negatives.

`custom_loss=ral` therefore optimises label-smoothed cross-entropy plus a non-monotonic reweighting, not
the RAL objective it is selected for. Worth re-deriving against the paper before patching, but something
is definitely wrong.

### 1.6 Two published Tara Pacific taxonomy blocks carry the wrong taxon

`planktonzilla/planktonzilla_dataset/utils/build_tara_pacific_taxonomy.py:455`

`_existing_indexes` builds the verbatim-reuse index with `by_raw.setdefault(row["Raw_Labels"], row)` —
**first row in file order wins**, with no check that the donor describes the same taxon the class dir is
keyed to, and no warning when two pre-existing rows disagree. `build_rows` rule 1 then copies all 17
non-key columns from that arbitrary donor. The reconciliation report only tabulates `derived` rows, so the
323 `verbatim` rows never reach the human-verify checkpoint.

Two confirmed in the committed CSV:

| `Raw_Labels` | Correct (6 other sources / EcoTaxa) | What the Tara Pacific rows say |
| --- | --- | --- |
| `Harpacticoida` | Order *harpacticoida*, aphia **1102** | Family *tachidiidae*, Genus ***euterpina***, aphia **115348** |
| `Creseidae` | Family *creseidae*, aphia **411905** | Family ***cliidae***, Genus *clio*, Species *pyramidata*, aphia **139033** |

In both cases `zooscan`'s over-specific row (lines 686, 389) precedes `global_uvp5`'s correct one
(lines 767, 466), so it wins on file order — and CSV lines 2063/2220 and 2055/2215 publish every
harpacticoid copepod as the genus *Euterpina*, and an entire pteropod family as the wrong family.

Related (#21): `_anchor_donor`'s bare-name fallback `by_raw.get(anchor["name"])` promotes a genus-level
class dir to a species row the same way — `Codonellopsis<Tintinnidiidae` (genus *Codonellopsis*) inherits
flowcamnet's `codonellopsis morchella`.

**Fix:** collect all rows per `Raw_Labels`, reject a donor whose asserted rank is deeper than the class
dir's own EcoTaxa rank, and route ambiguous keys into the reconciliation report for human resolution.

### 1.7 The guard that is supposed to catch 1.6 compares only one column

`tests/test_tara_pacific_taxonomy.py:270`

`test_only_the_documented_row_departs_from_the_ecotaxa_lineage` predicates on `derived["Kingdom"] !=
row["Kingdom"]` alone. A departure at Phylum, Class, Order, Family or Genus is invisible to it. Comparing
Kingdom→Genus, **19 class dirs depart, not 3** — including both rows in 1.6. The module docstring, the
generated reconciliation report, and KI-29 all rest on this guard ("asserted to be the *only* three"), so
the documented contract is not the one the code enforces.

### 1.8 `freeze_backbone=true` freezes the classification head too — nothing trains

`planktonzilla/train.py:218-225`

```python
if "classifier" in name or "head" in name:  param.requires_grad = True
else:                                       param.requires_grad = False
```

This matches on parameter *names*. On the open_clip path `clip_model.py:90` builds
`nn.Sequential(visual, nn.Linear(num_features, num_labels))`, so the head's parameters are named
`1.weight` / `1.bias` — neither contains "classifier" nor "head":

```
open_clip path param names: ['0.0.weight', '0.0.bias', '1.weight', '1.bias']
  -> params freeze_backbone would keep trainable: NONE  <-- nothing trains
timm path param names: ['head.weight', 'head.bias', 'trunk.weight', 'trunk.bias']
  -> trainable: ['head.weight', 'head.bias']
```

`freeze_backbone` is declared in `configs/model/default_clip.yaml:14` and set `true` in five shipped model
configs. On the timm path it works; on the open_clip path every parameter is frozen, including the
freshly-initialised head — the only thing that *must* train. **Fix:** select the head by identity
(`self.model[1]` / `visual.head`), not by substring.

### 1.9 The HF token is printed and written to disk in cleartext on every run

`planktonzilla/utils/hydra.py:98-99`

`task_wrapper` calls `extras(cfg)` on every decorated entry point; `extras` calls
`print_config_tree(cfg, resolve=True, save_to_file=True)`, and `configs/extras/default.yaml:8` sets
`print_config: true` — this is the **default** path, not an opt-in. `resolve=True` expands
`hf_token: ${oc.env:HF_TOKEN, null}` (`configs/dataset_import/default.yaml:50`) to the live value:

```
UNRESOLVED: ['hf_token: ${oc.env:HF_TOKEN, null}']
RESOLVED  : ['hf_token: hf_SECRET_TOKEN_ABC123']
extras.print_config = True -> print_config_tree(resolve=True, save_to_file=True)
```

So a write-scoped token is echoed to the terminal *and* persisted to `config_tree.log` in the output dir
— which on CI or a shared HPC filesystem outlives the run. **Fix:** redact known-secret keys before
rendering, or drop `hf_token` from the printed tree.

### 1.10 Nine of eleven `configs/experiment/*.yaml` cannot compose

`configs/experiment/` — verified by composing each against `train.yaml`:

```
experiment configs: 2 compose, 9 FAIL (of 11)
  FAIL base_cifar10   base_cifar100   base_inaturalist   base_jedi   base_whoi
  FAIL base_zoolake   base_zoolake_transformers-vit      base_zooscan   test
  OK   base_lensless  default
```

All nine fail with `MissingConfigException` on group files that do not exist (`augmentation/whoi.yaml`,
`augmentation/zoolake.yaml`, `tracking/many_loggers.yaml`, …). Several also name `dataset/whoi.yaml` and
`dataset/zooscan.yaml` where the files are `whoi-plankton.yaml` and `zooscannet.yaml`, and two carry
`_target_`s under the pre-rename `deep_plankton` package.

The two that *do* compose then inherit `experiment/default.yaml`'s `experiment_metadata` block, which
interpolates `${model.criterion._target_}`, `${dataset.sampler_type}` and `${model.net.model_name}` —
**none of which is defined by any config in the repo** (verified by grep). Since `extras()` resolves the
whole tree at startup, that is a startup-time failure for the remaining two as well.

No test composes the experiment group; `tests/test_dataset_import_configs.py` covers only
`configs/dataset_import/`. One parametrised "every config group member composes" test closes this
permanently.

---

## Tier 2 — silent-failure and data-integrity (confirmed by the panel)

| # | Severity | Location | Finding |
| --- | --- | --- | --- |
| 29 | medium | `dataset_importer.py:1093` | `imagefolder_is_complete()` means "holds ≥1 file", but WHOI and JEDI build incrementally and `rmtree` each release as they go. A run preempted at release 4 of 9 reports **complete**, and the next run splices 4/9 of the source into the published corpus. |
| 26 | medium | `dataset_importer.py:1876` | JEDI's `_prepare_imagefolder` unlinks archives and `rmtree`s release dirs **inside the shared `datasets` extraction cache**. `force_extract` never rebuilds it, so a rerun finds a hollow cache and copies nothing. |
| 13 | medium | `dataset_importer.py:2090` | `global_uvp5`'s copy worker logs `OSError` and returns `None` for both success and failure; nothing reconciles against `len(copy_tasks)`. A partial 7.4 M-image copy passes the only downstream gate (non-empty imagefolder). |
| 27 | medium | `dataset_importer.py:1848` | WHOI's `except OSError: logger.debug("… already in …")` names a cause that cannot occur — `copy2` overwrites silently (verified). What it actually swallows, at DEBUG, is `ENOSPC`/`EACCES`/`EIO`/`EMFILE`. |
| 25 | medium | `generate_planktonzilla.py:1131` | `refresh=rebuild` is a silent no-op: `build_overrides` appends `force_imagefolder_preparation=True`, but `import_and_redefine_source` skips `import_dataset()` entirely when the imagefolder is non-empty, so the flag never reaches its gate. |
| 7, 24 | medium | `make_planktonzilla.py:819` | Pre-flight decides "already built" with `os.listdir`, while the real run uses `imagefolder_is_complete()`. Its own docstring promises the two agree. A partial source is silently skipped by `check_downloads=needed`. |
| 14 | medium | `dataset_importer.py:817` | `is_valid_image_file` catches `(IOError, SyntaxError)`; `PIL.DecompressionBombError` derives directly from `Exception` and escapes, crashing the integrity pass mid-walk. |
| 28 | medium | `dataset_importer.py:1401` | A multi-file `manual_download_local_file_names` arrives as an `omegaconf.ListConfig`, which is not a `list` subclass, so `datasets`' `map_nested` stringifies it to `"['a.zip', …]"` and treats it as one relative path. |
| 19 | medium | `ecotaxa_client.py:310` | Every job is `submit`ted up front before `as_completed` sees a result — ~3.2 GiB of `Future` state for `tara_pacific_decknet`'s 1.58 M objects. A bounded submission window behaves identically. |
| 23 | medium | `gen_planktonzilla_only_plankton.py:191` | Second split sizes from `n` (pre-split) instead of `len(val_test_split)`; the `except ValueError` fallback retries with the identical `test_size` and re-raises, under a misleading "falling back to unstratified" warning. |
| 15 | medium | `scripts/mirror_planktonset1.py:153` | FTP `LIST`-supplied directory and file names become local write paths with no `..`/absolute-path check. `Path('/out') / '/etc/x'` is `/etc/x`. Low likelihood (NOAA over FTP), trivial fix, worth doing. |
| 17, 33, 36 | medium | `dataset.py:93-110` | `compute_mean_and_std_dev` decides its return arity from `image_array` *after* the loop — i.e. from whatever the last row happened to be — and never converts to RGB. A mixed RGB/L imagefolder publishes a one-element `Normalize` constant to the dataset card. Empty input raises `NameError`, not a clear error. |
| 10 | medium | `configs/training_arguments/experimental.yaml:3` | Passes `overwrite_output_dir`, removed in `transformers` 5.x — `pz_train training_arguments=experimental` raises after the dataset and weights have been fetched. |
| 9 | medium | `configs/model/beit-base.yaml:8` | timm weight id without the `timm/` namespace prefix. |
| 32 | medium | `frepj_validate.py:340` | VAL-02 "Overlap & Fidelity" passes vacuously for any class missing from the taxonomy CSV. |
| 31 | medium | `ecotaxa_client.py:168` | A missing/zero `total_ids` truncates a manifest to one window *and* skips the short-manifest guard. |
| 34 | medium | `tests/test_make_planktonzilla_hydra.py:1047` | `test_a_warning_alone_does_not_stop_the_run` asserts nothing about the warning it stages. |

---

## Tier 3 — documentation and contract drift (all confirmed by hand)

| # | Location | Drift |
| --- | --- | --- |
| 43 | `README.md:167` | The documented import commands never import. `configs/import_dataset.yaml:17` sets `action: show`, so `uv run pz_import_dataset dataset_import=isiisnet` only prints info — `action=import` is missing from every example. |
| 42 | `README.md:434` | `pz_train … model=resnet50` and `model=efficientnet`; neither config exists (`configs/model/` has `resnet18.yaml` and no efficientnet). |
| 41 | `README.md:541` | ZooLake's licence is `cc-by-4.0` in the README table and `cc0-1.0` in `configs/dataset_import/zoolake.yaml:12` + `constants.py`. |
| 39 | `README.md:27` | Opener says "two more are in the build registry"; there are six, and the README says so itself 490 lines later. |
| 38 | `README.md:341` | Claims `custom_metadata` is `{}` for every source but FREPJ; the four Tara Pacific sources populate it too. |
| 40 | `README.md:696` | Claims all tests mock the network. `tests/test_datasets.py` reaches the HF Hub — I watched it issue live requests to `huggingface.co` and `datasets-server.huggingface.co`. |
| 37 | `README.md:612` | "five different sets of terms"; the table directly below lists six. |
| 50 | `ecotaxa_client.py:285` | `download_vault_images` annotated `-> tuple[int, list[str]]`, returns a 3-tuple. |
| 45 | `configs/hydra/launcher/local_submitit.yaml:12` | Interpolates `${train_params}`, defined nowhere. |
| 44 | `.github/workflows/ci.yml:74` | The `--ignore` pair drops the one train-side test that was not already self-skipping. Both excluded suites pass in ~7 min (I ran them) — they are network-bound, not broken. |
| 78 | `generate_planktonzilla.py:20` | Docstring says three sources are omitted from `cfg.datasets`; all three are active entries. |

Plus five confirmed low-severity defects in `templates/sankey_flow.html` and the CLIP-export path
(#46-48, #52-53), and #35 (`generate_planktonzilla.py:659` stringifies metadata `NaN` to the literal
`'nan'` rather than null — related to KI-1 but distinct).

---

## Contested findings (one of two refuters rejected them)

Findings #57-78 each survived one refuter and were rejected by the other. They are listed in the appendix
data rather than argued here — treat them as leads, not conclusions. The two worth a look regardless are
**#72** (custom losses ignore `num_items_in_batch`, so gradient accumulation scales gradients by the accum
factor) and **#64** (`check_image_file_integrity` deletes vignettes that `imagefolder_is_complete()`
counts, so one undecodable image can make a source permanently "incomplete").

---

## Suggested order of work

1. **Stop the bleeding on the taxonomy CSV** — 1.1 and its sibling 1.2. Both destroy data on a command
   the project documents as safe/idempotent. Neither has a test.
2. **1.3** — every incremental build against a versioned artifact is currently blocked.
3. **1.4 / 1.5 / 1.8** — three of the seven selectable training configurations are silently wrong. A
   permutation-invariance property test over `loss.py` and one `freeze_backbone` assertion cover all of it.
4. **1.6 / 1.7** — the mislabels are already in a published artifact, so this needs the golden-diff gate
   that `KNOWN_ISSUES.md` itself flags as not yet built. Fix the *guard* (1.7) first so nothing new lands.
5. **1.9** — one-line redaction, removes a credential from logs.
6. **1.10 + Tier 3** — mostly mechanical; a "every config group member composes" test prevents recurrence.

Two structural gaps deserve naming, because most of Tier 1 traces back to them:

- **`imagefolder_is_complete()` means "non-empty"**, and two importers violate that premise by
  construction (#29, #24, #7, #25, #26). A per-importer expected-count check would collapse five findings
  into one fix.
- **The golden-diff harness still does not exist.** `KNOWN_ISSUES.md` already says so. 1.1 and 1.6 are
  both cases where a diff against the published reference would have caught silent corruption immediately.

---

## Method notes

- Reviewers were given the `KNOWN_ISSUES.md` / `RESOLVED_ISSUES.md` ledger up front and instructed to drop
  anything already recorded as a deliberate deferral unless they had something new. Three findings carry a
  KI reference where they extend a known entry.
- Each finding faced two independent refuters with distinct lenses (reachability; consequence), each
  defaulting to *refuted* when it could not positively confirm the defect. 49 of 123 died there.
- Every Tier 1 finding was then re-verified by hand against the working tree — code read, and where a
  reproduction was possible, executed. The commands and their outputs are quoted inline above.
- One finding the panel marked CONTESTED (1.5, RAL) is promoted to Tier 1 on the strength of my own
  measurement.
