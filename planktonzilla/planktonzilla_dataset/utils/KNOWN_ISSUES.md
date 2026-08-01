# Known Issues — `dataset_generation`

These are **deliberately deferred, behavior-changing** improvements identified during the
v1.0 tech-debt cleanup. They are **not implemented**, because applying them could change
what the generation scripts produce, and the planktonzilla datasets/models are already
published and **frozen** on HuggingFace Hub ([project-oceania](https://huggingface.co/project-oceania)).

The v1.0 cleanup was strictly **output-preserving**: it added observability (a `logger.warning`
before each silent fallback — see Phase 4) without changing control flow, caught exception
types, return values, or which records get populated. Everything below goes one step further
and *would* alter behavior, so it is recorded here instead.

**How to apply these later (v2 / `HARDEN-01`, `HARDEN-02`):** implement each behind a flag that
**defaults to today's behavior**, and accept it only after a **golden-output diff against the
published HuggingFace reference**. Never regenerate or re-publish the frozen artifacts from a
changed code path without that diff.

> Observability note: every site below already emits a `logger.warning`/`logger.debug` as of
> Phase 4, so these failures are no longer silent — only their *handling* is unchanged.

---

## KI-1 — Narrow the broad `except Exception` clauses to specific types

**Where:** `extract_cox.py` (esearch/efetch), `gen_planktonzilla.py` (`retrieve_whoi_metadata`,
`retrieve_ecotaxa_metadata`, `_flatten_metadata` JSON parse, `WHOIRedefiner` future handling,
`clean_corrupt_examples_optimized`), `extract_taxon_ids.py` (`search_wikidata_taxon`,
`_extract_property`, `fetch_external_ids`).

**Today:** broad `except Exception` swallows transient network/JSON/IO failures and falls back
to NaN/empty/`None`, conflating "the API failed" with "there is genuinely no data."

**Proposed:** catch only the expected types (e.g. `requests.RequestException`,
`orjson.JSONDecodeError`/`json.JSONDecodeError`, `KeyError`, `IndexError`, `TypeError`,
PIL/decode errors).

**Frozen-output risk: HIGH.** Narrowing lets previously-swallowed errors propagate and abort
runs that previously completed with NaN/empty rows — or, conversely, changes which rows the
corrupt-image filter drops — altering metadata columns and **row counts**. → `HARDEN-01`.

## KI-2 — Add retry/backoff + socket timeouts to the external fetchers

**Where:** `extract_cox.py` NCBI Entrez `esearch`/`efetch` (currently no retry, no timeout — a
failed batch is silently dropped or truncated); `gen_planktonzilla.py` WHOI/EcoTaxa GETs.

**Today:** a transient failure means those sequences/records are simply missing from the output.

**Proposed:** bounded retry with exponential backoff on 429/5xx, plus explicit socket timeouts.

**Frozen-output risk: HIGH.** Retrying can **recover records the original run dropped**, changing
the produced FASTA / `summary.csv` / metadata columns and row counts versus the frozen
reference. → `HARDEN-01`.

## KI-3 — Bound the Wikidata 429 recursion and tighten taxon disambiguation

**Where:** `extract_taxon_ids.py` `search_wikidata_taxon`.

**Today:** (a) on HTTP 429 it `time.sleep(2)` and **recurses unbounded**; (b) it accepts the
first result whose description loosely contains a biological keyword (substrings like `order`,
`organism`), which can match the wrong entity.

**Proposed:** convert the 429 path to a bounded backoff loop; require exact / word-boundary
keyword matching for disambiguation.

**Frozen-output risk: HIGH.** Tighter disambiguation changes **which Qcodes resolve**, hence the
resolved `aphia_ID` / `NCBI_ID` / `BOLD_ID` values. → `HARDEN-01`.

## KI-4 — Honor `--noexp` and revisit `skip_empty` in `process_csv`

**Where:** `extract_cox.py` `process_csv` / `get_cox_sequences`.

**Today:** `process_csv` calls `get_cox_sequences(..., expand_to_children=True)` hard-coded — the
`--noexp` CLI flag is not threaded into the batch path. The `skip_empty=False` branch also
changes which "no-ID" rows are written to `summary.csv`.

**Proposed:** thread `--noexp` through to the batch path; make the `skip_empty` semantics
explicit and consistent.

**Frozen-output risk: MEDIUM.** The frozen artifacts were produced on the success path
*without* `--noexp` and with default `skip_empty`, so a clean re-run with today's invocation is
often inert — but the change is genuinely behavior-altering for other invocations. → `HARDEN-01`.

## KI-5 — Don't cache a `None` that came from a transport error

**Where:** `extract_taxon_ids.py` `search_wikidata_taxon` + `_SEARCH_CACHE`.

**Today:** a `None` produced by a network/transport failure is cached identically to a genuine
"no match," so a later retry within the same run cannot recover it.

**Proposed:** cache only genuine no-match results; leave transport failures uncached (retryable).

**Frozen-output risk: MEDIUM.** Changes which taxa eventually resolve. → `HARDEN-01`.

## KI-6 — Distinguish "API failed" from "no ID" in `fetch_external_ids`

**Where:** `extract_taxon_ids.py` `fetch_external_ids` batch loop.

**Today:** when a batch ultimately fails, every Qcode in it is filled with `None` IDs —
indistinguishable from Qcodes that legitimately have no external ID.

**Proposed:** add a status indicator (e.g. a column) so downstream consumers can retry only the
true failures.

**Frozen-output risk: MEDIUM.** Adds/changes columns and alters the `None`-fill fallback. →
`HARDEN-01`.

## KI-7 — Reconcile null / separator / pandas-vs-polars CSV handling

**Where:** `extract_taxon_ids.py` (the two output CSVs, empty-string vs `null` asymmetry; polars)
vs `update_planktonzilla.py` (`build_sync_dict`, pandas) and the `";"` vs `","` separators.

**Today:** the two `extract_taxon_ids` output CSVs differ in empty-string vs null representation,
and the suite mixes pandas and polars with different separators, so null/dtype representation is
not uniform.

**Proposed:** unify on one CSV engine + separator convention and a single null representation.

**Frozen-output risk: MEDIUM.** Shifts null/dtype representation in the produced CSVs. →
`HARDEN-01`.

**Partially resolved (taxonomy-CSV half).** The `generate` vs `update` divergence is gone:
both now read `planktonzilla_taxonomy.csv` through the single polars reader
`generate_planktonzilla.build_taxonomy_lookup`, and `update_planktonzilla.build_sync_dict`
is a thin projection of it. The pandas implementation was deleted, so the two paths **cannot**
drift again — there is only one.

This was provably zero-drift: over the shipped CSV the two implementations agree on all 16
synced columns × 1485 rows, in value **and** Python type.
`tests/test_taxonomy_lookup_equivalence.py` pins that against a verbatim copy of the deleted
pandas reader, and keeps pinning it forward — the two would *not* agree on every possible CSV
(a numeric-only `ecotaxa_ID` column with blanks makes polars infer `Int64` → `328` where
pandas infers `float64` → `"328.0"`, exactly the KI-12 shape), so a future CSV edit that
enters that regime turns the test red instead of silently rewriting ID values.

One deliberate behavior change came with it: a duplicate `(Dataset, Raw_Labels)` key used to
hard-raise in the pandas path and be silently last-wins in the polars path. It now **warns and
keeps the last row** — the generation path's long-standing behavior, made visible. The shipped
CSV has no duplicates.

**Still open:** the `extract_taxon_ids.py` output CSVs (empty-string vs `null` asymmetry) and
the `";"` vs `","` separator convention. Those are untouched. → `HARDEN-01`.

## KI-16 — The split probe in the build path reads the repository root, not the imagefolder

**Where:** `generate_planktonzilla.py`, `import_and_redefine_source` — the `split_path = root /
alias` probe.

**Today:** `root` there is the module-level pyrootutils **repository** root, not the source's
`imagefolder_dir`. `DatasetImporter.import_dataset` runs the identical probe correctly rooted at
its own imagefolder, so this is a copy-paste slip. No `train/`, `validation/`, `val/` or `test/`
directory exists at the repository root, so `data_files` is always empty and control always
reaches the single-split fallback. Consequences:

- `n_splits` is always `1`, so `original_path` is always the last **two** path chunks
  (`/<class>/<file>`) rather than three. Pinned by
  `tests/test_gen_planktonzilla_lensless_e2e.py`.
- A stray `train/` directory at the repository root would hijack `data_files` for **every**
  source at once.
- The depth-2 fallback glob cannot read the split layouts `LenslessDatasetImporter` (renames
  `TRAIN_IMAGE`/`TEST_IMAGE` → `train`/`test`) and `ZooLakeDatasetImporter` produce.

**Do NOT fix.** `original_path` values are frozen in the published dataset, and a per-source
refresh places rebuilt rows *beside* rows carried over from that published dataset — correcting
the probe would make the two disagree within one artifact. Carried verbatim into
`import_and_redefine_source` under a `# KNOWN ISSUE:` comment. → gate any correction on a golden
diff (`HARDEN-01`).

## KI-17 — Two `dataset_import` configs named classes that do not exist

**Where:** `configs/dataset_import/medplanktonset.yaml`, `configs/dataset_import/sykezooscan2024.yaml`.

**Was:** `medplanktonset.yaml` targeted `MedPlanktonSetDatasetImporter`, which had never been
written, and `sykezooscan2024.yaml` targeted `SYKEZooScan2024` instead of
`SYKEZooScan2024DatasetImporter`. `medplanktonset` is the **5th of the 12 active entries** in
the `datasets` registry, so a full build died partway through — after four sources had already
been downloaded and processed.

**Resolved.** The `sykezooscan2024` target is corrected, and `MedPlanktonSetDatasetImporter` is
implemented. `tests/test_dataset_import_configs.py` now instantiates every config in the group,
so a `_target_` naming a missing class fails in CI instead of mid-build.

**Caveat on the new importer:** the internal layout of MedPlanktonSet's `IFCB_images.zip` could
not be verified — Zenodo was unreachable from the environment it was written in. Rather than
hard-code a guessed path from the extraction root to the class folders (as its sibling importers
do), it locates them with `find_class_root`, which scans for the directory whose subdirectories
hold images. That is layout-independent and covered by tests, but **the first real run should
still be checked**: its reported class count should match the 139 `medplanktonset` rows in
`planktonzilla_taxonomy.csv`. A mismatch means the scan picked the wrong level.

## KI-18 — A Hub push that never succeeded reported success

**Where:** `dataset_importer.py` `_push_to_hub`.

**Was:** the retry loop caught every exception per attempt and logged a warning. After the
last attempt it fell through to `update_dataset_metadata()` and returned normally, so a push
that never succeeded looked like a successful one — and the dataset card was refreshed for a
dataset that was never uploaded.

**Resolved.** The last error is retained and re-raised as a `RuntimeError` naming the attempt
count; `update_dataset_metadata()` is not reached on failure. Pinned by
`tests/test_dataset_import_configs.py::test_push_to_hub_raises_after_exhausting_retries`.

## KI-19 — `check_image_file_integrity` crashed on the layouts that need it

**Where:** `dataset_importer.py` `import_dataset`, the `check_image_file_integrity` block.

**Was:** a fixed two-level walk (`os.listdir(imagefolder)` then
`os.listdir(imagefolder / class_dir)`). On a **split** layout (`train/<class>/<image>`), which
`LenslessDatasetImporter` and `ZooLakeDatasetImporter` both produce, the inner entries are class
**directories**. `is_valid_image_file` returns False for a directory (`IsADirectoryError`
subclasses `OSError`, which `IOError` aliases), so the next line called `os.remove` on a
directory and raised uncaught. The adjacent warning was also missing its `f` prefix, so it
logged the literal text `{file}` / `{class_dir}`.

**Resolved.** The walk is now `rglob("*")` filtered to files, so it is layout-independent;
empty class folders left behind are cleaned up afterwards. Both fixed together since they sit
on adjacent lines. Off by default, so no published artifact is affected.

## KI-20 — The manual-download comment conflated three different identifiers

**Where:** `configs/generate_planktonzilla.yaml`, the commented block above `datasets`.

**Was:** the bullet read `jedi_oceans_cpics (import_name: jedi, redefiner: jedi, ...)`. The
leading token was actually the `import_name`, `import_name: jedi` was actually the *redefiner*
key, and the real `name` — which must equal the taxonomy CSV `Dataset` value — is a third
string, `jedioceans` (95 rows). Following the comment produced an entry that could not resolve.

**Resolved.** Replaced with a table naming all three identifiers per row, plus a note that 4 of
the 12 active entries have `name != import_name`. Comment only; the `datasets` table itself is
unchanged.

## KI-21 — "Requires a manual download" was true of one source, not three

**Where:** `configs/generate_planktonzilla.yaml` (comment above `datasets`),
`generate_planktonzilla.py` (module docstring), `configs/dataset_import/{zoolake,
jedi_oceans_cpics,sykezooscan2024}.yaml`.

**Was:** all three omitted sources were documented as needing a hand-downloaded `.zip`
because of "anti-bot protection". The configs say otherwise:

- **`zoolake`** has a direct `download_uris` and **no** manual override. Nothing forces it
  manual — it is simply absent from the `datasets` table. The automatic path appears never
  to have been tried.
- **`jedioceans`** has a direct `download_uris` **and** a
  `manual_download_local_file_names` that shadows it (the manual branch is checked first in
  `_download_and_extract`), so the direct URL is never attempted either.
- **`sykezooscan2024`** is the only genuine case: `download_uris` was the empty string.
  Fairdata serves a generated package rather than a stable URL, so there was nothing to
  fetch.

**Resolved, and verified against the live services on 2026-08-01. None of the three needs
a manual download.**

- **`sykezooscan2024`** — `SYKEZooScan2024DatasetImporter` resolves the archive through
  the Fairdata Download API (`fairdata_pid`). Exercised end to end: resolved, downloaded
  the 79,363,785-byte package, unwrapped it, and produced 20 class folders / 22,753
  images matching the 20 `sykezooscan2024` rows in the taxonomy CSV exactly.
- **`zoolake`** — its `download_uris` serves a 492 MB `application/zip` whose first entry
  is `data/Fig_All_plankton_images.png`. Reachability and archive shape verified; a full
  import was not run.
- **`jedioceans`** — its `download_uris` serves a zip whose first entry is
  `CPICS_Validated/20141001-07.zip`, the nested layout its importer expects. The manual
  override that shadowed this now defaults to null. Same caveat: reachability and shape
  verified, full import not run.

The "anti-bot protection" premise was wrong for all three. What was actually true is that
one source (`sykezooscan2024`) had no direct URL because Fairdata packages on demand.

## KI-22 — `SYKEZooScan2024DatasetImporter` globbed PlanktonSet1's path

**Where:** `dataset_importer.py` `SYKEZooScan2024DatasetImporter._prepare_imagefolder`.

**Was:** it globbed `0127422/2.3/data/FINAL_Plankton_Segments_12082014` — the NOAA
accession path belonging to **PlanktonSet1**, almost certainly copy-pasted. No such path
exists anywhere in the SYKE archive, so the loop iterated **nothing** and produced an
**empty imagefolder without raising**. Undetectable until now, because the source could
not be downloaded at all (KI-21), so the broken path was never reached.

**Today's real layout**, captured from the live download::

    <package>.zip
      SYKE-plankton_ZooScan_2024/readme.md
      SYKE-plankton_ZooScan_2024/SYKE-plankton_ZooScan_2024.zip     <- nested
        SYKE-plankton_ZooScan_2024/images/SYKE-plankton_ZooScan_2024/<class>/*.png
        SYKE-plankton_ZooScan_2024/class_splits/…
        __MACOSX/…

**Resolved.** The importer unwraps the nested archive, then locates the class folders with
`find_class_root` rather than a fixed path, so a re-release that renames a wrapper cannot
break it the same way. Pinned by
`tests/test_dataset_import_configs.py::test_syke_prepare_imagefolder_unwraps_the_nested_archive`
and by a test asserting the archive's 20 class names equal the CSV's 20 labels.

**Frozen-output risk: none.** `sykezooscan2024` is not in the active `datasets` table, and
its 20 CSV rows are 20 of the 1,485 — the published dataset's rows for this source came
from a build that must have used a hand-downloaded archive.

**Related improvement.** A missing hand-downloaded archive used to surface as an error from
inside `extract()` naming neither the file nor its source. `missing_manual_downloads()` /
`manual_download_instructions()` now report it up front — per source at import time, and
for a whole build via `pz_planktonzilla dry_run=true`, which lists every blocking archive
before anything is downloaded.

## KI-23 — The consolidated command could not perform the license migration

**Where:** `make_planktonzilla.py` `main`, against
`update_planktonzilla.add_license_columns`.

**Was:** an integration gap between the two changesets that landed together, invisible in
either one alone. `pz_planktonzilla` requires every part to match
`constants.CONSOLIDATED_COLUMNS`, which now includes `license` / `license_url`. The
**published** planktonzilla-17M predates those columns, so
`assert_consolidated_schema` rejected it as a base — and `add_license_columns` was only
ever called from `pz_update_planktonzilla`, never from the command that replaces it.

The documented migration was therefore broken end to end:

    pz_planktonzilla base=hub sources=[] sync_taxonomy=false   # -> Schema mismatch

and the deprecation warning on `pz_update_planktonzilla` pointed at that same failing
invocation.

**Resolved.** `ensure_license_columns` derives the pair from the `dataset` column when a
base lacks it, applied **before** the schema check and before any `select` —
`add_column` flattens an indices mapping, which on the full dataset would rewrite ~13.6M
rows for nothing. It is derivation, not invention (both values are a pure function of
the source), but it does change the published schema, so it logs a warning pointing at
`push_revision`. Pinned by two tests in `test_make_planktonzilla_splice.py` covering both
the pure-resync fast path and the splice path.

**Consequence worth knowing:** a base holding a source absent from `DATASET_LICENSES`
now fails during derivation rather than being carried over with a warning. That is
deliberate — a license cannot be guessed — but it means the carry-over of unregistered
sources only works on a base that already has the columns.

## KI-24 — `zoolake` and `jedioceans` joined the registry; the licence mix widened

**Where:** `configs/generate_planktonzilla.yaml` `datasets`.

**Change (2026-08-01, maintainer decision).** Both were added as active entries once
KI-21 established that neither needs a hand-downloaded archive. They are **appended, not
inserted**: registry order is the concatenation order of the output, so every existing
source keeps the index it already had.

`sykezooscan2024` followed the same day, once its Fairdata resolver (KI-21) and its
wrong-path `_prepare_imagefolder` (KI-22) were both fixed and verified end to end. The
registry now covers **all 15** sources of the published dataset, so a from-scratch build
reproduces it rather than 12 of its 15 parts.

**Licence consequence, stated plainly.** `jedioceans` is **CC-BY-SA-4.0** and is the only
ShareAlike source. A rebuild therefore mixes:

| terms | sources |
| --- | ---: |
| `cc-by-4.0` | 7 |
| `cc-by-nc-4.0` | 5 |
| `cc-by-sa-4.0` | 1 |
| `mit` | 1 |
| `other` | 1 |

ShareAlike and NonCommercial cannot both be satisfied by a single licence on a combined
work, so the aggregate **cannot** be relicensed as one thing — it could not before
either, but this makes it unambiguous. What makes the mix tractable is that every row
carries its own `license` / `license_url`, so a consumer filters rather than relying on a
dataset-level statement. The published dataset already contained all three of these
sources, so this changes what a *rebuild* produces, not what is published.

**The aggregate IS licensed on the Hub**, in
[`LICENSE.md`](https://huggingface.co/datasets/project-oceania/planktonzilla-17M/blob/main/LICENSE.md),
and that document is the authority — not this repository. It structures the corpus in
three layers: each image keeps its source collection's licence with no aggregate
override; the planktonzilla contributions (harmonised taxonomy, derived metadata,
splits, docs, scripts) are CC BY 4.0; and the compilation itself, including any sui
generis database right, is CC0 1.0. It reaches the same conclusion this entry does —
share-alike and non-commercial are mutually incompatible, so per-source licensing is the
only available structure — and it records that images are redistributed byte-identical,
which is what keeps the repository a *Collection* rather than an Adaptation and stops
CC-BY-SA propagating to the other fourteen collections.

**One discrepancy found and fixed (2026-08-01).** Comparing `DATASET_LICENSES` against
that LICENSE.md, fourteen of fifteen agreed; **`zoolake` did not**. It was recorded as
`cc-by-4.0` here (transcribed from its importer config) while the published notice
states **CC0 1.0 — no attribution required**, verified at the originating EAWAG deposit.
The repository value over-stated the restriction. Corrected in
`configs/dataset_import/zoolake.yaml` and `DATASET_LICENSES` together, since the drift
test compares them.

This mattered because `zoolake` had *just* become an active registry entry (above): a
rebuild would otherwise have stamped the wrong, more restrictive licence on every
zoolake row. The LICENSE.md notes that seven sources differ from earlier statements —
including the paper's Table 8 — always in the over-stating direction, so the repository
is the side that lags. **When the two disagree, the published LICENSE.md wins.**

---

*Recorded 2026-06-17 during the v1.0 `dataset_generation` cleanup (Phase 7, `KNOWN-01`).
See `.planning/REQUIREMENTS.md` `HARDEN-01` / `HARDEN-02` for the deferred v2 work.
KI-16 through KI-24 recorded 2026-08-01 during the `pz_planktonzilla` consolidation.*

---

## Data inconsistencies in `planktonzilla_taxonomy.csv` (KI-8 – KI-13)

KI-1..KI-7 and KI-16..KI-24 above concern **code behavior**. KI-8..KI-13 below concern **data** defects in the
frozen `planktonzilla_taxonomy.csv` itself, found by a two-method audit on **2026-07-13**
(deterministic checks + a 27-agent adversarially-verified multi-lens audit; every finding
below survived independent re-verification, and candidate findings explained by a legitimate
convention were discarded — see *Verified non-issues*). The CSV is **not edited**: the
datasets and models derived from it are published and frozen on HuggingFace Hub, so these are
recorded here and **pinned** by `tests/test_taxonomy_known_issues.py` rather than corrected.
Row numbers are **0-based data rows** (CSV line = row + 2).

## KI-8 — Rank-column contamination: a taxon in a rank slot its suffix contradicts

**Where:** rows 945 (`neomoelleria cornuta`), 153 (`azadinium caudatum`), 1126
(`pseudochattonella farcimen`), 817 (`katablepharis remigera`).

**Today:** a taxon name is placed in a rank column that its own name-suffix contradicts,
disagreeing with the same name's placement in dozens–hundreds of other rows:

- row 945: `bacillariophyceae` (a `-phyceae` **class**, correctly in `Class` in 225 other
  rows) is duplicated into both `Order` **and** `Family`.
- row 153: `dinophyceae` (**class**) appears in `Order` as well as `Class`.
- row 1126: `florenciellales` (an `-ales` **order**) appears in `Family` as well as `Order`.
- row 817: `cryptophyta` (a **phylum**) appears in `Class` (should be `cryptophyceae`).

**Frozen-output risk: data-side.** Correcting the slot changes that row's lineage in the
published table. Document only. → data fix gated on a golden diff (`HARDEN-01`).

## KI-9 — Uppercase value in a normalized column

**Where:** row 671 (Dataset=`global_uvp5`, Raw_Labels=`Eukaryota`).

**Today:** `proposed_label='Eukaryota'` — the *only* value with an uppercase letter across
every normalized column (Kingdom..Species, `proposed_label`, `root_class`, `qualifier`) in all
1,485 rows; the convention is lowercase. (`Raw_Labels` legitimately preserves source casing.)
Should read `eukaryota`.

**Frozen-output risk: data-side.** A case-sensitive consumer keying on `proposed_label` treats
this as a distinct class; changing it alters the label set. Document only.

## KI-10 — Contradictory `plankton` flag for identical fish-egg taxa

**Where:** rows 389/390 (`clupeiformes`, qualifier `egg`) and 645/646 (`engraulidae`,
qualifier `egg`).

**Today:** within each pair the rows are identical in `proposed_label`, `qualifier`, `living`,
`root_class`, and every `*_ID` column, yet one is `plankton=True` and the other `False`. Both
are fish eggs (ichthyoplankton), so no axis justifies the split. *Secondary (semantic, softer):*
for `teleostei`, adult `full_body` rows are `plankton=True` while `larvae` rows are `False` —
backwards, since larvae are the planktonic stage; this is a judgement call, not a strict
same-key contradiction.

**Frozen-output risk: data-side.** Correcting either flag changes the `plankton` column.
Document only.

## KI-11 — `qualifier` values outside the documented enumeration

**Where:** rows 122 (`part_carapace`), 446 & 512 (`part_skin`), 969 (`part_trunk`).

**Today:** the documented qualifier set is {`full_body`, `larvae`, `part`, `egg`, `like`,
`mix`, `part_tail`, `part_tentacle`, `part_head`, `parasite`, `part_leg`, `''`}. These four
rows use `part_carapace` / `part_skin` / `part_trunk`, which follow the `part_*` pattern and
are correctly `root_class=detritus` / `plankton=False` / `living=False`. The data is
internally consistent — the **enumeration is incomplete**.

**Frozen-output risk: none (docs/validator only).** Widening the documented set / any validator
to include these three values is not a data change. Lowest-risk item.

**Resolved 2026-07-13 (260713-n3e).** Investigation confirmed the repo had **no** qualifier
enumeration or validator — the generation pipeline only casts `qualifier` to string
(`generate_planktonzilla.py`), never validates its values. An authoritative vocabulary
`QUALIFIERS` (all 14 non-empty values, including `part_carapace` / `part_skin` / `part_trunk`)
was added to `planktonzilla_dataset/constants.py`; an empty cell means "unqualified". **No CSV
data changed** — this is documentation/validation only. Conformance (every CSV `qualifier` ∈
`QUALIFIERS`) is now pinned by `tests/test_taxonomy_known_issues.py`, which will fail if a
future CSV introduces an unrecognized qualifier without updating the constant.

## KI-12 — Integer IDs serialized as floats

**Where:** `aphia_ID` (1293/1293), `NCBI_ID` (1263/1263), `BOLD_ID` (1262/1262) — *every*
non-empty cell.

**Today:** numeric identifiers are stored as `"12345.0"` (a pandas `int→float` coercion in
NaN-bearing columns), not `"12345"`. `wikidata_ID` is clean (`Qxxxx`); `ecotaxa_ID` is clean
integers / `;`-joined lists. Extends **KI-7** (pandas-vs-polars null/dtype handling).

**Frozen-output risk: HIGH (systematic).** Re-serializing as ints rewrites every ID cell's
string form in the published CSV. Document only; if fixed, gate on a golden diff. → `HARDEN-01`.

## KI-13 — External ID reused across distinct taxa

**Where:** `NCBI_ID` `418941.0` (`discosphaera tubifera` + `rhabdosphaera clavigera`),
`418932.0` (`calciopappus caudatus` + `ophiaster`), `2723146.0` (`gonyaulax verior` +
`sourniaea diacantha`); `wikidata_ID` `Q25364681` (1 genuine collision).

**Today:** these taxid / QID values are stamped on genuinely different taxa (their other ID
columns differ, confirming distinctness). A larger bucket — ~25 NCBI, 5 aphia, 6 wikidata
cases — is **coarse-rank propagation**: a parent's ID reused on descendant/species rows (e.g.
the genus `chaetoceros` taxid on `chaetoceros dadayi`), an ID-**precision** limitation rather
than a hard error; and a few apparent collisions are real taxonomic **synonyms** correctly
sharing one taxid (e.g. `ceratoneis closterium` ≡ `cylindrotheca closterium`). The **forward**
direction is clean — no taxon carries two IDs in any column.

**Frozen-output risk: data-side.** Correcting an ID changes the published `*_ID` columns.
Document only.

---

## Source-license transcription (KI-14 – KI-15)

The `license` / `license_url` columns are transcribed verbatim from the `license:` field of
each `configs/dataset_import/*.yaml` into `constants.DATASET_LICENSES`, and
`tests/test_dataset_licenses.py` fails if the two ever disagree. Faithfulness to the importer
configs is the guarantee; whether each *config* states the right thing is a separate question,
and for two of the fifteen sources it is genuinely open. Both are recorded as stated and
carry a `license_url` pointing at the authoritative source record so a consumer can check
the real terms rather than act on a slug that does not carry them.

## KI-14 — `whoi` is recorded as `mit`, the license of a *code* repository

**Where:** `configs/dataset_import/whoi-plankton.yaml` (`license: "mit"`,
`source_url: https://github.com/hsosik/WHOI-Plankton`); 3,563,595 images, **20.5% of the
corpus** — the second-largest source.

**Today:** MIT is a software license, and the `source_url` it was taken from is a GitHub
repository. The repository's terms need not be the terms of the IFCB imagery hosted at
`ifcb-data.whoi.edu` and fetched by `retrieve_whoi_metadata`. `license_url` therefore points
at the repository rather than at a license deed.

**Risk: downstream-legal.** A fifth of the corpus is currently advertised as MIT — the most
permissive value in the table — on the strength of a code license. Confirm upstream before
anyone relies on it for redistribution. Correcting the slug changes a published column.

## KI-15 — `planktonset1.0` is recorded as `other`, which states nothing

**Where:** `configs/dataset_import/planktonset1.yaml` (`license: other`); 60,736 images, 0.35%.

**Today:** `other` is the HuggingFace placeholder for "not one of the known slugs" and gives a
consumer no terms at all. `license_url` therefore points at the NOAA NCEI DOI for accession
0127422 (`https://doi.org/10.7289/v5d21vjd`, already recorded in the config's citation), which
is the authoritative record for the actual terms.

**Risk: downstream-legal, bounded.** The smallest ambiguity in the table, but it is the one
value a license filter cannot act on: `other` can be neither included nor excluded on merit.

---

## Verified non-issues (checked and dismissed — do not re-open)

The audit tested and *rejected* these as legitimate conventions, not defects:

- `living` ⇔ `root_class == 'living'`: **0 mismatches** (1,276/1,276).
- **0** conflicting source mappings (a `(Dataset, Raw_Labels)` pair never maps two ways);
  **0** duplicate rows.
- Each `proposed_label` has exactly **one** lineage.
- Shared **species epithets** (`socialis`, `caudatum`, …) are normal — the `Species` column
  holds the epithet only, not the binomial.
- `ecotaxa_ID` `;`-multi-values and the coarse external-ID crosswalks are by design.
- `ctenophora`, `tripos`, and `siphonophora` at two ranks are legitimate biological
  **homonyms** (e.g. `siphonophora` the millipede genus vs `siphonophorae` the cnidarian
  order), not rank contamination.

*Recorded 2026-07-13 from the `planktonzilla_taxonomy.csv` consistency audit. Pinned by
`tests/test_taxonomy_known_issues.py`. Fixing any KI-8..KI-13 data item changes frozen output —
gate on a golden-output diff against the published HuggingFace reference (`HARDEN-01` /
`HARDEN-02`).*
