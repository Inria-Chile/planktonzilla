# Known Issues — dataset generation (`planktonzilla/planktonzilla_dataset`)

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

> **Two caveats on that gate, both true as of 2026-08-04 — read before relying on it.**
>
> 1. **The golden-diff harness does not exist.** No test in `tests/` compares a build against
>    the published `project-oceania/planktonzilla-17M`. The `test_taxonomy_known_issues.py` /
>    `test_taxonomy_lookup_equivalence.py` suites *pin* today's values so a change goes red —
>    valuable, but that is not the same as diffing against the published reference. Every
>    `→ HARDEN-01` below is therefore an IOU against a gate nobody has built yet. **Building it
>    is the prerequisite for closing any HIGH-risk item**, not a step inside closing one.
> 2. **`HARDEN-01` / `HARDEN-02` are defined in `.planning/REQUIREMENTS.md`, which is
>    gitignored** (`.gitignore:248`) and therefore absent from a fresh clone. The identifiers
>    are stable enough to cite, but a reader outside the maintainer's working tree cannot
>    resolve them. Treat the risk labels and exit conditions **in this file** as the
>    authoritative statement of what each deferral means.

**Observability note:** every site below already emits a `logger.warning`/`logger.debug` as of
Phase 4, so these failures are no longer silent — only their *handling* is unchanged.

---

## Index

Entries are numbered in the order they were found, not the order they are read: KI-1..7 and
KI-16..25 are **code behavior**, KI-8..13 and KI-31 are **data** defects in the frozen taxonomy CSV,
KI-14..15 are **source-license** questions, and KI-26 is a **data** defect in a source's own
sidecar tables; KI-27 through KI-30 are decision logs like KI-24. Numbers are never reused or renumbered — commits,
code comments and tests cite them.

**This file lists only what is still open.** Nine resolved entries — KI-11, KI-17..KI-23 and
KI-25 — were moved verbatim to [`RESOLVED_ISSUES.md`](RESOLVED_ISSUES.md) on 2026-08-04. A
number missing from the table below is *resolved*, not withdrawn; look for it there.

| # | Status | Frozen-output risk | Subject |
| --- | --- | --- | --- |
| KI-1 | open, deferred | HIGH | broad `except Exception` swallows transport failures |
| KI-2 | open, deferred | HIGH | no retry/backoff or socket timeouts on external fetchers |
| KI-3 | open, deferred | HIGH | unbounded Wikidata 429 recursion; loose taxon disambiguation |
| KI-4 | open, deferred | MEDIUM | `--noexp` not threaded into the batch path |
| KI-5 | open, deferred | MEDIUM | a transport-error `None` is cached as a genuine no-match |
| KI-6 | open, deferred | MEDIUM | "API failed" indistinguishable from "no ID" |
| KI-7 | **partly resolved** | MEDIUM | null/separator/engine handling; taxonomy-CSV half is done |
| KI-8 | open, wontfix | data-side | a taxon in a rank slot its suffix contradicts |
| KI-9 | open, wontfix | data-side | the one uppercase value in a normalized column |
| KI-10 | open, wontfix | data-side | contradictory `plankton` flag on identical fish-egg taxa |
| KI-12 | open, wontfix | HIGH | integer IDs serialized as `"12345.0"` |
| KI-13 | open, wontfix | data-side | one external ID stamped on distinct taxa |
| KI-14 | **open, escalate** | downstream-legal | `whoi` recorded as `mit` — 20.5% of the corpus |
| KI-15 | open, bounded | downstream-legal | `planktonset1.0` recorded as `other` — states nothing |
| KI-16 | open, **do not fix** | HIGH | split probe reads the repo root; splits discarded |
| KI-24 | decision log | MEDIUM (rebuild) | `zoolake`/`jedioceans` joined; the licence mix widened |
| KI-26 | open, source-side | none (17M) / republished (frepj) | FREPJ `Sampling date` is free text; 7.1% not a date — normalized, 1.9% null |
| KI-27 | decision log | MEDIUM (rebuild) | `frepj` joined the registry (16th); sidecar inputs became an importer protocol |
| KI-28 | decision log | MEDIUM (rebuild) | `daplankton` joined the registry (17th); Fairdata-resolved, doubly-nested archive, 44 merged classes |
| KI-29 | decision log | MEDIUM (rebuild) | the four Tara Pacific deposits joined (18th–21st, last); the first sources with **no archive** |
| KI-30 | decision log | none (same rows) | `planktonset1.0` is fetched from a mirror we keep; NCEI's on-demand generator cannot resume or be size-checked |
| KI-31 | open, wontfix | data-side | 20 source labels publish two different taxa across datasets |

Two obligations belong to archived entries but are **still open**, and are restated here so
archiving cannot bury them:

| from | open obligation |
| --- | --- |
| KI-21 / KI-24 | `zoolake` and `jedioceans` are verified for reachability and archive shape only; **no full import has completed** |
| KI-23 | deriving the two licence columns is safe; **re-pushing** the published artifact from that schema is still gated |

*KI-17's obligation is **discharged** (2026-08-28): MedPlanktonSet has now had a real run, and
its imagefolder holds exactly **139** class directories, matching the 139 `medplanktonset` rows
in `planktonzilla_taxonomy.csv`. `find_class_root` picked the right level on the archive layout
that could not be verified when the importer was written.*

**The three that want action, in order:** KI-14 (largest open legal exposure), the missing
golden-diff harness (blocks every HIGH item above), and KI-16's discarded split provenance
(silent, and not fixed by the archived KI-25).

---

## KI-1 — Narrow the broad `except Exception` clauses to specific types

**Where:** `extract_cox.py` (esearch/efetch), `generate_planktonzilla.py` (`retrieve_whoi_metadata`,
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
failed batch is silently dropped or truncated); `generate_planktonzilla.py` WHOI/EcoTaxa GETs.

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
  `tests/test_gen_planktonzilla_lensless_e2e.py`, now over **both** imagefolder layouts.
- **Split provenance is discarded at build time.** The consolidated dataset records no trace
  of which upstream split an image came from, so a lensless `test/` image is indistinguishable
  from a `train/` one. Anyone splitting planktonzilla-17M for evaluation cannot honour the
  upstream boundary, and may train on images a source reserved for testing. This is silent —
  nothing in the schema signals the information was dropped.
- A stray `train/` directory at the repository root would hijack `data_files` for **every**
  source at once. The repo already contains `tests/`; the trigger name is `test/`, one
  character away.
- ~~The depth-2 fallback glob cannot read the split layouts…~~ — that consequence was real but
  understated, and it is **fixed**. See **KI-25** in [`RESOLVED_ISSUES.md`](RESOLVED_ISSUES.md), a separate defect in the fallback,
  not in this probe.

**Do NOT fix.** `original_path` values are frozen in the published dataset. Carried verbatim
into `import_and_redefine_source` under a `# KNOWN ISSUE:` comment. → gate any correction on a
golden diff (`HARDEN-01`).

**Correction to the original freeze rationale (2026-08-04).** This entry used to argue that a
per-source refresh places rebuilt rows "beside rows carried over from that published dataset,
so the two would disagree **within one artifact**", which reads as one image acquiring two
identities. That cannot happen, and `IDENTITY_COLS` is a misleading name for what these
columns do: nothing joins, dedups or matches on them — the constant is referenced only by its
own definition and by `CONSOLIDATED_COLUMNS`. Splicing is **whole-source**, keyed on the
`dataset` column alone (`make_planktonzilla.py`, `base_indices[name]`), so a rebuilt source
replaces *all* of its rows and a carried source keeps *all* of its rows. `original_path` is
per-source relative provenance, and its only two readers take the last chunk — the filename —
so path depth is inert to them.

The real breakage is one level up, and it still justifies the freeze:

- A from-scratch build would emit the corrected 3-chunk paths, while an incremental run that
  carries a source over from base would emit 2-chunk. That violates the invariant stated in
  `make_planktonzilla.py` — *"an incremental run is row-for-row identical to a from-scratch
  one"* — which `tests/test_make_planktonzilla_splice.py` asserts.
- One artifact would carry one source's paths in a different shape from the other fourteen.

**Frozen-output risk: HIGH.** Changes a published column for every row of any rebuilt source.
→ `HARDEN-01`.

## KI-24 — `zoolake` and `jedioceans` joined the registry; the licence mix widened

**Where:** `configs/generate_planktonzilla.yaml` `datasets`.

**Change (2026-08-01, maintainer decision).** Both were added as active entries once
KI-21 (archived) established that neither needs a hand-downloaded archive. They are **appended, not
inserted**: registry order is the concatenation order of the output, so every existing
source keeps the index it already had.

`sykezooscan2024` followed the same day, once its Fairdata resolver (KI-21) and its
wrong-path `_prepare_imagefolder` (KI-22) — both archived — were fixed and verified end to end. The
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

**Frozen-output risk: MEDIUM, rebuild-only.** Nothing published changes; the published dataset
already contained all three sources. But a from-scratch build now emits three sources it did
not before, and the corrected `zoolake` slug changes the `license` value every zoolake row
would be stamped with on a rebuild. Registry order was preserved by appending, so no existing
source's index moved.

**Buildability caveat (2026-08-04).** "The registry now covers all 15 sources … so a
from-scratch build reproduces it" was true of the *registry* and false of the *build path*:
two of the newly active entries could not be built at all. See **KI-25**
(in [`RESOLVED_ISSUES.md`](RESOLVED_ISSUES.md)), now fixed. The claim
holds as of this date, with the caveat that `zoolake` and `jedioceans` have still only been
verified for reachability and archive shape, not by a completed import.

---

## KI-26 — FREPJ `Sampling date` is hand-typed free text; 7.1% of rows are not a date

**Where:** the upstream sidecar tables `Table_S3.csv` (40x) / `Table_S4.csv` (100x), column
`Sampling date` — md5-pinned, fetched by `FREPJDatasetImporter.ensure_sidecars` into the
gitignored `<data_dir>/frepj_tables/` (the crosswalk CLI's `DEFAULT_TABLES_DIR` on a default
run) — consumed by `FrepjRedefiner` through `frepj_tables.parse_sampling_date`.

**Found 2026-08-25** while assessing the v1.2 lifecycle. The `project-oceania/planktonzilla-frepj`
published on 2026-07-11 carried the column verbatim as `date` and left the consolidated
`timestamp` null for every FREPJ row. 6,273 of its 88,686 rows (7.07%) held something other
than `YYYY.MM.DD` — `20200917`, `2022.06,10`, `230815inba_funato`, `2021.11.011`, `2020.08.dd`,
`akanko1`, `tsuruoka_100`, … — copied straight from the upstream tables (5,906 rows in
Table_S3, 367 in Table_S4). The VAL-02 gate tested the column for non-null only, so it passed.

**What the build does now (maintainer decision, 2026-08-25).** The date lives in `timestamp`
only, as ISO `YYYY-MM-DD`; no raw copy is kept (the tables are pinned and refetchable), and
the magnification and raw site token moved into the `custom_metadata` JSON object.
`parse_sampling_date` applies fixed rules and **never guesses**:

| family | rows | rule | result |
| --- | ---: | --- | --- |
| `YYYY.MM.DD` (incl. `2019.11.6`) | 82,427 | the intended format | date |
| `YYYYMMDD` | 3,704 | dots omitted | date |
| `YYMMDD<site>` (`230815inba_funato`) | 430 | six-digit 20YY prefix | date |
| `YYYY.MM,DD` | 65 | a comma for the second dot | date |
| embedded run (`biwako_20211122(462)`, `biwa230213_100`) | 94 | first 8- or 6-digit run | date |
| `YYYY.MM.DDD` (`2021.11.011`) | 611 | drop one digit → candidates; kept only if Table_S1 records exactly **one** of them for the site | 259 dated, 352 null |
| month-only (`2020.08.dd`, `2016.11.`, `2311asahiyama_dai`) | 464 | — | null |
| bare site token (`akanko1`, `tsuruoka_100`, `akanko_557,558`) | 891 | — | null |

Net: **86,979 rows (98.1%) dated, 1,707 null.** VAL-02 now fails on any non-ISO `timestamp`
(`Timestamp Shape`) and on coverage below the 98% floor (`Timestamp Coverage`,
`--timestamp-floor`), so a regression in either direction blocks publish.

**Frozen-output risk: none for `planktonzilla-17M`** — FREPJ is in the registry since
2026-08-25 (KI-27) and enters the published artifact only with the v1.2 push. The
intermediate `planktonzilla-frepj` repo was **republished** from this build on 2026-08-25
with the Phase-19 helper (`v1.0.0-frepj` tags the old revision, `v1.2.0-frepj` the new),
which changed its schema: `date` / `magnification` / `site` became `timestamp` +
`custom_metadata`, and the `license` / `license_url` columns it predated were added.

**Upstream.** Worth reporting to the FREPJ authors (the 352 Lake Biwa rows are unrecoverable
without them); not blocking.

---

## KI-27 — `frepj` joined the registry (sixteenth); sidecar inputs became an importer protocol

**Where:** `configs/generate_planktonzilla.yaml` `datasets`; `DatasetImporter.sidecar_targets` /
`missing_sidecars` / `ensure_sidecars`; `RedefineDataset.attach_sidecars`;
`make_planktonzilla.ensure_source_sidecars` and the `sidecars:<source>` pre-flight check.

**Change (2026-08-25, maintainer decision).** `{name: frepj, import_name: frepj, cleanup: false,
redefiner: frepj}` is **appended, not inserted** — registry order is the concatenation order of
the output, so the fifteen published sources keep the index they already had. It is
`cc-by-4.0`, so the licence mix's set of terms is unchanged.

FREPJ is the first source whose *redefine* step needs inputs outside its archive on every
run: three md5-pinned geodata tables (Table_S1/S3/S4, 8.6 MB, figshare article 26891563) and
the committed site crosswalk. Rather than special-case the source name in the pipeline, the
importer base class gained a three-method **sidecar protocol** (declare / check / obtain) and
the redefiner base class a receiving hook; `make_planktonzilla.py` calls them for every source
and knows no source by name. Consequences, stated plainly:

- A from-scratch `pz_planktonzilla` needs no manual step: the FREPJ importer fetches the
  tables into `<data_dir>/frepj_tables` — with its own download config and the project
  User-Agent — **before the first import**, and hands them to `FrepjRedefiner` (which never
  downloads and now reads them lazily, so constructing every redefiner up front is free).
- A reuse run whose imagefolder exists but whose tables are missing is decided in seconds,
  up front, not hours in at the sixteenth source; `check_downloads=needed` probes the tables
  and counts such a source as one the run fetches. `dry_run=true` reports `sidecars:frepj`.
- A verified table is never re-fetched, `refresh=redownload` included — delete the file to
  force one. A committed file that is gone is a blocking failure (no run can repair a checkout).
- `pz_generate_planktonzilla --config-name generate_frepj_only` (the standalone republish path)
  goes through the same seam and behaves identically.

**Frozen-output risk: MEDIUM, rebuild-only.** Nothing published changes; a from-scratch build now
emits a sixteenth source. (It was appended last when this was recorded; `daplankton` was appended
after it — see KI-28 — which left its index unchanged.) The fifteen existing sources' rows are byte-identical
(`sidecar_targets() == []`, `ensure_sidecars() == {}` — pinned by
`tests/test_dataset_import_configs.py::test_sources_without_sidecars_are_unchanged`; `attach_sidecars({})`
is a no-op — pinned by `tests/test_frepj_redefiner.py::test_attach_sidecars_is_a_no_op_on_the_other_redefiners`).

**Known imprecision (accepted).** On a `check_downloads=needed` run with the imagefolder built and
only the tables missing, `download:frepj` probes all five targets, archive included, so a dead
archive host blocks a run that would not have fetched it, and the 963 MB inflates the free-space
estimate (a non-blocking WARN at worst). `global_uvp5` has the same shape on `refresh=rebuild`.

**Three more consequences, accepted and stated here (review of 2026-08-25):**

- An explicit `sources=[…the fifteen…]` list is now a *strict subset* of the registry, so the
  partial-overwrite guard applies to it: over an existing `output_dir` with `base=null` it is
  refused unless `allow_partial_overwrite=true`. `sources=all` (the default) is unaffected.
- The deprecated `pz_generate_planktonzilla` obtains a source's sidecars at that source's turn
  (through the shared seam), not up front — for the sixteenth entry, hours in on a full build.
  `pz_planktonzilla` is where fail-in-seconds lives; the deprecated command is left as is.
- `pz_planktonzilla` verifies the sidecars twice per source — up front in `ensure_source_sidecars`
  and again in the seam, which also serves the deprecated command. Both calls are idempotent
  (a streamed md5 per table, no download when the pin holds), so the second costs milliseconds.

---

## KI-28 — `daplankton` joined the registry (seventeenth); a doubly-nested archive and a five-root class merge

**Where:** `configs/generate_planktonzilla.yaml` `datasets`; `configs/dataset_import/daplankton.yaml`;
`dataset_import.daplankton_layout` / `daplankton_importer`;
`dataset_import.dataset_importer.FairdataPackagedDatasetImporter`; the 44 `daplankton` rows appended
to `planktonzilla_taxonomy.csv`.

**Change (2026-08-27, maintainer decision).** `{name: daplankton, import_name: daplankton,
cleanup: false, redefiner: none}` is **appended, not inserted** — registry order is the
concatenation order of the output, so the sixteen entries above keep the index they already had.
It is `cc-by-4.0`, so the licence mix's set of terms is unchanged; the slug was read from three
independent places (the Metax record's `access_rights.license`, the Etsin landing page, and the
`readme.md` bundled inside the archive).

DAPlankton is the second Fairdata-published source, so the Download API flow that had lived inside
`SYKEZooScan2024DatasetImporter` — request a package, poll, authorize a **single-use** URL, fetch it
exactly once — was extracted unchanged into `FairdataPackagedDatasetImporter` and is now shared.
Copying it was the alternative; its whole reason for existing is a failure mode (a second GET on a
consumed token returns 401) that is invisible until it happens.

Two properties of the source drive the importer, both established by downloading the real
2,739,982,748-byte package on 2026-08-27, matching its declared sha256, and enumerating all 112,040
entries:

- **The archive is doubly nested.** The package holds exactly one member,
  `DAPlankton/DAPlankton.zip`, which is itself the real archive, so a single unwrap yields a zip
  rather than class folders. The importer unwraps twice and *locates* the subset roots
  (`find_archive_root`) instead of hard-coding a path. Consequence: a from-scratch import wants
  roughly **9 GB** free under `data_dir` (package + unwrapped archive + merged imagefolder), and
  `cleanup: false` keeps the first two.
- **The label space is five-fold redundant.** Images are laid out
  `<subset>/<instrument>/<class>/` over two subsets (`DAPlankton_lab`, 15 cultured classes;
  `DAPlankton_sea`, 31 Baltic field classes) and three instruments (IFCB, CytoSense, FlowCam —
  which imaged the cultures only, hence five roots, not six). Subset and instrument are per-image
  provenance, not five label spaces, so the five roots are merged into **one class dir per taxon**,
  the policy KI-27's FREPJ-Z already applies to its two magnifications. That is 46 class slots but
  **44 distinct taxa**: `Aphanizomenon_flosaquae` and `Pseudopedinella_sp` are imaged in both
  subsets. Provenance survives as a filename prefix (`lab_cs_`, `sea_ifcb_`, …) readable in
  `original_path`, and the prefix is load-bearing — per-folder counters restart, so basenames
  collide five ways without it.

**Taxonomy.** 44 rows appended at EOF, after the frepj block, leaving the sha256-frozen first 1,486
lines untouched. 31 reuse `syke_ifcb_2022` rows verbatim (DAPlankton_SEA follows that dataset's
label scheme and its 31 classes are a strict subset of those 50); 4 more reuse an existing row; 9
were resolved against WoRMS, NCBI, Wikidata and BOLD, with anything an authority could not confirm
left **blank rather than guessed**. Reuse is not tidiness:
`test_forward_id_mapping_is_clean` fails if one `proposed_label` carries two different values in an
ID column, so an independent re-lookup differing by a digit would break the suite.

**Frozen-output risk: MEDIUM, rebuild-only.** Nothing published changes — `daplankton` is absent
from `samples.json` and listed in `_RECORDED_BUT_NOT_YET_PUBLISHED` alongside `frepj` — but a
from-scratch build now emits a seventeenth source, appended last.

**Known imprecision (accepted).** The importer's completeness check counts the images *present in
the imagefolder* against `N_IMAGES` and warns on a mismatch. It is a diagnostic only: nothing
raises, and the taxonomy join does not read the count. A deliberately partial or manually staged
import will therefore emit one WARNING per run.
## KI-29 — the four Tara Pacific deposits joined the registry (18th–21st, last); the first sources with **no archive**

**Where:** `configs/generate_planktonzilla.yaml` `datasets`; `configs/dataset_import/tara_pacific_*.yaml`;
`planktonzilla/dataset_import/{tara_pacific_layout,ecotaxa_client,tara_pacific_importer}.py`;
`generate_planktonzilla.TaraPacificRedefiner`; the 600 `tara_pacific_*` rows of the taxonomy CSV.

**Change (2026-08-26, answering issue #10).** The four SEANOE deposits of Mériguet et al. 2025
(`essd-17-2761-2025`) are **appended, not inserted**, after `frepj` — registry order is the
concatenation order of the output. All four are `cc-by-4.0`, so the licence mix's set of terms is
unchanged. Together they add ~2.35 M objects, `tara_pacific_decknet` alone ~1.58 M.

They are the first sources with **nothing to download**. Each deposit publishes an EcoTaxa **TSV
export** — per-object metadata and morphological features, no vignettes (verified by opening every
archive) — and names the public EcoTaxa projects that hold the images. EcoTaxa's archive export
needs an account (`POST /api/object_set/export` → `403 Not authenticated`), so the importer walks
the public read API: a per-object manifest, then one vignette per object from `/vault`. The
manifests are declared through the KI-27 **sidecar protocol**, so the pipeline needed no new seam.

Consequences, stated plainly:

- `download_uris` is empty for all four and `_download_and_extract` is a no-op that says so. There
  is no `.zip` to hand-download if EcoTaxa is unreachable; `dry_run=true` reports `sidecars:` and
  `check_downloads` probes the seven project endpoints.
- `redefiner: tara_pacific` reads the same manifests **offline**. Using `redefiner: ecotaxa` here
  would have issued one `GET /api/object/{objid}` per image — 2.35 million requests to a public
  service — to re-learn what the manifest already states.
- Class folders are named from the committed `tara_pacific_classes.tsv` map keyed by the EcoTaxa
  taxon id, **not** by the live display name: EcoTaxa renames taxa in place (the 2024 exports and
  the live API already disagree on 12+ labels per source for the same taxa), and a renamed folder
  would silently repoint every `Raw_Labels` join key. A rename is reported; a taxon new to EcoTaxa
  is skipped, because it has no taxonomy row.
- The vignette fetch is resumable and refuses to finish while more than
  `ecotaxa_max_missing_images` (default 0) are missing.

**Upstream defect, recorded not worked around.** The DeckNet deposit's `100% > 501 pixels` archive
(`.../00915/102697/data/114288.zip`) serves its full advertised 281,669,134 bytes and is still
unreadable: its end-of-central-directory offset overshoots the file by exactly 4,000,000 bytes
(`unzip`: "missing 4000000 bytes in zipfile"), and the entries straddling the gap cannot be read.
Confirmed on two independent downloads, 2026-08-26. Nothing here reads it.

**Inherited data issue, propagated knowingly.** Three of the 600 taxonomy rows take a lineage that
contradicts EcoTaxa's tree, because rules 1–2 of the curation engine reuse the lineage the master
CSV already records for a `proposed_label` (the table's own one-label-one-lineage invariant).
`Odontella sp.` is a repair — EcoTaxa hangs it under the *springtail* genus. `Ctenophora<Animalia`
and `part<Ctenophora` are **not**: the CSV has mapped `ctenophora` to the DIATOM genus (aphia
163921) since long before this milestone, across nine rows of six zooplankton-imager sources where
the comb jelly is the only plausible reading. These two rows follow the table rather than mint a
second `ctenophora` lineage; correcting the homonym is a separate change to all eleven rows, gated
on a golden-output diff like every other data item here. All three are enumerated in
[`TARA_PACIFIC_TAXONOMY_RECONCILIATION.md`](TARA_PACIFIC_TAXONOMY_RECONCILIATION.md) and asserted
to be the *only* three by `tests/test_tara_pacific_taxonomy.py`.

**Frozen-output risk: MEDIUM, rebuild-only.** Nothing published changes; a from-scratch build now
emits twenty sources, the four appended last. The first 1,715 lines of the taxonomy CSV are still
byte-frozen, and the fifteen archive-only sources still declare no sidecars.

---

*Recorded 2026-06-17 during the v1.0 dataset-generation cleanup (Phase 7, `KNOWN-01`).
`HARDEN-01` / `HARDEN-02` are defined in `.planning/REQUIREMENTS.md`, which is **gitignored**
and absent from a clone — see the caveat at the top of this file.
KI-16 through KI-24 recorded 2026-08-01 during the `pz_planktonzilla` consolidation.
KI-25, and the corrections dated 2026-08-04 throughout, from a full re-audit of every entry
against the code as it stands — the same pass that archived the nine resolved entries to
[`RESOLVED_ISSUES.md`](RESOLVED_ISSUES.md). KI-26 and KI-27 recorded 2026-08-25 during the v1.2
(FREPJ) lifecycle assessment and the registry join that followed it. KI-28 recorded 2026-08-27
with the daplankton registry join (issue #17). KI-29 recorded 2026-08-26 with the Tara Pacific
registry join (issue #10).*

---

## KI-30 — `planktonset1.0` is fetched from a mirror we keep, not from NCEI's on-demand generator

**Where:** `configs/dataset_import/planktonset1.yaml`; `PlanktonSet1DatasetImporter` and
`DatasetImporter._fetch_archive_verified` in `planktonzilla/dataset_import/dataset_importer.py`;
`scripts/mirror_planktonset1.py`. Output rows are **unchanged** — same 60,736 images.

**Change (2026-08-28).** `download_uris` never named a stored file. It names NCEI's Archive
Management System *download* endpoint, which tars accession 0127422 **on demand**. Measured against
the live service 2026-08-27: ~22 s to first byte, then ~0.6 MB/s (≈1 h for the archive), `Range`
**ignored** (a ranged GET answers `200`, not `206`), `Transfer-Encoding: chunked` with **no
`Content-Length`**, and a sustained `503` that was still returning `503` a full day later.

So every run was an hour-long transfer that could not resume and could not be size-checked, and any
interruption lost all of it. Two things hid this: `datasets` 5.0.1 declares but never reads
`resume_download` or the `DownloadConfig` `max_retries` the project was passing (so the promised
resume and five retries did not exist), and `datasets` + fsspec probe HEAD-then-GET, making the
server build the tarball two or three times per attempt.

Three consequences, recorded because they change how this source behaves:

- The archive is fetched by `_fetch_archive_verified`, not `DownloadManager`: one request per
  attempt, real retries, and a **gzip-framing check fed from the chunks as they arrive** — the only
  truncation signal available when the server discloses no size.
- The class tree is **located** (`FINAL_Plankton_Segments*`, then `find_class_root`) instead of
  walked to via `0127422/2.3/data/0-data/FINAL_Plankton_Segments_12082014`. That path pinned the
  accession **version**, and upstream already publishes 1.1, 2.2 and 2.3; `Path.glob` on a missing
  path yields nothing rather than raising, so a 2.4 re-release would have silently produced an empty
  imagefolder — the KI-22 / WHOI failure mode, latent here.
- The data is **121 classes / 60,736 JPEGs / 108,723,866 bytes and immutable since 2015**, so it is
  now mirrored once (`scripts/mirror_planktonset1.py`, over FTP) and imported from that archive via
  `manual_download_local_file_names`. NCEI's uptime leaves the critical path entirely.

FTP, not the HTTPS form of the same mirror: `ncei.noaa.gov/robots.txt` says `Disallow: /data*`,
which covers the HTTPS mirror path, while the FTP tree is the bulk route the accession landing page
itself advertises. Mirroring is ~4.3 h at 4 workers (~4 files/s) and resumable; it is slower than
the archive whenever the archive is up, which is why the generator stays the default route.

*Verified 2026-08-28 end to end: mirrored 60,736 files / 108,723,866 bytes with 0 failures, packaged
to a 91,292,764-byte `.tar.gz`, imported through `pz_import_dataset action=import`, and the built
imagefolder holds 121 classes / 60,736 images / 108,723,866 bytes whose names match the 121
`planktonset1.0` rows of the taxonomy CSV exactly, in both directions. Unrelated to
[KI-15](#ki-15--planktonset10-is-recorded-as-other-which-states-nothing), which concerns this same
source's `license: other`.*

---

## Data inconsistencies in `planktonzilla_taxonomy.csv` (KI-8 – KI-13, KI-31)

KI-1..KI-7, KI-16 and KI-24 above concern **code behavior**. KI-8..KI-13 below concern **data**
defects in the frozen `planktonzilla_taxonomy.csv` itself, found by a two-method audit on
**2026-07-13**
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

## KI-31 — One source label, two different taxa: `Raw_Labels` disagreements across datasets

**Where:** 20 of the 1,622 distinct `Raw_Labels` values, touching 80 rows. Enumerated with their
adjudication in [`LABEL_CONSISTENCY_WAIVERS.json`](LABEL_CONSISTENCY_WAIVERS.json); reported by
`utils/verify_label_consistency.py` and gated by `tests/test_taxonomy_label_consistency.py`.

**Today:** `Raw_Labels` is the key `build_taxonomy_lookup` joins on, so two rows sharing one are
describing the same imagefolder class directory in two different source datasets. For 20 labels
they publish different taxa:

- **5 `lineage_contradiction`** — the two lineages name different taxa at a rank both populate.
  `Creseidae` is published as the family `creseidae` by `global_uvp5` and as `clio pyramidata`
  (family `cliidae`) by three others. `cladocera` — the crustacean group — is published by
  `zoocamnet` as the genus `cladoceramus`, `Class=bivalvia`, `Phylum=mollusca`, carrying an
  `aphia_ID` the authority snapshot resolves to the class *Bivalvia*.
- **13 `rank_inflation`** — one side extends the other with nothing contradicting, so the label is
  published finer than it supports. `Harpacticoida` is the order `harpacticoida` (aphia 1102) in
  five datasets and the genus `euterpina` (aphia 115348) in three. The starkest are the buckets:
  `unknown` carries `Class=thecofilosea` in `global_uvp5`, `other_living` carries
  `Order=monstrilloida` in `zooscan`, and `filament` — a morphology — carries
  `Class=cyanophyceae` in `zoolake`.
- **2 `label_disagreement`** (WARN) — two non-taxonomic buckets named differently
  (`darkrods` → `other` / `shape`), with no rank cell on either side. No image gets a taxon it
  should not have; vocabulary drift, not a taxonomy defect.

Three of the twenty are **not** defects and are recorded as such: `heterocapsa triquetra` /
`kryptoperidinium triquetrum` is one organism under two names (the snapshot has aphia 110153 as
`unaccepted` with that valid name, and all three rows share NCBI taxid 66468), and the two
bucket-naming cases above.

**Why the existing checks do not catch it.** `verify_taxonomy_ids.py` works *vertically* — it takes
the taxon a row claims and asks WoRMS / NCBI / Wikidata whether the identifiers agree — and groups
its findings by `proposed_label`, which files the two disagreeing rows under *different* taxa and
so never puts them side by side. Both `Harpacticoida` rows pass all 23 of its checks, because aphia
115348 genuinely *is* Euterpina; it is simply attached to a class directory that says
Harpacticoida. The 2026-07-13 audit's two clean results are on adjacent axes and both still hold:
a `(Dataset, Raw_Labels)` pair never maps two ways (0 duplicates), and each `proposed_label` has
exactly one lineage (0 conflicts). The direction nothing tested is `Raw_Labels` → one taxon
*across* datasets.

**Consequence:** this is a ceiling on measured accuracy, not only a tidiness problem. Images in the
80 affected rows carry a taxon their folder name does not support, so a classifier is trained and
scored against partly-wrong ground truth — and the affected labels are coarse, high-count ones
(`foraminifera` spans 9 rows, `annelida` and `neoceratium` 7 each).

**Frozen-output risk: data-side.** Correcting a row changes its published lineage. Document only,
and gate a data fix on a golden diff (`HARDEN-01`). The check itself is network-free — one pass
over the committed CSV — so it runs on every push, and a NEW disagreement (a CSV edit, or a new
source dataset reusing an existing label for a different taxon) arrives unwaived and turns the
suite red.

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

**The Hub LICENSE.md does not settle this (2026-08-04).** KI-24 established that LICENSE.md is
the authority over this repository, and that on comparison "fourteen of fifteen agreed" with
only `zoolake` differing. `whoi` is one of the fourteen — so the published notice states `mit`
too. That is agreement, not corroboration: both sides were transcribed from the same GitHub
repository, so the authority carries the same unverified inference. **KI-14 is not closed by
KI-24, and the escalation still points upstream** — to the IFCB imagery terms at
`ifcb-data.whoi.edu`, not to a document that inherited the slug. It is the single largest open
question in this file by affected rows (3,563,595).

## KI-15 — `planktonset1.0` is recorded as `other`, which states nothing

**Where:** `configs/dataset_import/planktonset1.yaml` (`license: other`); 60,736 images, 0.35%.

**Today:** `other` is the HuggingFace placeholder for "not one of the known slugs" and gives a
consumer no terms at all. `license_url` therefore points at the NOAA NCEI DOI for accession
0127422 (`https://doi.org/10.7289/v5d21vjd`, already recorded in the config's citation), which
is the authoritative record for the actual terms.

**Risk: downstream-legal, bounded.** The smallest ambiguity in the table, but it is the one
value a license filter cannot act on: `other` can be neither included nor excluded on merit.

---

---

## Verified non-issues (checked and dismissed — do not re-open)

The audit tested and *rejected* these as legitimate conventions, not defects:

- `living` ⇔ `root_class == 'living'`: **0 mismatches** (1,276/1,276).
- **0** conflicting source mappings (a `(Dataset, Raw_Labels)` pair never maps two ways);
  *(still true, and a different axis from **KI-31**, which is one `Raw_Labels` mapping two ways
  across DIFFERENT datasets — untested until 2026-09-01);*
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
