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

---

*Recorded 2026-06-17 during the v1.0 `dataset_generation` cleanup (Phase 7, `KNOWN-01`).
See `.planning/REQUIREMENTS.md` `HARDEN-01` / `HARDEN-02` for the deferred v2 work.*
