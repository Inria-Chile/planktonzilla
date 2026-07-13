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

---

## Data inconsistencies in `planktonzilla_taxonomy.csv` (KI-8 – KI-13)

KI-1..KI-7 above concern **code behavior**. KI-8..KI-13 below concern **data** defects in the
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
