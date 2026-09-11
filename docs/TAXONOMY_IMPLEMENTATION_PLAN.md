# Implementation plan: normalised taxonomy table package

*Engineering plan, 2026-09-11. Turns the recommendation of
[`docs/TAXONOMY_REPRESENTATION.md`](TAXONOMY_REPRESENTATION.md) §8 into an ordered sequence of pull requests,
each with the files it touches, the gate that proves it, and the way back out. It changes no code by itself.*

The research report established **what** to build and **why**: a normalised plain-text table package (design A)
with Darwin Core and SSSOM column names, replacing the 19-column
`planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv`. This document is the **how**: what lands in
which order, what each step is allowed to break, and what has to be true before the next one starts.

Every claim the plan leans on was re-measured against the committed tree at `8d7f635` before it was written;
§1 records what reproduced, the one claim that needed refining, and a gap the report does not name. Numbers
marked *(measured)* were computed here; the rest are cited from the report.

---

## 1. Re-verification of the report's load-bearing claims

The plan's whole shape depends on §5 of the report (the CSV decomposes losslessly) and §3 (the frozen
contract). Both were re-derived from scratch rather than taken on trust.

**Reproduced exactly.** A standalone `csv`-only decomposition, written without reference to the report's
prototype, split the committed file into
907 concepts keyed by `proposed_label`, 2,358 mappings in file order, and 13 per-mapping id overrides, then
re-rendered it. The result is byte-identical: sha256 `95de6c49…`, matching the committed file. The
decomposition found **zero** lineage conflicts, confirming that `proposed_label` functionally determines the
seven rank columns, and all 13 id overrides are FREPJ rows. Also reproduced: 1,297 distinct rank-path nodes;
999 class concepts `(label, root_class, qualifier)`; 26 labels with more than one `root_class` and 58 with
more than one `qualifier`; `living ⇔ root_class == living` with zero exceptions; the 14-term qualifier
vocabulary plus blank; `root_class` at living 2,013 / detritus 165 / artefact 158 / inert 22; 850 distinct
training strings over 21 sources including the empty-string class; the 12 concepts that are not the deepest
node of their own lineage, name for name; and the two collation exceptions in the frozen prefix
(`Eukaryota`, `pseudo-nitzschia`).

**Refined — the frozen row order is partly derivable.** The report states as fact that `row_order.tsv` must
store physical order "since no sort key reproduces the 1,485-row prefix" (§7.2). That is too strong. The
prefix is in **non-descending order under the collation key `proposed_label.lower().replace("-", "")`, with
zero descents** *(measured)* — that single normalisation absorbs both documented exceptions. What is *not*
derivable is the order of rows **within** a label: of the 240 prefix labels carrying more than one row, 56
runs covering 545 rows are not in `(Dataset, Raw_Labels)` order *(measured)*, and no tried key recovers them.

The practical consequence is not that the manifest goes away — it stays, because a stored sequence number is
simpler and more robust than a derivation with an exception list. The consequence is that the manifest becomes
**checkable instead of merely trusted**: CI can assert that the stored order is non-descending under the
collation key, which catches a corrupted or hand-edited manifest that an opaque 2,358-line order file would
absorb silently. It also answers open decision §10.6 (row order for post-freeze sources) with a rule the
existing data already follows, rather than with a new convention.

**Confirmed still live.** `build_frepj_taxonomy.write_csv`
(`planktonzilla/planktonzilla_dataset/utils/build_frepj_taxonomy.py:435-454`) copies the byte prefix up to the
first `frepj` row and then writes only the frepj block, discarding everything after it — the 44 daplankton and
600 tara_pacific rows, 644 in total. `docs/CODE_REVIEW.md` lists finding 1.1 as **open**. This is a live
data-loss bug in a tool a curator is invited to re-run, and the plan promotes fixing it out of the migration
and into step 0, where it stands alone.

**Gap the report does not name.** There is **no test anywhere in `tests/` that pins the 850 / 599 label
vocabulary** *(measured: no test references either count, or builds a `ClassLabel` from the committed table)*.
M7 is the contract whose breach is least visible — it silently renumbers the class ids of four released models
— and it is today the least protected. `tests/test_taxonomy_lookup_equivalence.py` already pins the 16-column
lookup against a frozen pandas implementation, and `tests/test_frepj_taxonomy_coverage.py:170-177` pins the
first 1,486 bytes-frozen lines, but nothing pins the label strings. That reorders the plan: the characterization
test lands before the migration, not with it.

---

## 2. Target state

Code as an importable package, data beneath it. This separates the two from the report's §8 sketch, which put
the TSVs at the top of `taxonomy/`; keeping them in `taxonomy/data/` lets `taxonomy/` be a normal Python
package, which is what `from …taxonomy import load_taxonomy` needs. `[tool.hatch.build.targets.wheel]`
declares `packages = ["planktonzilla"]` with no exclude list (`pyproject.toml:107-108`), so everything below
ships in the wheel with no packaging change.

```
planktonzilla/planktonzilla_dataset/taxonomy/
├── __init__.py            load_taxonomy, TaxonomyStore — the only public surface
├── model.py               Taxon / Mapping / Identifier records; the in-memory store
├── loader.py              package-dir backend AND legacy wide-CSV backend
├── render.py              render_wide_csv() -> bytes; published projection; label vocabularies
├── validate.py            the polars rule module (the enforcer)
├── write.py               upsert_source, set_mapping, add_taxon, add_id, rename/merge/split, fmt, diff, release
├── cli.py                 pz_taxonomy console script
├── migrate.py             the one-shot migration (step 1), with its own render-back verification
└── data/
    ├── datapackage.json           the schema of record, EXECUTED by validate.py + source registry
    ├── taxon.tsv                  1,352 rows (as built: 1,297 lineage nodes + 12 finer-than-their-rank
    │                              concepts as `unranked` children + 43 lineage-less buckets)
    ├── identifier.tsv             4,032 rows (measured; 132 ';'-joined ecotaxa cells exploded one id per line)
    ├── mappings/<dataset>.tsv     21 files, 2,358 rows (measured; largest planktoscope 266, global_uvp5 254, frepj 229)
    ├── vocab/{rank,qualifier,root_class,authority,kind}.tsv
    ├── merged.tsv                 retired id -> replacement, date, reason
    ├── waivers/{horizontal,authority,tree}.tsv
    └── release/v1.0/              legacy_row_order.tsv (2,358 keys), legacy_overrides.tsv (13 id rows + concept pins), sha256
planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv    GENERATED until the pins retire
.gitattributes                      merge=union on taxon.tsv, identifier.tsv, merged.tsv; eol=lf on *.tsv
```

`tara_pacific` is four registered sources (`bongo` 137, `decknet` 132, `hsn` 159, `manta` 172 *(measured)*), so
porting its builder writes four mapping files, not one.

The `taxon.tsv` figure is 1,352 rather than the 1,297 + 43 = 1,340 this section first stated. The difference is
the 12 concepts that name something finer than their deepest legacy rank (`brachyura` under `decapoda`,
`alciopini` under `phyllodocidae`, …). Folding them into that deepest node, which the 1,340 figure assumed,
makes concept → taxon non-injective: 10 nodes end up named by two or three concepts each *(measured)*. Giving
them `unranked` child nodes is what report §7.2 asks for, and it costs the seven frozen columns nothing —
`project7()` fills a slot only from an ancestor whose rank *has* a `legacy_slot`, so an `unranked` node
contributes no cell. Their true ranks (infraorder, tribe, subclass) are left unassigned: that is a curation
act, not a migration one.

### 2.1 The public API every consumer moves onto

```python
def load_taxonomy(source: Path | str | None = None) -> TaxonomyStore
```

`source` is a **directory** (package backend), a **file** (legacy wide-CSV backend), or `None` (the bundled
package). The dual backend is not a courtesy: it is what keeps the Hydra key `taxonomy_csv_path`
(`configs/{generate,update}_planktonzilla.yaml`, `configs/generate_frepj_only.yaml`) and the fixture-writing
tests working untouched. Ten test modules fabricate a wide CSV and four more reference its header directly
*(measured — the report's count of ten reproduces)*, several with 18 columns, capitalised ranks and
`root_class: zoo`; the wide backend must keep accepting all of it.

| Method | Serves | Replaces |
| --- | --- | --- |
| `.lookup()` | the published build | `generate_planktonzilla.build_taxonomy_lookup` (16 cols, value- and type-identical) |
| `.rows()` | sankey, verifiers, builders | eight ad-hoc `csv.DictReader` / `pl.read_csv` parses |
| `.frame()` | `verify_taxonomy_ids` | its `pl.read_csv(csv_path, infer_schema_length=0)` at `:534` |
| `.render_wide_csv() -> bytes` | the golden gate, the committed artefact | the three builders' byte-splicing writers |
| `.published_projection()` | `RedefineDataset._taxonomy_row` | inline column expressions, in `CONSOLIDATED_COLUMNS` order |
| `.label_vocabulary(name)` | `gen_planktonzilla_only_plankton` | the emergent `sorted(set(...))` at `:95` |
| `.has() / .labels_for() / .datasets()` | `make_planktonzilla` pre-flight | `check_taxonomy_csv`'s bare `csv.reader` at `:536` |

`build_taxonomy_lookup(csv_path)` survives as a thin alias so no caller outside the taxonomy package changes
its import. Note the two id orders that must not be conflated: the CSV header is wikidata, aphia, NCBI, BOLD,
ecotaxa; `constants.CONSOLIDATED_COLUMNS` is wikidata, **ecotaxa**, aphia, NCBI, BOLD (`constants.py:266-275`).

---

## 3. Sequence

Ten PRs in four stages. The rule that orders them: **no consumer moves before a gate can prove the move was
lossless, and the published path changes last.** Stage 0 is new relative to the report's §9, which starts at
what is here PR 3.

### Stage 0 — Protect the contract (no new representation yet)

Both PRs are independently valuable and independently revertable. Neither mentions the new package. If the
migration were cancelled tomorrow, both should still land.

**PR 0.1 — Pin the label vocabulary (≈0.5 d).** New `tests/test_taxonomy_label_vocabulary.py` asserting, from
the committed CSV: 850 distinct `ClassLabel` names over 21 sources and 599 over the 15 v1.0 sources, in
`sorted(set())` order, each including the empty-string class at index 0 *(both counts re-measured)*; the six
repeated-token strings present verbatim — four from KI-8's rank-slot contamination (`cryptophyta`,
`bacillariophyceae`, `dinophyceae`, `florenciellales`) and two legitimate tautonyms (`porpita porpita`,
`eudactylota eudactylota`) that the report lists among the conventions a representation change must *not*
"fix"; and the `Kingdom != ""` filter's `None`-admitting behaviour
(`gen_planktonzilla_only_plankton.py:83-85`) reproduced rather than corrected. Commit the two vocabularies as
test fixtures so a diff shows *which* names moved, not just that the count changed.
*Gate:* green on today's tree. *Why first:* it is the only frozen contract with no test, and everything after
this point can renumber it.

**PR 0.2 — Disable the destructive builder write paths (≈0.5 d).** `build_frepj_taxonomy.write_csv` and
`build_tara_pacific_taxonomy.append_to_master` refuse to write without an explicit `--i-know-this-rewrites-the-csv`
flag, and `write_csv` gains a guard that raises if the rendered output would drop any row whose `Dataset` is
not its own. Fixes `CODE_REVIEW.md` finding 1.1 (verified live: 644 rows destroyed on a no-op re-run).
*Gate:* a regression test that runs `write_csv` on the committed CSV and asserts all 2,358 rows survive —
red before the fix, green after.

*Superseded by PR 6.* The flag is gone with the splice it guarded, and the gate changed shape with it: it no
longer asserts that the builder **refuses**, it asserts that the builder **runs and all 644 rows are still
there**. `assert_owns_every_change` stays, now over the whole re-render, where it should be unreachable and
runs anyway.

### Stage 1 — Build the store beside the CSV (nothing reads it yet)

**PR 1 — Migration script and the generated package (≈3 d).** `taxonomy/migrate.py`, run once, producing
`taxonomy/data/`. The report's refuters found this step is where design A's prototype was quietly
hand-finished, so budget it as new code: expect ~500 lines here and ~1,200 across the model, validator and
write side, not the 900 the design estimated.

Two properties decide whether this PR is sound, and both get a test:

- **Id minting is idempotent.** Ids are seeded from the committed `taxon.tsv` when one exists and minted only
  for paths absent from it; a re-run on the committed package is a byte no-op. The failure mode this closes is
  real and silent: the panel's script minted in sorted-path order, so inserting one row re-pointed 1,342 of
  1,351 ids while every foreign key still resolved.
- **The manifest is consistent, not just present.** `legacy_row_order.tsv` stores the sequence number, and a
  check asserts the stored prefix order is non-descending under `proposed_label.lower().replace("-", "")`
  (§1: zero descents today). A manifest that drifts from the collation is then a failing check rather than a
  silent reordering.

*Gate:* `migrate.py` run twice produces identical bytes; the package parses; row counts match §2's measured
figures. The wide CSV is untouched and still the only thing anything reads.

**PR 2 — Loader, model and renderer, with the golden gate (≈3 d).** `model.py`, `loader.py`, `render.py`, and
`tests/test_taxonomy_render_golden.py`:

1. `load_taxonomy().render_wide_csv()` equals the committed CSV **byte for byte**, reporting the first
   divergent row (not a bare hash mismatch) on failure;
2. the first 1,486 rendered lines match the existing sha pin in `tests/fixtures/frepj/pre_frepj_taxonomy.sha256`;
3. the 16-column lookup built **from the model, not from the rendered bytes** equals `build_taxonomy_lookup`
   in value *and Python type* on all 2,358 keys — the distinction matters, because rendering and re-parsing
   would prove only that the renderer round-trips;
4. the label vocabularies from PR 0.1 reproduce from the model;
5. the reproject fixed point: `project7(parse(csv)) == csv`.

The gate is deliberately **not** a whole-file hash. Design B's was, and adding a 50-class source turned it red
and crashed its own error reporter. Whole-file hashing belongs only in the per-release sha over that release's
own row-order keys.

*Gate:* the five assertions above. *Rollback:* delete the package directory; nothing imports it.

### Stage 2 — Make it enforceable, then move the readers

**PR 3 — Schema of record and validation (≈3 d).** `validate.py` plus `datapackage.json` plus a seven-defect
fixture package. *As built*, `validate.py` **executes** the descriptor rather than re-implementing it: two
statements of one constraint is a drift hazard with nothing to make them agree, so there is a single
execution path and a constraint declared in the descriptor cannot be silently unenforced. The descriptor is
plain JSON, parsed with the standard library, which is what actually delivers "a laptop without Frictionless
gives CI's verdict". Four checks exist **because tools the
design trusted do not enforce them**, and each needs its own failing fixture:

| Check | Why polars must own it |
| --- | --- |
| `(scientificName, taxonRank, parentNameUsageID)` unique | Frictionless 5.19 parses `uniqueKeys` but does not enforce it — an injected duplicate was reported valid |
| exactly one `exact` id per (concept, published authority) | a second id is schema-legal in any SSSOM-shaped table and is silently dropped by the exporter, decided by sort order |
| `Dataset` column == mapping file stem | a foreign source's row in the wrong file passes a per-file primary key; the exporter keeps whichever file the glob visits last |
| the `NoTermFound` absence row validates | the design's own id pattern and relation enum reject the row it offers for "not in this register" |

Also here: booleans normalised by `fmt` and read through one typed loader that **raises** on anything but
`true` / `false` (Excel writes `TRUE`; the panel measured that silently inverting `plankton` for 120 rows with
no failing check); dates stored as strings; a `draft` status the renderer and coverage rule skip, so a
partially mapped source is committable; the lineage-name-repeat check scoped to legacy-slot ranks. Port the 20
horizontal waivers and the 59 authority waivers *(both counts measured)* from their JSON files onto stable
concept ids, preserving each `category` and `reason` verbatim.

**PR 4 — Concurrency and diff ergonomics (≈1 d).** `.gitattributes` with `merge=union` on the append-only
ledgers and `eol=lf` on `*.tsv`; a duplicate-id rule in `validate.py` to catch what union merge lets through;
per-source descriptor fragments, since `merge=union` fixes the ledgers but both branches still insert their
source stanza at the same position in a single descriptor.

*As built*, the descriptor's source enumeration is **removed** rather than fragmented. Fragments would have
solved the conflict while leaving a second source registry beside `constants.DATASET_IMPORT_CONFIGS` — two
lists to keep in step, and R9 already notes that adding a source must touch `constants` regardless, for
`validate_license_coverage`. The mapping resource becomes a `pathGlob`, registry drift is measured against
`constants` in both directions, and adding a source is one new file plus the `constants` entry it already
needed. That also retires the runbook caveat the refuters asked for: with no shared stanza, there is no
stanza to resolve by hand. The measurement is in the gate rather than in prose — `tests/test_taxonomy_concurrency.py`
performs a real `git merge` of two add-source branches, and a control test proves the same two branches
conflict on both ledgers without `.gitattributes`.

**PR 5 — Switch every reader (≈2 d).** `build_taxonomy_lookup` becomes an alias
(`generate_planktonzilla.py:105-172` deleted); the near-verbatim copy in `frepj_validate.py:113-132` deleted;
`check_taxonomy_csv` (`make_planktonzilla.py:519-567`) rewritten on `.has()` / `.datasets()` instead of its bare
`csv.reader` header peek; `sankey.py:575-579`, `utils/verify_taxonomy_ids.py:522-535` and
`utils/verify_label_consistency.py:135-138` move to `.rows()` / `.frame()`. The published path keeps every call
site and column expression — its only edit is the loader block becoming a delegating alias.

*Gate:* the full CI suite, unchanged, plus the Stage 1 gates. The wide CSV stays committed with a staleness
test. **No published bytes move in this PR**, which is the property that makes it revertable.

**PR 5.5 — Per-source coverage, generalised (≈2 d).** Every source gets the guarantee FREPJ has: a frozen
list of the class directories the source actually contains, and a test that every one of them has a mapping
row. Today **two of twenty-one sources have this** — `frepj` (229 rows) and `daplankton` (44) — and the gap is
not cosmetic. A class directory with no mapping row yields sixteen `None`s for every image of that class and
nothing raises; `check_taxonomy_csv` only warns when a source has *no* rows at all, so a source missing twenty
class dirs passes clean. The 15 sources in the published artifact — 1,485 rows, **63 % of the table** — are all
in that state.

What is NOT adopted is FREPJ's mechanism. Its taxonomy half is 1,709 lines (`build_frepj_taxonomy.py` 667,
`resolve_frepj_ids.py` 1,042) of source-specific curation, and Tara Pacific's is another 1,019 written to a
different design. Cloning that per source is how the repository got three precedents with three different
methods; a fourth would be a fourth dialect with its own bugs, and `write_csv` destroying 644 rows is precisely
a bespoke-builder bug. Those builders are also the *write* path that PR 6 replaces. FREPJ's rule tables exist
because FREPJ labels have a grammar (`Ge._unk` sentinels, comma-separated five-field paths); most sources have
flat irregular labels with nothing for a rule table to key on, so the same builder there is hand-curation with
extra ceremony.

So: one generic coverage test parameterised over whatever frozen lists exist, not eighteen builders.

The constraint that sets the order is **evidence**. A frozen class-dir list is worth having only if it is
independent of the CSV; deriving one from the CSV asserts the table against itself. Three sources already have
independent evidence — `tests/fixtures/frepj/frepj_class_dirs.tsv` (229, an archive scan with per-magnification
counts), `tests/fixtures/daplankton/daplankton_class_dirs.tsv` (44), and
`planktonzilla/dataset_import/tara_pacific_classes.tsv` (600, an EcoTaxa export). That is 873 rows, 37 %. The
other 15 sources have none, and `samples.json` does not supply it: it records `(dataset, proposed_label,
root_class, n)` — the *harmonised* label, not the raw class directory.

Hence two halves, and only the first is code:

- **5.5a** the generic coverage test over the three sources whose evidence exists, replacing the two bespoke
  coverage modules with one parameterised over the fixtures;
- **5.5b** a one-time scan of `original_label` in the published artifact to mint frozen class-dir lists for the
  15 v1.0 sources. A network job, not a code job, and the unlock for the other 63 %.

*Gate:* the generic test is red when a class dir in any frozen list has no mapping row — exercised, not
asserted.

*Refined once 5.5a was written.* "Verified by deleting a row" turned out to name the wrong failure: since PR 6a
the loader refuses a package whose frozen row-order manifest names a key no mapping file carries, so deleting a
*published* row never reaches the coverage check at all. The realistic failure is an archive that grows a class
the curator has not mapped — the frozen list names it, the table does not — and that is what the gate now does.
A second gate covers the other direction (a row for a class directory renamed upstream) and a third records
which failures the release layer catches first, so the two lines of defence are not confused for one.

"Replacing the two bespoke coverage modules with one" was also too strong. Only their **coverage** tests are
the shared contract; the rest of each module is source knowledge that does not generalise — FREPJ's `Ge._unk`
sentinel cascade, DAPlankton's reuse-vs-resolution split and the sea ⊂ syke subset relation. Those stay where
they are. The four Tara Pacific sources, which had evidence and no coverage test at all, gain one: **six of
twenty-one sources, 873 of 2,358 rows.** The 15 without evidence are pinned by name and by row count, so 5.5b
minting a list turns the suite red rather than passing silently at 37 % forever.

### Stage 3 — Write side and the long tail

**PR 6 — Write API and CLI (≈5 d, the risky step).** `write.py` and `pz_taxonomy`: `upsert-source`,
`set-mapping`, `add-taxon`, `add-id`, `rename` / `merge` / `split` with tombstones into `merged.tsv` and
automatic pins, `fmt`, `diff`, `release`. Dry-run by default; never blank an id without an explicit flag;
print a cell-level change list. Port the three builders — `build_tara_pacific_taxonomy.py` (1,019 lines),
`resolve_frepj_ids.py` (1,042), `build_frepj_taxonomy.py` (667), **2,728 lines total** *(measured)* — onto it,
retire `extract_taxon_ids.py` (its output is `identifier.tsv`), re-express the Tara Pacific "rebuild equals
committed block" test on records rather than bytes, and add the end-to-end test on a synthetic 50-class source.
This is the step to schedule alone; it is where the panel's estimates were judged floors, and where the guard
flag from PR 0.2 finally comes off because the write path is no longer byte-splicing.

*Refined once the port was measured.* "2,728 lines of builders" is the wrong unit and made this step look
larger than it is. Those lines are overwhelmingly **curation** — rule tables, label grammars, donor
resolution, reconciliation reports — which is source knowledge and stays where it is. The **write surfaces**
are `build_frepj_taxonomy._encode_rows` + `write_csv`, `build_tara_pacific_taxonomy._serialize` +
`append_to_master`, and `resolve_frepj_ids.backfill_csv`. Porting all three deleted **150 lines and added
102** *(measured: the diff of this step)* — 5.5 % of the 2,728. The port is one adapter,
`write.records_from_wide_rows`, which inverts the seven legacy rank columns through the same `lineage_of` the
migration uses, so the builders go on producing 19-column rows and only the last call changes.

*Refined again on the release layer.* The loader required the frozen row-order manifest to name every
accepted mapping, which made `upsert_source` unusable: adding a source left every consumer broken until the
next release was cut. A release now freezes the order of the rows **it published** and nothing more, and rows
mapped afterwards render after the frozen block in collation order — §10.6's rule, applied at read time
rather than only at migration time. Appending rather than interleaving is what the wide CSV itself did for
its three post-freeze blocks, and it is what keeps the 1,486-line prefix pin meaningful: a new source can
never push a frozen row down the file. The count check the manifest used to get is replaced by its own sha
pin, so truncating it is still refused, by a check that says which release drifted.

*Gate, all measured on the committed table:* re-deriving all 21 sources from what is committed and upserting
them back is a **zero-change no-op**, through both the record API and the wide-row adapter; running all three
ported builders writes **0 changes** and leaves the package and the CSV byte-identical; the synthetic
50-class source lands, mints 65 nodes from the next free id with no gaps, moves no frozen byte, and is a
no-op on its second run. A dry run allocates no id at all.

`upsert-source` is deliberately absent from the CLI: a source's records come from its builder, and there is
no way to express 229 FREPJ records on a command line. The CLI carries the acts that ARE one line — `check`,
`show`, `fmt`, `rename`, `retire`, `clear-id`, `render`, `diff` — and `release` moves to PR 9, where the
retirement switch it serves is decided.

**PR 7 — Provenance and waivers as data (≈2 d).** This is where FREPJ's and Tara Pacific's reconciliation
reports stop being prose and become columns. Provenance for *why a row says what it says* exists for 229 of
2,358 rows today, inside a Markdown file, plus 95 of the 600 rows Tara Pacific computes and writes out; for the
remaining 63 % nobody can answer the question at all. The `method` / `donor` / `identifiedBy` / `dateIdentified`
columns are already on every mapping row and already empty, so this step needs no new external input and no new
schema — which is why it is the cheapest half of generalising the FREPJ process and should not wait on 5.5b.

Backfill `method` / `donor` for the 600 Tara Pacific and 229
FREPJ decisions from the existing Markdown reports; move the seven `RANK_DEPARTURES` prose entries
(`build_tara_pacific_taxonomy.py:373-421`) onto the rows they explain; seed `broadMatch` rows from the
authority findings; derive the Markdown reports **from** the ledger instead of string-concatenating them.

*Refined once it was written.* The backfill did not need the Markdown reports at all — both builders already
**computed** the provenance per row and then threw it away into prose. So it is not a parse of a report, it is
`method` / `donor` / `identificationReferences` coming out of `build_rows` where they were always available.
The reach is larger than the plan estimated: Tara Pacific records a rule for all **600** of its rows and not
only the 95 `derived` ones, so **829 of 2,358 rows (35 %)** carry a method, against the 324 predicted.

Two things the plan did not say. **`method` is a controlled vocabulary** (`vocab/method.tsv`, six terms,
declared in the descriptor and enforced at the write): free text makes 2,358 answers incomparable, which is
the same as having none. And **blank is not a term.** Writing `unknown` on the 1,529 rows nobody can answer
for would dress an absence up as provenance; the count is pinned by a test instead, so it can only go down.

The `broadMatch` seeding covers all 147 coarse identifiers — **449 identifier rows**, one per finer concept —
and **no published byte moves**: the renderer never looks at the predicate, because the 19-column CSV has no
way to say "broader than". Withdrawing 449 published ids to express a nuance the format cannot carry would be
the worse answer. The validator's backlog goes from 168 findings to 21, and `unjustified_broad_match` now
guards the re-predication, so relabelling a real cross-branch collision cannot make it disappear.

The descriptor earned its keep here: seeding `semapv:LogicalReasoning` produced **449 errors** until the
justification enum was extended. That is the schema of record working, not a formality. It also exposed a
general defect — the validator applied enum and vocabulary constraints to an EMPTY value on an OPTIONAL field,
which `method` is the first field to hit.

*Gate:* the render is byte-identical with every provenance column filled; the package validates with zero
errors; both backfills are idempotent; a write that says nothing about provenance leaves it alone, the same
rule ids already had.

**PR 8 — Label vocabularies as data (≈1.5 d).** Write `v1.0_taxpath` (599 names) and `v1.2_taxpath` (21
sources) as versioned files; `gen_planktonzilla_only_plankton` reads the file rather than computing
`sorted(set(...))`. PR 0.1's test becomes the proof that the file equals today's expression. Released
vocabularies are frozen snapshots keyed to a release tag — only new vocabularies are regenerated, because
design C's regenerated vocabulary renumbered 537 of 599 class ids on its first data fix.

**PR 9 — Authority snapshot, review tooling, retirement switch (≈3 d).** Now also carries `release`,
moved here from PR 6: cutting a release freezes the current order into a new manifest, which is only
meaningful once the switch that stops committing the wide CSV is decided. Re-key the authority snapshot to the
sorted distinct-id set with incremental harvest, so a spelling fix no longer forces a network re-harvest and an
86,618-line JSON diff (needs the maintainers' HARDEN decision, §10.4). Add the CI semantic-diff comment
("N published cells changed on M rows", computed by rendering the legacy view before and after), the generated
curation runbook, and the README / KNOWN_ISSUES updates. Document — but do not yet throw — the switch that
stops committing the wide CSV, which waits on a golden diff against the Hub artefact.

Only after PR 9 is correcting a frozen defect a one-line act: delete the pin row, run `release`, and the diff
shows exactly the published rows that move.

---

## 4. Risk register

| # | Risk | Mitigation | Lands in |
| --- | --- | --- | --- |
| R1 | Migration re-run silently re-points ids while every FK still resolves | seed from committed `taxon.tsv`, mint only new paths; "run twice is a byte no-op" test | PR 1 |
| R2 | A builder destroys 644 rows on a no-op re-run (live today) | guard flag + drop-detection, before the migration starts | PR 0.2 |
| R3 | A spreadsheet save inverts `plankton` with no failing check | `fmt` normalises; typed loader raises on anything but `true`/`false`; dates as strings | PR 3 |
| R4 | Released model class ids renumber unnoticed | vocabulary pinned *before* any change; frozen per-release snapshots | PR 0.1, PR 8 |
| R5 | Golden gate breaks on every legitimate new source | render-equality + frozen-prefix pin + per-release sha; never a whole-file hash | PR 2 |
| R6 | Declared constraints that the named tool does not enforce | polars module re-implements all four; each gets a failing fixture | PR 3 |
| R7 | Concurrent add-source PRs conflict | `merge=union` + duplicate-id rule + per-source descriptor fragments; manual-resolution case documented | PR 4 |
| R8 | Pin layer freezes the concept but not the mapping, so a curator cannot land a KI-10 fix | three nullable `root_class` / `qualifier` / `plankton` columns on the mapping pin | PR 1 |
| R11 | A class directory with no mapping row publishes sixteen nulls per image, silently | a frozen class-dir list and a coverage test per source; only 2 of 21 have one today | PR 5.5 |
| R9 | Adding a source needs edits outside the taxonomy directory | `validate_license_coverage` raises without a `DATASET_LICENSES` entry (`constants.py:212-228`); the runbook says so | PR 9 |
| R10 | ~~Frictionless in the dev group drifts~~ — **retired**: no Frictionless dependency was added | the descriptor is stdlib-parsed and executed directly; nothing to pin | PR 3 |

R10 deserves its own note: `pyproject.toml:78-92` already carries a long comment explaining that a `>=` floor
plus an absent lockfile is not a pin at all. Any new dev dependency inherits that reasoning, not a floor.

---

## 5. Decisions, and the step each one blocks

Twelve decisions are open in the report's §10. Nine can ride their defaults; **three must be answered before
the PR that depends on them**, because reversing them later is a data migration rather than a code change.

| Decision | Blocks | Why it cannot wait |
| --- | --- | --- |
| §10.1 id style (opaque CURIE + display-name column) | **PR 1** | ids are minted once; changing the style later rewrites every foreign key |
| §10.6 row order for post-freeze sources | **PR 1** | the manifest is written during migration. §1's measurement supplies a default the existing data already obeys: `proposed_label.lower().replace("-", "")`, then registry order, then raw-label bytes |
| §10.2 frozen-defect policy (keep the pin layer) | **PR 2** | decides whether the golden gate targets today's bytes or a corrected v1.1 |
| §10.3 unpublished Tara Pacific blocks | PR 6 | nothing pins them yet, so the window to correct Harpacticoida / Creseidae without a release closes when the write API ports the builder |
| §10.5 Frictionless in the dev group | **PR 3, resolved: no** | the package could not be installed in this environment at all, and an unverifiable dev dependency is not a gate. The descriptor is executed directly instead, so Frictionless stays an optional cross-check on the same file |
| §10.4, .7–.12 | PR 6–9 | defaults hold; each is reversible in the step that consumes it |

§10.12's second half is already spent: PR #35 merged on 2026-09-10, so rebasing `verify_label_consistency.py`
onto the loader is simply part of PR 5.

---

## 6. Effort

| Stage | PRs | Days |
| --- | --- | ---: |
| 0 Protect the contract | 0.1, 0.2 | 1 |
| 1 Build the store | 1, 2 | 6 |
| 2 Enforce and switch readers | 3, 4, 5 | 6 |
| 2 Per-source coverage | 5.5a | 2 |
| 3 Write side and tail | 6, 7, 8, 9 | 11.5 |
| | | **26.5** |

5.5b (the `original_label` scan for the 15 published sources) is deliberately not costed here: it is a network
job against the Hub artefact, not engineering time, and it gates nothing else.

Against the report's 20.5 listed days: Stage 0 is new (+1), PR 1 absorbs the refuters' finding that the
migration is new code rather than a port (+1), and PR 4 separates concurrency work the report folded into other
steps (+1). The report's own framing still holds and should be the one quoted to stakeholders — the panel's
estimates were judged **floors** by every refuter, so plan for **four to six weeks of one engineer**, with PR 6
the step most likely to overrun.

---

## 7. What this plan deliberately does not do

- **No change to the published dataset, the paper, or the four released models.** Every gate in Stage 1 and 2
  exists to prove that. The wide CSV stays committed and byte-identical throughout.
- **No new runtime dependency.** polars (`>=1.41.2`) and the stdlib `csv` module carry the whole store;
  Frictionless, if adopted, is dev-only.
- **No orthogonal facet model beyond `lifeStage` and `bodyPart`** defaulted from the qualifier vocabulary
  (S6 follows once the vocabulary is agreed), **no name-usage table for synonyms** (S8: `acceptedNameUsageID`
  plus `taxonomicStatus` cover the three known pairs), and **no export adapters** (the concept table is already
  ColDP NameUsage-shaped, so each is one polars write later).
- **No correction of a frozen defect.** The *Ctenophora* homonym, Harpacticoida / Creseidae, KI-8 through
  KI-13 and the 20 KI-31 disagreements are all reproduced unchanged. The point of the work is to make
  correcting them a one-line reviewable act — not to do it in the same breath as the migration.
