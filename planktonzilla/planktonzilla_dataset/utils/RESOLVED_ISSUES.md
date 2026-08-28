# Resolved Issues — dataset generation (`planktonzilla/planktonzilla_dataset`)

Entries from [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) that are **fixed and verified**, moved here
so that file lists only what is still true. Nothing is deleted: each entry is reproduced
verbatim, including its caveats, because the record of *why* a fix was made is what stops it
being undone.

**KI numbers are never reused or renumbered.** Commits, code comments and tests cite them —
`generate_planktonzilla.py` carries a `# KNOWN ISSUE:` block naming KI-16,
`tests/test_taxonomy_known_issues.py` names KI-8..KI-13, and
`tests/test_dataset_import_configs.py` names KI-16 and KI-25. A number appearing here rather
than in `KNOWN_ISSUES.md` means *resolved*, not *withdrawn*.

## Live follow-ups that survived their entry

Three obligations outlived the fixes they belong to. They are **open**, and are listed in
`KNOWN_ISSUES.md`'s index so they are not lost by being archived here:

- **KI-17** — MedPlanktonSet's importer was written against an unverifiable archive layout
  (Zenodo was unreachable). Its first real run must be checked: the reported class count
  should equal the **139** `medplanktonset` rows in `planktonzilla_taxonomy.csv`. A mismatch
  means `find_class_root` picked the wrong level.
- **KI-21 / KI-24** — `zoolake` and `jedioceans` are verified for **reachability and archive
  shape only**; neither has completed a full import. KI-25 removed one known blocker on that
  path, which is not the same as having run it.
- **KI-23** — `ensure_license_columns` adds two columns the published planktonzilla-17M does
  not have. Deriving them is safe; **re-pushing** the artifact from that schema is the part
  still gated.

---

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

## KI-17 — Two `dataset_import` configs named classes that do not exist

**Where:** `configs/dataset_import/medplanktonset.yaml`, `configs/dataset_import/sykezooscan2024.yaml`.

**Was:** `medplanktonset.yaml` targeted `MedPlanktonSetDatasetImporter`, which had never been
written, and `sykezooscan2024.yaml` targeted `SYKEZooScan2024` instead of
`SYKEZooScan2024DatasetImporter`. `medplanktonset` is the **5th of the 15 active entries** in
the `datasets` registry, so a full build died partway through — after four sources had already
been downloaded and processed. (It was the 5th of 12 when this was recorded; `zoolake`,
`jedioceans` and `sykezooscan2024` were appended later — see KI-24 — which left its index
unchanged.)

**Resolved.** The `sykezooscan2024` target is corrected, and `MedPlanktonSetDatasetImporter` is
implemented. `tests/test_dataset_import_configs.py` now instantiates every config in the group,
so a `_target_` naming a missing class fails in CI instead of mid-build.

**Frozen-output risk: none.** Both configs named classes that could not be instantiated, so
neither ever produced a row. Fixing them changes what a build *reaches*, not what it emits.

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

**Frozen-output risk: none.** Affects the per-source Hub push and its dataset card, not the
consolidated dataset's rows. The change is a failure that now surfaces rather than one that
was reported as success.

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

**Resolved.** Replaced with a table naming all three identifiers per row, plus a note that
**5 of the 15** active entries have `name != import_name` (`whoi`/`whoi-plankton`,
`zooscan`/`zooscannet`, `planktonset1.0`/`planktonset1`, `global_uvp5`/`global_uvp5net`,
`jedioceans`/`jedi_oceans_cpics`). Comment only; the `datasets` table itself is unchanged.
The count read "4 of the 12" when recorded, before the three appends of KI-24.

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

**Frozen-output risk: none by itself** — it corrects documentation and un-shadows a URL. But
it is the precondition for KI-24, which *does* change what a rebuild produces by making these
three active registry entries.

**Caveat that outlived this entry:** "reachability and archive shape verified, full import not
run" was the right hedge. For `zoolake` a full import would have failed — not on the download,
but on the imagefolder load. See **KI-25**.

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

**Frozen-output risk: none.** The published dataset's rows for this source came from a build
that must have used a hand-downloaded archive, so nothing published was produced by the broken
path. Its 20 CSV rows are 20 of the 1,485.

*Superseded detail (2026-08-04):* this paragraph originally opened "`sykezooscan2024` is not
in the active `datasets` table", which was true when recorded and is **contradicted by KI-24
below** — it became an active entry the same day. The risk verdict is unaffected: it is
`none` because the broken path never produced published rows, not because the source was
inactive.

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

**Frozen-output risk: MEDIUM — schema, not values.** `ensure_license_columns` adds two columns
the published planktonzilla-17M does not have. It is derivation (both values are a pure
function of the `dataset` column), so no existing column changes, but a consumer pinned to the
published schema sees two new fields. The code logs a warning pointing at `push_revision` for
exactly this reason. → `HARDEN-01` if the published artifact is ever re-pushed from it.

## KI-25 — The no-splits fallback glob could not see a split layout, so two active sources could not be built

**Where:** `generate_planktonzilla.py` `import_and_redefine_source` and
`dataset_importer.py` `import_dataset` — the `if not data_files:` fallback in each.

**Was:** both fell back to a hard-coded depth-2 glob, `<imagefolder>/*/*[!._]*`, which matches
the flat `<class>/<image>` layout only. On a split layout that pattern matches the class
**directories**; `datasets.data_files.resolve_pattern` keeps only entries whose type is
`file`, so **zero files resolved** and the load died with:

    ValueError: Instruction "train" corresponds to no data!

Not a degraded read — a hard stop. Because of KI-16 the repo-root probe never matches, so this
fallback is the *only* branch `import_and_redefine_source` ever takes, and it alone decided
what every source loaded. Two **active** registry entries produce split layouts:

- **`lensless`** — `_prepare_imagefolder` renames `TRAIN_IMAGE`/`TEST_IMAGE` to `train`/`test`.
  Reproduced against the real bundled zip: the old glob raised; a correctly-depthed one reads
  5,000 + 1,400 images.
- **`zoolake`** — `_prepare_imagefolder` writes `train_split/`, `val_split/`, `test_split/`.
  None of those is in `split_aliases` (`train`, `validation`, `val`, `test`), so it also failed
  the *correctly rooted* probe inside `import_dataset`, one step earlier than lensless did.

This contradicted KI-24's claim that the registry "covers all 15 sources … so a from-scratch
build reproduces it": the registry *listed* 15, but the build path could execute 13.

**Resolved (2026-08-04).** Both fallbacks now call `resolve_imagefolder_glob`, which tries
class-folder depths shallowest-first (`IMAGEFOLDER_CLASS_DEPTHS = (1, 2)`) and returns the
first that yields files. Depth 1 is the pattern every flat source already used, so those
resolve **exactly** as before; only a layout that yields nothing at depth 1 — i.e. one that
raises today — falls through.

When **no** depth yields a file the resolver logs a warning and returns the depth-1 pattern
anyway, rather than raising. That is deliberate and was corrected during implementation: an
early raise read better, but it invented a failure mode for every caller that never resolves
the pattern against a real filesystem. Eleven Hydra tests in `test_gen_planktonzilla_hydra.py`
and `test_make_planktonzilla_hydra.py` monkeypatch `load_dataset` and build stub imagefolders
with no images; they went red immediately. An empty imagefolder must keep failing exactly
where and how it always did — this fix is for split layouts only.

**Why a depth ladder and not a recursive `**` glob.** `imagefolder` infers each label from the
file's parent directory and emits a `label` column only when the matched files sit at a
*uniform* depth. A recursive glob also matches any stray image at the imagefolder root, which
breaks that uniformity — the loader then silently drops `label` altogether and
`_taxonomy_row`'s `class_names[example["label"]]` raises `KeyError`. Measured on a flat tree
with one loose root image: recursive gave `n=10, cols=['image']`; the depth ladder gave
`n=9, cols=['image', 'label']`. A fixed depth cannot hit that failure mode. Pinned by five
tests in `tests/test_dataset_import_configs.py`, including the anti-recursion one.

**Frozen-output risk: none — verified, not asserted.** `n_splits` still comes from the
caller's single-split fallback and is still `1`, so `original_path` is still the last two
chunks. Confirmed on the real lensless archive: the ladder yields
`/ACTINOSPHAERIUM NUCLEOFILUM/act114.jpg`, the same shape as the published rows, with all 10
classes correctly inferred — the `train`/`test` level is transparent to label inference.
`tests/test_gen_planktonzilla_lensless_e2e.py` is now parametrized over both layouts and
asserts they produce identical rows.

**Consequence worth knowing.** Fixing this does *not* recover split provenance — see KI-16's
second bullet. Both sources now load, with their upstream split boundary still discarded.

---

*KI-11 resolved 2026-07-13. KI-17..KI-23 resolved 2026-08-01 during the `pz_planktonzilla`
consolidation. KI-25 resolved 2026-08-04. Archived here 2026-08-04, from a full re-audit of
every entry against the code as it stands.*

---

## The 2026-08-27 taxonomy repair pass (KI-8 – KI-10, KI-29 – KI-35)

Ten data entries fixed in one maintainer-directed pass, test-first: the enforced data
contract was written as `tests/test_taxonomy_validation.py` (red on the broken table),
every proposed cell value was adversarially verified by an independent 14-decision review
with live WoRMS / NCBI / GBIF / Wikidata lookups, the repairs were applied to 70 rows —
38 base, 32 frepj — (237 cells; the commit diff is the authoritative manifest; row count
and join keys unchanged), and the Tara Pacific block plus its reconciliation report were
regenerated with `build_tara_pacific_taxonomy` so the block stays byte-reproducible. Five
labels left the label set (`cladoceramus`, `clio pyramidata`, `neoceratium`,
`heterocapsa triquetra`, and `Eukaryota` — replaced by its casing twin `eukaryota`), one
changed its reading (`ctenophora`: diatom genus → comb-jelly phylum), and the frepj
byte-freeze baseline was re-baselined
(`tests/fixtures/frepj/pre_frepj_taxonomy.sha256`). The published HuggingFace artifacts
are unchanged; the fixed table ships with the next dataset build. Entries below are
reproduced verbatim as they stood in `KNOWN_ISSUES.md`, each followed by its resolution.

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

**Resolved 2026-08-27.** All four rows re-slotted with registry-verified values, each
guarded by `test_ki8_resolved_rank_slots` and generically by the suffix-convention test:
row 153 `Order=gonyaulacales` (NCBI, GBIF and Wikidata are unanimous; WoRMS abstains with
"Dinophyceae incertae sedis" — which is where the old class-name-in-Order actually came
from, reframing this entry's "contamination" reading); row 945
`Order=hemiaulales, Family=hemiaulaceae` (the table's own *Eucampia* vocabulary); row
1126 `Family="florenciellales incertae sedis"` (the review REFUTED the obvious fix — no
family "Florenciellaceae" has ever been published; the WoRMS-verbatim placeholder keeps
the rank ladder gap-free without fabricating a name); row 817
`Class=cryptophyceae, Order=katablepharidales` (per the row's own NCBI lineage;
`kathablepharidacea` turned out to be NCBI's verbatim order node, not a table typo).

## KI-9 — Uppercase value in a normalized column

**Where:** row 671 (Dataset=`global_uvp5`, Raw_Labels=`Eukaryota`).

**Today:** `proposed_label='Eukaryota'` — the *only* value with an uppercase letter across
every normalized column (Kingdom..Species, `proposed_label`, `root_class`, `qualifier`) in all
1,485 rows; the convention is lowercase. (`Raw_Labels` legitimately preserves source casing.)
Should read `eukaryota`.

**Frozen-output risk: data-side.** A case-sensitive consumer keying on `proposed_label` treats
this as a distinct class; changing it alters the label set. Document only.

**Resolved 2026-08-27.** `Eukaryota` → `eukaryota`. Lowercase normalization is now
enforced over every normalized column with zero exceptions
(`test_normalized_columns_are_lowercase`).

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

*2026-08-26:* the same-key contradiction class is larger than the fish-egg pairs this entry
records — `chordata` and `hexapoda` carry it too, and the four labels together are now pinned
as the complete set. See KI-30.

**Resolved 2026-08-27.** Both egg rows → `plankton=True` (fish eggs are
ichthyoplankton), and the "backwards" secondary was repaired too: all seven fish-larvae
rows (`chordata`/`teleostei`/`leptocephalus`/`myctophidae`, rows 329, 829, 912,
1315–1318) → `True`. Two rules now enforce this class of defect generically:
`(proposed_label, qualifier)` → `(plankton, living)` must be a function, and an `egg` row
is plankton exactly when a taxon anchors it. Directions guarded by
`test_ki10_ki30_resolved_plankton_directions`.

## KI-29 — zoocamnet `Cladocera` is mapped to an extinct fossil bivalve genus

**Where:** row 378 (Dataset=`zoocamnet`, Raw_Labels=`Cladocera`).

**Today:** the water-flea label carries the lineage
`animalia/mollusca/bivalvia/pteriida/inoceramidae/cladoceramus` — *Cladoceramus* is an
inoceramid bivalve known only from Cretaceous fossils — flagged `plankton=True`,
`living=True`. Three internal contradictions mark it as a name-similarity mismatch
(*Cladocera* → *Cladoceramus*) rather than a reading of the images: global_uvp5's identical
raw label maps to `branchiopoda` (row 201); zoocamnet's own cladoceran genus class
(`Penilia`, row 1050) sits under `branchiopoda`; and the row's `aphia_ID`/`NCBI_ID`
(105.0 / 6544.0) are **class Bivalvia's** identifiers — byte-identical to the `bivalvia`
label's rows — not *Cladoceramus*'s.

**Frozen-output risk: data-side.** Every zoocamnet image of that class ships with the wrong
kingdom-to-genus lineage; correcting it changes published rank columns. Document only. →
data fix gated on `HARDEN-01`.

**Resolved 2026-08-27.** Row 378 now carries the `branchiopoda` payload of the
global_uvp5 row (the crustacean reading); no row carries genus `cladoceramus` any more.
Guarded by `test_ki29_resolved_cladocera_is_branchiopoda`.

## KI-30 — Contradictory `plankton` flag: `chordata` and `hexapoda` (completes KI-10)

**Where:** rows 318–328 (`chordata`, qualifier `full_body`) and rows 785/786 (`hexapoda`,
`full_body`).

**Today:** the same strict-key contradiction KI-10 records for the fish-egg pairs —
identical `proposed_label`, `qualifier`, `living`, `root_class` and every `*_ID` column, yet
`plankton` takes both values — exists at two more labels, both already present in the
1,485-row table the 2026-07-13 audit covered (it missed them). `chordata`: **True** for
global_uvp5 / isiisnet / planktonset1.0 / uvp6net / zoocamnet / zooscan
(*Actinopterygii* / *Gnathostomata* / *Tunicata* / `chordate_type1`), **False** for
jedioceans (`LClass_32.1Adult_Fish` and `LClass_fish_lavae`) and zoolake (`fish`) — fish
larvae flagged not-plankton is the same "backwards" ichthyoplankton reading KI-10 already
notes for `teleostei`. `hexapoda`: **True** for global_uvp5 (*Chaeteessa*), **False** for
zooscan (*Insecta*). The appended blocks inherit the split (three Tara `chordata` rows took
True, two Tara `Insecta` rows took False). With these, the complete same-key contradiction
set in the table is exactly four labels — chordata, clupeiformes, engraulidae, hexapoda —
and the pin asserts that completeness, so a fifth cannot appear silently.

**Frozen-output risk: data-side.** Correcting either side changes the `plankton` column.
Document only.

**Resolved 2026-08-27.** `chordata`/`full_body` → `True` everywhere (the table's
adult-fish convention); `hexapoda`/`full_body` → `False` everywhere (insects are not
plankton — the *Chaeteessa* row flipped). The complete-set property is now generic:
`test_flags_are_a_function_of_label_and_qualifier` fails on ANY contradictory key.

## KI-31 — The frepj append gave four hierarchy nodes two parents

**Where:** Family `bosminidae` → Order `anomopoda` (rows 196–197, 1486–1490) AND
`diplostraca` (row 1491, *Bosminopsis deitersi*); Family `daphniidae` → `anomopoda`
(rows 266, 510–512, 1512–1526) AND `diplostraca` (row 1527, *Scapholeberis smirnovi*);
Family `sididae` → `ctenopoda` (rows 551, 1049–1051, 1536–1542) AND `diplostraca`
(row 1543, *Sida*); Order `arcellinida` → Class `tubulinea` (row 94, base) AND `lobosa`
(rows 1710–1711, frepj).

**Today:** `FREPJ_TAXONOMY_RECONCILIATION.md` records the order-granularity judgment call
(Section B.1: new-to-FREPJ cladocerans keep GBIF's `diplostraca`; overlapping genera reuse
the base's finer orders) and the curated `lobosa` proposal (Section B.7) — but not the
emergent consequence: the rank columns no longer form a tree. The one-lineage-per-label pin
cannot see this, because the conflicting rows carry different `proposed_label`s. Grouping by
a rank column or walking parent→child edges (hierarchical metrics, sankey roll-ups, the
label graph) now splits three families across two orders each, and *Arcellinida* across two
classes. Related, documented in Section B.5 of the same file: *Simocephalus*
(rows 1510–1511) spells its family `daphniida`, leaving it outside the `daphniidae` node
entirely.

**Frozen-output risk: data-side.** Choosing one parent per node changes rank columns in
published rows. Document only; pinned as the exact four-node set so a fifth two-parent node
turns the test red.

**Resolved 2026-08-27.** All 30 frepj `diplostraca` rows normalized to the finer
cladoceran orders the base table already uses (per-family assignments WoRMS-confirmed:
Bosminidae/Chydoridae/Daphniidae/Moinidae/Macrothricidae → anomopoda,
Holopediidae/Sididae → ctenopoda); *Simocephalus*'s `daphniida` → `daphniidae`;
*Centropyxis*'s `lobosa` → `tubulinea`. `diplostraca`, `daphniida` and `lobosa` no longer
occur anywhere. The tree property is enforced generically
(`test_rank_columns_form_a_tree`); the chosen vocabulary is guarded by
`test_ki31_resolved_rank_vocabulary`.

## KI-32 — One raw label, divergent mappings across sources; four look misaligned

**Where:** 20 `Raw_Labels` strings map to different `proposed_label`s depending on the
source dataset (never within one dataset — the zero-conflicting-source-mappings invariant
still holds).

**Today:** sixteen of the twenty are granularity or context judgment calls — uvp6net
`Annelida` → `poeobius`, `Trachymedusae` → `botrynema`, `Thecosomata` → `cavolinia inflexa`;
zooscan `Foraminifera` → `globigerinidae`, `Harpacticoida` → `euterpina`, `Penilia` →
`penilia avirostris`, `actinula` → `solmundella bitentaculata`; flowcamnet `Dinophyceae` →
`gonyaulacales`, `Ornithocercus` → `ornithocercus magnificus`; `nauplii` → `arthropoda` vs
`copepoda`; `filament`, `fiber_detritus`, `darkrods`; plus `Cladocera` (KI-29) and
`Neoceratium` / `Heterocapsa_triquetra` (KI-33). Four have internal evidence of being wrong
rather than contextual:

- row 908: zooscan `other_living` → `monstrilloida` — the catch-all bucket declared one
  copepod order, `plankton=True`, while flowcamnet / isiisnet / planktoscope / zoocamnet map
  `other_living` → `other` and global_uvp5 maps its own generic `living` / `other<living`
  buckets → `other` (rows 1000–1001).
- row 1339: global_uvp5 `unknown` → `thecofilosea` — the only dataset whose `unknown`
  resolves to a concrete taxon (with full IDs); zoolake's `unknown` → `unknown` / artefact.
- row 37: flowcamnet `Acantharia` → `amphibelone` — a class-level source label rendered as
  one genus; every other dataset's `Acantharia` → class `acantharia`.
- row 387: zooscan `Creseidae` → `clio pyramidata` — a species of a *different* family (the
  row's own Family column says `cliidae`, contradicting the family the label names);
  global_uvp5's `Creseidae` keeps family `creseidae` (row 464), and zooscan's own
  `Creseidae acicula` resolves inside Creseidae (row 466).

**Mechanism, guarded forward (2026-08-26):** `build_tara_pacific_taxonomy` resolved an
ambiguous verbatim donor by FILE POSITION (`_existing_indexes` keeps the first row per
`Raw_Labels`), which is how the Tara rows inherited `Creseidae` → `clio pyramidata`
(rows 2009/2169) and `Harpacticoida` → `euterpina` (rows 2017/2174). The builder now
refuses an ambiguous donor whose pick is not recorded in its `DIVERGENT_DONORS` table
(eight acknowledged picks today), so the next append cannot repeat the accident.

**Frozen-output risk: data-side.** Re-mapping any of the four changes `proposed_label` for
every image of that class. Document only.

**Resolved 2026-08-27 (the four suspects; the judgment tier stays, deliberately).**
zooscan `other_living` → `other`; global_uvp5 `unknown` → `unknown`; flowcamnet
`Acantharia` → `acantharia`; zooscan `Creseidae` → `creseidae` (each payload
registry-verified). The 13 remaining per-dataset divergences are deliberate granularity
choices, pinned in both directions by
`test_ki32_remaining_divergences_are_exactly_the_acknowledged_ones`;
`DIVERGENT_DONORS` in the tara builder shrank from eight entries to five. One knock-on to
carry forward: the tara `Monstrilloida` row's donor was the misaligned zooscan row, so on
regeneration it kept only its EcoTaxa-derived `aphia_ID` (1106.0) — its
wikidata/NCBI/BOLD/ecotaxa cells are blank until the next ID-fill pass re-derives them
for the label's own row.

## KI-33 — Synonym splits: one taxon ships as two label classes

**Where:** `neoceratium` vs `tripos`; `heterocapsa triquetra` vs
`kryptoperidinium triquetrum`.

**Today:** raw *Neoceratium* → label `neoceratium` in flowcamnet / planktoscope / zoocamnet
(rows 940–942, plus three inherited Tara rows) but → `tripos` in zooscan (row 1393);
sharpest inside planktoscope itself, whose two species-mix classes land in different labels
(`neoceratium gibberum concilians mix` → `neoceratium`, row 943;
`neoceratium falcatum inflatum mix` → `tripos`, row 1395). And raw `Heterocapsa_triquetra`
→ `heterocapsa triquetra` (syke_ifcb_2022, row 783) but → `kryptoperidinium triquetrum`
(whoi, row 822) — one species, as the shared NCBI taxid 66468 confirms. KI-13 blesses a
shared taxid across synonyms as ID-side correct; the label-space consequence — a classifier
trained on this table treats one taxon as two classes — was unrecorded.

**Frozen-output risk: data-side.** Merging either pair changes the label set. Document only.

**Resolved 2026-08-27.** Merged by accepted name, both directions registry-verified:
raw *Neoceratium* → `tripos` (junior synonym; the freshwater `ceratium` rows are
untouched), raw `Heterocapsa_triquetra` → `kryptoperidinium triquetrum`. Neither retired
label remains in the table. Guarded by `test_ki33_resolved_synonym_merges`.

## KI-34 — 23 Tara rows carry a label finer than their recorded ranks

**Where:** 23 `tara_pacific_*` rows; 12 distinct labels (`cirripedia`, `brachyura`,
`achelata`, `gammaridea`, `alciopini`, `globorotalidae`, `anthozoa`, `coscinodiscids`,
`dinophyceae x`, `odontella sp.`, `chaetoceros inter. calothrix`,
`chaetoceros inter ciliate`).

**Today:** in the base and frepj blocks `proposed_label` always equals the lowest filled
rank (or the Genus+Species binomial) — an unwritten but previously universal invariant. The
Tara block breaks it where a taxon sits at a rank the seven-column ladder cannot hold:
`cirripedia` (infraclass; ranks stop at Class=`thecostraca`, row 2019), `brachyura`
(infraorder; Order=`decapoda`, row 2047), `achelata` (raw `phyllosoma`, row 2071),
`gammaridea` (suborder, row 2043), `alciopini` (tribe, row 2158), plus EcoTaxa morpho /
open-nomenclature nodes (`dinophyceae x`, `odontella sp.`, …). Two have an internal
precedent showing the ranks were representable: `globorotalidae` (row 2135) is a FAMILY
label whose Family column is empty, while zooscan's foram family `globigerinidae` (row 733)
fills `Class=globothalamea / Order=rotaliida`; and `anthozoa` (rows 2130/2306) is a
CLASS-rank taxon whose Class column is empty because the table's cnidarian Class vocabulary
is `hexacorallia` (rows 22, 265, 1482). A consumer that derives the class partition from the
rank columns gets a coarser partition than `proposed_label` for exactly these classes, and a
label lookup in the rank columns fails for all 23.

**Frozen-output risk: data-side.** Filling ranks or coarsening labels changes published
columns. Document only; pinned as the exact 23-row / 12-label set.

**Resolved 2026-08-27.** Split by expressibility: `globorotalidae` and `anthozoa` ARE
rank-expressible and got their fills (via `RANK_GAP_FILLS`, WoRMS/NCBI-verified:
Globorotaliidae sits in Globothalamea/Rotaliida; Anthozoa fills the Class slot with
`hexacorallia` remaining a parallel finer class under cnidaria). The other ten labels sit
at ranks the seven-column ladder cannot hold (WoRMS ranks Cirripedia a subclass;
Brachyura/Achelata infraorders; Gammaridea a suborder; Alciopini a tribe) and are
registered with their parents in `constants.SUB_RANK_LABELS`. The invariant is now
enforced: a ranked row's label must be its lowest filled rank, the Genus+Species
binomial, or a registered sub-rank label whose declared parent matches
(`test_label_is_lowest_rank_binomial_or_registered_sub_rank`).

## KI-35 — `ctenophora`: comb-jelly raw labels carried on the diatom genus lineage

**Where:** 12 rows with `proposed_label='ctenophora'` — 9 base (rows 473–481) and 3 Tara
(rows 2107, 2144, 2284).

**Today:** the label maps to the DIATOM genus *Ctenophora* (aphia 163921,
`chromista/heterokontophyta/bacillariophyceae/fragilariales/fragilariaceae`), yet its raw
labels come from zooplankton imagers and include `Ctenophora<Animalia` (the source names the
kingdom itself, row 2144), `comb_Ctenophora` (row 477) and `tentacle<Ctenophora` (row 481,
qualifier `part_tentacle`) — comb plates and tentacles are ctenophore anatomy diatoms do not
have. The same rows carry the comb-jelly `ecotaxa_ID` pair (`456;559`, identical to the
*Beroe* / *Lobata* / *Cydippida* rows) beside the diatom aphia/NCBI/BOLD ids, and the comb
jellies' own subtaxa sit under Phylum `ctenophora` in `animalia` elsewhere in the table.
This SUPERSEDES the *Verified non-issues* dismissal below (the 2026-07-13 audit read the
two-rank appearance as a legitimate homonym);
[`TARA_PACIFIC_TAXONOMY_RECONCILIATION.md`](TARA_PACIFIC_TAXONOMY_RECONCILIATION.md) §B3
reached the same conclusion independently and records how the three Tara rows inherited it.

**Frozen-output risk: data-side.** Re-reading these 12 rows as the comb jelly means a new
label (the phylum is `ctenophora` too) or a changed lineage — either changes published
columns, exactly the golden-diff-gated change §B3 describes. Document only.

**Resolved 2026-08-27.** All 12 rows re-mapped to the comb-jelly phylum:
`animalia/ctenophora`, label `ctenophora`, with live-verified IDs `wikidata=Q102778`,
`aphia=1248.0`, `NCBI=10197.0` (BOLD left blank — unconfirmable), keeping each row's
flags/qualifier and the comb-jelly `ecotaxa_ID` pair `456;559`. The diatom genus reading
is gone table-wide; the two builder `HOMONYM_NOTES` entries that inherited it were
retired, and the *Verified non-issues* dismissal in `KNOWN_ISSUES.md` is formally
superseded. Guarded by `test_ki35_resolved_ctenophora_is_the_comb_jelly`.


---

## The daplankton merge (2026-08-28) — the enforcement suite's first live catch

The `daplankton` source (KI-36) was authored on a branch that forked BEFORE the
2026-08-27 repair pass, so three of its 44 rows were built by copying rows that still
carried defects the pass had since fixed. Merging the two branches put those copies next
to the repaired originals, and the enforcement suite caught all of it — this is the first
time these tests judged data they were not written against:

- `tests/test_taxonomy_validation.py::test_each_label_has_one_lineage` and
  `::test_rank_columns_form_a_tree` failed on `Katablepharis_remigera`: the daplankton row
  had been copied from the pre-repair row 817 and carried the KI-8 defect verbatim
  (`cryptophyta` — a phylum name — in the Class slot, `kathablepharidacea` in Order). Given
  the same registry-verified values as its original: `Class=cryptophyceae`,
  `Order=katablepharidales`.
- `tests/test_taxonomy_known_issues.py::test_ki33_resolved_synonym_merges` failed on
  `Heterocapsa_triquetra`, which re-introduced the label KI-33 had merged into
  `kryptoperidinium triquetrum`. Given the accepted-name payload byte-exactly, as its two
  siblings were.
- `tests/test_taxonomy_external_ids.py::test_labels_are_names_the_gbif_backbone_recognizes`
  flagged the new label `rhinomonas nottbeckii`. Checked and benign: GBIF's backbone does
  not carry that species, only the genus, and answers with the genus at
  `matchType=HIGHERRANK` — which the checker refuses rather than accept as a name match.
  The row's genus lineage is exactly GBIF's. Recorded in that test's allowlist.

Nothing else in the 44 rows moved: the Tara Pacific block rebuilds byte-identically with
daplankton in the donor pool, the block's 120 identifiers score clean against the
registries, and the frepj byte-freeze prefix is untouched by the insert (the daplankton
rows land after it, at lines 1716-1759).

*The value here is the mechanism, not the three rows.* A defect fixed on one branch can
be re-introduced by another branch that copied it before the fix — silently, because the
copy is internally consistent. Generic invariants over the whole table are what make that
loud at merge time instead of at publication.
