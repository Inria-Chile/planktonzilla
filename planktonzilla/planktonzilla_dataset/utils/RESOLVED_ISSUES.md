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

Two obligations outlived the fixes they belong to. They are **open**, and are listed in
`KNOWN_ISSUES.md`'s index so they are not lost by being archived here:

- ~~**KI-17**~~ — **discharged 2026-08-28.** MedPlanktonSet's importer was written against an
  unverifiable archive layout (Zenodo was unreachable), so its first real run had to be
  checked. It has now run: the imagefolder holds exactly **139** class directories, matching
  the 139 `medplanktonset` rows in `planktonzilla_taxonomy.csv`. `find_class_root` picked the
  right level.
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

**Checked, and it holds (2026-08-28).** MedPlanktonSet has since had a real run against the
live Zenodo archive, and `medplanktonsetdatasetimporter_imagefolder` holds exactly **139**
class directories — equal to the 139 `medplanktonset` rows in the taxonomy CSV. The guessed
layout was never relied on, and `find_class_root` resolved the real one correctly. This
caveat is discharged.

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
