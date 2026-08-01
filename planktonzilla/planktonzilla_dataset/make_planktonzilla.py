"""
(c) Inria

Create or update the consolidated planktonzilla dataset — the ``pz_planktonzilla``
entry point.

Supersedes ``pz_generate_planktonzilla`` (build everything from scratch) and
``pz_update_planktonzilla`` (re-sync the taxonomy of the published dataset), which
were two commands that shared a taxonomy CSV and half a pipeline. Here the run is
described by three orthogonal parameters instead of a mode:

    base            where already-built rows come from   null | hub | local | <path>
    sources         which sources are rebuilt now        all  | []   | [whoi]
    sync_taxonomy   re-apply the CSV to carried rows     true | false

so the three real scenarios are parameterisations rather than separate code paths::

    pz_planktonzilla                                              # create from scratch
    pz_planktonzilla base=hub   sources=[]                        # taxonomy CSV changed
    pz_planktonzilla base=local sources=[whoi] refresh=redownload # re-import one source

INVARIANT: the output holds exactly one contribution per source — freshly built for
the sources in ``sources``, carried over from ``base`` for every other one —
concatenated in the ``datasets`` declaration order. Reassembling in registry order
rather than appending rebuilt rows at the end is what makes an incremental run
row-for-row identical to a from-scratch one, which is what
``tests/test_make_planktonzilla_splice.py`` asserts.

Prerequisites:

  - Taxonomy CSV with the taxonomy and external ID columns, indexed by
    (Dataset, Raw_Labels).
  - Network access for ``base=hub``, and for any source whose imagefolder must be
    downloaded. ``base=null`` with existing imagefolders is fully offline.
"""

import os
import shutil
from pathlib import Path

import hydra
import numpy as np
import pyarrow.compute as pc
import pyrootutils
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datasets.utils import Version
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError
from omegaconf import DictConfig

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.generate_planktonzilla import (
    REDEFINERS,
    REFRESH_MODES,
    build_overrides,
    build_taxonomy_lookup,
    clean_corrupt_examples_optimized,
    import_and_redefine_source,
)
from planktonzilla.planktonzilla_dataset.update_planktonzilla import (
    add_license_columns,
    build_sync_dict,
    sync_columns,
)
from planktonzilla.utils.logger import get_pylogger

root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

logger = get_pylogger(__name__)

CLEAN_SCOPES = ("fresh", "all", "none")
UNMATCHED_POLICIES = ("keep", "clear")


def select_sources(cfg) -> list:
    """Registry entries to rebuild, in REGISTRY order (never the order the user typed).

    Registry order matters: it is the concatenation order of the output, so honouring
    the user's ordering here would silently reorder the dataset.

    Accepts ``"all"``, ``None`` / ``[]``, a list of names, or a bare string (treated
    as a one-element list — otherwise ``sources=whoi`` would iterate characters).

    Raises:
        ValueError: On an unknown source name, or a name in both ``sources`` and
            ``drop``. An ``import_name`` passed by mistake is called out by name,
            since 4 of the 12 entries have an ``import_name`` that differs from their
            ``name``.
    """
    registry = list(cfg.datasets)
    by_name = {entry["name"]: entry for entry in registry}

    requested = cfg.get("sources")
    if isinstance(requested, str) and requested != "all":
        requested = [requested]

    if requested == "all":
        selected_names = [entry["name"] for entry in registry]
    elif requested is None:
        selected_names = []
    else:
        selected_names = [str(name) for name in requested]

    by_import_name = {entry["import_name"]: entry["name"] for entry in registry}
    for name in selected_names:
        if name in by_name:
            continue
        if name in by_import_name:
            raise ValueError(
                f"Unknown source {name!r}: that is an `import_name` (the config stem), not a source name. "
                f"Did you mean {by_import_name[name]!r}? Valid names: {', '.join(sorted(by_name))}."
            )
        raise ValueError(f"Unknown source {name!r}. Valid names: {', '.join(sorted(by_name))}.")

    dropped = {str(name) for name in (cfg.get("drop") or [])}
    for name in dropped:
        if name in selected_names:
            raise ValueError(f"Source {name!r} is in both `sources` and `drop`; it cannot be rebuilt and removed at once.")

    # Deduplicate while keeping registry order.
    wanted = set(selected_names)
    return [entry for entry in registry if entry["name"] in wanted]


def resolve_base_location(cfg, output_dir: Path):
    """Resolve ``cfg.base`` to ``("hub", repo_id)`` / ``("disk", path)`` / ``None``."""
    base = cfg.get("base")
    if base is None:
        return None

    base = str(base)
    if base == "hub":
        return ("hub", cfg.base_repo_id)
    if base == "local":
        return ("disk", output_dir)
    return ("disk", Path(base))


def load_base(location) -> Dataset:
    """Load the base dataset from the Hub or from disk, unwrapping a single-split dict."""
    kind, target = location

    if kind == "hub":
        logger.info(f"Loading base dataset from the HuggingFace Hub: {target}.")
        return load_dataset(target, split="train")

    logger.info(f"Loading base dataset from disk: {target}.")
    ds = load_from_disk(str(target))

    if isinstance(ds, Dataset):
        return ds

    splits = list(ds.keys())
    if len(splits) != 1:
        raise ValueError(f"Base dataset at {target} has splits {splits}; expected a single split to carry over.")
    return ds[splits[0]]


def source_row_indices(ds: Dataset) -> dict:
    """Map each ``dataset`` column value to the row indices carrying it.

    Reads only the ``dataset`` column through Arrow, so no image is decoded — the
    difference between seconds and hours on a 17M-row dataset.
    """
    column = ds.select_columns("dataset").with_format("arrow")[:]["dataset"]

    indices = {}
    for name in pc.unique(column).to_pylist():
        if name is None:
            continue
        # int64, NOT the uint64 Arrow hands back: datasets' interpolation search does
        # `(j - i) * (x - arr[i]) // (...)` on the index array, and mixing numpy uint64
        # with a Python int promotes the result to float64, so the computed index is a
        # float and Dataset.select raises IndexError.
        positions = pc.indices_nonzero(pc.equal(column, name)).to_numpy(zero_copy_only=False)
        indices[name] = positions.astype(np.int64, copy=False)
    return indices


def ensure_license_columns(ds: Dataset, *, where: str) -> Dataset:
    """Derive the ``license`` / ``license_url`` columns when a base predates them.

    The published planktonzilla-17M was built before per-image license provenance
    existed, so it has none. Freshly built rows always carry them (``_taxonomy_row``
    stamps them during the taxonomy pass), which means without this a run that carries
    rows over from the frozen dataset would be rejected outright by
    :func:`assert_consolidated_schema` — making the very migration that adds the columns
    impossible to express with this command.

    Both values are a pure function of the ``dataset`` column, so this is derivation, not
    invention. It is applied to the base BEFORE any ``select``: ``add_column`` flattens an
    indices mapping, which on the full dataset would rewrite ~13.6M rows for nothing.

    It DOES change the published schema, so it is logged loudly and the caller is pointed
    at ``push_revision``.
    """
    missing = [col for col in constants.LICENSE_COLS if col not in ds.column_names]
    if not missing:
        return ds

    logger.warning(
        f"{where} predates the license columns {missing}; deriving them from the `dataset` column. "
        f"This CHANGES the published schema — publish it with push_revision=<branch> rather than over "
        f"the revision the paper and released models are pinned to."
    )
    return add_license_columns(ds)


def assert_consolidated_schema(ds: Dataset, *, where: str, reference=None) -> None:
    """Fail loudly when a dataset's column SET diverges from the consolidated schema.

    Necessary, not decorative: ``concatenate_datasets`` silently NULL-FILLS a column
    that one side is missing rather than raising, so a base that gained or lost a
    column would be blanked for exactly the rows just rebuilt — a corruption that
    reports success.

    Column ORDER is handled correctly and silently by ``concatenate_datasets``, so
    only the set is checked.
    """
    expected = set(reference if reference is not None else constants.CONSOLIDATED_COLUMNS)
    actual = set(ds.column_names)

    if actual == expected:
        return

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    detail = []
    if missing:
        detail.append(f"missing {missing}")
    if extra:
        detail.append(f"unexpected {extra}")
    raise ValueError(
        f"Schema mismatch in {where}: {'; '.join(detail)}. Concatenating it would silently null-fill "
        f"the differing columns instead of failing, so the run is stopped here."
    )


def conform_schema(ds: Dataset, reference_features) -> Dataset:
    """Reorder columns to the reference and cast only when the features actually differ."""
    if list(ds.column_names) != list(reference_features):
        ds = ds.select_columns(list(reference_features))
    if ds.features != reference_features:
        ds = ds.cast(reference_features)
    return ds


def log_lookup_coverage(part: Dataset, source_name: str, lookup: dict) -> None:
    """Report how many of a rebuilt source's labels have no taxonomy CSV entry."""
    labels = set(part.unique("original_label"))
    unmatched = sorted(label for label in labels if (source_name, label) not in lookup)

    if not unmatched:
        logger.info(f"╰─ {source_name}: all {len(labels)} labels resolved against the taxonomy CSV.")
        return

    shown = ", ".join(unmatched[:10])
    more = f" (+{len(unmatched) - 10} more)" if len(unmatched) > 10 else ""
    logger.warning(
        f"╰─ {source_name}: {len(unmatched)}/{len(labels)} labels have NO taxonomy CSV entry and get null "
        f"taxonomy + null IDs: {shown}{more}"
    )


def resolve_version(cfg):
    """Validate ``cfg.version`` and report whether it can be embedded in the artifact.

    Called before any build work so a malformed version costs seconds, not hours.

    Returns:
        ``(version, embeddable)`` — the version string (``None`` when unset) and
        whether it has the ``x.y.z`` form ``datasets.utils.Version`` accepts. A
        non-embeddable version is still valid as a Hub tag, which is free-form.

    Raises:
        ValueError: If the version is blank, or is not ``x.y.z`` while
            ``version_strict`` is set.
    """
    version = cfg.get("version")
    if version is None:
        return None, False

    version = str(version).strip()
    if not version:
        raise ValueError("version is set but empty; use null to build an unversioned dataset.")

    try:
        normalised = str(Version(version))
        embeddable = True
    except ValueError:
        normalised, embeddable = version, False

    if not embeddable:
        if cfg.get("version_strict", False):
            raise ValueError(
                f"version={version!r} is not the x.y.z form required to embed it in the dataset, and "
                f"version_strict=true. Use a version like 1.4.0, or set version_strict=false to use it "
                f"as a Hub tag only."
            )
        logger.warning(
            f"version={version!r} is not the x.y.z form datasets.utils.Version accepts, so it will NOT be "
            f"embedded in the saved artifact. It will still be pushed as a Hub tag. "
            f"Set version_strict=true to make this an error."
        )
    elif normalised != version:
        # e.g. "2026.08.01" -> "2026.8.1". The Hub tag keeps the string the user typed.
        logger.warning(f"version={version!r} normalises to {normalised!r} when embedded in the dataset.")

    return version, embeddable


def apply_version(ds: Dataset, version: str, embeddable: bool) -> Dataset:
    """Stamp ``version`` into the dataset's ``DatasetInfo`` when it can be embedded."""
    if version is None or not embeddable:
        return ds

    ds.info.version = version
    logger.info(f"Embedded version {str(ds.info.version)!r} in the dataset info.")
    return ds


def tag_hub_release(repo_id: str, version: str, *, token, message=None, overwrite=False, revision=None) -> None:
    """Create (or move) a git tag on the Hub dataset repo for this version.

    Runs only after a successful push, so the tag always points at data that exists.
    ``revision`` names the branch to tag — pass the same value as ``push_revision`` so a
    schema-change run tags the branch it actually wrote, not the repo default.
    """
    api = HfApi(token=token)
    tag_message = message or f"planktonzilla dataset version {version}"

    if overwrite:
        try:
            api.delete_tag(repo_id, tag=version, repo_type="dataset")
            logger.warning(f"Deleted the existing Hub tag «{version}» before re-tagging (version_overwrite=true).")
        except RevisionNotFoundError:
            # No such tag yet, which is the normal case for a first release — there is
            # nothing to move aside, so fall through to create_tag below. Only this one
            # exception means "absent"; anything else is a real failure and propagates.
            logger.info(f"No existing Hub tag «{version}» to replace; creating it.")

    try:
        api.create_tag(repo_id, tag=version, tag_message=tag_message, repo_type="dataset", revision=revision)
    except Exception as e:
        # Broad on purpose: the push already happened, so whatever went wrong here the
        # user needs to be told the data IS uploaded but is not tagged — otherwise they
        # would reasonably assume the whole run failed and repeat it.
        raise RuntimeError(
            f"The dataset was pushed to «{repo_id}», but tagging it {version!r} failed: {e}. "
            f"The upload itself succeeded — do not re-run the build. The tag may already exist; "
            f"pick a new version, or set version_overwrite=true to move it."
        ) from e

    logger.info(f"Tagged «{repo_id}» as «{version}» on the HuggingFace Hub.")


def atomic_replace(final: Dataset, output_dir: Path) -> None:
    """Save to ``output_dir``, tolerating that it may be the dataset's own source.

    ``save_to_disk`` raises ``PermissionError`` when the target is the directory the
    dataset is memory-mapped from, which is exactly the ``base=local`` case. Writing
    beside it and swapping also means there is no window where neither a complete old
    nor a complete new copy exists.
    """
    if not output_dir.exists():
        final.save_to_disk(str(output_dir))
        return

    staged = output_dir.with_name(f"{output_dir.name}.new-{os.getpid()}")
    previous = output_dir.with_name(f"{output_dir.name}.old-{os.getpid()}")

    logger.info(f"Output exists; staging the new dataset in {staged} before swapping it in.")
    try:
        final.save_to_disk(str(staged))
    except BaseException:
        # A half-written staging tree helps nobody and would accumulate on every retry
        # (running out of disk on a 17M-image write is the realistic way to get here).
        # output_dir has not been touched yet, so the existing dataset is intact.
        shutil.rmtree(staged, ignore_errors=True)
        raise

    output_dir.rename(previous)
    staged.rename(output_dir)
    # The old tree is only removed once the new one is in place. On Linux the mmaps
    # `final` still holds into it survive the unlink.
    shutil.rmtree(previous, ignore_errors=True)


def log_plan(*, selected, registry, base_location, output_dir, cfg, dropped) -> None:
    """Print one banner describing everything the run will do, before it does any of it."""
    names = [entry["name"] for entry in selected]

    if base_location is None:
        base_desc = "nothing (building only what `sources` rebuilds)"
    elif base_location[0] == "hub":
        base_desc = f"HuggingFace Hub «{base_location[1]}»"
    else:
        base_desc = f"disk «{base_location[1]}»"

    logger.info("=" * 78)
    logger.info(f"Base          : {base_desc}")
    logger.info(f"Rebuilding    : {len(names)}/{len(registry)} sources" + (f" ({', '.join(names)})" if names else ""))
    logger.info(f"Refresh depth : {cfg.refresh}")
    logger.info(f"Dropping      : {', '.join(sorted(dropped)) if dropped else '(nothing)'}")
    logger.info(f"Taxonomy sync : {'carried-over rows, unmatched=' + cfg.sync_unmatched if cfg.sync_taxonomy else 'off'}")
    logger.info(f"Corrupt scan  : {cfg.clean}")
    logger.info(f"Version       : {cfg.get('version') or '(unversioned)'}")
    logger.info(f"Output        : {output_dir}")
    logger.info(f"Hub push      : {cfg.repo_id if cfg.get('push_to_hub', False) else '(no push)'}")
    logger.info("=" * 78)


def _report_dry_run(selected, cfg) -> list:
    """Instantiate each selected importer and report what a real run would touch.

    Returns the archives a real run would block on — sources whose imagefolder must be
    built but whose hand-downloaded input is not on disk.
    """
    logger.info("DRY RUN: resolving the plan only. Nothing is read, written or pushed.")
    blockers = []

    for entry in selected:
        overrides = build_overrides(
            cfg.data_dir,
            entry["import_name"],
            entry["cleanup"],
            entry.get("extra_overrides", []),
            refresh=cfg.refresh,
            import_overrides=cfg.import_overrides,
        )
        import_cfg = hydra.compose(config_name="import_dataset", overrides=overrides)
        importer = hydra.utils.instantiate(import_cfg.dataset_import)

        imagefolder = Path(importer.imagefolder_dir)
        exists = imagefolder.exists() and bool(os.listdir(imagefolder))
        state = f"{len(os.listdir(imagefolder))} categories" if exists else "absent/empty -> would be imported"
        removal = " (would be REMOVED first)" if cfg.refresh == "redownload" and imagefolder.exists() else ""

        logger.info(f"╰─ {entry['name']:16s} {imagefolder} [{state}]{removal}")

        # A source that needs a hand-downloaded archive only fails once the run reaches
        # it, which on a full build can be hours in. Report it now, while the whole plan
        # is on screen and nothing has been downloaded yet.
        if not exists or cfg.refresh == "redownload":
            missing = importer.missing_manual_downloads()
            if missing:
                blockers.extend(missing)
                for line in importer.manual_download_instructions().splitlines():
                    logger.warning(f"   {line}")

    return blockers


def _refuse_empty_result(*, dropped, fresh_names, base_source_names=None) -> None:
    """Raise a clear error when the assembled dataset would have no rows at all.

    Reachable two ways, both of which used to surface as something opaque: with no base
    and nothing rebuilt (``concatenate_datasets`` on an empty list), and with every
    source the base holds listed in ``drop`` (an ``IndexError`` picking a reference
    part). Neither loses data — the save never happens — but this command's whole
    contract is to fail understandably before doing work, so say what happened.
    """
    detail = [f"rebuilt sources {sorted(fresh_names)}" if fresh_names else "no sources selected for rebuild"]
    if dropped:
        detail.append(f"drop={sorted(dropped)}")
    if base_source_names is not None:
        detail.append(f"base holds {sorted(base_source_names) or 'no sources'}")

    raise ValueError(
        "The run would produce an EMPTY dataset, so nothing was written: "
        + "; ".join(detail)
        + ". Check `sources` and `drop` — dropping every source the base holds leaves nothing to save."
    )


def _reference_features(parts):
    """Features every part is conformed to: those of the LARGEST part.

    Conforming can cast, and a cast rewrites the whole table. Taking the reference from
    the biggest part keeps any cast on the smaller side. Using ``parts[0]`` instead made
    it depend on registry position: rebuilding the first entry made the small fresh part
    the reference, so every carried-over base block — up to ~13.6M rows — would be cast
    rather than the new one.
    """
    return max(parts, key=len).features


def assemble(*, base, fresh, registry, dropped, sync_dict, cfg, num_proc_arg) -> Dataset:
    """Reassemble the output in registry order from rebuilt parts and carried-over rows."""
    registry_names = [entry["name"] for entry in registry]

    if base is None:
        parts = [fresh[name] for name in registry_names if name in fresh]
        if not parts:
            _refuse_empty_result(dropped=dropped, fresh_names=fresh)
        logger.info(f"Concatenating {len(parts)} freshly built sources.")
        return concatenate_datasets(parts)

    # Pure re-sync: no rebuild, no removal. Return the base as-is (optionally synced)
    # without select/reorder, so row order survives even if the published dataset is
    # not in registry order.
    if not fresh and not dropped:
        if not cfg.sync_taxonomy:
            logger.warning("Nothing to rebuild and sync_taxonomy=false: the output is the base, unchanged.")
            return base
        logger.info(f"Re-syncing the taxonomy of all {len(base)} rows from the CSV.")
        return sync_columns(base, sync_dict, num_proc_arg, unmatched=cfg.sync_unmatched)

    base_indices = source_row_indices(base)
    logger.info("Base composition: " + ", ".join(f"{name}={len(idx)}" for name, idx in sorted(base_indices.items())))

    # A rebuilt source that matches 0 base rows is either a genuine addition (the base
    # predates that registry entry) or a rename — and a rename silently DOUBLES the
    # source, because its old rows get carried over under their old name while the
    # fresh part is appended under the new one. The two cases are distinguishable: a
    # rename leaves a name in the base that the registry does not know, so treat that
    # combination as fatal and a clean addition as a logged fact.
    registry_name_set = set(registry_names)
    unknown_in_base = sorted(name for name in base_indices if name not in registry_name_set and name not in dropped)
    added = [name for name in fresh if name not in base_indices]

    if added and unknown_in_base:
        raise ValueError(
            f"Rebuilding {added} would add rows for sources the base has none of, while the base also holds "
            f"{unknown_in_base}, which the registry does not know. That is the signature of a renamed source: "
            f"the old rows would be carried over AND the rebuilt rows appended, doubling the source. "
            f"Resolve it explicitly with drop={unknown_in_base} if those rows are the old name, "
            f"or base=null to rebuild without carrying anything over."
        )

    for name in added:
        logger.info(f"Source {name!r} has no rows in the base; its {len(fresh[name])} rebuilt rows are an addition.")

    for name in dropped:
        if name not in base_indices:
            logger.warning(f"drop={name!r} matched 0 rows in the base; nothing to remove.")

    # Which sources come from the base, in registry order, then any unregistered
    # source the base carries (so a base is never silently truncated).
    carried_names = [name for name in registry_names if name not in fresh and name not in dropped and name in base_indices]
    for name in unknown_in_base:
        logger.warning(f"Carrying over {len(base_indices[name])} rows of source {name!r}, which is not in the registry.")
    carried_names += unknown_in_base

    if carried_names:
        kept_indices = np.concatenate([base_indices[name] for name in carried_names])
        kept = base.select(kept_indices)

        if cfg.sync_taxonomy:
            logger.info(f"Re-syncing the taxonomy of the {len(kept)} carried-over rows from the CSV.")
            kept = sync_columns(kept, sync_dict, num_proc_arg, unmatched=cfg.sync_unmatched)

        # Offsets into `kept`, which follows carried_names order by construction.
        offsets, cursor = {}, 0
        for name in carried_names:
            size = len(base_indices[name])
            offsets[name] = (cursor, cursor + size)
            cursor += size
    else:
        kept, offsets = None, {}

    blocks = []
    for name in registry_names:
        if name in fresh:
            blocks.append(("fresh", name))
        elif name in offsets:
            blocks.append(("base", name))
    blocks += [("base", name) for name in unknown_in_base]

    if not blocks:
        _refuse_empty_result(dropped=dropped, fresh_names=fresh, base_source_names=base_indices)

    parts = []
    for kind, name in blocks:
        if kind == "fresh":
            parts.append(fresh[name])
        else:
            start, end = offsets[name]
            parts.append(kept.select(range(start, end)))

    reference = _reference_features(parts)
    parts = [conform_schema(part, reference) for part in parts]

    for kind, name in blocks:
        if kind == "fresh":
            before = len(base_indices.get(name, ()))
            logger.info(f"╰─ {name}: {before} rows dropped, {len(fresh[name])} rebuilt rows added.")

    final = concatenate_datasets(parts)
    assert sum(len(part) for part in parts) == len(final), "row count changed during concatenation"
    return final


@hydra.main(
    version_base="1.3",
    config_path=str(root / "configs"),
    config_name="planktonzilla.yaml",
)
def main(cfg: DictConfig) -> None:
    """Create or update the consolidated planktonzilla dataset.

    Resolves the run from ``base`` / ``sources`` / ``sync_taxonomy``, rebuilds the
    selected sources, splices them into the rows carried over from the base in
    registry order, saves to ``output_dir`` and — when ``push_to_hub`` is set —
    pushes to ``repo_id``.
    """
    taxo_csv_path = (
        cfg.taxonomy_csv_path if cfg.get("taxonomy_csv_path") is not None else str(constants.DEFAULT_TAXONOMY_CSV_FILENAME)
    )
    num_proc_arg = cfg.num_proc if cfg.get("num_proc") is not None else constants.default_num_proc()
    output_dir = (
        Path(cfg.output_dir)
        if cfg.get("output_dir") is not None
        else Path(cfg.data_dir) / constants.DEFAULT_PLANKTONZILLA_DATASET_NAME
    )

    if cfg.refresh not in REFRESH_MODES:
        raise ValueError(f"refresh must be one of {REFRESH_MODES}, got {cfg.refresh!r}")
    if cfg.clean not in CLEAN_SCOPES:
        raise ValueError(f"clean must be one of {CLEAN_SCOPES}, got {cfg.clean!r}")
    if cfg.sync_unmatched not in UNMATCHED_POLICIES:
        raise ValueError(f"sync_unmatched must be one of {UNMATCHED_POLICIES}, got {cfg.sync_unmatched!r}")

    version, version_embeddable = resolve_version(cfg)

    registry = list(cfg.datasets)
    selected = select_sources(cfg)
    dropped = {str(name) for name in (cfg.get("drop") or [])}
    base_location = resolve_base_location(cfg, output_dir)

    # Checked before any download or API call: a source wired into cfg.datasets without a
    # recorded license would otherwise only surface hours in, having already written rows
    # whose redistribution terms we cannot state. Only the sources this run rebuilds are
    # checked — rows carried over from the base already carry their own license columns.
    constants.validate_license_coverage(entry["name"] for entry in selected)

    if not selected and base_location is None and not dropped:
        raise ValueError("Nothing to do: `sources` selects no source and `base` is null.")

    # save_to_disk overwrites a different dataset directory silently, so a partial
    # rebuild with no base would replace the consolidated artifact with a fragment and
    # report success. A full rebuild is exempt: it is a superset, not a fragment.
    is_partial = 0 < len(selected) < len(registry)
    if base_location is None and is_partial and output_dir.exists() and not cfg.allow_partial_overwrite:
        raise ValueError(
            f"Refusing to overwrite {output_dir} with a PARTIAL rebuild of "
            f"{[entry['name'] for entry in selected]} while `base` is null — the other sources would be lost. "
            f"Pass base=local to splice into what is already there, choose a fresh output_dir=, "
            f"or set allow_partial_overwrite=true if shrinking the target really is the intent."
        )

    log_plan(
        selected=selected,
        registry=registry,
        base_location=base_location,
        output_dir=output_dir,
        cfg=cfg,
        dropped=dropped,
    )

    if cfg.dry_run:
        blockers = _report_dry_run(selected, cfg)
        if blockers:
            logger.warning(
                f"DRY RUN complete; nothing was modified. {len(blockers)} archive(s) must be downloaded by "
                f"hand before a real run can get past them (listed above)."
            )
        else:
            logger.info("DRY RUN complete; nothing was modified.")
        return

    lookup = build_taxonomy_lookup(taxo_csv_path)
    sync_dict = build_sync_dict(taxo_csv_path) if (base_location is not None and cfg.sync_taxonomy) else None

    fresh = {}
    for entry in selected:
        name = entry["name"]
        logger.info(f"Start importing dataset «{name}».")

        part = import_and_redefine_source(
            entry,
            data_dir=cfg.data_dir,
            redefiner=REDEFINERS[entry["redefiner"]](csv_taxonomies_path=taxo_csv_path),
            num_proc_arg=num_proc_arg,
            refresh=cfg.refresh,
            import_overrides=list(cfg.import_overrides),
        )

        log_lookup_coverage(part, name, lookup)

        if cfg.clean == "fresh":
            logger.info(f"╰─ Cleaning up corrupt examples in «{name}».")
            part = clean_corrupt_examples_optimized(part, batch_size=1000, n_jobs=-1)

        assert_consolidated_schema(part, where=f"rebuilt source {name!r}")
        fresh[name] = part

    base = None
    if base_location is not None:
        base = load_base(base_location)
        # Before the schema check, so a base that predates per-image licensing can still
        # be brought up to date rather than rejected for lacking the columns.
        base = ensure_license_columns(base, where="the base dataset")
        assert_consolidated_schema(base, where="the base dataset")

    ds = assemble(
        base=base,
        fresh=fresh,
        registry=registry,
        dropped=dropped,
        sync_dict=sync_dict,
        cfg=cfg,
        num_proc_arg=num_proc_arg,
    )

    if cfg.clean == "all":
        logger.info("Cleaning up corrupt examples across the whole assembled dataset.")
        ds = clean_corrupt_examples_optimized(ds, batch_size=1000, n_jobs=-1)

    ds = apply_version(ds, version, version_embeddable)

    logger.info(f"Saving consolidated Planktonzilla dataset ({len(ds)} rows) to {output_dir}.")
    atomic_replace(ds, output_dir)

    if cfg.get("push_to_hub", False):
        # A schema change (adding the license columns, say) belongs on its own branch,
        # not over the revision the paper and the released models are pinned to.
        push_revision = cfg.get("push_revision", None)
        revision_kwargs = {"revision": push_revision} if push_revision else {}
        target = f"{cfg.repo_id}@{push_revision}" if push_revision else str(cfg.repo_id)

        logger.info(f"Pushing consolidated Planktonzilla dataset to HuggingFace Hub as «{target}».")
        ds.push_to_hub(
            cfg.repo_id,
            private=cfg.get("push_as_private", True),
            token=cfg.get("hf_token", None),
            **revision_kwargs,
        )

        # Tagged only after a successful push, so the tag always points at real data —
        # and at the revision this run actually wrote, not the repo default.
        if version is not None:
            tag_hub_release(
                cfg.repo_id,
                version,
                token=cfg.get("hf_token", None),
                message=cfg.get("version_message"),
                overwrite=cfg.get("version_overwrite", False),
                revision=push_revision,
            )
    else:
        logger.warning("Skipping pushing dataset to HuggingFace Hub, set push_to_hub=True to change this.")
        if version is not None:
            logger.warning(f"Version {version!r} was not pushed as a Hub tag: this run did not push (push_to_hub=false).")

    logger.info("Process completed!")


if __name__ == "__main__":
    main()
