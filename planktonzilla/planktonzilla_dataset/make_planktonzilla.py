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

A run that can take hours should not discover its problems hours in, so the same command
also answers "would this work?" before doing any of it::

    pz_planktonzilla dry_run=true check_downloads=all             # verify, build nothing

``dry_run`` resolves the plan and checks its local prerequisites; ``check_downloads``
adds the network ones — one HEAD (or ranged GET) per file a real run would fetch, the
Fairdata packaging API, the Hub base, the push target and the version tag. Anything
blocking is raised as a single summarised error, so the command doubles as a CI gate.
``check_downloads`` is independent of ``dry_run``: set on a real run, it refuses to start
rather than failing four sources in. See :func:`run_preflight`.

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

import concurrent.futures
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pyarrow.compute as pc
import pyrootutils
import requests
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datasets import config as datasets_config
from datasets.utils import Version
from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from humanize import naturalsize
from omegaconf import DictConfig

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.generate_planktonzilla import (
    LOOKUP_COLS,
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
            since 5 of the 17 entries have an ``import_name`` that differs from their
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


def ensure_custom_metadata(ds: Dataset, *, where: str) -> Dataset:
    """Add an empty ``custom_metadata`` column when a base predates it.

    ``custom_metadata`` (v1.2) holds, per row, the JSON object of whatever source-specific
    metadata has no consolidated column — see ``constants.CUSTOM_METADATA_COL``. Every
    source published before it existed had nothing to put there, so the column is filled
    with the literal ``constants.EMPTY_CUSTOM_METADATA`` (``"{}"``): the same value a
    from-scratch rebuild of those sources writes, which keeps carried-over and rebuilt
    rows indistinguishable. Derivation, not invention.

    Same mechanics and caveats as :func:`ensure_license_columns`: applied to the base
    BEFORE any ``select`` (``add_column`` is a zero-copy Arrow concat; the image column
    is never touched), and it changes the published schema, so publish the result onto a
    ``push_revision`` rather than over the frozen revision.
    """
    column = constants.CUSTOM_METADATA_COL
    if column in ds.column_names:
        return ds

    logger.warning(
        f"{where} predates the `{column}` column; filling it with {constants.EMPTY_CUSTOM_METADATA!r} for all "
        f"{len(ds)} rows. This CHANGES the published schema — publish it with push_revision=<branch> rather than "
        "over the revision the paper and released models are pinned to."
    )
    return ds.add_column(column, [constants.EMPTY_CUSTOM_METADATA] * len(ds))


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


# ==================================================================== pre-flight ===
#
# ``log_plan`` says what the run WOULD do; the checks below say whether it COULD. They
# come in two tiers, because they cost different things:
#
#   local    the taxonomy CSV, a ``base`` on disk, the output and data directories,
#            every hand-downloaded archive, each source's imagefolder — free
#   remote   one HEAD (or ranged GET) per file a real run would fetch, the Fairdata
#            packaging API, the Hub base, the push target and the version tag — gated
#            on ``check_downloads``, since this is the only part that uses the network
#
# Nothing here downloads data, writes a dataset or POSTs anything; the single write is
# a zero-byte file created and removed to prove a directory is writable.

CHECK_SCOPES = ("none", "needed", "all")

# Free space wanted per byte of download. A source's archive lands on disk, is extracted
# beside itself, and is then copied into the imagefolder, so three copies of it exist at
# the high-water mark of an import.
DISK_SPACE_FACTOR = 3


@dataclass(frozen=True)
class Check:
    """One pre-flight verdict.

    ``ok=False`` with ``blocking=False`` is a warning: reported loudly, but not a reason
    to stop. Checks that PASS are kept and reported too — a pre-flight that printed only
    its failures could not be read as evidence that everything else is fine, which is
    the question it is asked.
    """

    name: str
    ok: bool
    detail: str
    blocking: bool = True


def check_taxonomy_csv(csv_path, selected) -> list:
    """Check the taxonomy CSV exists, parses, has every column, and covers each source.

    The column check is not decoration: ``build_taxonomy_lookup`` resolves an absent
    column to ``None`` for every row instead of raising, so a CSV that lost a rank
    builds the whole dataset with that rank blank and reports success.

    A selected source with NO row at all in the CSV is reported as a warning rather than
    a failure — it is what adding a source before curating its labels looks like — but
    it is worth saying up front, because the alternative is discovering afterwards that
    a few hundred thousand images have null taxonomy and null IDs.
    """
    path = Path(csv_path)
    if not path.exists():
        return [Check("taxonomy-csv", False, f"missing: {path}")]

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
    except OSError as e:
        return [Check("taxonomy-csv", False, f"unreadable: {path} ({e})")]

    required = ("Dataset", "Raw_Labels", *LOOKUP_COLS)
    absent = [column for column in required if column not in header]
    if absent:
        return [Check("taxonomy-csv", False, f"{path} is missing the column(s) {absent}")]

    try:
        lookup = build_taxonomy_lookup(str(path))
    except Exception as e:
        return [Check("taxonomy-csv", False, f"{path} could not be parsed: {type(e).__name__}: {e}")]

    checks = [Check("taxonomy-csv", True, f"{len(lookup)} (dataset, label) rows, all {len(required)} columns present")]

    covered = {dataset for dataset, _ in lookup}
    uncovered = [entry["name"] for entry in selected if entry["name"] not in covered]
    if uncovered:
        checks.append(
            Check(
                "taxonomy-coverage",
                False,
                f"no CSV row at all for {uncovered}; every image of those sources would get null taxonomy and null IDs",
                blocking=False,
            )
        )
    elif selected:
        checks.append(Check("taxonomy-coverage", True, f"all {len(selected)} rebuilt source(s) appear in the CSV"))

    return checks


def check_base_on_disk(location) -> list:
    """Check a ``base`` on disk really is a saved dataset, without loading it.

    Mirrors ``datasets.load_from_disk``'s own dispatcher — ``dataset_info.json`` AND
    ``state.json`` means a ``Dataset``, ``dataset_dict.json`` means a ``DatasetDict``,
    neither means the path is not a saved dataset at all. Reading those two small JSON
    files also yields the row count, the embedded version and the shard list for free,
    so a splice run can be checked against the numbers it is about to build on without
    touching a byte of Arrow.
    """
    path = Path(location)
    if not path.exists():
        return [Check("base", False, f"base points at {path}, which does not exist")]

    info_file = path / datasets_config.DATASET_INFO_FILENAME
    state_file = path / datasets_config.DATASET_STATE_JSON_FILENAME
    dict_file = path / datasets_config.DATASETDICT_JSON_FILENAME

    if not (info_file.is_file() and state_file.is_file()):
        if dict_file.is_file():
            return [Check("base", True, f"{path} is a saved DatasetDict")]
        return [Check("base", False, f"{path} is not a saved dataset: it has neither {info_file.name} nor {dict_file.name}")]

    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [Check("base", False, f"{path} has an unreadable {info_file.name}/{state_file.name}: {e}")]

    # `splits` is written by a builder, so the real artifact carries it and an ad-hoc
    # Dataset.from_dict(...).save_to_disk() does not. Absent means "not recorded", never
    # "empty" — reporting 0 rows for a full dataset would be worse than saying nothing.
    splits = info.get("splits") or {}
    rows = sum(split.get("num_examples") or 0 for split in splits.values()) if splits else None
    version = (info.get("version") or {}).get("version_str")
    shards = [entry.get("filename") for entry in state.get("_data_files") or []]
    missing_shards = [name for name in shards if name and not (path / name).is_file()]

    detail = f"{path}: {len(shards)} shard(s)"
    if rows is not None:
        detail = f"{path}: {rows} rows in {len(shards)} shard(s)"
    if version:
        detail += f", version {version}"
    checks = [Check("base", True, detail)]
    if missing_shards:
        gone = f"{len(missing_shards)} shard(s) named in {state_file.name} are gone: {missing_shards[:3]}"
        checks.append(Check("base-shards", False, gone))
    return checks


def check_base_on_hub(repo_id, *, token, timeout, api=None) -> list:
    """Check the Hub dataset a ``base=hub`` run would read is there and readable.

    A private or unauthorised repo answers 404, so ``RepositoryNotFoundError`` means
    "missing OR invisible to this token" and is worded that way. ``GatedRepoError`` —
    which SUBCLASSES it in huggingface_hub 1.x, hence the order of the handlers — is the
    one case where the repo is known to exist and only access is missing.
    """
    api = api or HfApi(token=token)

    try:
        info = api.dataset_info(repo_id, timeout=timeout)
    except GatedRepoError:
        return [Check("base-hub", False, f"«{repo_id}» is gated; request access on the Hub before a base=hub run")]
    except RepositoryNotFoundError:
        return [Check("base-hub", False, f"«{repo_id}» not found, or invisible to this token (a private repo needs HF_TOKEN)")]
    except Exception as e:
        # Broad on purpose: huggingface_hub 1.x speaks httpx, whose transport errors are
        # neither requests exceptions nor OSError, so nothing narrower catches "the
        # network is down". A pre-flight has to report that, not crash on it.
        return [Check("base-hub", False, f"«{repo_id}» could not be read: {type(e).__name__}: {e}")]

    detail = f"«{repo_id}» readable"
    if info.sha:
        detail += f", revision {info.sha[:7]}"
    if info.last_modified:
        detail += f", last modified {info.last_modified:%Y-%m-%d}"
    if info.private:
        detail += ", private"
    return [Check("base-hub", True, detail)]


def check_push_target(cfg, version, *, api=None) -> list:
    """Check the run could push — and tag — what it is about to spend hours building.

    The tag check is the point of this one. ``tag_hub_release`` only discovers a name
    collision AFTER the entire dataset has been uploaded, and the error it raises then
    exists purely to stop the user re-running the build. Reading the repo's refs up front
    turns that into a five-second failure with the same advice.

    Takes no ``timeout``, deliberately: ``auth_check`` and ``list_repo_refs`` accept none
    in huggingface_hub 1.x, and a parameter that silently bounds nothing would read as a
    guarantee this cannot make. (``dataset_info`` does accept one — see
    :func:`check_base_on_hub`, which passes ``check_timeout``.)
    """
    repo_id = str(cfg.repo_id)
    token = cfg.get("hf_token", None)
    api = api or HfApi(token=token)

    if not token and not get_token():
        return [Check("push-token", False, "push_to_hub=true but no token: set HF_TOKEN, or run `hf auth login`")]

    checks = []
    try:
        api.auth_check(repo_id, repo_type="dataset", write=True)
        checks.append(Check("push-target", True, f"«{repo_id}» exists and this token may write to it"))
    except GatedRepoError:
        checks.append(Check("push-target", False, f"«{repo_id}» is gated for this token"))
    except RepositoryNotFoundError:
        # Deliberately NOT blocking, and deliberately ambiguous: the Hub answers 404 both
        # for a repo that does not exist — which push_to_hub would simply create — and
        # for one this token cannot see, where the push fails. Nothing distinguishes them
        # from here, so the message names both instead of guessing, and it is also what a
        # typo in repo_id looks like, which is only cheap to notice before the upload.
        absent = (
            f"«{repo_id}» is not visible to this token: either it does not exist (the push would create it — "
            f"check the spelling) or the token lacks access to it (the push would fail)"
        )
        checks.append(Check("push-target", False, absent, blocking=False))
    except Exception as e:
        unconfirmed = f"write access to «{repo_id}» could not be confirmed: {type(e).__name__}: {e}"
        checks.append(Check("push-target", False, unconfirmed))

    push_revision = cfg.get("push_revision", None)
    if version is None and not push_revision:
        return checks

    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
    except Exception as e:
        unlisted = f"the refs of «{repo_id}» could not be listed: {type(e).__name__}: {e}"
        checks.append(Check("hub-refs", False, unlisted, blocking=False))
        return checks

    if version is not None:
        tags = {ref.name for ref in refs.tags}
        if version not in tags:
            checks.append(Check("version-tag", True, f"«{version}» is free to tag on «{repo_id}»"))
        elif cfg.get("version_overwrite", False):
            checks.append(Check("version-tag", True, f"«{version}» exists and would be MOVED (version_overwrite=true)"))
        else:
            checks.append(
                Check(
                    "version-tag",
                    False,
                    f"tag «{version}» already exists on «{repo_id}» — pick another version, or set version_overwrite=true. "
                    f"Left as is, this run uploads the whole dataset and only then fails to tag it",
                )
            )

    if push_revision:
        exists = push_revision in {ref.name for ref in refs.branches}
        state = "exists" if exists else "does not exist yet; the push creates it"
        checks.append(Check("push-revision", True, f"branch «{push_revision}» {state}"))

    return checks


def check_writable_dir(name, path, *, needed_bytes=0) -> list:
    """Check a directory can be written to, and report the free space on its filesystem.

    Writability is tested by creating and deleting a zero-byte file rather than with
    ``os.access``, which answers for the real uid and lies on NFS and ACL mounts. The
    nearest EXISTING ancestor is used, because the target itself is usually the thing
    the run would create.
    """
    target = Path(path)

    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent

    probe = existing / f".pz-preflight-{os.getpid()}"
    try:
        probe.touch()
        probe.unlink()
    except OSError as e:
        return [Check(name, False, f"{existing} is not writable: {e}")]

    free = shutil.disk_usage(existing).free
    detail = f"{target} writable, {naturalsize(free)} free"

    if needed_bytes and free < needed_bytes * DISK_SPACE_FACTOR:
        return [
            Check(
                name,
                False,
                f"{detail} — the downloads alone are {naturalsize(needed_bytes)}, and an import needs about "
                f"{DISK_SPACE_FACTOR}x that (archive + extraction + imagefolder)",
                blocking=False,
            )
        ]
    return [Check(name, True, detail)]


def instantiate_selected(selected, cfg) -> list:
    """Compose and instantiate every selected source's importer. Touches nothing.

    Composing is the only way to see EFFECTIVE values: a registry entry's
    ``extra_overrides`` (a hand-downloaded archive, say), the ``refresh`` flags and
    ``import_overrides`` all change what a source would fetch, and none of them is
    visible in ``configs/dataset_import/*.yaml``.
    """
    importers = []
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
        importers.append((entry, hydra.utils.instantiate(import_cfg.dataset_import)))
    return importers


def ensure_source_sidecars(importers) -> dict:
    """Obtain every selected source's build-time sidecar inputs, BEFORE the first import.

    Not part of the pre-flight — a plain run never pre-flights — but the plain run doing
    in seconds what it would otherwise discover hours in, at the sixteenth source. A
    no-op for a source without any. Returns ``{name: {file: path}}`` so the log can say
    what was verified.
    """
    obtained = {}
    for entry, importer in importers:
        name = entry["name"]
        sidecars = importer.ensure_sidecars()
        if sidecars:
            logger.info(f"╰─ {name:16s} {len(sidecars)} sidecar file(s) verified: {', '.join(sidecars)}")
        obtained[name] = sidecars
    return obtained


def report_source_state(importers, cfg) -> tuple:
    """Report each source's imagefolder, hand-downloaded archives and sidecar inputs.

    Returns ``(checks, fetch_names)`` — the second being the sources a real run would
    actually download, decided exactly as ``import_and_redefine_source`` decides it: a
    non-empty imagefolder short-circuits the import unless ``refresh=redownload``
    removes it first — or a sidecar input it lacks makes it fetch regardless.
    """
    checks, fetch_names = [], []

    for entry, importer in importers:
        name = entry["name"]
        imagefolder = Path(importer.imagefolder_dir)
        exists = imagefolder.exists() and bool(os.listdir(imagefolder))
        state = f"{len(os.listdir(imagefolder))} categories" if exists else "absent/empty -> would be imported"
        removal = " (would be REMOVED first)" if cfg.refresh == "redownload" and imagefolder.exists() else ""

        logger.info(f"╰─ {name:16s} {imagefolder} [{state}]{removal}")

        # A source that needs a hand-downloaded archive only fails once the run reaches
        # it, which on a full build can be hours in. Report it now, while the whole plan
        # is on screen and nothing has been downloaded yet.
        if not exists or cfg.refresh == "redownload":
            fetch_names.append(name)
            missing = importer.missing_manual_downloads()
            if missing:
                for line in importer.manual_download_instructions().splitlines():
                    logger.warning(f"   {line}")
                wanted = f"{len(missing)} archive(s) must be downloaded by hand: {missing}"
                checks.append(Check(f"manual:{name}", False, wanted))

        # Inputs a source needs on EVERY run, imagefolder or not (FREPJ's md5-pinned geodata
        # tables). Absent ones are not a failure — the run fetches them before its first
        # import — but they make the source a fetcher, so `check_downloads=needed` probes
        # it. A bundled one that is gone cannot be repaired by any run, so it blocks.
        targets = importer.sidecar_targets()
        if targets:
            gone = [location for kind, location in targets if kind == "bundled" and not Path(location).exists()]
            for location in gone:
                detail = (
                    f"{location} — not on disk, and it ships with the package rather than being downloaded: "
                    "restore the checkout"
                )
                checks.append(Check(f"sidecars:{name}", False, detail))

            absent = importer.missing_sidecars()
            if absent:
                if name not in fetch_names:
                    fetch_names.append(name)
                listed = ", ".join(path.name for path in absent)
                logger.info(f"   sidecars: {len(absent)} absent/unverified -> would be fetched: {listed}")
                fetched_into = absent[0].parent
                detail = (
                    f"{len(absent)} sidecar file(s) absent or failing their md5 pin ({listed}) -> would be fetched "
                    f"md5-verified into {fetched_into} before the first import"
                )
                checks.append(Check(f"sidecars:{name}", True, detail))
                for path in absent:
                    if path.exists():
                        drifted = (
                            f"{path.name} is on disk but fails its md5 pin — would be re-fetched before the first "
                            "import (an upstream re-upload, or a truncated copy)"
                        )
                        checks.append(Check(f"sidecars:{name}", False, drifted, blocking=False))
            elif not gone:
                fetched = sum(1 for kind, _ in targets if kind == "url")
                bundled = sum(1 for kind, _ in targets if kind == "bundled")
                detail = f"{fetched} sidecar file(s) on disk with their md5 pin, {bundled} bundled"
                checks.append(Check(f"sidecars:{name}", True, detail))

    return checks, fetch_names


def check_source_downloads(importers, fetch_names, *, scope, timeout, session=None, audit=True) -> tuple:
    """Probe every file a real run would obtain, without obtaining any of it.

    ``scope`` decides the question being asked. ``needed`` answers "can THIS run
    proceed?" and skips the sources whose imagefolder is already built, since those are
    never downloaded. ``all`` answers "is every selected source still downloadable?",
    which is the one worth running on a schedule — upstream archives move, and finding
    out at the start of the next rebuild is late.

    ``audit`` is what keeps ``all`` usable in both roles. When nothing is being built
    (a dry run) every failure blocks, so the audit exits non-zero on any breakage. On a
    REAL run only the sources this run would actually fetch can block it: refusing to
    build because an archive is unreachable for a source whose imagefolder is already on
    disk would stop a run that was going to succeed.

    Returns ``(checks, total_bytes)``. The byte total counts only what would really be
    downloaded, so the free-space check is not inflated by an ``all``-scope audit.
    """
    wanted = set(fetch_names)
    targets = [(entry["name"], importer) for entry, importer in importers if scope == "all" or entry["name"] in wanted]
    skipped = [entry["name"] for entry, _ in importers if scope != "all" and entry["name"] not in wanted]

    checks, total = [], 0

    def _probe(target):
        """Probe one source, converting any crash into a verdict about that source.

        probe_downloads only promises to swallow requests' own exceptions. Anything else
        (a malformed header, a bug here) would propagate out of executor.map and take the
        whole report down with it — losing the verdicts on every other source,
        which is exactly what this pre-flight exists to avoid.
        """
        name, importer = target
        try:
            if session is not None:
                return importer.probe_downloads(timeout=timeout, session=session)
            # One session per source rather than one shared across the pool: requests'
            # Session is not documented as thread-safe, and per-source is where the reuse
            # that matters happens anyway (whoi alone is 9 URLs on one host).
            with requests.Session() as own_session:
                return importer.probe_downloads(timeout=timeout, session=own_session)
        except Exception as e:
            logger.warning(f"Probing «{name}» raised {type(e).__name__}: {e}")
            return e

    if targets:
        # Concurrent because 17 sources spread over 9 hosts are ~28 mostly-idle requests;
        # sequentially that is a minute of waiting for a check meant to be instant.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(targets))) as executor:
            probed = list(executor.map(_probe, targets))
    else:
        probed = []

    for (name, _), results in zip(targets, probed):
        blocking = audit or name in wanted

        if isinstance(results, Exception):
            crashed = f"could not be probed: {type(results).__name__}: {results}"
            checks.append(Check(f"download:{name}", False, crashed, blocking))
            continue

        failures = [result for result in results if not result.ok]
        sizes = [result.size for result in results if result.size]
        if name in wanted:
            total += sum(sizes)

        if failures:
            shown = "; ".join(f"{result.location} — {result.detail}" for result in failures[:2])
            more = f" (+{len(failures) - 2} more)" if len(failures) > 2 else ""
            detail = f"{len(failures)}/{len(results)} unreachable: {shown}{more}"
            if not blocking:
                detail += " (not blocking: this run reuses the imagefolder it already has)"
            checks.append(Check(f"download:{name}", False, detail, blocking))
        else:
            volume = f", {naturalsize(sum(sizes))}" if sizes else ""
            checks.append(Check(f"download:{name}", True, f"{len(results)} target(s) reachable{volume}"))

        checks.extend(
            Check(f"download:{name}", False, f"{result.location} — {result.warning}", blocking=False)
            for result in results
            if result.warning
        )

    if skipped:
        # Named rather than counted: a probe that silently covered 3 of 17 sources reads
        # like a clean bill of health for all 17.
        checks.append(
            Check("downloads-skipped", True, f"imagefolder already built, not probed: {', '.join(skipped)}", blocking=False)
        )

    return checks, total


def run_preflight(*, selected, cfg, base_location, output_dir, taxo_csv_path, version) -> list:
    """Verify that a real run could do what ``log_plan`` just described.

    Every verdict is returned, passes included, in report order. The caller decides what
    to do with them; nothing here raises for a check that failed, because stopping at the
    first problem is what makes a multi-hour build take three attempts to start.
    """
    remote = cfg.check_downloads != "none"
    scope = "network + local checks" if remote else "local checks only (check_downloads=needed probes the network)"
    logger.info(f"PRE-FLIGHT: {scope}. Nothing is downloaded, built or pushed.")

    checks = list(check_taxonomy_csv(taxo_csv_path, selected))

    importers = instantiate_selected(selected, cfg)
    source_checks, fetch_names = report_source_state(importers, cfg)
    checks += source_checks

    estimated_bytes = 0
    if remote and importers:
        download_checks, estimated_bytes = check_source_downloads(
            importers,
            fetch_names,
            scope=cfg.check_downloads,
            timeout=cfg.check_timeout,
            # A dry run builds nothing, so every failure it finds is the answer it was
            # asked for; a real run may only be stopped by a source it would fetch.
            audit=bool(cfg.dry_run),
        )
        checks += download_checks

    if base_location is not None:
        kind, target = base_location
        if kind == "disk":
            checks += check_base_on_disk(target)
        elif remote:
            checks += check_base_on_hub(target, token=cfg.get("hf_token", None), timeout=cfg.check_timeout)

    checks += check_writable_dir("output-dir", output_dir)
    checks += check_writable_dir("data-dir", cfg.data_dir, needed_bytes=estimated_bytes)

    if cfg.get("push_to_hub", False) and remote:
        checks += check_push_target(cfg, version)

    return checks


def log_preflight(checks) -> None:
    """Print the whole verdict table, one line per check, in the banner style."""
    passed = [check for check in checks if check.ok]
    warned = [check for check in checks if not check.ok and not check.blocking]
    failed = [check for check in checks if not check.ok and check.blocking]

    logger.info("=" * 78)
    logger.info(f"PRE-FLIGHT: {len(passed)} ok, {len(warned)} warning(s), {len(failed)} blocking failure(s)")
    for check in checks:
        line = f"[{'ok  ' if check.ok else 'FAIL' if check.blocking else 'WARN'}] {check.name:20s} {check.detail}"
        if check.ok:
            logger.info(line)
        elif check.blocking:
            logger.error(line)
        else:
            logger.warning(line)
    logger.info("=" * 78)


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
    if cfg.check_downloads not in CHECK_SCOPES:
        raise ValueError(f"check_downloads must be one of {CHECK_SCOPES}, got {cfg.check_downloads!r}")
    # `not isinstance(...)` first: `check_timeout=null` reaches here as None, and the
    # comparison alone would answer a mistyped value with a TypeError from inside the
    # guard written to reject it.
    if not isinstance(cfg.check_timeout, (int, float)) or isinstance(cfg.check_timeout, bool) or cfg.check_timeout <= 0:
        raise ValueError(f"check_timeout must be a positive number of seconds, got {cfg.check_timeout!r}")

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

    # The pre-flight runs for a dry run (which is nothing else) and whenever the network
    # checks were asked for on a real run — there, refusing to start beats failing four
    # sources in. A plain run skips it entirely and behaves exactly as it always has.
    if cfg.dry_run or cfg.check_downloads != "none":
        checks = run_preflight(
            selected=selected,
            cfg=cfg,
            base_location=base_location,
            output_dir=output_dir,
            taxo_csv_path=taxo_csv_path,
            version=version,
        )
        log_preflight(checks)

        blocking = [check for check in checks if not check.ok and check.blocking]
        if blocking:
            raise RuntimeError(
                f"Pre-flight found {len(blocking)} blocking problem(s), so nothing was downloaded, built or pushed: "
                + " | ".join(f"{check.name}: {check.detail}" for check in blocking)
            )

        if cfg.dry_run:
            logger.info("DRY RUN complete; nothing was modified.")
            return
        logger.info("Pre-flight passed; starting the real run.")

    lookup = build_taxonomy_lookup(taxo_csv_path)
    sync_dict = build_sync_dict(taxo_csv_path) if (base_location is not None and cfg.sync_taxonomy) else None

    # Every selected importer and redefiner, built BEFORE the first import. Construction
    # is free (compose + dataclass; the CSV lookup is cached) and it is where a source's
    # build-time inputs become known — a sidecar table its redefiner needs on every run,
    # a committed crosswalk that is gone. Obtaining them now means the last source cannot
    # fail hours in on an 8 MB file.
    importers = instantiate_selected(selected, cfg)
    redefiners = {entry["name"]: REDEFINERS[entry["redefiner"]](csv_taxonomies_path=taxo_csv_path) for entry in selected}
    for name, sidecars in ensure_source_sidecars(importers).items():
        redefiners[name].attach_sidecars(sidecars)

    fresh = {}
    for entry, importer in importers:
        name = entry["name"]
        logger.info(f"Start importing dataset «{name}».")

        part = import_and_redefine_source(
            entry,
            data_dir=cfg.data_dir,
            redefiner=redefiners[name],
            num_proc_arg=num_proc_arg,
            refresh=cfg.refresh,
            import_overrides=list(cfg.import_overrides),
            importer=importer,
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
        base = ensure_custom_metadata(base, where="the base dataset")
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
