"""
(c) Inria

Re-sync the taxonomy and external IDs of the published planktonzilla dataset.

Loads the frozen consolidated dataset from the HuggingFace Hub, overwrites its
taxonomy ranks, label/classification extras and external-database ID columns from
the taxonomy CSV (matched per example on ``(dataset, original_label)``), adds the
per-image ``license`` / ``license_url`` provenance columns derived from the
``dataset`` column, then saves the result back to disk and, when ``push_to_hub`` is
set, pushes it to the Hub.

The taxonomy/ID re-sync only updates columns that already exist, and every synced
column must be present — a missing one is an error, not a skip. The license columns
are the one addition, and they are appended without reading the image column at all
(see ``add_license_columns``). No rows are added or removed by either step.

The re-sync is a full-table ``map``: it rewrites every row, taxonomy columns included,
from the current CSV. To add *only* the license columns — leaving the published taxonomy
exactly as it is, and never even reading the CSV — run with ``sync_taxonomy=false``.

DEPRECATED: superseded by ``pz_planktonzilla`` (``make_planktonzilla.py``), which does
this and the from-scratch build and the per-source refresh in one command. This module
stays for one minor release. Two differences when porting an invocation across:
it saves to the bare ``cfg.data_dir`` where ``pz_planktonzilla`` defaults to
``<data_dir>/planktonzilla-17M``, and its ``sync_taxonomy=false`` license-only run is
spelled ``pz_planktonzilla base=hub sources=[] sync_taxonomy=false`` there.
"""

from pathlib import Path

import hydra
import pyrootutils
from datasets import Dataset, Value, load_dataset
from omegaconf import DictConfig

from planktonzilla.utils.logger import get_pylogger

from .constants import (
    DATASET_LICENSES,
    DEFAULT_TAXONOMY_CSV_FILENAME,
    EXTRA_COLS,
    ID_NUM_COLS,
    ID_STR_COLS,
    LICENSE_COLS,
    TAXONOMY_RANKS,
    default_num_proc,
    validate_license_coverage,
)
from .generate_planktonzilla import build_taxonomy_lookup

root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

logger = get_pylogger(__name__)

# Taxonomy columns that get re-synced (seven ranks + label/classification extras).
TAXO_COLS = list(TAXONOMY_RANKS) + list(EXTRA_COLS)

# ID columns from external databases. All are stored as string.
STR_ID_COLS = list(ID_STR_COLS)  # already come as string in the CSV
NUMERIC_ID_COLS = list(ID_NUM_COLS)  # come as float in the CSV -> string without decimals
ID_COLS = STR_ID_COLS + NUMERIC_ID_COLS

# All the columns to update. They already exist in the dataset.
SYNC_COLS = TAXO_COLS + ID_COLS


def build_sync_dict(csv_path: str | Path) -> dict:
    """Load the CSV and build the (Dataset, Raw_Labels) -> values-to-update dictionary.

    Delegates to ``generate_planktonzilla.build_taxonomy_lookup``, the single
    taxonomy-CSV reader for the repo, and narrows its columns to ``SYNC_COLS``.

    This used to be a separate pandas implementation while the generation path used
    polars, so the two could drift on ID stringification and null representation
    (KI-7/KI-12). They are provably value-identical on the shipped CSV — all 16 sync
    columns, all 1485 rows — which ``tests/test_taxonomy_lookup_equivalence.py`` pins
    against a verbatim copy of the pandas version replaced here.

    One deliberate behavior change: a duplicate ``(Dataset, Raw_Labels)`` key used to
    raise from ``DataFrame.set_index(...).to_dict("index")``; it now warns and keeps
    the last row, matching what the generation path has always done. The shipped CSV
    has no duplicates, so nothing changes for it today.
    """
    logger.info("Loading CSV and preparing dictionary...")
    lookup = build_taxonomy_lookup(csv_path)
    return {key: {col: row[col] for col in SYNC_COLS} for key, row in lookup.items()}


def sync_columns(ds: Dataset, sync_dict: dict, num_proc: int, *, unmatched: str = "keep") -> Dataset:
    """Update the values of the already-existing columns from the CSV.

    Args:
        ds: Dataset to re-sync. Must expose ``dataset``, ``original_label`` and every
            column in ``SYNC_COLS``.
        sync_dict: ``(dataset, original_label) -> {column: value}`` from
            :func:`build_sync_dict`.
        num_proc: Workers for the map.
        unmatched: What to do with a row whose key is absent from the CSV.
            ``"keep"`` preserves its current taxonomy and nulls only the external IDs
            (the long-standing behavior of this script). ``"clear"`` nulls the
            taxonomy too, which is what a from-scratch rebuild produces for such a row
            (``_taxonomy_row`` defaults every lookup column to ``None``). Keyword-only
            so the historical 3-positional call shape is unchanged.
    """
    if unmatched not in ("keep", "clear"):
        raise ValueError(f"unmatched must be 'keep' or 'clear', got {unmatched!r}")

    missing = [c for c in ["dataset", "original_label", *SYNC_COLS] if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset is missing columns required to re-sync the taxonomy: {', '.join(missing)}")

    # All the ID columns end up as string.
    new_features = ds.features.copy()
    for c in ID_COLS:
        new_features[c] = Value("string")

    # Text columns that should not be left with empty strings: taxonomy (without the
    # plankton boolean) plus the IDs.
    text_cols = [c for c in TAXO_COLS if c != "plankton"] + ID_COLS

    def update_example(example):
        key = (example["dataset"], example["original_label"])
        updates = sync_dict.get(key)

        if updates is not None:
            for col in SYNC_COLS:
                example[col] = updates[col]
        elif unmatched == "clear":
            # Null everything, matching what a from-scratch rebuild emits for a row
            # whose (dataset, label) has no CSV entry.
            for col in SYNC_COLS:
                example[col] = None
        else:
            # No match in the CSV: we leave the taxonomy as is and null out the IDs.
            for col in ID_COLS:
                example[col] = None

        # Any empty or blank string becomes None (null, not "" or nan).
        for col in text_cols:
            v = example[col]
            if isinstance(v, str) and v.strip() == "":
                example[col] = None

        return example

    logger.info("Updating columns...")
    return ds.map(
        update_example,
        num_proc=num_proc,
        features=new_features,
        desc="Re-syncing taxonomy and external IDs",
    )


def add_license_columns(ds: Dataset) -> Dataset:
    """Add the per-image ``license`` / ``license_url`` columns from the ``dataset`` column.

    Both values are a pure function of the source dataset (``DATASET_LICENSES``), so this
    reads *only* the ``dataset`` column and appends the results with ``add_column``, which
    is a zero-copy Arrow ``concat_tables(axis=1)``. The image column is never materialized.

    That is deliberate and load-bearing on a 17M-image dataset. The alternative — a ``map``
    over the whole example — would decode all 17M images to PIL and rewrite the entire
    table, image bytes included, to a fresh Arrow cache. (It would *not* corrupt them:
    datasets 5.0.1 round-trips the original bytes through ``map``, single- and
    multi-process alike. The cost is the decode and the rewrite, not the encoding.)

    Peak extra memory is the source-name list plus two lists of pointers into the fifteen
    interned license strings — a few GB on the full dataset, well under what the
    taxonomy re-sync already costs.

    Re-runnable: pre-existing license columns are dropped and rebuilt, never duplicated.

    Args:
        ds: Dataset exposing the ``dataset`` column.

    Returns:
        ``ds`` with ``license`` and ``license_url`` appended as ``string`` columns.

    Raises:
        KeyError: If the dataset contains a source with no recorded license.
    """
    source_names = ds["dataset"]
    distinct_sources = set(source_names)
    # Every distinct source must be accounted for before anything is written: an
    # unrecorded one must fail, not ship as a null license on published images.
    validate_license_coverage(distinct_sources)

    already_present = [col for col in LICENSE_COLS if col in ds.column_names]
    if already_present:
        logger.info(f"License column(s) {already_present} already present, rebuilding them.")
        ds = ds.remove_columns(already_present)

    logger.info(f"Adding license columns {list(LICENSE_COLS)} for {len(distinct_sources)} source datasets...")
    for col in LICENSE_COLS:
        # Every recorded value is a non-null str, so Arrow infers Value("string") on
        # its own. Deliberately no follow-up cast(): casting rewrites the entire table,
        # image column included, to reach a type add_column already gives us.
        ds = ds.add_column(col, [DATASET_LICENSES[name][col] for name in source_names])

    return ds


@hydra.main(
    version_base="1.3",
    config_path=str(root / "configs"),
    config_name="update_planktonzilla.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point: load, re-sync from the taxonomy CSV, save and optionally push.

    Loads ``cfg.repo_id`` from the Hub, rebuilds the ``(dataset, label) -> values``
    lookup from the taxonomy CSV, re-syncs the taxonomy/ID columns onto every
    example, adds the ``license`` / ``license_url`` columns, saves the result to
    ``cfg.data_dir`` and, when ``cfg.push_to_hub`` is true, also pushes it to
    ``cfg.repo_id`` (visibility from ``push_as_private``, token from ``hf_token`` /
    the ``HF_TOKEN`` env var).

    Set ``cfg.sync_taxonomy`` false to skip the re-sync entirely and add only the license
    columns — a strictly additive run that never reads the CSV and leaves every published
    taxonomy/ID value exactly as it is. It defaults true, which is the historical behavior.

    Set ``cfg.push_revision`` to publish onto a branch other than the repo default,
    which is how a schema change reaches the Hub without overwriting the revision the
    paper and the released models are pinned to.
    """
    logger.warning(
        "pz_update_planktonzilla is DEPRECATED and will be removed in the next minor "
        "release. Use `pz_planktonzilla` instead — it creates or updates the dataset "
        "with one command. The equivalent of this run is "
        "`pz_planktonzilla base=hub sources=[] output_dir='${data_dir}'` (this script "
        "saves to the bare data_dir; pz_planktonzilla defaults one level down, into "
        "<data_dir>/planktonzilla-17M)."
    )

    repo_id = cfg.repo_id
    taxo_csv_path = cfg.taxonomy_csv_path if cfg.get("taxonomy_csv_path") is not None else DEFAULT_TAXONOMY_CSV_FILENAME
    num_proc = cfg.num_proc if cfg.get("num_proc") is not None else default_num_proc()
    output_dir = cfg.data_dir

    sync_taxonomy = cfg.get("sync_taxonomy", True)

    logger.info(f"Loading dataset {repo_id}.")
    ds = load_dataset(repo_id, split="train")

    if sync_taxonomy:
        logger.info(f"Re-syncing taxonomy and external IDs on {repo_id} from taxonomy CSV {taxo_csv_path}.")
        sync_dict = build_sync_dict(taxo_csv_path)
        dataset_final = sync_columns(ds, sync_dict, num_proc)
    else:
        # The CSV is never even read: no lookup is built, no full-table map runs, and the
        # published taxonomy/ID columns are carried through byte for byte. This makes the
        # run strictly additive — the license columns are the only change.
        logger.info("sync_taxonomy=false: leaving the published taxonomy and external IDs untouched.")
        dataset_final = ds

    # After the re-sync, so the frozen taxonomy/ID path keeps operating on exactly the
    # columns it always has and the license columns simply ride along to disk.
    dataset_final = add_license_columns(dataset_final)

    logger.info(f"Saving dataset to disk ({output_dir})...")
    dataset_final.save_to_disk(output_dir)

    if cfg.get("push_to_hub", False):
        # revision is only forwarded when set, so the default call stays byte-identical.
        push_revision = cfg.get("push_revision", None)
        revision_kwargs = {"revision": push_revision} if push_revision else {}
        target = f"{cfg.repo_id}@{push_revision}" if push_revision else str(cfg.repo_id)
        logger.info(f"Pushing updated Planktonzilla dataset to HuggingFace Hub as «{target}».")
        dataset_final.push_to_hub(
            cfg.repo_id,
            private=cfg.get("push_as_private", True),
            token=cfg.get("hf_token", None),
            **revision_kwargs,
        )
    else:
        logger.warning("Skipping pushing dataset to HuggingFace Hub, set push_to_hub=True to change this.")

    logger.info("Process finished!")


if __name__ == "__main__":
    main()
