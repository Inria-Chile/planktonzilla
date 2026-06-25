"""
(c) Inria

Re-sync the taxonomy and external IDs of the published planktonzilla dataset.

Loads the frozen consolidated dataset from the HuggingFace Hub, overwrites its
taxonomy ranks, label/classification extras and external-database ID columns from
the taxonomy CSV (matched per example on ``(dataset, original_label)``), then saves
the result back to disk and, when ``push_to_hub`` is set, pushes it to the Hub.

Only columns that already exist in the dataset are updated; no rows are added or
removed.
"""

import math
from pathlib import Path

import hydra
import pandas as pd
import pyrootutils
from datasets import Dataset, Value, load_dataset
from omegaconf import DictConfig

from planktonzilla.utils.logger import get_pylogger

from .constants import (
    DEFAULT_TAXONOMY_CSV_FILENAME,
    EXTRA_COLS,
    ID_NUM_COLS,
    ID_STR_COLS,
    TAXONOMY_RANKS,
    default_num_proc,
)

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
    """Load the CSV and build the (Dataset, Raw_Labels) -> values-to-update dictionary."""
    logger.info("Loading CSV and preparing dictionary...")
    df = pd.read_csv(csv_path, sep=",")

    # wikidata_ID / ecotaxa_ID: string as is (e.g. "Q3386609" or "274;1231;15123").
    for c in STR_ID_COLS:
        df[c] = df[c].apply(lambda v: str(v) if pd.notna(v) else None)

    # aphia/NCBI/BOLD: the CSV reads them as float (135336.0); we turn them into a
    # string without decimals ("135336"), not int, because the column is saved as string.
    for c in NUMERIC_ID_COLS:
        df[c] = df[c].apply(lambda v: str(int(v)) if pd.notna(v) else None)

    rows = df.set_index(["Dataset", "Raw_Labels"])[SYNC_COLS].to_dict("index")

    # Empty -> None (null): both NaN (float) and blank strings. This is done on the
    # Python dict because at the DataFrame level pandas turns the None back into NaN.
    # The plankton boolean is not affected.
    def to_null(v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    return {key: {col: to_null(val) for col, val in row.items()} for key, row in rows.items()}


def sync_columns(ds: Dataset, sync_dict: dict, num_proc: int) -> Dataset:
    """Update the values of the already-existing columns from the CSV."""
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


@hydra.main(
    version_base="1.3",
    config_path=str(root / "configs"),
    config_name="update_planktonzilla.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point: load, re-sync from the taxonomy CSV, save and optionally push.

    Loads ``cfg.repo_id`` from the Hub, rebuilds the ``(dataset, label) -> values``
    lookup from the taxonomy CSV, re-syncs the taxonomy/ID columns onto every
    example, saves the result to ``cfg.data_dir`` and, when ``cfg.push_to_hub`` is
    true, also pushes it to ``cfg.repo_id`` (visibility from ``push_as_private``,
    token from ``hf_token`` / the ``HF_TOKEN`` env var).
    """
    repo_id = cfg.repo_id
    taxo_csv_path = cfg.taxonomy_csv_path if cfg.get("taxonomy_csv_path") is not None else DEFAULT_TAXONOMY_CSV_FILENAME
    num_proc = cfg.num_proc if cfg.get("num_proc") is not None else default_num_proc()
    output_dir = cfg.data_dir

    logger.info(f"Updating Planktonzilla dataset on {repo_id} with taxonomy CSV {taxo_csv_path}.")

    logger.info(f"Loading dataset {repo_id}.")
    ds = load_dataset(repo_id, split="train")

    sync_dict = build_sync_dict(taxo_csv_path)
    dataset_final = sync_columns(ds, sync_dict, num_proc)

    logger.info(f"Saving dataset to disk ({output_dir})...")
    dataset_final.save_to_disk(output_dir)

    if cfg.get("push_to_hub", False):
        logger.info(f"Pushing updated Planktonzilla dataset to HuggingFace Hub as «{cfg.repo_id}».")
        dataset_final.push_to_hub(cfg.repo_id, private=cfg.get("push_as_private", True), token=cfg.get("hf_token", None))
    else:
        logger.warning("Skipping pushing dataset to HuggingFace Hub, set push_to_hub=True to change this.")

    logger.info("Process finished!")


if __name__ == "__main__":
    main()
