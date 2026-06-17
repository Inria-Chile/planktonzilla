import math
import os

import pandas as pd
from datasets import Dataset, Value, load_dataset

from .constants import (
    EXTRA_COLS,
    ID_NUM_COLS,
    ID_STR_COLS,
    REPO_ID,
    TAXONOMY_CSV_FILENAME,
    TAXONOMY_RANKS,
    default_num_proc,
)

# Configuration
# On-disk copy of the re-synced dataset, written to the shared storage space.
OUTPUT_DIR = "/home/acontreras/group_storage_rennes/acontreras/planktonzilla_17M_updated"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "data", TAXONOMY_CSV_FILENAME)

# Taxonomy columns that get re-synced (seven ranks + label/classification extras).
TAXO_COLS = list(TAXONOMY_RANKS) + list(EXTRA_COLS)

# ID columns from external databases. All are stored as string.
STR_ID_COLS = list(ID_STR_COLS)  # already come as string in the CSV
NUMERIC_ID_COLS = list(ID_NUM_COLS)  # come as float in the CSV -> string without decimals
ID_COLS = STR_ID_COLS + NUMERIC_ID_COLS

# All the columns to update. They already exist in the dataset.
SYNC_COLS = TAXO_COLS + ID_COLS


def build_sync_dict(csv_path: str) -> dict:
    """Load the CSV and build the (Dataset, Raw_Labels) -> values-to-update dictionary."""
    print("Loading CSV and preparing dictionary...")
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


def sync_columns(ds: Dataset, sync_dict: dict) -> Dataset:
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

    print("Updating columns...")
    return ds.map(
        update_example,
        num_proc=default_num_proc(),
        features=new_features,
        desc="Re-syncing taxonomy and external IDs",
    )


def main() -> None:
    """Load the dataset, re-sync taxonomy/ID columns from the CSV, and save it."""
    print(f"Loading dataset {REPO_ID}...")
    ds = load_dataset(REPO_ID, split="train")

    sync_dict = build_sync_dict(CSV_PATH)
    dataset_final = sync_columns(ds, sync_dict)

    print(f"Saving dataset to disk ({OUTPUT_DIR})...")
    dataset_final.save_to_disk(OUTPUT_DIR)

    print("\nProcess finished!")


if __name__ == "__main__":
    main()
