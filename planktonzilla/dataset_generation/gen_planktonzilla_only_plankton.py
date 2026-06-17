"""
(c) Inria
"""

import argparse
import logging
import os
from collections import Counter

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)

from planktonzilla.utils.logger import get_pylogger

from .constants import REPO_ID, TAXONOMY_RANKS, default_num_proc

logger = get_pylogger(__name__)

# Configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "planktonzilla_17M_only_plankton")

SEED = 42
TEST_FRAC = 0.2
VAL_FRAC = 0.2
MIN_CLASS_FREQ = 5  # classes with fewer examples are kept whole in train

# Taxonomy ranks used to build the label, from highest to lowest in the hierarchy.
TAXONOMY_COLS = list(TAXONOMY_RANKS)


def build_only_plankton(ds: Dataset, num_proc: int = 1) -> Dataset:
    """Keep only plankton with taxonomy and encode the taxonomy label."""

    # Plankton mask: marked as plankton and with a Kingdom assigned.
    ds = ds.filter(
        lambda x: x["plankton"] is True and x["Kingdom"] != "",
        num_proc=num_proc,
    )

    # The label is the full taxonomy path, skipping the empty ranks.
    def build_tax_string(example):
        tax = [example[c] for c in TAXONOMY_COLS if example[c] not in ("", None)]
        return {"tax_label": " ".join(tax)}

    ds = ds.map(build_tax_string, num_proc=num_proc)

    # Encode each unique taxonomy string into an integer class.
    unique_labels = sorted(set(ds["tax_label"]))
    class_label = ClassLabel(names=unique_labels)

    def encode_label(example):
        return {"label": class_label.str2int(example["tax_label"])}

    ds = ds.map(encode_label, num_proc=num_proc)

    # Keep only what we need for training.
    ds = ds.remove_columns([c for c in ds.column_names if c not in ["image", "label", "dataset"]])

    ds = ds.cast(
        Features(
            {
                "image": ds.features["image"],
                "label": class_label,
                "dataset": Value("string"),
            }
        )
    )

    return ds


def stratified_split_by_dataset(
    ds: Dataset,
    num_proc: int,
    seed: int = SEED,
    test_frac: float = TEST_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[Dataset, Dataset | None, Dataset | None]:
    """Train/val/test split stratified by dataset and by taxonomy.

    The split is done independently within each source dataset, and within each
    one it is stratified by label. Classes with fewer than MIN_CLASS_FREQ examples
    are sent whole to train.
    """
    train_splits = []
    val_splits = []
    test_splits = []

    for dname in sorted(set(ds["dataset"])):
        ds_sub = ds.filter(lambda x: x["dataset"] == dname, num_proc=num_proc)

        labels = ds_sub["label"]
        counts = Counter(labels)

        # Minority classes: held back for train so we don't lose them in val/test.
        minority = {k for k, v in counts.items() if v < MIN_CLASS_FREQ}
        minority_idx = [i for i, y in enumerate(labels) if y in minority]
        remaining_idx = [i for i, y in enumerate(labels) if y not in minority]

        ds_minority = ds_sub.select(minority_idx) if minority_idx else None
        ds_remaining = ds_sub.select(remaining_idx) if remaining_idx else None

        # If nothing is left to split after removing the minority ones, all goes to train.
        if ds_remaining is None or len(ds_remaining) == 0:
            train_splits.append(ds_sub)
            continue

        n = len(ds_remaining)

        # First cut: train against the block reserved for val + test.
        try:
            splits = ds_remaining.train_test_split(
                test_size=int(n * (test_frac + val_frac)),
                shuffle=True,
                seed=seed,
                stratify_by_column="label",
            )
        except ValueError as e:
            logger.warning(f"Stratified train/val-test split failed for dataset {dname!r}, falling back to unstratified: {e}")
            splits = ds_remaining.train_test_split(
                test_size=int(n * (test_frac + val_frac)),
                shuffle=True,
                seed=seed,
            )

        train_split = splits["train"]
        val_test_split = splits["test"]

        # Second cut: we separate val and test inside the reserved block.
        try:
            splits = val_test_split.train_test_split(
                test_size=int(n * val_frac),
                shuffle=True,
                seed=seed,
                stratify_by_column="label",
            )
        except ValueError as e:
            logger.warning(f"Stratified val/test split failed for dataset {dname!r}, falling back to unstratified: {e}")
            splits = val_test_split.train_test_split(
                test_size=int(n * val_frac),
                shuffle=True,
                seed=seed,
            )

        test_split = splits["train"]
        val_split = splits["test"]

        # Add the reserved minority classes to train.
        if ds_minority is not None:
            train_split = concatenate_datasets([train_split, ds_minority])

        train_splits.append(train_split)
        val_splits.append(val_split)
        test_splits.append(test_split)

    train_ds = concatenate_datasets(train_splits)
    val_ds = concatenate_datasets(val_splits) if val_splits else None
    test_ds = concatenate_datasets(test_splits) if test_splits else None

    return train_ds, val_ds, test_ds


def main() -> None:
    """Load the dataset, keep plankton, stratify-split, and save the DatasetDict."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID, help="HuggingFace Hub dataset repo to load.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to save the split DatasetDict to.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for the stratified split.")
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC, help="Fraction of each dataset reserved for test.")
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC, help="Fraction of each dataset reserved for validation.")
    parser.add_argument("--num-proc", type=int, default=default_num_proc(), help="Number of processes for dataset ops.")
    args = parser.parse_args()

    num_proc = args.num_proc

    ds = load_dataset(args.repo_id, split="train")
    ds = build_only_plankton(ds, num_proc=num_proc)

    train_ds, val_ds, test_ds = stratified_split_by_dataset(
        ds,
        num_proc=num_proc,
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )

    dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    dataset.save_to_disk(args.output_dir)
    logger.info("DONE")


if __name__ == "__main__":
    main()
