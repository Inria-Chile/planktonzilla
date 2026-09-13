"""
(c) Inria

``stratified_split_by_dataset`` cuts train/val/test independently inside each source.

Both cuts size themselves as a fraction of ``n``, the pre-split total, which is what
makes the second one land correctly — the reserved block IS ``(test+val)*n``, so
``val_frac*n`` of it is exactly the intended validation share. What that arithmetic does
not survive is a SMALL source: rounding produces a requested size of 0, or of the whole
block, and ``train_test_split`` raises ``ValueError`` — which the fallback could not fix,
because it dropped the stratification and kept the size, then re-raised under a warning
saying it was "falling back to unstratified".
"""

import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from datasets import ClassLabel, Dataset, Features, Value

from planktonzilla.planktonzilla_dataset.gen_planktonzilla_only_plankton import (
    _splittable_size,
    stratified_split_by_dataset,
)

LABELS = ["a", "b"]


def make(per_label, dataset_name="src"):
    """A dataset with a real ClassLabel, which is what `stratify_by_column` requires."""
    labels = [index for index, _ in enumerate(LABELS) for _ in range(per_label)]
    return Dataset.from_dict(
        {"label": labels, "dataset": [dataset_name] * len(labels)},
        features=Features({"label": ClassLabel(names=LABELS), "dataset": Value("string")}),
    )


# --- the clamp -----------------------------------------------------------------------


def test_a_size_that_already_works_passes_through_untouched():
    """The whole corpus is this case: the fix must not move the normal answer."""
    assert _splittable_size(20, 100) == 20


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        (0, 10, 1),  # rounded down to nothing -> ValueError before the clamp
        (10, 10, 9),  # the whole block, leaving the other side empty
        (99, 10, 9),
        (0, 1, None),  # one row cannot be cut in two at all
        (0, 0, None),
    ],
)
def test_a_size_that_cannot_work_is_clamped_or_refused(requested, available, expected):
    assert _splittable_size(requested, available) == expected


# --- the split -----------------------------------------------------------------------


def test_the_ordinary_case_splits_to_the_requested_fractions():
    """1000 rows at 10% + 10%: the second cut takes val_frac*n out of the reserved block."""
    train, val, test = stratified_split_by_dataset(make(500), num_proc=1, test_frac=0.1, val_frac=0.1)

    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_every_row_survives_the_split():
    train, val, test = stratified_split_by_dataset(make(500), num_proc=1, test_frac=0.1, val_frac=0.1)

    assert len(train) + len(val) + len(test) == 1000


def test_a_source_too_small_to_reserve_a_block_goes_entirely_to_train():
    """Six rows at 10%+10% reserves int(6*0.2) = 1, which cannot be cut into val and test.

    This used to raise ValueError out of the second cut, several hours into a build.
    """
    train, val, test = stratified_split_by_dataset(make(3), num_proc=1, test_frac=0.1, val_frac=0.1)

    assert len(train) == 6
    assert val is None
    assert test is None


def test_a_source_that_can_reserve_but_not_divide_keeps_a_test_split():
    """20 rows at 10%+10% reserves 4 and wants int(20*0.1) = 2 of them: both splits exist."""
    train, val, test = stratified_split_by_dataset(make(10), num_proc=1, test_frac=0.1, val_frac=0.1)

    assert len(train) + len(val) + len(test) == 20
    assert len(val) == 2 and len(test) == 2


def test_sources_are_split_independently_of_each_other():
    from datasets import concatenate_datasets

    both = concatenate_datasets([make(500, "one"), make(500, "two")])

    train, val, test = stratified_split_by_dataset(both, num_proc=1, test_frac=0.1, val_frac=0.1)

    assert len(train) == 1600 and len(val) == 200 and len(test) == 200
    assert sorted(set(val["dataset"])) == ["one", "two"]
