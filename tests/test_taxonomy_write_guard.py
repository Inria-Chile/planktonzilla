"""
(c) Inria

Regression gate for ``docs/CODE_REVIEW.md`` finding 1.1 — the taxonomy builder that
destroys 644 rows on a no-op re-run.

``build_frepj_taxonomy.write_csv`` documents itself as append-only and idempotent, and was,
while ``frepj`` was the last block in the file. It has not been since ``daplankton`` landed:
the writer copies the byte prefix up to the first ``frepj`` row and appends its own block, so
everything appended AFTER frepj falls outside the copied prefix and is dropped. Re-running the
builder on the committed table turns 2358 rows into 1714 — the 44 daplankton and 600
``tara_pacific_*`` rows — and exits 0.

The reproduction is the first test below; it is what this module exists to keep failing-proof.
The remaining tests cover the two guards that now stand in front of both in-place writers, and
the CLI opt-in they read.

Scope, deliberately narrow: this is step 0.2 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md`` —
make the loss impossible rather than silent. It does NOT repair the writer. On today's table
``write_csv`` now refuses every call, which is the intended state: the builder re-derives a
block that is already committed, so refusing costs no work, and step 6 replaces the
byte-splicing with a key-addressed write API rather than patching this boundary scan.
"""

import csv
import io
from collections import Counter

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.utils import build_frepj_taxonomy as frepj_builder
from planktonzilla.planktonzilla_dataset.utils import build_tara_pacific_taxonomy as tara_builder
from planktonzilla.planktonzilla_dataset.utils import taxonomy_write_guard as write_guard

REAL_CSV = constants.DEFAULT_TAXONOMY_CSV_FILENAME

# The blocks appended after frepj — the rows the unguarded writer discards.
APPENDED_AFTER_FREPJ = {
    "daplankton": 44,
    "tara_pacific_bongo": 137,
    "tara_pacific_decknet": 132,
    "tara_pacific_hsn": 159,
    "tara_pacific_manta": 172,
}


def _rows(text: str) -> list:
    return list(csv.DictReader(io.StringIO(text)))


def _by_dataset(text: str) -> dict:
    return dict(Counter(row["Dataset"] for row in _rows(text)))


@pytest.fixture
def working_copy(tmp_path):
    """A scratch copy of the committed CSV. Never let these tests touch the real file."""
    path = tmp_path / "planktonzilla_taxonomy.csv"
    path.write_text(REAL_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.fixture
def frepj_rows(working_copy):
    """The committed frepj block, read back in the shape ``write_csv`` expects.

    A re-run of the builder produces these same rows, so writing them back is the exact
    "no-op re-run" that loses 644 rows.
    """
    return [
        {column: row[column] for column in frepj_builder.CSV_COLUMNS}
        for row in _rows(working_copy.read_text(encoding="utf-8"))
        if row["Dataset"] == frepj_builder.DATASET_NAME
    ]


def test_the_no_op_rerun_that_destroyed_644_rows_is_now_refused(working_copy, frepj_rows):
    """THE GATE. Re-running the frepj builder must not be able to drop another source's rows.

    Before the guard this call returned None, exited 0 and left 1714 of 2358 rows behind.
    """
    before = working_copy.read_text(encoding="utf-8")
    assert sum(_by_dataset(before)[name] for name in APPENDED_AFTER_FREPJ) == 644

    with pytest.raises(write_guard.TaxonomyWriteRefusedError) as refusal:
        frepj_builder.write_csv(working_copy, frepj_rows, unlocked=True)

    message = str(refusal.value)
    assert "DESTROY 644 row(s)" in message
    for name, count in APPENDED_AFTER_FREPJ.items():
        assert f"{name} ({count} row(s))" in message, f"{name} missing from the refusal message"

    assert working_copy.read_text(encoding="utf-8") == before, "the file was modified despite the refusal"


def test_the_opt_in_flag_alone_does_not_authorise_losing_rows(working_copy, frepj_rows):
    """``--i-know-this-rewrites-the-csv`` widens the blast radius; it never waives integrity.

    The two guards are independent on purpose. There is no flag for the second one, because
    there is no legitimate reason for one builder to move another source's rows.
    """
    for unlocked in (True, False):
        with pytest.raises(write_guard.TaxonomyWriteRefusedError):
            frepj_builder.write_csv(working_copy, frepj_rows, unlocked=unlocked)


def test_an_unauthorised_write_is_refused_before_anything_is_read_or_rendered(working_copy):
    """Without the flag the refusal names the flag, and the file is untouched."""
    before = working_copy.read_text(encoding="utf-8")

    with pytest.raises(write_guard.TaxonomyWriteRefusedError) as refusal:
        frepj_builder.write_csv(working_copy, [], unlocked=False)

    assert write_guard.UNLOCK_FLAG in str(refusal.value)
    assert working_copy.read_text(encoding="utf-8") == before


def test_the_tara_writer_preserves_foreign_rows_and_still_asserts_it(working_copy):
    """The Tara Pacific writer keeps foreign lines, so the same guard passes and it writes.

    This is the control: it shows the guard permits a correct in-place rewrite rather than
    blocking every write, and it holds that writer to its own docstring.
    """
    committed = _rows(working_copy.read_text(encoding="utf-8"))
    tara_rows = [
        {column: row[column] for column in tara_builder.CSV_COLUMNS}
        for row in committed
        if row["Dataset"] in set(tara_builder.DATASET_NAMES)
    ]
    assert len(tara_rows) == 600

    written = tara_builder.append_to_master(tara_rows, working_copy, unlocked=True)

    assert written == 600
    assert working_copy.read_text(encoding="utf-8") == REAL_CSV.read_text(encoding="utf-8"), (
        "a no-op re-run of the Tara Pacific builder is no longer byte-idempotent"
    )


def test_the_tara_writer_also_refuses_without_the_flag(working_copy):
    before = working_copy.read_text(encoding="utf-8")

    with pytest.raises(write_guard.TaxonomyWriteRefusedError):
        tara_builder.append_to_master([], working_copy, unlocked=False)

    assert working_copy.read_text(encoding="utf-8") == before


def test_the_guard_catches_an_edit_that_keeps_the_row_count(working_copy):
    """Counting rows per source would miss this; comparing the lines does not.

    An in-place edit reads as a loss, and rightly so: the committed line is gone from the
    render. The message names the source that lost it rather than the one that gained a
    variant, which is the side a reviewer needs.
    """
    original = working_copy.read_text(encoding="utf-8")
    edited = original.replace("zooscan,eudoxie_Abylopsis tetragona", "zooscan,eudoxie_Abylopsis tetragonx", 1)
    assert edited != original
    assert len(original.splitlines()) == len(edited.splitlines()), "the row count must be unchanged"

    with pytest.raises(write_guard.TaxonomyWriteRefusedError) as refusal:
        write_guard.assert_owns_every_change(original, edited, owner={"frepj"}, tool="test")

    assert "zooscan (1 row(s))" in str(refusal.value)


def test_the_guard_catches_a_lost_header(working_copy):
    """The header is foreign to every builder, so losing it is caught by the same rule."""
    original = working_copy.read_text(encoding="utf-8")
    headerless = original.split("\n", 1)[1]

    with pytest.raises(write_guard.TaxonomyWriteRefusedError):
        write_guard.assert_owns_every_change(original, headerless, owner=set(tara_builder.DATASET_NAMES), tool="test")


def test_the_guard_permits_a_render_that_only_moves_its_own_rows():
    """The permissive case, so the guard is not merely refusing everything."""
    original = "Dataset,Raw_Labels\nfrepj,a\nother,b\n"
    rendered = "Dataset,Raw_Labels\nother,b\nfrepj,a\nfrepj,c\n"

    write_guard.assert_owns_every_change(original, rendered, owner={"frepj"}, tool="test")


def test_a_quoted_field_is_never_mistaken_for_a_dataset_column():
    """WR-04's reasoning, applied to the guard: the first field is parsed, not sliced."""
    original = 'Dataset,Raw_Labels\nzooscan,"frepj,not a dataset"\n'

    with pytest.raises(write_guard.TaxonomyWriteRefusedError):
        write_guard.assert_owns_every_change(original, "Dataset,Raw_Labels\n", owner={"frepj"}, tool="test")


@pytest.mark.parametrize(
    "module",
    [
        "planktonzilla.planktonzilla_dataset.utils.build_frepj_taxonomy",
        "planktonzilla.planktonzilla_dataset.utils.build_tara_pacific_taxonomy",
    ],
)
def test_both_builders_advertise_the_opt_in_flag_on_their_cli(module):
    """The flag has to reach the command line, or the writers are simply unreachable.

    Asserted against the real ``--help`` of each entry point rather than against a parser
    this test builds itself, which would prove only that argparse works.
    """
    import subprocess
    import sys

    help_text = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout

    assert write_guard.UNLOCK_FLAG in help_text


def test_the_flag_round_trips_through_argparse():
    """Absent means locked; present means unlocked. No third state."""
    import argparse

    parser = argparse.ArgumentParser()
    write_guard.add_unlock_argument(parser)

    assert write_guard.is_unlocked(parser.parse_args([])) is False
    assert write_guard.is_unlocked(parser.parse_args([write_guard.UNLOCK_FLAG])) is True
