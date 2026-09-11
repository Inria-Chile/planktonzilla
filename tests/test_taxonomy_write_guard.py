"""
(c) Inria

Regression gate for ``docs/CODE_REVIEW.md`` finding 1.1 — the taxonomy builder that destroyed 644
rows on a no-op re-run.

``build_frepj_taxonomy.write_csv`` documented itself as append-only and idempotent, and was, while
``frepj`` was the last block in the file. It had not been since ``daplankton`` landed: the writer
copied the byte prefix up to the first ``frepj`` row and appended its own block, so everything
appended AFTER frepj fell outside the copied prefix and was dropped. Re-running the builder on the
committed table turned 2358 rows into 1714 — the 44 daplankton and 600 ``tara_pacific_*`` rows — and
exited 0.

Step 0.2 made that loss impossible rather than silent, with a guard in front of the splice and a
CLI opt-in in front of the guard. Step 6 removed the splice, and this module changed shape with it.
The gate is no longer "the builder refuses"; it is **"the builder runs, and all 644 rows are still
there"** — asserted on the real class-dir fixture through the real write path, on a scratch package
and a scratch CSV.

The opt-in flag is gone with the splice it guarded, as the plan said it would be: there is no byte
range to get wrong now, and ``--apply`` carries the deliberateness the long flag used to. What is
kept is the check that never needed a flag — ``assert_owns_every_change``, which on the new path
should be unreachable and runs anyway, because "this writer keeps foreign lines" was documented of
the frepj writer too, right up until it wasn't.
"""

import csv
import io
import shutil
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
from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR, loader
from planktonzilla.planktonzilla_dataset.utils import build_frepj_taxonomy as frepj_builder
from planktonzilla.planktonzilla_dataset.utils import build_tara_pacific_taxonomy as tara_builder
from planktonzilla.planktonzilla_dataset.utils import taxonomy_write_guard as write_guard

REAL_CSV = constants.DEFAULT_TAXONOMY_CSV_FILENAME

# The blocks appended after frepj — the rows the unguarded writer discarded.
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
def workspace(tmp_path):
    """A scratch package and a scratch CSV. Never let these tests touch the committed files."""
    package = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, package)
    csv_path = tmp_path / "planktonzilla_taxonomy.csv"
    csv_path.write_bytes(REAL_CSV.read_bytes())
    loader.cache_clear()
    yield package, csv_path
    loader.cache_clear()


@pytest.fixture
def frepj_rows(workspace):
    """The committed frepj block, in the shape the builder's own curation produces.

    A re-run of the builder produces these same rows, so writing them back is the exact "no-op
    re-run" that used to lose 644 rows.
    """
    _package, csv_path = workspace
    return [
        {column: row[column] for column in frepj_builder.CSV_COLUMNS}
        for row in _rows(csv_path.read_text(encoding="utf-8"))
        if row["Dataset"] == frepj_builder.DATASET_NAME
    ]


def test_the_no_op_rerun_that_destroyed_644_rows_now_changes_nothing(workspace, frepj_rows):
    """THE GATE. The frepj builder runs to completion and every foreign row is still there.

    Before step 0.2 this returned None, exited 0 and left 1714 of 2358 rows behind. Under the guard
    it raised. Now it writes, and the file it writes is byte-identical to the one it read: the
    builder upserts ``mappings/frepj.tsv`` and the CSV is re-rendered whole from the package, so
    there is no boundary to misjudge.
    """
    package, csv_path = workspace
    before = csv_path.read_text(encoding="utf-8")
    assert sum(_by_dataset(before)[name] for name in APPENDED_AFTER_FREPJ) == 644

    changes = frepj_builder.write_rows(frepj_rows, package_dir=package, csv_path=csv_path, apply=True)

    assert not changes, changes.describe(5)
    after = csv_path.read_text(encoding="utf-8")
    assert _by_dataset(after) == _by_dataset(before)
    assert after == before, "a no-op re-run of the frepj builder is not byte-idempotent"


def test_the_frepj_builder_writes_nothing_without_apply(workspace, frepj_rows):
    """Dry run is the default. The long opt-in flag is gone; ``--apply`` carries the intent."""
    package, csv_path = workspace
    fingerprint = {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()}
    before = csv_path.read_bytes()

    frepj_builder.write_rows(frepj_rows, package_dir=package, csv_path=csv_path)

    assert {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()} == fingerprint
    assert csv_path.read_bytes() == before


def test_a_frepj_rerun_cannot_reach_another_sources_mapping_file(workspace, frepj_rows):
    """Per-source partition, on the files themselves: one mapping file is opened, and it is frepj's.

    The property the byte-range writer could not have. It is not that the builder is careful about
    the other 20 sources — it never names them.
    """
    package, csv_path = workspace
    others = {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if path.stem != "frepj"}

    # A row the committed table does not carry, so the write is real rather than a no-op.
    grown = [*frepj_rows, {**frepj_rows[0], "Raw_Labels": "Arachnida,Trombidiformes,Hydrachnidia,Ge._new,Ge._new"}]
    changes = frepj_builder.write_rows(grown, package_dir=package, csv_path=csv_path, apply=True)

    assert changes and changes.applied
    assert {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if path.stem != "frepj"} == others
    assert _by_dataset(csv_path.read_text(encoding="utf-8"))["frepj"] == len(frepj_rows) + 1


def test_the_tara_builder_writes_its_four_sources_and_only_those(workspace):
    """This builder curates four sources at once, which is a partition, not a filter."""
    package, csv_path = workspace
    committed = _rows(csv_path.read_text(encoding="utf-8"))
    tara_rows = [
        {column: row[column] for column in tara_builder.CSV_COLUMNS}
        for row in committed
        if row["Dataset"] in set(tara_builder.DATASET_NAMES)
    ]
    assert len(tara_rows) == 600
    others = {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if "tara" not in path.stem}

    changes = tara_builder.write_rows(tara_rows, package_dir=package, csv_path=csv_path, apply=True)

    assert not changes, changes.describe(5)
    assert {path: path.read_bytes() for path in (package / "mappings").glob("*.tsv") if "tara" not in path.stem} == others
    assert csv_path.read_bytes() == REAL_CSV.read_bytes(), "a no-op re-run is no longer byte-idempotent"


# render_master — the replacement for every builder's in-place splice
def test_render_master_refuses_a_render_that_would_move_a_foreign_row(workspace):
    """Unreachable on the new path, and it runs anyway. That is the point of asserting an invariant.

    Forced here by claiming ownership of one source while the package has been curated in another,
    which is what a mis-set ``owner`` would look like.
    """
    package, csv_path = workspace
    from planktonzilla.planktonzilla_dataset.taxonomy import write

    write.rename(package, "pzt:000071", "bosmina renamed", apply=True)
    before = csv_path.read_bytes()

    with pytest.raises(write_guard.TaxonomyWriteRefusedError, match="frepj"):
        write_guard.render_master(package, csv_path, owner={"zooscan"}, tool="test")

    assert csv_path.read_bytes() == before, "the file was modified despite the refusal"


def test_render_master_writes_the_package_render_when_ownership_holds(workspace):
    """The permissive case: the render comes from the package and the row count says so."""
    package, csv_path = workspace
    written = write_guard.render_master(package, csv_path, owner={"frepj"}, tool="test")

    assert written == 2358
    assert csv_path.read_bytes() == REAL_CSV.read_bytes()


# assert_owns_every_change — kept from step 0.2, unchanged
def test_the_guard_catches_an_edit_that_keeps_the_row_count(workspace):
    """Counting rows per source would miss this; comparing the lines does not.

    An in-place edit reads as a loss, and rightly so: the committed line is gone from the render.
    The message names the source that lost it rather than the one that gained a variant, which is
    the side a reviewer needs.
    """
    _package, csv_path = workspace
    original = csv_path.read_text(encoding="utf-8")
    edited = original.replace("zooscan,eudoxie_Abylopsis tetragona", "zooscan,eudoxie_Abylopsis tetragonx", 1)
    assert edited != original
    assert len(original.splitlines()) == len(edited.splitlines()), "the row count must be unchanged"

    with pytest.raises(write_guard.TaxonomyWriteRefusedError) as refusal:
        write_guard.assert_owns_every_change(original, edited, owner={"frepj"}, tool="test")

    assert "zooscan (1 row(s))" in str(refusal.value)


def test_the_guard_catches_a_lost_header(workspace):
    """The header is foreign to every builder, so losing it is caught by the same rule."""
    _package, csv_path = workspace
    original = csv_path.read_text(encoding="utf-8")
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
        "planktonzilla.planktonzilla_dataset.utils.resolve_frepj_ids",
    ],
)
def test_no_builder_writes_without_apply_on_its_cli(module):
    """``--apply`` has to reach the command line, and the retired opt-in must not linger beside it.

    Asserted against the real ``--help`` of each entry point rather than against a parser this test
    builds itself, which would prove only that argparse works.
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

    assert "--apply" in help_text
    assert "--i-know-this-rewrites-the-csv" not in help_text, "the retired step-0.2 flag is still advertised"
