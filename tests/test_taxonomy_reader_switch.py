"""
(c) Inria

Gate for step 5 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md`` — the nine CSV readers become one.

Before this step the repository parsed ``planktonzilla_taxonomy.csv`` in six places on the read
side, each with its own idea of how to stringify an id or represent a blank. The worst of them was
``frepj_validate``, which existed to VALIDATE that published FREPJ rows match the taxonomy and did
so against a hand-written copy of the reader it was checking. That is how two readers drift: the
one place guaranteed to notice was using a second implementation of the thing it compared to.

This module is the zero-drift evidence for deleting them, and it is built the way
``tests/test_taxonomy_lookup_equivalence.py`` was: **each deleted reader is reconstructed here
verbatim**, and the surviving loader is asserted to reproduce it exactly on the committed table.
Do not "clean up" the copies below — their value is being unchanged copies of the implementations
whose output is being pinned.

What this step does NOT do is move a published byte. The wide CSV is still the source of record and
is still committed; a staleness test asserts the package renders it unchanged. That is the property
that makes the step revertable.
"""

import csv
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import polars as pl
import pytest

from planktonzilla.planktonzilla_dataset import constants, frepj_validate, make_planktonzilla
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp
from planktonzilla.planktonzilla_dataset.taxonomy import load_taxonomy, loader
from planktonzilla.planktonzilla_dataset.utils import verify_label_consistency, verify_taxonomy_ids

REAL_CSV = Path(constants.DEFAULT_TAXONOMY_CSV_FILENAME)


@pytest.fixture(scope="module")
def rows():
    return load_taxonomy(REAL_CSV).rows()


# Verbatim copies of the deleted readers
def _deleted_polars_all_string_reader(csv_path):
    """VERBATIM copy of the reader deleted from ``verify_taxonomy_ids.read_csv_rows``."""
    frame = pl.read_csv(csv_path, infer_schema_length=0)
    return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in frame.to_dicts()]


def _deleted_dictreader(csv_path):
    """VERBATIM copy of the reader deleted from ``sankey.main`` and ``verify_label_consistency``."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# Each deleted reader, pinned against its replacement
def test_rows_reproduces_the_deleted_polars_reader(rows):
    """``verify_taxonomy_ids`` compared ids exactly as committed; a typing change would be silent."""
    assert rows == _deleted_polars_all_string_reader(REAL_CSV)


def test_rows_reproduces_the_deleted_dictreader(rows):
    """``sankey`` and ``verify_label_consistency`` both had their own ``csv.DictReader``."""
    assert rows == _deleted_dictreader(REAL_CSV)


def test_the_package_backend_agrees_with_both_row_for_row():
    """The store computed FROM THE MODEL yields the same rows the two deleted readers did.

    Which is what makes the switch safe to land before the package becomes the source of record:
    whichever backend a consumer ends up on, it sees the same rows.
    """
    assert load_taxonomy().rows() == _deleted_dictreader(REAL_CSV)


def test_the_frepj_validator_no_longer_carries_its_own_reader():
    """The copy is gone, not merely unused — an unused copy is one import away from coming back."""
    source = Path(frepj_validate.__file__).read_text(encoding="utf-8")

    assert "def _build_taxonomy_lookup" not in source
    assert "pl.read_csv" not in source
    assert not hasattr(frepj_validate, "_build_taxonomy_lookup")


def test_no_switched_module_kept_a_private_taxonomy_parse():
    """The six readers this step moved must no longer parse the taxonomy themselves.

    Scoped to those six on purpose. A tree-wide grep for ``csv.DictReader`` would be the wrong
    question — ``frepj_tables``, ``frepj_crosswalk``, ``sankey`` and ``extract_cox`` all parse other
    CSVs entirely, and a guard that flags them teaches people to ignore it.

    The three builders still splice the taxonomy byte stream and are NOT switched here: porting them
    onto the write API is step 6, and until then they are held by ``utils/taxonomy_write_guard.py``,
    which is what makes leaving them safe.
    """
    switched = {
        "generate_planktonzilla.py": gp,
        "frepj_validate.py": frepj_validate,
        "utils/verify_taxonomy_ids.py": verify_taxonomy_ids,
        "utils/verify_label_consistency.py": verify_label_consistency,
    }

    offenders = []
    for name, module in switched.items():
        text = Path(module.__file__).read_text(encoding="utf-8")
        if "pl.read_csv(" in text:
            offenders.append(name)

    assert offenders == [], f"still parsing the taxonomy privately: {offenders}"


# The reader of record's surviving surface
def test_build_taxonomy_lookup_is_an_alias_that_kept_its_contract():
    """Every caller keeps its import, and the lookup is unchanged."""
    assert gp.build_taxonomy_lookup(str(REAL_CSV)) == load_taxonomy(REAL_CSV).lookup()
    assert len(gp.build_taxonomy_lookup(str(REAL_CSV))) == 2358


def test_the_three_names_callers_reach_for_survived_the_deletion():
    """``LOOKUP_COLS``, ``_norm`` and the cache handle. Each has a live caller outside this module."""
    assert gp.LOOKUP_COLS == (
        *constants.TAXONOMY_RANKS,
        *constants.EXTRA_COLS,
        *constants.ID_STR_COLS,
        *constants.ID_NUM_COLS,
    )
    assert gp._norm("  ") is None and gp._norm(" x ") == "x"
    # The same cached callable, so the ten `.cache_clear()` call sites reach the loader's cache.
    assert gp._build_taxonomy_lookup_cached is loader.load_cached


def test_the_lookup_is_still_cached_per_resolved_path():
    """Not a detail: a build makes one redefiner per source, and the legacy reader cached for it.

    Dropping the cache while switching the readers would have traded a silent 20x regression for a
    tidier signature.
    """
    loader.cache_clear()
    first = load_taxonomy(REAL_CSV)
    assert load_taxonomy(REAL_CSV) is first, "a second load re-read the taxonomy"
    assert load_taxonomy(str(REAL_CSV)) is first, "a str and a Path missed each other in the cache"

    loader.cache_clear()
    assert load_taxonomy(REAL_CSV) is not first


# Each switched consumer still behaves
def test_the_verifiers_read_through_the_loader(rows):
    assert verify_taxonomy_ids.read_taxonomy(REAL_CSV) == rows
    assert verify_label_consistency.read_rows(REAL_CSV) == rows


def test_the_label_consistency_gate_still_reports_the_pinned_inventory():
    """KI-31's twenty adjudicated disagreements, through the new reader."""
    findings = verify_label_consistency.check_label_consistency(verify_label_consistency.read_rows(REAL_CSV))

    assert verify_label_consistency.summarize(findings)["total"] == 20


def test_check_taxonomy_csv_answers_from_the_store(tmp_path):
    """Rewritten off its bare ``csv.reader`` header peek — two parses of one file became one."""
    selected = [{"name": "zooscan"}, {"name": "frepj"}]
    checks = make_planktonzilla.check_taxonomy_csv(REAL_CSV, selected)

    assert [c.name for c in checks] == ["taxonomy-csv", "taxonomy-coverage"]
    assert all(c.ok for c in checks)
    assert "2358 (dataset, label) rows" in checks[0].detail


def test_check_taxonomy_csv_still_catches_a_missing_column(tmp_path):
    """The check that is not decoration: an absent column resolves to None for every row."""
    path = tmp_path / "short.csv"
    path.write_text("Dataset,Raw_Labels,Kingdom\nsrc,lbl,animalia\n")

    checks = make_planktonzilla.check_taxonomy_csv(path, [])

    assert not checks[0].ok
    assert "is missing the column(s)" in checks[0].detail


def test_check_taxonomy_csv_still_reports_a_missing_file_and_an_unparseable_one(tmp_path):
    absent = make_planktonzilla.check_taxonomy_csv(tmp_path / "nope.csv", [])
    assert not absent[0].ok and "missing:" in absent[0].detail

    junk = tmp_path / "junk.csv"
    junk.write_text("not,a,taxonomy\n1,2,3\n")
    checks = make_planktonzilla.check_taxonomy_csv(junk, [])
    assert not checks[0].ok and "could not be parsed" in checks[0].detail


def test_check_taxonomy_csv_warns_without_blocking_on_an_uncovered_source():
    """Adding a source before curating its labels is a warning, not a failure."""
    checks = make_planktonzilla.check_taxonomy_csv(REAL_CSV, [{"name": "not_a_source"}])
    coverage = next(check for check in checks if check.name == "taxonomy-coverage")

    assert not coverage.ok and not coverage.blocking
    assert "not_a_source" in coverage.detail


# No published byte moved
def test_the_committed_csv_is_not_stale():
    """The wide CSV stays the source of record and stays committed, so it must still be current."""
    assert load_taxonomy().render_wide_csv() == REAL_CSV.read_bytes()


def test_every_switched_reader_also_accepts_the_package_directory():
    """The point of routing through one loader: each of these now reads either form."""
    package_rows = load_taxonomy(loader.PACKAGE_DIR).rows()

    assert verify_label_consistency.read_rows(loader.PACKAGE_DIR) == package_rows
    assert verify_taxonomy_ids.read_taxonomy(loader.PACKAGE_DIR) == package_rows


def test_a_taxonomy_rewritten_at_a_path_already_read_is_not_served_stale(tmp_path):
    """The cache is keyed on content, not just on path — found by a test, not by review.

    Routing EVERY reader through one cache widens the blast radius of the legacy reader's own
    staleness: before this step only the generation path cached, and the verifiers and builders
    write a table and read it back in the same process. A path-only key served the first read to
    the second call and turned a real authority check green on data it had never seen.
    """
    path = tmp_path / "taxonomy.csv"
    header = "Dataset,Raw_Labels,Kingdom,proposed_label,plankton,root_class\n"
    path.write_text(header + "src,lbl,animalia,first,True,living\n")

    assert load_taxonomy(path).lookup()[("src", "lbl")]["proposed_label"] == "first"

    path.write_text(header + "src,lbl,chromista,second,False,detritus\n")

    assert load_taxonomy(path).lookup()[("src", "lbl")]["proposed_label"] == "second"


def test_the_cache_still_serves_one_read_when_the_taxonomy_has_not_changed(tmp_path):
    """The fingerprint must not defeat the cache it protects."""
    loader.cache_clear()
    first = load_taxonomy(REAL_CSV)

    assert load_taxonomy(REAL_CSV) is first
    assert load_taxonomy(loader.PACKAGE_DIR) is load_taxonomy(loader.PACKAGE_DIR)
