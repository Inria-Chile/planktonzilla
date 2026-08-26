"""
(c) Inria

Zero-drift proof for the taxonomy-CSV reader unification.

``generate_planktonzilla`` read the taxonomy CSV with polars and
``update_planktonzilla`` read it with pandas, with independently written
stringification and null handling. That divergence is recorded as KI-7 (mixed
pandas/polars null + dtype handling) and KI-12 (integer IDs serialized as floats).
Both now go through ``generate_planktonzilla.build_taxonomy_lookup``, so they cannot
drift apart again — but collapsing two readers into one is only safe if they agreed
in the first place.

This module pins that they did. ``_legacy_pandas_sync_dict`` below is a VERBATIM copy
of the pandas ``build_sync_dict`` that was deleted from ``update_planktonzilla``; the
tests assert the surviving polars reader reproduces it exactly — same keys, same
values, same Python types — over the real shipped CSV.

Its lasting job is forward-looking. The two readers agree on today's CSV but would
NOT agree on every possible CSV: a numeric-only ``ecotaxa_ID`` column containing
blanks makes polars infer ``Int64`` (``328``) where pandas infers ``float64``
(``"328.0"``) — exactly the KI-12 shape. If a future CSV edit moves a column into
that regime, this test goes red instead of the change silently rewriting ID values in
the published dataset.
"""

import math

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pandas as pd
import pytest

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp
from planktonzilla.planktonzilla_dataset import update_planktonzilla as up

REAL_CSV = constants.DEFAULT_TAXONOMY_CSV_FILENAME


def _legacy_pandas_sync_dict(csv_path) -> dict:
    """VERBATIM copy of the pandas ``build_sync_dict`` deleted from update_planktonzilla.

    Do not "clean up" or refactor this: its value is being an unchanged copy of the
    implementation whose output we are pinning. Kept in the test file precisely so
    production code carries only one reader.
    """
    df = pd.read_csv(csv_path, sep=",")

    for c in up.STR_ID_COLS:
        df[c] = df[c].apply(lambda v: str(v) if pd.notna(v) else None)

    for c in up.NUMERIC_ID_COLS:
        df[c] = df[c].apply(lambda v: str(int(v)) if pd.notna(v) else None)

    rows = df.set_index(["Dataset", "Raw_Labels"])[up.SYNC_COLS].to_dict("index")

    def to_null(v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    return {key: {col: to_null(val) for col, val in row.items()} for key, row in rows.items()}


def test_sync_cols_match_the_generation_lookup_cols():
    """The two paths pull the same 16 columns, in the same order.

    If this drifts, the equivalence assertion below would be comparing different
    column sets and could pass vacuously.
    """
    assert list(gp.LOOKUP_COLS) == list(up.SYNC_COLS)
    assert len(up.SYNC_COLS) == 16


def test_polars_reader_reproduces_the_deleted_pandas_reader_on_the_real_csv():
    """The surviving reader is byte-equivalent to the deleted one on the shipped CSV.

    This is the zero-drift evidence for removing pandas from the re-sync path: over
    all 2314 rows (1485 frozen + 229 appended frepj + 600 appended tara_pacific) and all
    16 synced columns, keys, values AND Python types match.
    """
    new = up.build_sync_dict(REAL_CSV)
    legacy = _legacy_pandas_sync_dict(REAL_CSV)

    assert set(new) == set(legacy), "key sets differ"
    assert len(new) == 1485 + 229 + 600

    value_diffs, type_diffs = [], []
    for key in legacy:
        for col in up.SYNC_COLS:
            a, b = new[key][col], legacy[key][col]
            if a != b:
                value_diffs.append((key, col, a, b))
            elif type(a) is not type(b):
                type_diffs.append((key, col, type(a), type(b)))

    assert not value_diffs, f"{len(value_diffs)} value differences, first 5: {value_diffs[:5]}"
    assert not type_diffs, f"{len(type_diffs)} type differences, first 5: {type_diffs[:5]}"


def test_numeric_ids_are_decimal_free_strings():
    """KI-12's shape, pinned: the CSV stores 135336.0 and the lookup yields "135336"."""
    lookup = gp.build_taxonomy_lookup(REAL_CSV)

    seen = 0
    for row in lookup.values():
        for col in constants.ID_NUM_COLS:
            v = row[col]
            if v is not None:
                assert isinstance(v, str), f"{col} should be str, got {type(v)}"
                assert "." not in v, f"{col} kept a decimal point: {v!r}"
                seen += 1

    assert seen > 3000, "expected thousands of populated numeric IDs; fixture may be wrong"


def test_blank_cells_become_none_not_empty_string():
    """Blank CSV cells resolve to None, never "" or NaN."""
    lookup = gp.build_taxonomy_lookup(REAL_CSV)

    for row in lookup.values():
        for col, v in row.items():
            assert v != "", f"{col} kept an empty string"
            assert not (isinstance(v, float) and math.isnan(v)), f"{col} kept a NaN"


def test_lookup_is_cached_per_path(monkeypatch):
    """Repeated builds read the CSV once, so a 12-source run does not read it 12 times."""
    gp._build_taxonomy_lookup_cached.cache_clear()

    calls = []
    real_read_csv = gp.pl.read_csv

    def counting_read_csv(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("source"))
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(gp.pl, "read_csv", counting_read_csv)

    for _ in range(5):
        gp.build_taxonomy_lookup(REAL_CSV)
    up.build_sync_dict(REAL_CSV)

    assert len(calls) == 1, f"expected 1 CSV read, got {len(calls)}"

    gp._build_taxonomy_lookup_cached.cache_clear()


def test_duplicate_keys_warn_and_keep_the_last_row(tmp_path, caplog):
    """A duplicate (Dataset, Raw_Labels) warns and keeps the last row.

    The deleted pandas reader hard-raised here while the polars one silently kept the
    last row. Unifying had to pick one; this pins the choice (warn + last-wins, so the
    generation path's long-standing behavior is preserved) and makes the condition
    visible instead of silent.
    """
    header = (
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
    )
    rows = [
        "ds,dup,animalia,arthropoda,,,,,,first,True,living,,Q1,1,1.0,1.0,",
        "ds,other,animalia,cnidaria,,,,,,other,True,living,,Q2,2,2.0,2.0,",
        "ds,dup,animalia,mollusca,,,,,,second,False,living,,Q3,3,3.0,3.0,",
    ]
    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n")

    gp._build_taxonomy_lookup_cached.cache_clear()
    with caplog.at_level("WARNING"):
        lookup = gp.build_taxonomy_lookup(csv_path)

    assert "duplicate" in caplog.text.lower()
    assert "ds/dup" in caplog.text

    assert len(lookup) == 2
    assert lookup[("ds", "dup")]["proposed_label"] == "second", "last row should win"
    assert lookup[("ds", "other")]["proposed_label"] == "other"

    gp._build_taxonomy_lookup_cached.cache_clear()


@pytest.mark.parametrize("dataset_name,raw_label", [("x", "y")])
def test_equivalence_on_the_single_row_fixture(tmp_path, dataset_name, raw_label):
    """The two readers also agree on the 1-row fixture the other suites use.

    A single-row CSV is the risky dtype-inference case: whole columns are blank, so
    polars and pandas each have to guess a type with almost no evidence.
    """
    header = (
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
    )
    row = f"{dataset_name},{raw_label},Animalia,Arthropoda,,,,,,Copepoda,True,zoo,,Q3386609,274;1231,135336.0,6854.0,"
    csv_path = tmp_path / "taxo.csv"
    csv_path.write_text(header + "\n" + row + "\n")

    gp._build_taxonomy_lookup_cached.cache_clear()
    new = up.build_sync_dict(csv_path)
    legacy = _legacy_pandas_sync_dict(csv_path)

    assert set(new) == set(legacy)
    for key in legacy:
        for col in up.SYNC_COLS:
            assert new[key][col] == legacy[key][col], f"{col} differs"
            assert type(new[key][col]) is type(legacy[key][col]), f"{col} type differs"

    gp._build_taxonomy_lookup_cached.cache_clear()
