"""
(c) Inria

Network-free arithmetic guard for ``docs/PERIALPINE_ZOOSCAN_2024.md``, the assessment of
the deposit named by issue #13 (Recherche Data Gouv ``10.57745/9ZD7JW``; see KI-32).

That record's three inventory tables — per label, per lake, per sampling date — are three
views of the SAME 11,079 objects. Editing one of them without the others produces a
document that contradicts itself in a way no reader would catch, and nothing else in the
tree references this source, so no other test would either.

Deliberately narrow: the tables' numbers, and the fact that the README and KI-32 still
point at the file. Prose, headings and wording are NOT pinned — rewording a record should
never fail CI.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import re

import pytest

RECORD_PATH = "docs/PERIALPINE_ZOOSCAN_2024.md"
RECORD = root / RECORD_PATH
README = root / "README.md"
KNOWN_ISSUES = root / "planktonzilla" / "planktonzilla_dataset" / "utils" / "KNOWN_ISSUES.md"

# Measured on the deposit, 2026-09-18. 20 label strings over 19 lineages, because
# `nauplii_Copepoda` and `nauplii<Copepoda` are one taxon spelled two ways.
TOTAL_OBJECTS = 11_079
N_LABELS = 20
N_LINEAGES = 19
N_DATES = 12
LAKE_OBJECTS = {"ANN74": 3_859, "BOU73": 4_149, "LEM74": 3_071}

LABEL_TABLE_HEADER = "Label (`annotation`)"


def _tables(markdown):
    """Every pipe table in `markdown`, as (header cells, data rows), separator dropped."""
    tables, current = [], []
    for line in [*markdown.splitlines(), ""]:
        if line.startswith("|"):
            current.append([cell.strip() for cell in line.strip().strip("|").split("|")])
            continue
        if current:
            tables.append((current[0], current[2:]))
            current = []
    return tables


def _rows(markdown, first_cell):
    """The data rows of the one table headed `first_cell`, split into (body, total row).

    Exact header match, not a prefix: §6.5's authority table is also headed `Label`.
    """
    matches = [t for t in _tables(markdown) if t[0][0] == first_cell]
    assert len(matches) == 1, f"expected exactly one table headed {first_cell!r}, found {len(matches)}"
    rows = matches[0][1]
    body = [r for r in rows if not r[0].startswith("**")]
    totals = [r for r in rows if r[0].startswith("**")]
    assert len(totals) == 1, f"table {first_cell!r} needs exactly one total row"
    return body, totals[0]


def _number(cell):
    """The integer in a cell, ignoring markdown bold and thousands separators."""
    return int(re.sub(r"[^0-9]", "", cell))


@pytest.fixture(scope="module")
def record():
    return RECORD.read_text(encoding="utf-8")


def test_label_table_is_twenty_labels_over_nineteen_lineages(record):
    labels, total = _rows(record, LABEL_TABLE_HEADER)

    assert len(labels) == N_LABELS
    assert len({lineage for _label, _count, lineage in labels}) == N_LINEAGES
    assert sum(_number(r[1]) for r in labels) == TOTAL_OBJECTS
    assert _number(total[1]) == TOTAL_OBJECTS


def test_lake_and_date_tables_count_the_same_objects(record):
    lakes, lake_total = _rows(record, "Lake")
    assert {r[1].strip("`"): _number(r[2]) for r in lakes} == LAKE_OBJECTS
    assert _number(lake_total[2]) == TOTAL_OBJECTS == sum(LAKE_OBJECTS.values())

    dates, date_total = _rows(record, "Sampling date")
    assert len(dates) == N_DATES

    per_lake = dict.fromkeys(LAKE_OBJECTS, 0)
    for row in dates:
        sampled = [(lake, cell) for lake, cell in zip(sorted(LAKE_OBJECTS), row[1:-1]) if cell != "—"]
        # One lake per date, so the row's total is that lake's count.
        assert len(sampled) == 1
        lake, cell = sampled[0]
        per_lake[lake] += _number(cell)
        assert _number(row[-1]) == _number(cell)

    assert per_lake == LAKE_OBJECTS
    assert _number(date_total[-1]) == TOTAL_OBJECTS


def test_the_readme_and_ki_32_still_point_at_the_record():
    assert RECORD.is_file()
    for path in (README, KNOWN_ISSUES):
        assert RECORD_PATH in path.read_text(encoding="utf-8"), f"{path.name} no longer links the record"
