"""
(c) Inria

Network-free tests for ``docs/PERIALPINE_ZOOSCAN_2024.md``, the assessment of the deposit
named by issue #13 (Recherche Data Gouv ``10.57745/9ZD7JW``).

That deposit is the one source this repository has measured and deliberately NOT joined to
the registry: it publishes a table of 11,079 ZooScan objects and no images, so there is no
imagefolder for an importer to build (see KI-32). Because nothing in ``configs/`` or
``planktonzilla/`` references it, no existing test touches it either — the document *is*
the artifact, and an unpinned document drifts. These tests are what keep it honest:

  (a) ARITHMETIC — the three inventory tables agree with each other and with the totals
      the prose states. The per-label counts, the per-lake counts and the per-date matrix
      are three views of the same 11,079 objects, so any hand edit that touches one of them
      and not the others is caught here rather than read as fact years later.
  (b) THE 20-vs-19 TRAP — the file carries 20 label strings over 19 lineages, because
      ``nauplii_Copepoda`` and ``nauplii<Copepoda`` are one taxon spelled two ways. §6.1
      warns about it; this pins the warning against the table it describes, so the two
      cannot disagree.
  (c) PROVENANCE — the identifiers and integrity values that make the record checkable
      (dataset DOI, file DOI, byte size, MD5) are present verbatim. They were measured
      against the live repository on 2026-09-18; the constants below are a second copy, so
      a typo in either place fails instead of quietly becoming the record.
  (d) CROSS-REFERENCES — the README subsection and the KI-32 entry that point at the
      document still point at it, and still state the same headline count.

Reads three committed markdown files and nothing else. No network, no fixtures, no build.
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

RECORD = root / "docs" / "PERIALPINE_ZOOSCAN_2024.md"
README = root / "README.md"
KNOWN_ISSUES = root / "planktonzilla" / "planktonzilla_dataset" / "utils" / "KNOWN_ISSUES.md"

# The deposit, as measured on 2026-09-18. Kept here as well as in the document on purpose:
# these are the values a reader would re-verify with curl, and two copies that must agree
# beat one copy nobody checks.
TOTAL_OBJECTS = 11_079
N_LABELS = 20
N_LINEAGES = 19
N_DATES = 12
DUPLICATED_LINEAGE_LABELS = {"nauplii_Copepoda", "nauplii<Copepoda"}
LAKE_OBJECTS = {"ANN74": 3_859, "BOU73": 4_149, "LEM74": 3_071}
DATASET_DOI = "10.57745/9ZD7JW"
FILE_DOI = "10.57745/AHK6AU"
FILE_BYTES = "620,557 bytes"
FILE_MD5 = "b3b64ac23ec5a718b2827f8a754edc1b"

# The inventory tables, keyed by their first header cell.
LABEL_TABLE_HEADER = "Label (`annotation`)"


def _tables(markdown):
    """Every pipe table in `markdown`, as (header cells, data rows) with cells stripped.

    A table is a run of lines starting with `|`; the second one is the `---` separator and
    is dropped. Cells keep their markdown (backticks, bold) — the tests that need a number
    call `_number` on them.
    """
    tables, current = [], []
    for line in [*markdown.splitlines(), ""]:
        if line.startswith("|"):
            current.append([cell.strip() for cell in line.strip().strip("|").split("|")])
            continue
        if current:
            tables.append((current[0], current[2:]))
            current = []
    return tables


def _table_with_header(markdown, first_cell):
    """The one table whose first header cell is exactly `first_cell`.

    Exact, not a prefix: §6.5's authority table is also headed `Label`, and matching
    loosely would silently read the wrong one.
    """
    matches = [t for t in _tables(markdown) if t[0][0] == first_cell]
    assert len(matches) == 1, f"expected exactly one table headed {first_cell!r}, found {len(matches)}"
    return matches[0]


def _number(cell):
    """The integer in a cell, ignoring markdown bold and thousands separators."""
    return int(re.sub(r"[^0-9]", "", cell))


@pytest.fixture(scope="module")
def record():
    return RECORD.read_text(encoding="utf-8")


def test_the_record_exists_and_states_its_verdict(record):
    """The document's whole point is the verdict; a rewrite that loses it loses the record."""
    assert "assessed, recorded, *not* a registry source" in record
    assert "one table and no" in record and "images" in record


def test_label_table_counts_sum_to_the_total(record):
    header, rows = _table_with_header(record, LABEL_TABLE_HEADER)
    assert header[1] == "Objects"
    labels = [r for r in rows if not r[0].startswith("**")]
    totals = [r for r in rows if r[0].startswith("**")]

    assert len(labels) == N_LABELS
    assert sum(_number(r[1]) for r in labels) == TOTAL_OBJECTS
    assert len(totals) == 1 and _number(totals[0][1]) == TOTAL_OBJECTS
    assert f"{N_LABELS} label strings, {N_LINEAGES} distinct lineages" in totals[0][2]


def test_two_labels_share_one_lineage_and_the_prose_says_which(record):
    """(b) The 20-vs-19 trap: the table and §6.1 must name the same pair."""
    _, rows = _table_with_header(record, LABEL_TABLE_HEADER)
    labels = [r for r in rows if not r[0].startswith("**")]

    by_lineage = {}
    for label, _count, lineage in labels:
        by_lineage.setdefault(lineage, []).append(label.strip("`"))

    assert len(by_lineage) == N_LINEAGES
    shared = {lineage: names for lineage, names in by_lineage.items() if len(names) > 1}
    assert len(shared) == 1
    ((lineage, names),) = shared.items()
    assert set(names) == DUPLICATED_LINEAGE_LABELS
    assert lineage.endswith("Copepoda>nauplii`")

    # §6.1 names the same pair, with the same counts, as the table just did.
    counts = {r[0].strip("`"): _number(r[1]) for r in labels}
    trap = record.split("**6.1 One taxon, two label strings.**")[1].split("**6.2")[0]
    for name in DUPLICATED_LINEAGE_LABELS:
        assert f"`{name}` ({counts[name]})" in trap


def test_lake_table_matches_the_recorded_per_lake_counts(record):
    header, rows = _table_with_header(record, "Lake")
    assert header[1] == "`lake_id`" and header[2] == "Objects"
    lakes = [r for r in rows if not r[0].startswith("**")]
    totals = [r for r in rows if r[0].startswith("**")]

    assert {r[1].strip("`"): _number(r[2]) for r in lakes} == LAKE_OBJECTS
    assert sum(LAKE_OBJECTS.values()) == TOTAL_OBJECTS
    assert len(totals) == 1 and _number(totals[0][2]) == TOTAL_OBJECTS


def test_date_matrix_agrees_with_the_lake_table(record):
    """(a) The third view of the same objects: 12 dates x 3 lakes, no date shared."""
    header, rows = _table_with_header(record, "Sampling date")
    lake_ids = [cell.strip("`") for cell in header[1:-1]]
    assert lake_ids == sorted(LAKE_OBJECTS)

    dates = [r for r in rows if not r[0].startswith("**")]
    totals = [r for r in rows if r[0].startswith("**")]
    assert len(dates) == N_DATES
    assert len(totals) == 1

    per_lake = dict.fromkeys(lake_ids, 0)
    for row in dates:
        counted = [(lake, cell) for lake, cell in zip(lake_ids, row[1:-1]) if cell != "—"]
        # One lake was sampled per date; the row total is that lake's count.
        assert len(counted) == 1
        lake, cell = counted[0]
        per_lake[lake] += _number(cell)
        assert _number(row[-1]) == _number(cell)

    assert per_lake == LAKE_OBJECTS
    assert [_number(cell) for cell in totals[0][1:-1]] == [LAKE_OBJECTS[lake] for lake in lake_ids]
    assert _number(totals[0][-1]) == TOTAL_OBJECTS


def test_provenance_is_recorded_verbatim(record):
    """(c) What a reader re-verifies with curl: the DOIs, the size and the checksum."""
    for value in (DATASET_DOI, FILE_DOI, FILE_BYTES, FILE_MD5):
        assert value in record, f"{value} is missing from the record"
    assert "etalab-2.0" in record
    # The one command that reproduces the checksum has to name the file endpoint it reads.
    assert "api/access/datafile/708600?format=original" in record


def test_the_readme_points_at_the_record_with_the_same_count():
    """(d) The source table lists what builds; this subsection says why one deposit does not."""
    readme = README.read_text(encoding="utf-8")
    assert "#### Evaluated, not in the registry" in readme
    assert "docs/PERIALPINE_ZOOSCAN_2024.md" in readme
    assert f"{TOTAL_OBJECTS:,}" in readme
    assert DATASET_DOI in readme


def test_ki_32_is_indexed_and_points_at_the_record():
    """(d) The decision log carries the entry, and the entry carries the link."""
    known_issues = KNOWN_ISSUES.read_text(encoding="utf-8")
    assert "| KI-32 | decision log |" in known_issues

    entry = known_issues.split("## KI-32 —")[1]
    assert "docs/PERIALPINE_ZOOSCAN_2024.md" in entry
    assert DATASET_DOI in entry
    assert f"{TOTAL_OBJECTS:,}" in entry
