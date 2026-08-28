"""
(c) Inria

The enforced data contract for `planktonzilla_taxonomy.csv`.

Where `tests/test_taxonomy_known_issues.py` PINS defects that are documented but not yet
fixed, this suite asserts the invariants the table is REQUIRED to satisfy. It was
established 2026-08-27, together with the maintainer-directed repair pass that fixed the
2026-08-26 full-table audit findings (KI-29..KI-35 and the KI-8/KI-9/KI-10 data items —
see RESOLVED_ISSUES.md); these tests are what keeps them fixed. Every check is a general
rule over all rows, never a row-specific pin, so a future append is held to the same
contract automatically.

Two deliberate exclusions, so nobody "hardens" them by accident:

- `root_class` is NOT part of the flag-coherence key: a non-living bucket label
  (`other`, `mix`) legitimately spans `detritus` and `artefact` depending on which
  source class was coarsened into it. Per-row coherence is still enforced through
  `living` <=> `root_class == 'living'`.
- Numeric IDs are enforced in the table's existing `"X.0"` serialization, not as bare
  integers: canonicalizing rewrites ~3,700 cells and shifts dtype inference in both
  committed readers, so it stays open as KI-12 until the reader-equivalence suite can be
  run against the change.

Network-free: reads only the committed CSV and `constants`.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import csv
import re
from collections import defaultdict

import pytest

from planktonzilla.planktonzilla_dataset import constants

_CSV_PATH = root / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"

RANKS = constants.TAXONOMY_RANKS
NORMALIZED_COLUMNS = (*RANKS, "proposed_label", "root_class", "qualifier")
ID_COLUMNS = ("wikidata_ID", "aphia_ID", "NCBI_ID", "BOLD_ID", "ecotaxa_ID")
EXPECTED_COLUMNS = (
    "Dataset",
    "Raw_Labels",
    *RANKS,
    "proposed_label",
    "plankton",
    "living",
    "root_class",
    "qualifier",
    *ID_COLUMNS,
)
ROOT_CLASSES = ("living", "detritus", "artefact", "inert")

# The one legitimate exception to the `-aceae` -> Family suffix rule: in Rotifera,
# Flosculariaceae is a valid ORDER name (sister to Ploima and Bdelloidea).
_ACEAE_ORDER_ALLOW = frozenset({"flosculariaceae"})


@pytest.fixture(scope="module")
def rows():
    with _CSV_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        loaded = list(reader)
        assert tuple(reader.fieldnames) == EXPECTED_COLUMNS
        return loaded


def _lowest_filled(row):
    """``(rank, value)`` of the deepest non-empty rank column, or ``None``."""
    filled = [(rank, row[rank].strip().lower()) for rank in RANKS if row[rank].strip()]
    return filled[-1] if filled else None


# --- Row and key integrity ------------------------------------------------------------


def test_join_keys_are_unique_and_rows_are_distinct(rows):
    """(Dataset, Raw_Labels) is the importer join key; a duplicate silently miscounts images."""
    keys = [(r["Dataset"], r["Raw_Labels"]) for r in rows]
    assert len(keys) == len(set(keys))
    payloads = [tuple(r.values()) for r in rows]
    assert len(payloads) == len(set(payloads))


def test_every_dataset_is_registered(rows):
    """A Dataset value outside DATASET_IMPORT_CONFIGS has no importer and no license."""
    assert {r["Dataset"] for r in rows} <= set(constants.DATASET_IMPORT_CONFIGS)


def test_no_stray_whitespace(rows):
    """No cell may carry leading/trailing whitespace; internal runs only in Raw_Labels."""
    offenders = []
    for i, r in enumerate(rows):
        for column, value in r.items():
            if value != value.strip():
                offenders.append((i, column, repr(value)))
            if column != "Raw_Labels" and "  " in value:
                offenders.append((i, column, repr(value)))
    assert offenders == []


# --- Vocabularies and domains ---------------------------------------------------------


def test_boolean_columns_are_boolean(rows):
    assert {r["plankton"] for r in rows} <= {"True", "False"}
    assert {r["living"] for r in rows} <= {"True", "False"}


def test_root_class_vocabulary(rows):
    assert {r["root_class"] for r in rows} <= set(ROOT_CLASSES)


def test_qualifier_vocabulary(rows):
    """An empty cell means "unqualified"; everything else must be a registered qualifier."""
    assert {r["qualifier"] for r in rows} - {""} <= set(constants.QUALIFIERS)


def test_normalized_columns_are_lowercase(rows):
    """Raw_Labels keeps source casing; every derived column is lowercase."""
    offenders = [(i, col, r[col]) for i, r in enumerate(rows) for col in NORMALIZED_COLUMNS if r[col] != r[col].lower()]
    assert offenders == []


def test_id_formats(rows):
    """wikidata `Qd+`; aphia/NCBI/BOLD in the table's `X.0` serialization (KI-12 stays
    open for canonicalization); ecotaxa an integer or `;`-joined integers."""
    float_int = re.compile(r"^\d+\.0$")
    offenders = []
    for i, r in enumerate(rows):
        offenders.extend(
            (i, col, r[col]) for col in ("aphia_ID", "NCBI_ID", "BOLD_ID") if r[col] and not float_int.match(r[col])
        )
        if r["wikidata_ID"] and not re.match(r"^Q\d+$", r["wikidata_ID"]):
            offenders.append((i, "wikidata_ID", r["wikidata_ID"]))
        if r["ecotaxa_ID"] and not re.match(r"^\d+(;\d+)*$", r["ecotaxa_ID"]):
            offenders.append((i, "ecotaxa_ID", r["ecotaxa_ID"]))
    assert offenders == []


# --- The taxonomy graph ---------------------------------------------------------------


def test_each_label_has_one_lineage(rows):
    lineages = defaultdict(set)
    for r in rows:
        lineages[r["proposed_label"].strip().lower()].add(tuple(r[rank].strip().lower() for rank in RANKS))
    conflicting = {label: sorted(values) for label, values in lineages.items() if len(values) > 1}
    assert not conflicting


def test_each_label_has_one_value_per_id_column(rows):
    for col in ID_COLUMNS:
        by_label = defaultdict(set)
        for r in rows:
            if r[col]:
                by_label[r["proposed_label"].strip().lower()].add(r[col])
        conflicting = {label: sorted(values) for label, values in by_label.items() if len(values) > 1}
        assert not conflicting, f"{col}: {conflicting}"


def test_rank_columns_form_a_tree(rows):
    """Every rank value (Kingdom..Genus) sits under exactly ONE upper path — no node with
    two parents, so grouping by any rank column partitions cleanly. Species is exempt:
    epithets legitimately repeat across genera."""
    conflicts = {}
    for depth, rank in enumerate(RANKS[:-1]):
        by_value = defaultdict(set)
        for r in rows:
            value = r[rank].strip().lower()
            if value:
                by_value[value].add(tuple(r[k].strip().lower() for k in RANKS[:depth]))
        for value, paths in by_value.items():
            if len(paths) > 1:
                conflicts[(rank, value)] = sorted(paths)
    assert conflicts == {}


def test_no_rank_ladder_gaps(rows):
    """An empty rank never has a filled rank below it — the label graph stays walkable."""
    offenders = []
    for i, r in enumerate(rows):
        values = [r[rank].strip() for rank in RANKS]
        first_gap = next((index for index, value in enumerate(values) if not value), len(values))
        if any(values[first_gap:]):
            offenders.append((i, r["Dataset"], r["Raw_Labels"], values))
    assert offenders == []


def test_rank_suffix_conventions(rows):
    """Latin rank suffixes land in their own column: `-phyceae` (class) only in Class,
    `-ales` (order) only in Order, `-idae` (animal family) only in Family, `-aceae`
    (plant/protist family) only in Family — except the registered rotifer order."""
    offenders = []
    for i, r in enumerate(rows):
        for rank in RANKS:
            value = r[rank].strip().lower()
            if not value:
                continue
            if value.endswith("phyceae") and rank != "Class":
                offenders.append((i, rank, value))
            elif value.endswith("ales") and rank != "Order":
                offenders.append((i, rank, value))
            elif value.endswith("idae") and rank != "Family":
                offenders.append((i, rank, value))
            elif value.endswith("aceae") and rank != "Family" and not (rank == "Order" and value in _ACEAE_ORDER_ALLOW):
                offenders.append((i, rank, value))
    assert offenders == []


# --- Flags ----------------------------------------------------------------------------


def test_living_agrees_with_root_class(rows):
    mismatched = [
        (i, r["Dataset"], r["Raw_Labels"])
        for i, r in enumerate(rows)
        if (r["living"] == "True") != (r["root_class"] == "living")
    ]
    assert mismatched == []


def test_plankton_implies_living(rows):
    offenders = [
        (i, r["Dataset"], r["Raw_Labels"]) for i, r in enumerate(rows) if r["plankton"] == "True" and r["living"] != "True"
    ]
    assert offenders == []


def test_flags_are_a_function_of_label_and_qualifier(rows):
    """Two rows agreeing on (proposed_label, qualifier) must agree on (plankton, living).
    root_class is deliberately excluded — see the module docstring."""
    flags = defaultdict(set)
    where = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["proposed_label"].strip().lower(), r["qualifier"])
        flags[key].add((r["plankton"], r["living"]))
        where[key].append(i)
    conflicting = {key: (sorted(values), where[key][:10]) for key, values in flags.items() if len(values) > 1}
    assert conflicting == {}


def test_egg_rows_are_plankton_iff_anchored(rows):
    """An egg of a known taxon is ichthyo-/meroplankton; an unattributable egg is not —
    the same rule `build_tara_pacific_taxonomy._flags_for` applies at build time."""
    offenders = []
    for i, r in enumerate(rows):
        if r["qualifier"] != "egg":
            continue
        anchored = any(r[rank].strip() for rank in RANKS)
        if (r["plankton"] == "True") != anchored:
            offenders.append((i, r["Dataset"], r["Raw_Labels"], r["proposed_label"], r["plankton"]))
    assert offenders == []


# --- The label column -----------------------------------------------------------------


def test_label_is_lowest_rank_binomial_or_registered_sub_rank(rows):
    """For a ranked row, `proposed_label` is the lowest filled rank value, the
    Genus+Species binomial, or a label registered in `constants.SUB_RANK_LABELS` whose
    declared parent equals the row's lowest filled rank. Rows with no ranks at all are
    the non-taxonomic buckets (artefact, detritus, egg, other, ...) and are exempt."""
    offenders = []
    for i, r in enumerate(rows):
        lowest = _lowest_filled(r)
        if lowest is None:
            continue
        rank, value = lowest
        label = r["proposed_label"].strip().lower()
        expected = f"{r['Genus'].strip().lower()} {r['Species'].strip().lower()}" if rank == "Species" else value
        if label == expected:
            continue
        if constants.SUB_RANK_LABELS.get(label) == (rank, value):
            continue
        offenders.append((i, r["Dataset"], r["Raw_Labels"], label, (rank, value)))
    assert offenders == []


def test_sub_rank_registry_has_no_dead_entries(rows):
    """Every registered sub-rank label occurs in the CSV with exactly the declared parent."""
    seen = defaultdict(set)
    for r in rows:
        lowest = _lowest_filled(r)
        if lowest is not None:
            seen[r["proposed_label"].strip().lower()].add(lowest)
    for label, parent in constants.SUB_RANK_LABELS.items():
        assert seen.get(label) == {parent}, f"{label!r}: registry says {parent}, CSV has {sorted(seen.get(label, ()))}"
