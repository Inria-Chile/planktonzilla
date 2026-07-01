"""
(c) Inria

Network-free tests for the pure polars data layer of planktonzilla/explorer/taxonomy_explorer.py.

These tests import ONLY the pure helpers (no gradio, no plotly) so they run in the
project test env, which ships polars but not gradio/plotly. We load the module by
file path with importlib (rather than a package import) so this test never triggers
the module's function-local gradio/plotly imports. A tiny hand-built CSV fixture
written to tmp_path exercises every transform with predictable, hand-computed
expectations.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the explorer tool module by absolute path (avoids triggering its lazy viz imports).
_TOOL_PATH = Path(__file__).resolve().parent.parent / "planktonzilla" / "explorer" / "taxonomy_explorer.py"
_spec = importlib.util.spec_from_file_location("taxonomy_explorer", _TOOL_PATH)
taxonomy_explorer = importlib.util.module_from_spec(_spec)
sys.modules["taxonomy_explorer"] = taxonomy_explorer
_spec.loader.exec_module(taxonomy_explorer)

load_taxonomy = taxonomy_explorer.load_taxonomy
apply_filters = taxonomy_explorer.apply_filters
distinct_values = taxonomy_explorer.distinct_values
build_sankey_index = taxonomy_explorer.build_sankey_index
build_hierarchy_table = taxonomy_explorer.build_hierarchy_table
build_table_rows = taxonomy_explorer.build_table_rows
search_table = taxonomy_explorer.search_table

# Full 19-column header in exact CSV order.
HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,living,root_class,qualifier,"
    "wikidata_ID,aphia_ID,NCBI_ID,BOLD_ID,ecotaxa_ID"
)

# Hand-built rows as field tuples (column order matches HEADER). Kept as tuples
# so each line stays well within ruff's 128-char limit; joined to CSV below.
# Fields: Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,
# proposed_label,plankton,living,root_class,qualifier,
# wikidata_ID,aphia_ID,NCBI_ID,BOLD_ID,ecotaxa_ID
ROW_FIELDS = [
    # animalia/cnidaria full depth (plankton True), ds A
    (
        "dsA",
        "raw1",
        "animalia",
        "cnidaria",
        "hydrozoa",
        "siphonophorae",
        "abylidae",
        "abylopsis",
        "tetragona",
        "abylopsis tetragona",
        "True",
        "True",
        "living",
        "full",
        "Q1",
        "1.0",
        "2.0",
        "3.0",
        "10",
    ),
    # animalia/cnidaria duplicate (Dataset,Kingdom) transition -> link count 2; ragged (blank Species)
    (
        "dsA",
        "raw2",
        "animalia",
        "cnidaria",
        "hydrozoa",
        "siphonophorae",
        "abylidae",
        "abylopsis",
        "",
        "abylidae",
        "True",
        "True",
        "living",
        "full",
        "Q2",
        "4.0",
        "5.0",
        "6.0",
        "11",
    ),
    # animalia/arthropoda (plankton False), ds A
    (
        "dsA",
        "raw3",
        "animalia",
        "arthropoda",
        "copepoda",
        "calanoida",
        "",
        "",
        "",
        "copepod",
        "False",
        "True",
        "living",
        "part",
        "Q3",
        "7.0",
        "8.0",
        "9.0",
        "12",
    ),
    # chromista/ochrophyta (plankton True), ds B
    (
        "dsB",
        "raw4",
        "chromista",
        "ochrophyta",
        "bacillariophyceae",
        "",
        "",
        "",
        "",
        "diatom",
        "True",
        "True",
        "living",
        "full",
        "Q4",
        "10.0",
        "11.0",
        "12.0",
        "13",
    ),
    # animalia/cnidaria (plankton True), ds B -> shares cnidaria prefix
    (
        "dsB",
        "raw5",
        "animalia",
        "cnidaria",
        "scyphozoa",
        "",
        "",
        "",
        "",
        "jelly",
        "True",
        "True",
        "living",
        "full",
        "Q5",
        "13.0",
        "14.0",
        "15.0",
        "14",
    ),
    # fully blank taxonomy below Kingdom (ragged at Phylum), plankton False, ds B
    (
        "dsB",
        "raw6",
        "animalia",
        "",
        "",
        "",
        "",
        "",
        "",
        "unknown",
        "False",
        "False",
        "non-living",
        "none",
        "Q6",
        "16.0",
        "17.0",
        "18.0",
        "15",
    ),
]
ROWS = [",".join(fields) for fields in ROW_FIELDS]


@pytest.fixture()
def csv_path(tmp_path):
    p = tmp_path / "tiny_taxonomy.csv"
    p.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    return p


@pytest.fixture()
def df(csv_path):
    return load_taxonomy(csv_path)


def test_load(df):
    assert df.height == 6
    assert df.columns == HEADER.split(",")
    assert len(df.columns) == 19
    # Blank cells normalized to "" (never None).
    species = df.get_column("Species").to_list()
    assert species[1] == ""  # raw2 blank species
    assert None not in species
    assert df.get_column("Phylum").to_list()[5] == ""  # raw6 blank phylum


def test_filter_plankton(df):
    only = apply_filters(df, plankton="plankton")
    assert only.height == 4
    assert set(only.get_column("plankton").to_list()) == {"True"}

    non = apply_filters(df, plankton="non-plankton")
    assert non.height == 2
    assert set(non.get_column("plankton").to_list()) == {"False"}

    assert apply_filters(df, plankton="All").height == 6
    # "True"/"False" aliases behave the same as the friendly labels.
    assert apply_filters(df, plankton="True").height == 4
    assert apply_filters(df, plankton="False").height == 2


def test_filter_kingdom_phylum(df):
    animalia = apply_filters(df, kingdom="animalia")
    assert animalia.height == 5
    assert set(animalia.get_column("Kingdom").to_list()) == {"animalia"}

    chromista = apply_filters(df, kingdom="chromista")
    assert chromista.height == 1

    cnidaria = apply_filters(df, phylum="cnidaria")
    assert cnidaria.height == 3
    assert set(cnidaria.get_column("Phylum").to_list()) == {"cnidaria"}

    # Combined + case-insensitive.
    combo = apply_filters(df, kingdom="ANIMALIA", phylum="Cnidaria")
    assert combo.height == 3


def test_distinct_values(df):
    assert distinct_values(df, "Kingdom") == ["animalia", "chromista"]
    # Blanks are excluded and result is sorted.
    assert distinct_values(df, "Phylum") == ["arthropoda", "cnidaria", "ochrophyta"]


def test_sankey_wellformed(df):
    stages = ["Dataset", "Kingdom", "Phylum"]
    index = build_sankey_index(df, stages)
    n_nodes = len(index["nodes"])
    # Every link endpoint is a valid node index (no orphans / out-of-range).
    for s, t in zip(index["source"], index["target"], strict=True):
        assert 0 <= s < n_nodes
        assert 0 <= t < n_nodes
    # Two-stage selection: total link value == used_rows (every row = one transition).
    two = build_sankey_index(df, ["Dataset", "Kingdom"])
    assert sum(two["value"]) == two["used_rows"] == 6
    # dsA -> animalia happens 3 times (raw1, raw2, raw3) => a link of value 3 exists.
    da_nodes = two["nodes"]
    src_idx = next(i for i, n in enumerate(da_nodes) if n["stage"] == 0 and n["label"] == "dsA")
    tgt_idx = next(i for i, n in enumerate(da_nodes) if n["stage"] == 1 and n["label"] == "animalia")
    link_value = next(
        v for s, t, v in zip(two["source"], two["target"], two["value"], strict=True) if s == src_idx and t == tgt_idx
    )
    assert link_value == 3


def test_sankey_requires_two_stages(df):
    with pytest.raises(ValueError):
        build_sankey_index(df, ["Dataset"])


def test_drop_blank_changes_counts(df):
    stages = ["Kingdom", "Phylum", "Class", "Species"]
    keep = build_sankey_index(df, stages, drop_blank=False)
    drop = build_sankey_index(df, stages, drop_blank=True)
    # Only raw1 has a non-blank Species; drop_blank keeps just that row.
    assert keep["used_rows"] == 6
    assert drop["used_rows"] == 1
    assert drop["used_rows"] < keep["used_rows"]
    assert len(drop["source"]) < len(keep["source"])


def test_min_flow(df):
    stages = ["Dataset", "Kingdom"]
    full = build_sankey_index(df, stages, min_flow=1)
    thresh = build_sankey_index(df, stages, min_flow=2)
    # dsA->animalia (3) survives min_flow=2; singleton links are dropped.
    assert all(v >= 2 for v in thresh["value"])
    assert len(thresh["source"]) < len(full["source"])


def test_hierarchy_counts(df):
    table = build_hierarchy_table(df)
    by_id = dict(zip(table["ids"], table["values"], strict=True))
    # animalia appears in raw1,raw2,raw3,raw5,raw6 => 5.
    assert by_id["animalia"] == 5
    # animalia/cnidaria => raw1,raw2,raw5 => 3.
    assert by_id["animalia/cnidaria"] == 3
    # chromista => only raw4.
    assert by_id["chromista"] == 1
    # Ragged: raw6 stops at animalia (blank Phylum) so no animalia/<blank> node exists.
    assert all("animalia/" not in i or i.split("/")[1] for i in table["ids"])
    # No orphan parents: every non-root parent id is itself an id.
    ids = set(table["ids"])
    for parent in table["parents"]:
        if parent:
            assert parent in ids


def test_table_rows(df):
    columns, rows = build_table_rows(df)
    assert columns[0] == "Dataset"
    assert columns[1] == "Raw_Labels"
    assert "proposed_label" in columns
    assert len(rows) == 6
    assert all(len(r) == len(columns) for r in rows)


def test_search(df):
    _, rows = build_table_rows(df)
    # Case-insensitive substring match.
    hits = search_table(rows, "CNIDARIA")
    assert len(hits) == 3
    # Empty query returns all rows.
    assert len(search_table(rows, "")) == len(rows)
    assert len(search_table(rows, "   ")) == len(rows)
    # No match returns empty.
    assert search_table(rows, "zzz-no-such-token") == []
