"""
(c) Inria

Network-free tests for the pure polars layer of planktonzilla.explorer.shapes (FND-05).

These tests import ONLY the pure layer (``from planktonzilla.explorer import
shapes``) — which proves the layer imports in the core env where gradio/plotly are
absent. A tiny hand-built CSV fixture written to ``tmp_path`` exercises every
transform with predictable, hand-computed expectations covering full-depth lineage,
a ragged/blank rank, a "(blank)" Sankey cell, and at least two datasets. The geo
aggregation is exercised with small measured (uppercase) + inferred (lowercase)
frames to pin the casing reconciliation and na-drop behavior.
"""

import polars as pl
import pytest

from planktonzilla.explorer import shapes

# Full 19-column header in exact CSV order (matches the frozen taxonomy CSV).
HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,living,root_class,qualifier,"
    "wikidata_ID,aphia_ID,NCBI_ID,BOLD_ID,ecotaxa_ID"
)

# Hand-built rows as field tuples (column order matches HEADER). Joined to CSV below.
# Fields: Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,
# proposed_label,plankton,living,root_class,qualifier,
# wikidata_ID,aphia_ID,NCBI_ID,BOLD_ID,ecotaxa_ID
ROW_FIELDS = [
    # dsA animalia/cnidaria full depth.
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
    # dsA animalia/cnidaria again, ragged (blank Species) -> dsA->animalia link count grows.
    (
        "dsA",
        "raw2",
        "animalia",
        "cnidaria",
        "hydrozoa",
        "siphonophorae",
        "abylidae",
        "abylopsis",
        "  ",
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
    # dsA animalia/arthropoda (plankton False), blank Family onward.
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
    # dsB chromista/ochrophyta, ragged at Order.
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
    # dsB animalia/cnidaria (shares cnidaria prefix), ragged at Order.
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
    # dsB ragged at Phylum (blank) -> "(blank)" Sankey cell at the Phylum stage.
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
    return shapes.load_taxonomy(csv_path)


def test_load_normalizes_blanks_and_strips(df):
    assert df.height == 6
    assert df.columns == HEADER.split(",")
    species = df.get_column("Species").to_list()
    assert species[1] == ""  # "  " stripped to ""
    assert None not in species  # blanks normalized to "" (never None)
    assert df.get_column("Phylum").to_list()[5] == ""  # raw6 blank phylum


def test_load_reads_frozen_csv_by_default():
    """The default path reads the frozen committed CSV (D4) with many rows."""
    frozen = shapes.load_taxonomy()
    assert frozen.height > 1000
    assert "Dataset" in frozen.columns and "Species" in frozen.columns


def test_sankey_wellformed_no_orphans(df):
    index = shapes.build_sankey_index(df, ["Dataset", "Kingdom", "Phylum"])
    n_nodes = len(index["nodes"])
    assert n_nodes > 0
    for s, t in zip(index["source"], index["target"], strict=True):
        assert 0 <= s < n_nodes
        assert 0 <= t < n_nodes
    # Two-stage: total link value == used_rows (each row contributes one transition).
    two = shapes.build_sankey_index(df, ["Dataset", "Kingdom"])
    assert sum(two["value"]) == two["used_rows"] == 6
    # dsA -> animalia happens 3 times (raw1, raw2, raw3) -> a link of value 3.
    src = next(i for i, n in enumerate(two["nodes"]) if n["stage"] == 0 and n["label"] == "dsA")
    tgt = next(i for i, n in enumerate(two["nodes"]) if n["stage"] == 1 and n["label"] == "animalia")
    link_value = next(v for s, t, v in zip(two["source"], two["target"], two["value"], strict=True) if s == src and t == tgt)
    assert link_value == 3


def test_sankey_blank_sentinel(df):
    """raw6 is blank at Phylum -> a "(blank)" node exists at the Phylum stage."""
    index = shapes.build_sankey_index(df, ["Kingdom", "Phylum"])
    blank_nodes = [n for n in index["nodes"] if n["stage"] == 1 and n["label"] == shapes.BLANK]
    assert len(blank_nodes) == 1


def test_sankey_requires_two_stages(df):
    with pytest.raises(ValueError):
        shapes.build_sankey_index(df, ["Dataset"])


def test_sankey_missing_stage_raises(df):
    with pytest.raises(ValueError):
        shapes.build_sankey_index(df, ["Dataset", "NotAColumn"])


def test_drop_blank_changes_counts(df):
    stages = ["Kingdom", "Phylum", "Class", "Species"]
    keep = shapes.build_sankey_index(df, stages, drop_blank=False)
    drop = shapes.build_sankey_index(df, stages, drop_blank=True)
    # Only raw1 has a non-blank Species -> drop_blank keeps just that row.
    assert keep["used_rows"] == 6
    assert drop["used_rows"] == 1
    assert len(drop["source"]) < len(keep["source"])


def test_min_flow_drops_small_links(df):
    stages = ["Dataset", "Kingdom"]
    full = shapes.build_sankey_index(df, stages, min_flow=1)
    thresh = shapes.build_sankey_index(df, stages, min_flow=2)
    assert all(v >= 2 for v in thresh["value"])
    assert len(thresh["source"]) < len(full["source"])


def test_hierarchy_ragged_and_no_orphans(df):
    table = shapes.build_hierarchy_table(df)
    by_id = dict(zip(table["ids"], table["values"], strict=True))
    # animalia appears in raw1,raw2,raw3,raw5,raw6 -> 5.
    assert by_id["animalia"] == 5
    # animalia/cnidaria -> raw1,raw2,raw5 -> 3.
    assert by_id["animalia/cnidaria"] == 3
    # chromista -> only raw4.
    assert by_id["chromista"] == 1
    # Ragged: raw6 stops at animalia (blank Phylum); no animalia/<blank> node.
    assert all("animalia/" not in i or i.split("/")[1] for i in table["ids"])
    # No orphan parents: every non-root parent id is itself an id.
    ids = set(table["ids"])
    for parent in table["parents"]:
        if parent:
            assert parent in ids


def test_aggregate_geo_measured_only_collapses_and_drops_null():
    measured = pl.DataFrame(
        {
            "Latitude": [1.0, 1.0, 2.0, None],
            "Longitude": [3.0, 3.0, 4.0, 5.0],
            "dataset": ["a", "a", "b", "c"],
        }
    )
    out = shapes.aggregate_geo(measured)
    # The five legacy columns keep their name/order; `category` is appended 6th (D4).
    assert out.columns == ["dataset", "Latitude", "Longitude", "count", "source", "category"]
    # Near-duplicate (a, 1.0, 3.0) collapses to one row with count 2.
    a_row = out.filter(pl.col("dataset") == "a")
    assert a_row.height == 1 and a_row.get_column("count").to_list()[0] == 2
    # The null-lat row (dataset c) is dropped.
    assert "c" not in out.get_column("dataset").to_list()
    assert set(out.get_column("source").to_list()) == {"measured"}
    # Measured-only -> every row's category is "measured".
    assert set(out.get_column("category").to_list()) == {"measured"}


def test_aggregate_geo_merges_inferred_and_reconciles_casing():
    measured = pl.DataFrame({"Latitude": [10.0], "Longitude": [20.0], "dataset": ["m"]})
    # Lowercase inferred frame (mirrors the committed CSV), with one na row to drop.
    inferred = pl.DataFrame(
        {
            "dataset": ["i1", "i2", "na_ds"],
            "latitude": ["45.5", "59.78", ""],
            "longitude": ["-2.5", "21.37", ""],
            "confidence": ["high", "low", "na"],
        }
    )
    out = shapes.aggregate_geo(measured, inferred)
    # Casing reconciled: output uses uppercase Latitude/Longitude; `category` appended 6th (D4).
    assert out.columns == ["dataset", "Latitude", "Longitude", "count", "source", "category"]
    sources = set(out.get_column("source").to_list())
    assert sources == {"measured", "inferred"}
    # na row dropped; two inferred + one measured survive.
    assert out.filter(pl.col("source") == "inferred").height == 2
    assert out.filter(pl.col("source") == "measured").height == 1
    assert "na_ds" not in out.get_column("dataset").to_list()
    # Graded category: i1=high -> inferred-high, i2=low -> inferred-low, m -> measured; na excluded.
    assert set(out.get_column("category").to_list()) == {"measured", "inferred-high", "inferred-low"}
    assert "inferred-high" in out.filter(pl.col("dataset") == "i1").get_column("category").to_list()
    assert "inferred-low" in out.filter(pl.col("dataset") == "i2").get_column("category").to_list()
