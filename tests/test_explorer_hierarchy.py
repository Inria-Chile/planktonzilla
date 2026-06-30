"""
(c) Inria

Two-environment, network-free tests for the explorer Hierarchy view (Phase 12, HIER-01..04).

Two groups:

* CORE-SAFE (run everywhere, no plotly/gradio): the pure data layer (SC5) — the module
  imports with viz ABSENT (D4); ``shapes.build_hierarchy_table`` over a tiny hand-built CSV
  yields the known ids/values the figure consumes; ``shapes.hierarchy_root_class`` equals an
  INDEPENDENT full-distribution-argmax reference (incl. a dedicated MERGE-case proving the
  b92d7d0 Sankey-bug guardrail — pre-argmax-then-merge would flip the winner — and a
  first-seen tie-break); and ``hierarchy.match_node`` (exact → contains → None).
* EXPLORER-GROUP (``pytest.importorskip("plotly")``/``"gradio"`` — SKIP in core, RUN under the
  explorer group CI job): ``make_hierarchy_figure`` default is a ``go.Sunburst`` with
  ``branchvalues == "total"`` (SC1); ``chart="icicle"`` toggles to ``go.Icicle`` over the SAME
  ids (HIER-02); a matched id sets the trace ``level`` (HIER-03, zoom-to-branch); the color
  toggle rank↔root_class yields per-node colors of length #ids that differ (HIER-04); and a
  ``render`` smoke.

All tests are network-free: the Hierarchy consumes the frozen committed taxonomy CSV via
shapes + the inline fixture; no live HF.
"""

from __future__ import annotations

import importlib
import os

import polars as pl
import pytest

from planktonzilla.explorer import shapes

# Disable gradio telemetry/analytics so render() stays network-free. Set before any gradio
# import in the explorer-group tests (the core tests never import gradio at all).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# --------------------------------------------------------------------------- #
# Tiny hand-built CSV fixture (reuses test_explorer_shapes.py rows so the hierarchy
# counts are deterministic and hand-verifiable). 19-column frozen-CSV header order.
# --------------------------------------------------------------------------- #
HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,living,root_class,qualifier,"
    "wikidata_ID,aphia_ID,NCBI_ID,BOLD_ID,ecotaxa_ID"
)

# dsA: 3 rows (animalia x3); dsB: 3 rows (chromista x1, animalia x2).
ROW_FIELDS = [
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
        "detritus",
        "part",
        "Q3",
        "7.0",
        "8.0",
        "9.0",
        "12",
    ),
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
        "inert",
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


def _reference_root_class(df: pl.DataFrame, ranks: tuple[str, ...]) -> dict[str, str]:
    """Independent full-distribution-argmax reference (no shapes/sankey internals reused).

    Walks each row left-to-right over present ranks (stop at first blank rank), tallies the
    FULL root_class distribution per "/"-joined prefix, then argmaxes ONCE with a strict-`>`,
    first-seen tie-break — the JS-faithful semantics the helper must match.
    """
    present = [r for r in ranks if r in df.columns]
    rc_col = df.get_column("root_class").to_list()
    cols = [df.get_column(r).to_list() for r in present]
    tallies: dict[str, dict[str, int]] = {}
    for ri in range(df.height):
        rc = (rc_col[ri] or "").strip() or "Unknown"
        parts: list[str] = []
        for ci in range(len(present)):
            val = (cols[ci][ri] or "").strip()
            if not val:
                break
            parts.append(val)
            node_id = "/".join(parts)
            t = tallies.setdefault(node_id, {})
            t[rc] = t.get(rc, 0) + 1

    def argmax(counts: dict[str, int]) -> str:
        best_k, best_v = "Unknown", -1
        for k, v in counts.items():
            if v > best_v:
                best_k, best_v = k, v
        return best_k

    return {nid: argmax(t) for nid, t in tallies.items()}


# --------------------------------------------------------------------------- #
# (a) CORE-SAFE: module import + table structure + root_class majority + match_node.
# --------------------------------------------------------------------------- #
def test_hierarchy_module_imports_in_core():
    """The module imports with NO viz at module scope (D4) — proves the lazy seam holds."""
    mod = importlib.import_module("planktonzilla.explorer.hierarchy")
    assert hasattr(mod, "make_hierarchy_figure")
    assert hasattr(mod, "render")
    assert hasattr(mod, "main")
    assert hasattr(mod, "match_node")
    assert hasattr(mod, "node_depth")
    assert hasattr(mod, "build_figure")


def test_hierarchy_table_structure(df):
    """build_hierarchy_table yields aligned-length columns + the known fixture values."""
    table = shapes.build_hierarchy_table(df)
    n = len(table["ids"])
    assert n == len(table["labels"]) == len(table["parents"]) == len(table["values"])
    by_id = dict(zip(table["ids"], table["values"], strict=True))
    # Known fixture counts (parity with test_explorer_shapes.test_hierarchy_ragged_and_no_orphans).
    assert by_id["animalia"] == 5
    assert by_id["animalia/cnidaria"] == 3
    assert by_id["chromista"] == 1
    # No orphans: every non-root parent is itself an id (the figure requires this).
    ids = set(table["ids"])
    for parent in table["parents"]:
        if parent:
            assert parent in ids


def test_hierarchy_root_class_full_distribution_matches_reference(df):
    """shapes.hierarchy_root_class == an independent full-distribution-argmax reference (HIER-04)."""
    out = shapes.hierarchy_root_class(df)
    ref = _reference_root_class(df, shapes.TAXONOMY_RANKS)
    assert out == ref
    # Keys align 1:1 with build_hierarchy_table ids (what node_colors consumes).
    table = shapes.build_hierarchy_table(df)
    assert set(out.keys()) == set(table["ids"])
    # Spot-check: animalia rows are living(raw1,raw2,raw5) + detritus(raw3) + inert(raw6)
    # -> {living:3, detritus:1, inert:1} -> 'living'.
    assert out["animalia"] == "living"


def test_hierarchy_root_class_merge_case():
    """The Sankey-bug guardrail (b92d7d0): a node's majority is the argmax of its FULL merged
    distribution, NOT a per-child pre-argmax then merge.

    Fixture: under Kingdom 'k', child 'k/a' carries {X:5, Y:4} and child 'k/b' carries {Y:3}.
    The merged distribution at 'k' is {X:5, Y:7} -> argmax 'Y'. A lossy pre-argmax path
    (a -> 'X', b -> 'Y', then credit each child's full rows) would give X:9, Y:3 -> wrongly 'X'.
    """
    rows = [("k", "a", "X")] * 5 + [("k", "a", "Y")] * 4 + [("k", "b", "Y")] * 3
    df = pl.DataFrame(
        {
            "Kingdom": [r[0] for r in rows],
            "Phylum": [r[1] for r in rows],
            "root_class": [r[2] for r in rows],
        }
    )
    out = shapes.hierarchy_root_class(df)
    assert out["k"] == "Y", f"merge-case must resolve to the merged argmax 'Y', got {out['k']!r}"
    assert out["k/a"] == "X"  # child a's own majority is X (5 > 4)
    assert out["k/b"] == "Y"


def test_hierarchy_root_class_first_seen_tie_break():
    """A tie resolves to the FIRST-seen root_class in row-walk order (delegated to _argmax_key)."""
    # 'k' sees living (2) then artefact (2): tie -> first-seen 'living'.
    rows = [("k", "living")] * 2 + [("k", "artefact")] * 2
    df = pl.DataFrame({"Kingdom": [r[0] for r in rows], "root_class": [r[1] for r in rows]})
    assert shapes.hierarchy_root_class(df)["k"] == "living"
    # Reverse the first-seen order -> 'artefact' now wins the tie.
    rows2 = [("k", "artefact")] * 2 + [("k", "living")] * 2
    df2 = pl.DataFrame({"Kingdom": [r[0] for r in rows2], "root_class": [r[1] for r in rows2]})
    assert shapes.hierarchy_root_class(df2)["k"] == "artefact"


def test_hierarchy_root_class_empty_when_column_absent():
    """No root_class column -> {} so the caller falls back to rank coloring."""
    df = pl.DataFrame({"Kingdom": ["animalia"], "Phylum": ["cnidaria"]})
    assert shapes.hierarchy_root_class(df) == {}


def test_match_node_exact_then_contains_then_none():
    """match_node: exact label (case-insensitive) wins; else contains; empty/no-match -> None."""
    from planktonzilla.explorer import hierarchy

    table = {
        "ids": ["animalia", "animalia/cnidaria", "chromista"],
        "labels": ["animalia", "cnidaria", "chromista"],
        "parents": ["", "animalia", ""],
        "values": [5, 3, 1],
    }
    assert hierarchy.match_node(table, "cnidaria") == "animalia/cnidaria"  # exact
    assert hierarchy.match_node(table, "CNIDARIA") == "animalia/cnidaria"  # case-insensitive
    assert hierarchy.match_node(table, "chrom") == "chromista"  # contains
    assert hierarchy.match_node(table, "anim") == "animalia"  # contains, ids order
    assert hierarchy.match_node(table, "zzz") is None
    assert hierarchy.match_node(table, "") is None
    assert hierarchy.match_node(table, "   ") is None


def test_node_depth_core():
    from planktonzilla.explorer import hierarchy

    assert hierarchy.node_depth("animalia") == 0
    assert hierarchy.node_depth("animalia/cnidaria") == 1
    assert hierarchy.node_depth("animalia/cnidaria/hydrozoa") == 2


# --------------------------------------------------------------------------- #
# (b) EXPLORER-GROUP: figure structure (sunburst/icicle/branchvalues/level/color) + render.
# --------------------------------------------------------------------------- #
def test_make_hierarchy_figure_sunburst_branchvalues_total(df):
    """Default trace is a go.Sunburst with branchvalues == 'total' (SC1, D1)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import hierarchy

    table = shapes.build_hierarchy_table(df)
    fig = hierarchy.make_hierarchy_figure(table)
    assert fig.data and fig.data[0].type == "sunburst"
    assert fig.data[0].branchvalues == "total"
    assert len(fig.data[0].ids) == len(table["ids"])
    assert list(fig.data[0].values) == table["values"]


def test_make_hierarchy_figure_icicle_toggle(df):
    """chart='icicle' yields a go.Icicle over the SAME ids/values (HIER-02)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import hierarchy

    table = shapes.build_hierarchy_table(df)
    fig = hierarchy.make_hierarchy_figure(table, chart="icicle")
    assert fig.data[0].type == "icicle"
    assert fig.data[0].branchvalues == "total"
    assert list(fig.data[0].ids) == table["ids"]


def test_make_hierarchy_figure_level_zoom(df):
    """A matched node id sets the trace level to that id (HIER-03, zoom-to-branch)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import hierarchy

    table = shapes.build_hierarchy_table(df)
    node = hierarchy.match_node(table, "cnidaria")
    assert node == "animalia/cnidaria"
    fig = hierarchy.make_hierarchy_figure(table, level=node)
    assert fig.data[0].level == "animalia/cnidaria"
    # No level -> trace re-roots to the full chart (level unset/None).
    fig_full = hierarchy.make_hierarchy_figure(table)
    assert not fig_full.data[0].level


def test_make_hierarchy_figure_color_toggle(df):
    """color_by rank vs root_class: both len == #ids and differ on at least one node (HIER-04)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import hierarchy

    table = shapes.build_hierarchy_table(df)
    rcbn = shapes.hierarchy_root_class(df)
    rank_colors = hierarchy.node_colors(table, color_by="rank")
    rc_colors = hierarchy.node_colors(table, color_by="root_class", root_class_by_node=rcbn)
    assert len(rank_colors) == len(table["ids"])
    assert len(rc_colors) == len(table["ids"])
    assert rank_colors != rc_colors
    # The figure marker colors reflect the chosen scheme.
    fig_rank = hierarchy.make_hierarchy_figure(table, color_by="rank")
    fig_rc = hierarchy.make_hierarchy_figure(table, color_by="root_class", root_class_by_node=rcbn)
    assert list(fig_rank.data[0].marker.colors) == rank_colors
    assert list(fig_rc.data[0].marker.colors) == rc_colors


def test_build_figure_end_to_end_and_note(df):
    """build_figure resolves query -> level and surfaces a not-found note for unmatched queries."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import hierarchy

    fig, note = hierarchy.build_figure(df, chart="sunburst", color_by="root_class", query="cnidaria")
    assert fig.data[0].type == "sunburst"
    assert fig.data[0].level == "animalia/cnidaria"
    assert note == ""
    # Unmatched, non-empty query -> full chart (no level) + a gentle note.
    fig2, note2 = hierarchy.build_figure(df, query="zzzz-not-a-taxon")
    assert not fig2.data[0].level
    assert note2 and "zzzz-not-a-taxon" in note2


def test_render_smoke(df):
    """render(df) builds a gr.Blocks fragment (network-free)."""
    pytest.importorskip("plotly")
    gr = pytest.importorskip("gradio")
    from planktonzilla.explorer import hierarchy

    fragment = hierarchy.render(df)
    assert isinstance(fragment, gr.Blocks)


def test_make_hierarchy_figure_on_frozen_csv():
    """The default render path reads the frozen committed CSV (network-free, real shape)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import hierarchy

    frozen = shapes.load_taxonomy()
    table = shapes.build_hierarchy_table(frozen)
    fig = hierarchy.make_hierarchy_figure(table)
    assert fig.data[0].type == "sunburst"
    assert fig.data[0].branchvalues == "total"
    assert len(fig.data[0].ids) > 100
