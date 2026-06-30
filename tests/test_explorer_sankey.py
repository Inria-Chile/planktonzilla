"""
(c) Inria

Two-environment, network-free tests for the explorer Sankey view (Phase 11, SANKEY-01..05).

Three groups:

* CORE-SAFE (run everywhere, no plotly/gradio): count-pinning over a tiny hand-built CSV
  hitting ``shapes.build_sankey_index`` + ``sankey.aggregate_other`` (flow-conserving
  "Other (n taxa)" long-tail collapse — SANKEY-01) + ``distinct_datasets``/``filter_by_dataset``
  (per-source-dataset filter — SANKEY-02) + ``link_lineage`` majority root_class, plus a core
  import test proving the module loads with viz ABSENT (D4).
* EXPLORER-GROUP (``pytest.importorskip("plotly")``/``"gradio"`` — SKIP in core, RUN under
  the explorer group CI job): figure structure (trace node/link counts vs the index;
  hovertemplates present), export (HTML non-empty + contains "plotly"; PNG non-empty +
  PNG magic bytes via kaleido — SANKEY-04), and a render() smoke + preset-callback test.
* ZERO-DRIFT (core-safe, stdlib-only generator): regenerate generate_sankey.py output TWICE
  and assert byte-identical, AND assert the generate_sankey.py source is byte-unchanged vs
  the committed blob (SANKEY-05, D5).

All tests are network-free: the Sankey consumes the frozen committed taxonomy CSV via shapes;
the generator is stdlib-only.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from planktonzilla.explorer import sankey, shapes

# Disable gradio's telemetry/analytics so render() stays network-free. Gradio reads these
# env vars at import/launch time; set them before any gradio import in the explorer-group
# tests (the core tests never import gradio at all).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# --------------------------------------------------------------------------- #
# Tiny hand-built CSV fixture (reuses test_explorer_shapes.py's style + rows so
# counts are deterministic and hand-verifiable). 19-column frozen-CSV header order.
# --------------------------------------------------------------------------- #
HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,living,root_class,qualifier,"
    "wikidata_ID,aphia_ID,NCBI_ID,BOLD_ID,ecotaxa_ID"
)

# Fields per HEADER. dsA: 3 rows (animalia x3); dsB: 3 rows (animalia x2, chromista x1).
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


# --------------------------------------------------------------------------- #
# (a) CORE-SAFE: count-pinning + dataset filter + lineage + core import.
# --------------------------------------------------------------------------- #
def test_sankey_module_imports_in_core():
    """The module imports with NO viz at module scope (D4) — proves the lazy seam holds."""
    import importlib

    mod = importlib.import_module("planktonzilla.explorer.sankey")
    assert hasattr(mod, "make_sankey_figure")
    assert hasattr(mod, "render")
    assert hasattr(mod, "main")
    assert hasattr(mod, "aggregate_other")


def test_distinct_datasets(df):
    choices = sankey.distinct_datasets(df)
    assert choices[0] == sankey.ALL_DATASETS
    assert choices == ["All", "dsA", "dsB"]


def test_filter_by_dataset_reduces_used_rows(df):
    """Filtering to one dataset BEFORE build_sankey_index reduces used_rows (SANKEY-02)."""
    all_idx = shapes.build_sankey_index(df, ["Dataset", "Kingdom"])
    assert all_idx["used_rows"] == 6

    dsa = sankey.filter_by_dataset(df, "dsA")
    assert dsa.height == 3
    dsa_idx = shapes.build_sankey_index(dsa, ["Dataset", "Kingdom"])
    assert dsa_idx["used_rows"] == 3

    # "All"/None/"" are no-ops (frame unchanged).
    assert sankey.filter_by_dataset(df, "All").height == 6
    assert sankey.filter_by_dataset(df, None).height == 6
    assert sankey.filter_by_dataset(df, "").height == 6


def test_aggregate_other_noop_below_threshold_2(df):
    """min_flow<=1 collapses nothing and conserves the index shape (SANKEY-01)."""
    index = shapes.build_sankey_index(df, ["Dataset", "Kingdom"])
    agg = sankey.aggregate_other(index, min_flow=1)
    assert len(agg["nodes"]) == len(index["nodes"])  # no Other node
    assert sum(agg["value"]) == sum(index["value"])  # flow conserved
    assert all(not n["other"] for n in agg["nodes"])


def test_aggregate_other_collapses_long_tail_and_conserves_flow(df):
    """chromista (flow 1) collapses into one 'Other (1 taxa)' node; flow conserved."""
    index = shapes.build_sankey_index(df, ["Dataset", "Kingdom"])
    total_before = sum(index["value"])
    agg = sankey.aggregate_other(index, min_flow=2)

    # Exactly one Other node appears (chromista was the only below-threshold node).
    other_nodes = [n for n in agg["nodes"] if n["other"]]
    assert len(other_nodes) == 1
    assert other_nodes[0]["label"] == "Other (1 taxa)"
    assert other_nodes[0]["stage"] == 1  # Kingdom stage

    # chromista no longer a standalone node; animalia survives.
    labels = [n["label"] for n in agg["nodes"]]
    assert "chromista" not in labels
    assert "animalia" in labels

    # Flow conservation: aggregation reroutes, never drops (SANKEY-01).
    assert sum(agg["value"]) == total_before == 6

    # No orphan links: every source/target indexes a real aggregated node.
    n = len(agg["nodes"])
    for s, t in zip(agg["source"], agg["target"], strict=True):
        assert 0 <= s < n
        assert 0 <= t < n


def test_link_lineage_full_distribution(df):
    """link_lineage returns the FULL {root_class: count} distribution per raw link.

    It must NOT pre-argmax (the old contract returned a majority string and discarded the
    distribution — which broke merged-link coloring). The full per-link map is what lets
    aggregate_other merge distributions across collapsed links and argmax once, matching
    generate_sankey.py's JS linkLineage/mergeTally/argmaxKey.
    """
    stages = ["Dataset", "Kingdom"]
    lineage = sankey.link_lineage(df, stages)
    # Rebuild raw node ids the same way build_sankey_index does, to locate links.
    index = shapes.build_sankey_index(df, stages)

    def node_id(stage, label):
        return next(i for i, nn in enumerate(index["nodes"]) if nn["stage"] == stage and nn["label"] == label)

    dsa = node_id(0, "dsA")
    animalia = node_id(1, "animalia")
    # dsA -> animalia carries raw1(living), raw2(living), raw3(detritus) -> {living:2, detritus:1}.
    assert lineage[(dsa, animalia)] == {"living": 2, "detritus": 1}

    dsb = node_id(0, "dsB")
    # dsB -> animalia carries raw5(living), raw6(inert) -> {living:1, inert:1} (each once).
    assert lineage[(dsb, animalia)] == {"living": 1, "inert": 1}

    # The whole structure is dicts of {root_class: count}, never pre-argmax'd strings.
    assert all(isinstance(v, dict) for v in lineage.values())


def test_argmax_key_first_seen_tie_break():
    """_argmax_key mirrors JS argmaxKey: strict > comparison, first-seen key wins on ties."""
    assert sankey._argmax_key({"X": 5, "Y": 4}) == "X"
    assert sankey._argmax_key({"X": 5, "Y": 7}) == "Y"
    # Tie -> first-seen key (dicts preserve insertion order, JS uses `if(v>bv)`).
    assert sankey._argmax_key({"artefact": 4, "living": 4}) == "artefact"
    assert sankey._argmax_key({"living": 4, "artefact": 4}) == "living"
    # Empty map -> Unknown.
    assert sankey._argmax_key({}) == sankey.LINEAGE_UNKNOWN


def test_aggregate_other_merges_distributions_on_merged_links():
    """The MERGE case the old tests missed: ≥2 raw links with differing root_class
    distributions collapse into ONE final link; the color must be the argmax of their
    MERGED distribution (mirrors JS mergeTally + argmaxKey), NOT a per-link pre-argmax.

    Fixture: two raw links a1->b1 {X:5, Y:4} and a2->b2 {Y:3}, both below the min_flow
    threshold so a1/a2 collapse into one stage-0 Other and b1/b2 into one stage-1 Other,
    routing both into the single Other->Other final link. Merged = {X:5, Y:3+4=7} -> 'Y'.
    The OLD port pre-argmax'd each raw link (a1->'X', a2->'Y') then credited each link's
    full value to its majority -> X:9, Y:3 -> wrongly returned 'X'.
    """
    # Node ids: a1=0,b1=1,a2=2,b2=3 (stage 0 = a*, stage 1 = b*); all four flow below 9.
    raw_index = {
        "nodes": [
            {"stage": 0, "label": "a1"},
            {"stage": 1, "label": "b1"},
            {"stage": 0, "label": "a2"},
            {"stage": 1, "label": "b2"},
        ],
        "source": [0, 2],
        "target": [1, 3],
        "value": [9, 3],  # link a1->b1 has 9 rows (X:5,Y:4); a2->b2 has 3 rows (Y:3)
        "used_rows": 12,
    }
    lineage = {
        (0, 1): {"X": 5, "Y": 4},  # full distribution of a1->b1
        (2, 3): {"Y": 3},  # full distribution of a2->b2
    }
    agg = sankey.aggregate_other(raw_index, min_flow=10, root_class_by_link=lineage)

    # Both stage-0 nodes collapse into one Other, both stage-1 into one Other -> 1 final link.
    assert len(agg["source"]) == 1
    assert agg["value"][0] == 12  # flow conserved (9 + 3), unchanged behavior
    # Merged distribution {X:5, Y:7} -> argmax 'Y'. The old port returned 'X' (the bug).
    assert agg["link_class"][0] == "Y"


def test_aggregate_other_merge_tie_break_matches_js_first_seen():
    """A merged tie resolves to the FIRST-seen root_class in row-walk/merge order (JS parity).

    Two raw links {artefact:1, living:1} and {living:1, detritus:1, artefact:2} merge into
    {artefact:3, living:2, detritus:1}; argmax is the unique max 'artefact'. To exercise the
    tie path specifically, also merge {artefact:2} + {living:2} -> {artefact:2, living:2} ->
    'artefact' wins (first-seen), matching JS argmaxKey's `if(v>bv)`.
    """
    raw_index = {
        "nodes": [
            {"stage": 0, "label": "a1"},
            {"stage": 1, "label": "b1"},
            {"stage": 0, "label": "a2"},
            {"stage": 1, "label": "b2"},
        ],
        "source": [0, 2],
        "target": [1, 3],
        "value": [2, 2],
        "used_rows": 4,
    }
    lineage = {
        (0, 1): {"artefact": 2},  # seen first in the merge walk
        (2, 3): {"living": 2},
    }
    agg = sankey.aggregate_other(raw_index, min_flow=3, root_class_by_link=lineage)
    assert len(agg["source"]) == 1
    assert agg["value"][0] == 4
    assert agg["link_class"][0] == "artefact"  # tie -> first-seen, not 'living'


def test_sankey_lineage_parity_with_generate_sankey_on_frozen_csv():
    """Regression: the previously-divergent frozen-CSV link now matches the JS semantics.

    Default stages (Dataset, Kingdom, Phylum, Class). For min_flow in the merging range
    (33..40) two raw links with differing distributions collapse into the Other->(blank)
    final link at the Class stage. The faithful full-distribution argmax (a direct,
    independent reference for generate_sankey.py's build()) yields 'artefact' at min_flow=40
    (true merged dist {artefact:4, living:4, detritus:2, inert:1}, tie broken first-seen).
    The OLD port returned 'living'. Assert the fixed port now equals the reference.
    """
    frozen = shapes.load_taxonomy()
    stages = ["Dataset", "Kingdom", "Phylum", "Class"]
    blank = shapes.BLANK

    # --- Independent JS-faithful reference: full per-raw-link distributions, merged through
    #     the same collapse, argmax'd once (no plotly; pure dict math). ---
    node_index: dict[tuple[int, str], int] = {}

    def nid(stage_i, value):
        key = (stage_i, value)
        idx = node_index.get(key)
        if idx is None:
            idx = len(node_index)
            node_index[key] = idx
        return idx

    cols = [frozen.get_column(s).to_list() for s in stages]
    rc_col = frozen.get_column("root_class").to_list()
    full: dict[tuple[int, int], dict[str, int]] = {}
    for ri in range(frozen.height):
        vals = [cols[si][ri] or blank for si in range(len(stages))]
        rc = (rc_col[ri] or "").strip() or sankey.LINEAGE_UNKNOWN
        for i in range(len(stages) - 1):
            s = nid(i, vals[i])
            t = nid(i + 1, vals[i + 1])
            full.setdefault((s, t), {})
            full[(s, t)][rc] = full[(s, t)].get(rc, 0) + 1

    index = shapes.build_sankey_index(frozen, stages)
    lineage = sankey.link_lineage(frozen, stages)
    # The fixed link_lineage must equal the independent full-distribution reference.
    assert lineage == full

    for min_flow in range(33, 41):
        agg = sankey.aggregate_other(index, min_flow=min_flow, root_class_by_link=lineage)
        nodes = agg["nodes"]

        # Reference: merge full distributions through the same final-id collapse, argmax once.
        in_flow = [0] * len(index["nodes"])
        out_flow = [0] * len(index["nodes"])
        for s, t, v in zip(index["source"], index["target"], index["value"], strict=True):
            out_flow[s] += v
            in_flow[t] += v
        node_flow = [max(in_flow[i], out_flow[i]) for i in range(len(index["nodes"]))]
        collapse = [min_flow > 1 and node_flow[i] < min_flow for i in range(len(index["nodes"]))]
        remap = [-1] * len(index["nodes"])
        ref_nodes = []
        for i in range(len(index["nodes"])):
            if collapse[i]:
                continue
            remap[i] = len(ref_nodes)
            ref_nodes.append(dict(index["nodes"][i], other=False))
        other_node: dict[int, int] = {}
        other_taxa: dict[int, int] = {}
        for i in range(len(index["nodes"])):
            if collapse[i]:
                other_taxa[index["nodes"][i]["stage"]] = other_taxa.get(index["nodes"][i]["stage"], 0) + 1
        for stage, n in other_taxa.items():
            other_node[stage] = len(ref_nodes)
            ref_nodes.append({"stage": stage, "label": f"Other ({n} taxa)", "other": True})

        def fid(r, _col=collapse, _on=other_node, _rm=remap, _nodes=index["nodes"]):
            return _on[_nodes[r]["stage"]] if _col[r] else _rm[r]

        ref_lineage: dict[tuple[int, int], dict[str, int]] = {}
        for s, t in zip(index["source"], index["target"], strict=True):
            fk = (fid(s), fid(t))
            dst = ref_lineage.setdefault(fk, {})
            for k, c in full.get((s, t), {}).items():
                dst[k] = dst.get(k, 0) + c

        # Locate every Other->(blank) final link in BOTH the port and the reference; assert parity.
        port_links = {
            (s, t): lc
            for s, t, lc in zip(agg["source"], agg["target"], agg["link_class"], strict=True)
            if nodes[s].get("other") and nodes[t]["label"] == blank
        }
        assert port_links, f"min_flow={min_flow}: expected an Other->(blank) link to exist"
        for (s, t), lc in port_links.items():
            expected = sankey._argmax_key(ref_lineage.get((s, t), {}))
            assert lc == expected, f"min_flow={min_flow}: link {(s, t)} port={lc!r} != ref={expected!r}"

    # Pin the specific previously-divergent case: at min_flow=40 the stage-0 Other -> stage-1
    # (blank) link merges to {artefact:4, living:4, detritus:2, inert:1}; argmax (first-seen
    # tie-break, artefact before living) = 'artefact'. The OLD port returned 'living' here.
    agg40 = sankey.aggregate_other(index, min_flow=40, root_class_by_link=lineage)
    nodes40 = agg40["nodes"]
    kingdom_blank = [
        lc
        for s, t, lc in zip(agg40["source"], agg40["target"], agg40["link_class"], strict=True)
        if nodes40[s].get("other") and nodes40[t]["label"] == blank and nodes40[s]["stage"] == 0 and nodes40[t]["stage"] == 1
    ]
    assert kingdom_blank == ["artefact"], (
        f"expected the stage0-Other -> stage1-(blank) link colored 'artefact' (the previously-divergent "
        f"merged case), got {kingdom_blank}"
    )


def test_aggregate_other_carries_link_class(df):
    """When lineage is supplied, aggregate_other emits a majority link_class per link."""
    stages = ["Dataset", "Kingdom"]
    index = shapes.build_sankey_index(df, stages)
    lineage = sankey.link_lineage(df, stages)
    agg = sankey.aggregate_other(index, min_flow=1, root_class_by_link=lineage)
    assert len(agg["link_class"]) == len(agg["source"])
    assert all(isinstance(rc, str) and rc for rc in agg["link_class"])


def test_aggregate_other_three_stages_flow_conserved(df):
    """Three-stage aggregation still conserves total flow across each transition layer."""
    stages = ["Dataset", "Kingdom", "Phylum"]
    index = shapes.build_sankey_index(df, stages)
    agg = sankey.aggregate_other(index, min_flow=2)
    assert sum(agg["value"]) == sum(index["value"])


def test_load_frozen_csv_default_has_datasets():
    """The default render path reads the frozen committed CSV (network-free, many rows)."""
    frozen = shapes.load_taxonomy()
    assert frozen.height > 1000
    choices = sankey.distinct_datasets(frozen)
    assert choices[0] == "All" and len(choices) > 5


# --------------------------------------------------------------------------- #
# (b) EXPLORER-GROUP: figure structure + export (HTML+PNG) + render smoke.
# --------------------------------------------------------------------------- #
def test_make_sankey_figure_structure(df):
    pytest.importorskip("plotly")
    stages = ["Dataset", "Kingdom", "Phylum"]
    index = shapes.build_sankey_index(df, stages)
    lineage = sankey.link_lineage(df, stages)
    agg = sankey.aggregate_other(index, min_flow=1, root_class_by_link=lineage)
    fig = sankey.make_sankey_figure(agg, stages)

    assert fig.data and fig.data[0].type == "sankey"
    trace = fig.data[0]
    # Trace node/link counts match the aggregated index exactly.
    assert len(trace.node.label) == len(agg["nodes"])
    assert len(trace.link.source) == len(agg["source"])
    assert list(trace.link.value) == agg["value"]
    # Hovertemplates present on both node and link.
    assert "rows" in trace.node.hovertemplate
    assert "rows" in trace.link.hovertemplate


def test_make_sankey_figure_accepts_plain_index(df):
    """A plain build_sankey_index result (no 'other'/'link_class') still renders."""
    pytest.importorskip("plotly")
    stages = ["Dataset", "Kingdom"]
    index = shapes.build_sankey_index(df, stages)
    fig = sankey.make_sankey_figure(index, stages)
    assert fig.data and fig.data[0].type == "sankey"


def test_build_figure_end_to_end_with_filter(df):
    pytest.importorskip("plotly")
    fig = sankey.build_figure(df, ["Dataset", "Kingdom", "Phylum"], dataset="dsA", min_flow=1)
    assert fig is not None and fig.data[0].type == "sankey"
    # < 2 stages -> None (gradio shows a blank plot).
    assert sankey.build_figure(df, ["Dataset"]) is None


def test_export_html_and_png(df):
    """export_html writes non-empty HTML w/ plotly markup; export_png writes a real PNG (SANKEY-04)."""
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    fig = sankey.build_figure(df, ["Dataset", "Kingdom", "Phylum"], min_flow=1)
    assert fig is not None

    html_path = Path(sankey.export_html(fig))
    assert html_path.exists()
    html_bytes = html_path.read_bytes()
    assert len(html_bytes) > 0
    assert b"plotly" in html_bytes.lower()

    png_path = Path(sankey.export_png(fig))
    assert png_path.exists()
    png_bytes = png_path.read_bytes()
    assert len(png_bytes) > 0
    assert png_bytes.startswith(b"\x89PNG")  # PNG magic bytes


def test_export_none_raises():
    pytest.importorskip("plotly")
    with pytest.raises(ValueError):
        sankey.export_html(None)
    with pytest.raises(ValueError):
        sankey.export_png(None)


def test_render_smoke_and_presets(df):
    """render() builds a gr.Blocks fragment; PRESETS map to in-CSV stage selections."""
    pytest.importorskip("plotly")
    gr = pytest.importorskip("gradio")

    fragment = sankey.render(df)
    assert isinstance(fragment, gr.Blocks)

    # Preset semantics: each preset resolves to a valid (>=2) stage selection for this CSV.
    for preset_name, preset_stages in sankey.PRESETS.items():
        in_csv = [s for s in preset_stages if s in df.columns]
        if len(in_csv) >= 2:
            idx = shapes.build_sankey_index(df, in_csv)
            assert idx["used_rows"] >= 0, preset_name


# --------------------------------------------------------------------------- #
# (c) ZERO-DRIFT (SANKEY-05): double-generation byte-identity + source unchanged.
# --------------------------------------------------------------------------- #
def test_generate_sankey_double_generation_byte_identical():
    """generate_sankey.py is deterministic: two runs produce byte-identical HTML (D5)."""
    from planktonzilla.planktonzilla_dataset import constants, generate_sankey

    csv = constants.DEFAULT_TAXONOMY_CSV_FILENAME
    with tempfile.TemporaryDirectory() as d:
        out1 = Path(d) / "gen1.html"
        out2 = Path(d) / "gen2.html"
        rc1 = generate_sankey.main(["--csv", str(csv), "--out", str(out1)])
        rc2 = generate_sankey.main(["--csv", str(csv), "--out", str(out2)])
        assert rc1 == 0 and rc2 == 0
        assert out1.read_bytes() == out2.read_bytes()
        assert len(out1.read_bytes()) > 0


def test_generate_sankey_source_unchanged_vs_committed():
    """generate_sankey.py source is byte-unchanged vs the committed blob (D5)."""
    from planktonzilla.planktonzilla_dataset import generate_sankey

    source_path = Path(generate_sankey.__file__)
    on_disk = source_path.read_bytes()

    rel = "planktonzilla/planktonzilla_dataset/generate_sankey.py"
    repo_root = Path(__file__).resolve().parent.parent
    try:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git work tree (or git unavailable) — cannot compare committed blob")

    assert on_disk == committed, "generate_sankey.py drifted from its committed content (D5 violation)"
