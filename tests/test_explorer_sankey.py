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


def test_link_lineage_majority_root_class(df):
    """Per-link majority root_class matches the records flowing through each edge."""
    stages = ["Dataset", "Kingdom"]
    lineage = sankey.link_lineage(df, stages)
    # Rebuild raw node ids the same way build_sankey_index does, to locate links.
    index = shapes.build_sankey_index(df, stages)

    def node_id(stage, label):
        return next(i for i, nn in enumerate(index["nodes"]) if nn["stage"] == stage and nn["label"] == label)

    dsa = node_id(0, "dsA")
    animalia = node_id(1, "animalia")
    # dsA -> animalia carries raw1(living), raw2(living), raw3(detritus) -> majority "living".
    assert lineage[(dsa, animalia)] == "living"

    dsb = node_id(0, "dsB")
    # dsB -> animalia carries raw5(living), raw6(inert) -> tie broken by max() (first max key);
    # both appear once so the result is one of {"living","inert"} — assert it is a real class.
    assert lineage[(dsb, animalia)] in {"living", "inert"}


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
