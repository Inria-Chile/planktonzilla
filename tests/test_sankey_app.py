"""
(c) Inria

Network-free tests for the pz_sankey_app explorer (planktonzilla/planktonzilla_dataset/sankey_app.py).

Two layers, mirroring the repo's dependency-isolation contract:

* PURE-FUNCTION tests (``row_path`` / ``build_graph`` / ``breadcrumb``) need NO explorer group —
  they import and run with only the frozen core, so they carry NO ``pytest.importorskip`` guard.
* FIGURE / APP tests reach ``plotly`` / ``gradio`` only behind a ``pytest.importorskip(...)`` FIRST
  line (so the file collects but SKIPS in the core ``test`` job and RUNS under the explorer job),
  with the autouse INET-blocking socket fixture + offline env from ``tests/test_app_compose.py``.

Hand-computed expectations on tiny fixtures pin the converging data model: the (unclassified)
living fallback, Images additive conservation, Taxa 1/N fractional attribution, threshold pooling,
and focus/breadcrumb.
"""

from __future__ import annotations

import os
import socket

import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

# Set before any gradio/plotly import (belt-and-suspenders with the socket fixture below).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from collections import Counter, defaultdict

from planktonzilla.planktonzilla_dataset import sankey_app as sa
from planktonzilla.planktonzilla_dataset.sankey_app import ALL_COLUMNS, breadcrumb, build_graph, row_path

RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def _row(dataset, proposed_label, root_class, **ranks):
    r = {"Dataset": dataset, "proposed_label": proposed_label, "root_class": root_class}
    for rk in RANKS:
        r[rk] = ranks.get(rk, "")
    return r


# A genus-level living label, a species under the same genus, and one non-living leaf — spread over
# two source datasets (mirrors tests/test_build_sankey.py).
ROWS = [
    _row(
        "whoi",
        "calanus finmarchicus",
        "living",
        Kingdom="animalia",
        Phylum="arthropoda",
        Genus="calanus",
        Species="finmarchicus",
    ),
    _row("whoi", "calanus", "living", Kingdom="animalia", Phylum="arthropoda", Genus="calanus"),
    _row("zoolake", "marine snow", "detritus"),
]
PER_SOURCE = Counter(
    {
        ("whoi", "calanus finmarchicus", "living"): 100,
        ("whoi", "calanus", "living"): 50,
        ("zoolake", "marine snow", "detritus"): 30,
    }
)


def _node(graph, col, label):
    for i, node in enumerate(graph.nodes):
        if node.col == col and node.label == label:
            return i, node
    return None, None


def _assert_conserves(graph):
    """Every non-root node's incoming link-value sum equals its node value (the core invariant).

    (Outgoing is NOT required to equal value: a taxonomy label can terminate on an INTERNAL node —
    e.g. a genus-level label under which species also sit — so incoming == value while some flow
    stops there. This mirrors build_sankey's cumulative ``s``.)
    """
    incoming = defaultdict(float)
    for link in graph.links:
        incoming[link.tgt] += link.value
    for i, node in enumerate(graph.nodes):
        if incoming[i] > 0:
            assert incoming[i] == pytest.approx(node.value)


def _assert_flow_invariants(graph, context=""):
    """The FULL conservation contract on the converging DAG, checked at every node.

    * incoming (when non-zero) equals the node value — a node can never keep flow that no surviving
      ribbon delivers (the M1 defect: edge-level pooling popped an edge but left the value behind).
    * outgoing never EXCEEDS the node value — a node can never emit flow it never received.
    """
    incoming = defaultdict(float)
    outgoing = defaultdict(float)
    for link in graph.links:
        incoming[link.tgt] += link.value
        outgoing[link.src] += link.value
    for i, node in enumerate(graph.nodes):
        if incoming[i] > 0:
            assert incoming[i] == pytest.approx(node.value), f"incoming != value for {node}{context}"
        assert outgoing[i] <= node.value + 1e-9, f"outgoing > value for {node}{context}"


# ----------------------------------------------------------------- pure functions
def test_row_path_living_and_non_living():
    living = _row("whoi", "calanus", "living", Kingdom="animalia", Genus="calanus")
    assert row_path(living) == [
        ("source_dataset", "whoi"),
        ("root_class", "living"),
        ("Kingdom", "animalia"),
        ("Genus", "calanus"),
    ]
    detritus = _row("zoolake", "marine snow", "detritus")
    assert row_path(detritus) == [
        ("source_dataset", "zoolake"),
        ("root_class", "detritus"),
        ("proposed_label", "marine snow"),
    ]


def test_row_path_unclassified_fallback_and_conservation():
    # An all-ranks-blank living row must fall back to (unclassified)->proposed_label (build_sankey parity).
    blank = _row("ecotaxa", "blob", "living")
    assert row_path(blank) == [
        ("source_dataset", "ecotaxa"),
        ("root_class", "living"),
        ("group", "(unclassified)"),
        ("proposed_label", "blob"),
    ]
    rows = [_row("whoi", "calanus", "living", Genus="calanus"), blank]
    counts = Counter({("whoi", "calanus", "living"): 50, ("ecotaxa", "blob", "living"): 20})
    graph = build_graph(rows, counts, size_metric="images")
    _assert_conserves(graph)
    living_idx, living = _node(graph, "root_class", "living")
    assert living.value == pytest.approx(70)
    # living is purely internal here (both rows continue past it) -> its OUTGOING must also equal its
    # value: nothing dangles on root_class:living thanks to the (unclassified) fallback.
    outgoing = defaultdict(float)
    for link in graph.links:
        outgoing[link.src] += link.value
    assert outgoing[living_idx] == pytest.approx(70)


def test_row_path_column_collapse():
    living = _row("whoi", "calanus", "living", Kingdom="animalia", Genus="calanus")
    enabled = [c for c in ALL_COLUMNS if c != "root_class"]
    assert row_path(living, enabled) == [
        ("source_dataset", "whoi"),
        ("Kingdom", "animalia"),
        ("Genus", "calanus"),
    ]


def test_build_graph_images_conservation():
    graph = build_graph(ROWS, PER_SOURCE, size_metric="images")
    _assert_conserves(graph)
    _, whoi = _node(graph, "source_dataset", "whoi")
    assert whoi.value == pytest.approx(150)
    _, calanus = _node(graph, "Genus", "calanus")
    assert calanus.value == pytest.approx(150)  # 50 own-row + 100 from the species below it
    _, fin = _node(graph, "Species", "finmarchicus")
    assert fin.value == pytest.approx(100)
    _, snow = _node(graph, "proposed_label", "marine snow")
    assert snow.value == pytest.approx(30)


def test_build_graph_taxa_fractional_attribution():
    # One leaf taxon shared by two datasets -> two 0.5 ribbons; the shared leaf node value == 1.
    rows = [
        _row("whoi", "calanus", "living", Genus="calanus"),
        _row("zoolake", "calanus", "living", Genus="calanus"),
    ]
    graph = build_graph(rows, size_metric="taxa")
    _, leaf = _node(graph, "Genus", "calanus")
    assert leaf.value == pytest.approx(1.0)
    _, whoi = _node(graph, "source_dataset", "whoi")
    _, zoolake = _node(graph, "source_dataset", "zoolake")
    assert whoi.value == pytest.approx(0.5)
    assert zoolake.value == pytest.approx(0.5)
    _assert_conserves(graph)


def test_build_graph_threshold_pooling():
    rows = [
        _row("whoi", "a", "living", Genus="a"),
        _row("whoi", "b", "living", Genus="b"),
        _row("whoi", "c", "living", Genus="c"),
    ]
    counts = Counter({("whoi", "a", "living"): 1, ("whoi", "b", "living"): 1, ("whoi", "c", "living"): 100})
    graph = build_graph(rows, counts, size_metric="images", min_threshold=10)
    others = [n for n in graph.nodes if n.is_other]
    assert len(others) == 1
    assert others[0].label == "+2 other"
    assert others[0].value == pytest.approx(2.0)
    assert _node(graph, "Genus", "a")[1] is None  # pooled away
    assert _node(graph, "Genus", "b")[1] is None
    assert _node(graph, "Genus", "c")[1] is not None  # above threshold, kept
    _assert_conserves(graph)


def test_focus_subtree_and_breadcrumb():
    rows = [
        _row("whoi", "calanus", "living", Genus="calanus"),
        _row("zoolake", "marine snow", "detritus"),
    ]
    counts = Counter({("whoi", "calanus", "living"): 50, ("zoolake", "marine snow", "detritus"): 30})
    full = build_graph(rows, counts, size_metric="images")
    focused = build_graph(rows, counts, size_metric="images", focus_key=("root_class", "living"))

    cols = {n.col for n in focused.nodes}
    assert "source_dataset" not in cols  # restricted to living + descendants
    assert "detritus" not in {n.label for n in focused.nodes}
    living_idx, living = _node(focused, "root_class", "living")
    assert living is not None
    incoming = defaultdict(float)
    for link in focused.links:
        incoming[link.tgt] += link.value
    assert incoming[living_idx] == 0  # living is the new root
    _assert_conserves(focused)

    assert breadcrumb(full, ("Genus", "calanus")) == [
        ("source_dataset", "whoi"),
        ("root_class", "living"),
        ("Genus", "calanus"),
    ]


# ------------------------------------------------ pooling: conservation on the DAG
# The same leaf genus reached through TWO different kingdoms in TWO datasets — the converging
# shape the real CSV has and the one that breaks edge-level pooling: at threshold 10 the ecotaxa
# branch (2 images) is sub-threshold while the whoi branch (100) survives, so a pooler that pops
# an edge without re-deriving node values leaves ``calanus`` holding 102 with only 100 incoming.
SHARED_CHILD_ROWS = [
    _row("whoi", "calanus", "living", Kingdom="animalia", Genus="calanus"),
    _row("ecotaxa", "calanus", "living", Kingdom="protista", Genus="calanus"),
]
SHARED_CHILD_COUNTS = Counter(
    {("whoi", "calanus", "living"): 100, ("ecotaxa", "calanus", "living"): 2},
)


def test_pooling_conserves_with_shared_child():
    for metric in ("images", "taxa"):
        for threshold in (0, 1, 5, 10, 50, 500):
            graph = build_graph(
                SHARED_CHILD_ROWS,
                SHARED_CHILD_COUNTS,
                size_metric=metric,
                min_threshold=threshold,
            )
            _assert_flow_invariants(graph, context=f" [metric={metric} threshold={threshold}]")


def test_pooling_is_per_parent():
    # Two kingdoms, each with two 1-image children and one 100-image child. A GLOBAL pooler emits a
    # single "+4 other"; the per-parent contract requires one bucket under EACH kingdom.
    rows = [
        _row("whoi", "a1", "living", Kingdom="animalia", Genus="a1"),
        _row("whoi", "a2", "living", Kingdom="animalia", Genus="a2"),
        _row("whoi", "a3", "living", Kingdom="animalia", Genus="a3"),
        _row("whoi", "p1", "living", Kingdom="protista", Genus="p1"),
        _row("whoi", "p2", "living", Kingdom="protista", Genus="p2"),
        _row("whoi", "p3", "living", Kingdom="protista", Genus="p3"),
    ]
    counts = Counter(
        {
            ("whoi", "a1", "living"): 1,
            ("whoi", "a2", "living"): 1,
            ("whoi", "a3", "living"): 100,
            ("whoi", "p1", "living"): 1,
            ("whoi", "p2", "living"): 1,
            ("whoi", "p3", "living"): 100,
        }
    )
    graph = build_graph(rows, counts, size_metric="images", min_threshold=10)
    others = [n for n in graph.nodes if n.is_other]
    assert len(others) == 2
    assert {n.label for n in others} == {"+2 other"}
    assert all(n.value == pytest.approx(2.0) for n in others)
    assert all(n.col == "Genus" for n in others)
    _assert_flow_invariants(graph)


def test_pooling_other_per_column():
    # ONE parent (root_class:living) whose small children live in DIFFERENT columns: a ranked row
    # continues into Kingdom, an all-ranks-blank row into the (unclassified) group column. Each
    # needs its OWN "+N other" in ITS child's column — never one bucket parked in the left-most.
    rows = [
        _row("whoi", "calanus", "living", Kingdom="animalia", Genus="calanus"),
        _row("whoi", "blob", "living"),
        _row("whoi", "big", "living", Kingdom="bigkingdom"),
    ]
    counts = Counter(
        {
            ("whoi", "calanus", "living"): 1,
            ("whoi", "blob", "living"): 2,
            ("whoi", "big", "living"): 100,
        }
    )
    graph = build_graph(rows, counts, size_metric="images", min_threshold=10)
    others = [n for n in graph.nodes if n.is_other]
    assert len(others) == 2
    assert {n.col for n in others} == {"Kingdom", "group"}
    assert all(n.label == "+1 other" for n in others)
    _assert_flow_invariants(graph)


def test_pooling_boundary():
    # STRICT `<`: a child sitting exactly AT the threshold is KEPT.
    rows = [_row("whoi", "a", "living", Genus="a"), _row("whoi", "b", "living", Genus="b")]
    counts = Counter({("whoi", "a", "living"): 10, ("whoi", "b", "living"): 100})

    kept = build_graph(rows, counts, size_metric="images", min_threshold=10)
    assert not any(n.is_other for n in kept.nodes)
    assert _node(kept, "Genus", "a")[1] is not None
    _assert_flow_invariants(kept)

    pooled = build_graph(rows, counts, size_metric="images", min_threshold=10.5)
    others = [n for n in pooled.nodes if n.is_other]
    assert len(others) == 1
    assert others[0].label == "+1 other"
    assert _node(pooled, "Genus", "a")[1] is None
    _assert_flow_invariants(pooled)


def test_images_count_key_dedupe():
    # Two CSV rows collapsing onto ONE load_sample_counts key (proposed_label differs only in case).
    # The single 100-image entry must be attributed exactly once, and the loser row's divergent
    # Kingdom must not materialise a phantom node.
    rows = [
        _row("whoi", "Calanus", "living", Kingdom="animalia", Genus="calanus"),
        _row("whoi", "calanus", "living", Kingdom="protista", Genus="calanus"),
    ]
    counts = Counter({("whoi", "calanus", "living"): 100})
    graph = build_graph(rows, counts, size_metric="images")
    _, whoi = _node(graph, "source_dataset", "whoi")
    assert whoi.value == pytest.approx(100)  # not 200
    assert _node(graph, "Kingdom", "animalia")[1] is not None
    assert _node(graph, "Kingdom", "protista")[1] is None
    _assert_flow_invariants(graph)


def test_breadcrumb_rows_branch():
    rows = [_row("whoi", "calanus", "living", Kingdom="animalia", Genus="calanus")]
    assert breadcrumb(rows, ("Genus", "calanus")) == [
        ("source_dataset", "whoi"),
        ("root_class", "living"),
        ("Kingdom", "animalia"),
        ("Genus", "calanus"),
    ]
    assert breadcrumb(rows, None) == []
    assert breadcrumb(rows, ("Genus", "nope")) == [("Genus", "nope")]

    # M2: with a column collapsed away, Zoom-out must never target a key from that column.
    enabled = [c for c in ALL_COLUMNS if c != "Kingdom"]
    chain = breadcrumb(rows, ("Genus", "calanus"), enabled)
    assert all(col != "Kingdom" for col, _label in chain)
    path = row_path(rows[0], enabled)
    assert chain == path[: len(chain)]
    assert chain[-1] == ("Genus", "calanus")


# --------------------------------------------------- figure layer (plotly-guarded)
@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Make any real INTERNET socket raise; allow AF_UNIX so a viz event loop still builds.

    Mirrors tests/test_app_compose.py: INET/INET6 sockets (and ``create_connection``) raise, so a
    live HF read fails LOUDLY, while local ``AF_UNIX`` socketpairs (gradio's asyncio loop) pass.
    """
    real_socket = socket.socket

    def _guarded_socket(family=socket.AF_INET, *args, **kwargs):
        if family in (socket.AF_INET, getattr(socket, "AF_INET6", socket.AF_INET)):
            raise RuntimeError("network access is blocked in the explorer test suite")
        return real_socket(family, *args, **kwargs)

    def _no_connection(*args, **kwargs):
        raise RuntimeError("network access is blocked in the explorer test suite")

    monkeypatch.setattr(socket, "socket", _guarded_socket)
    monkeypatch.setattr(socket, "create_connection", _no_connection)
    yield


def test_make_figure_returns_pinned_sankey():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    graph = build_graph(ROWS, PER_SOURCE, size_metric="images")
    fig = sa.make_figure(graph, theme="light", size_metric="images")
    assert isinstance(fig, go.Figure)
    sankeys = [trace for trace in fig.data if isinstance(trace, go.Sankey)]
    assert len(sankeys) == 1
    node_x = sankeys[0].node.x
    assert node_x is not None
    assert len(node_x) == len(graph.nodes)
    assert all(0.02 <= x <= 0.98 for x in node_x)  # columns pinned off the 0/1 edges
    assert sankeys[0].arrangement == "fixed"
    # Rouge is reserved: no data node may carry it.
    assert all((c or "").lower() != sa.ROUGE_RESERVED for c in sankeys[0].node.color)


def test_make_figure_pins_x_per_column():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    graph = build_graph(ROWS, PER_SOURCE, size_metric="images")
    sankey = next(t for t in sa.make_figure(graph).data if isinstance(t, go.Sankey))
    xs = list(sankey.node.x)

    # arrangement="fixed" is only meaningful if a column maps to exactly ONE x ...
    per_column = defaultdict(set)
    for node, x in zip(graph.nodes, xs, strict=True):
        per_column[node.col].add(round(float(x), 9))
    assert all(len(values) == 1 for values in per_column.values()), per_column

    # ... and if those x's march LEFT TO RIGHT in the fixed column order (a constant-x layout
    # would satisfy the 0.02..0.98 range check alone).
    ordered = sorted(per_column, key=lambda col: ALL_COLUMNS.index(col))
    x_by_column = [next(iter(per_column[col])) for col in ordered]
    assert len(set(x_by_column)) == len(x_by_column)
    assert x_by_column == sorted(x_by_column)
    assert all(0.02 <= x <= 0.98 for x in x_by_column)


def test_other_node_is_gray():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    rows = [
        _row("whoi", "a", "living", Genus="a"),
        _row("whoi", "b", "living", Genus="b"),
        _row("whoi", "c", "living", Genus="c"),
    ]
    counts = Counter({("whoi", "a", "living"): 1, ("whoi", "b", "living"): 1, ("whoi", "c", "living"): 100})
    graph = build_graph(rows, counts, size_metric="images", min_threshold=10)
    sankey = next(t for t in sa.make_figure(graph).data if isinstance(t, go.Sankey))
    colors = list(sankey.node.color)

    other_idx = [i for i, n in enumerate(graph.nodes) if n.is_other]
    assert other_idx
    assert all(colors[i] == sa.DATA_OTHER_COLOR for i in other_idx)
    # Rouge is the view's ONE signal — never a data encoding, in either notation.
    assert all((c or "").lower() != sa.ROUGE_RESERVED for c in colors)
    rouge_rgb = ",".join(str(int(sa.ROUGE_RESERVED[i : i + 2], 16)) for i in (1, 3, 5))
    assert all(rouge_rgb not in (c or "") for c in sankey.link.color)


def test_make_figure_theme_dark():
    pytest.importorskip("plotly")

    graph = build_graph(ROWS, PER_SOURCE, size_metric="images")
    light = sa.make_figure(graph, theme="light")
    dark = sa.make_figure(graph, theme="dark")
    assert dark.layout.paper_bgcolor != light.layout.paper_bgcolor
    assert dark.layout.paper_bgcolor == "#121517"
    assert dark.layout.font.color == "#e9edf0"
    assert light.layout.paper_bgcolor == "#ffffff"


def test_color_ramp_is_inria():
    # Pure (no gradio import): the 11 Tailwind-shaped stops the gr.themes.Color takes positionally.
    ramp = sa._color_ramp(sa.ROUGE_RESERVED)
    assert len(ramp) == 11
    assert ramp[5] == "#c9191e"  # c500 is the charter Rouge itself, untinted and unshaded
    assert all(isinstance(stop, str) and len(stop) == 7 and stop.startswith("#") for stop in ramp)
    assert all(int(stop[1:], 16) >= 0 for stop in ramp)
    assert ramp == sorted(ramp, key=lambda s: -sum(int(s[i : i + 2], 16) for i in (1, 3, 5)))


def test_style_constants_present():
    # Assert on the module's constant strings directly — no gradio/plotly import needed.
    assert "plotly_click" in sa.BRIDGE_JS
    assert "#pz_click" in sa.BRIDGE_JS
    assert "--rouge" in sa.INRIA_CSS
    assert "inria-motif" in sa.INRIA_CSS
    assert 'data-theme="dark"' in sa.INRIA_CSS and 'data-theme="light"' in sa.INRIA_CSS
    assert ".dark" in sa.INRIA_CSS  # Gradio toggles a CLASS, not a :root[data-theme] attribute
    for token in ("--sunken", "--ink-2", "--border-strong"):
        assert sa.INRIA_CSS.count(token) >= 3  # :root + both dark blocks — a partial override is a bug
    assert ".gradio-container" in sa.INRIA_CSS and "max-width:1200px" in sa.INRIA_CSS
    assert "#000091" in sa.HEADER_HTML  # RF State blue, outside the Inria 5-hue palette
    assert "#e1000f" in sa.HEADER_HTML  # RF State red


# --------------------------------------------------- gradio app (gradio-guarded)
def _radio_labels(radio):
    return [choice[0] if isinstance(choice, list | tuple) else choice for choice in radio.choices]


def _radio_by_label(demo, label):
    import gradio as gr

    found = [b for b in demo.blocks.values() if isinstance(b, gr.Radio) and getattr(b, "label", None) == label]
    assert len(found) == 1, f"expected exactly one Radio labelled {label!r}, got {len(found)}"
    return found[0]


def test_build_app_constructs_blocks_taxa_only():
    pytest.importorskip("gradio")
    import gradio as gr

    demo = sa.build_app()  # no rows -> bundled CSV, no .launch, network-free
    assert isinstance(demo, gr.Blocks)
    elem_ids = {getattr(block, "elem_id", None) for block in demo.blocks.values()}
    assert "pz_sankey" in elem_ids  # the go.Sankey Plot the JS bridge hooks
    assert "pz_click" in elem_ids  # the CSS-hidden gr.Number double-click sink

    labels = _radio_labels(_radio_by_label(demo, "Size by"))
    assert "Images" not in labels  # no counts -> "Images" filtered out, not greyed
    assert "Taxa" in labels
    assert _radio_labels(_radio_by_label(demo, "Theme")) == ["Light", "Dark"]


def test_build_app_offers_images_when_counts():
    pytest.importorskip("gradio")

    demo = sa.build_app(rows=ROWS, counts=PER_SOURCE)
    labels = _radio_labels(_radio_by_label(demo, "Size by"))
    assert "Images" in labels and "Taxa" in labels
