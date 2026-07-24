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


def test_style_constants_present():
    # Assert on the module's constant strings directly — no gradio/plotly import needed.
    assert "plotly_click" in sa.BRIDGE_JS
    assert "#pz_click" in sa.BRIDGE_JS
    assert "--rouge" in sa.INRIA_CSS
    assert "inria-motif" in sa.INRIA_CSS
    assert 'data-theme="dark"' in sa.INRIA_CSS and 'data-theme="light"' in sa.INRIA_CSS
    assert "#000091" in sa.HEADER_HTML  # RF State blue, outside the Inria 5-hue palette
    assert "#e1000f" in sa.HEADER_HTML  # RF State red


# --------------------------------------------------- gradio app (gradio-guarded)
def _radio_labels(radio):
    return [choice[0] if isinstance(choice, list | tuple) else choice for choice in radio.choices]


def test_build_app_constructs_blocks_taxa_only():
    pytest.importorskip("gradio")
    import gradio as gr

    demo = sa.build_app()  # no rows -> bundled CSV, no .launch, network-free
    assert isinstance(demo, gr.Blocks)
    elem_ids = {getattr(block, "elem_id", None) for block in demo.blocks.values()}
    assert "pz_sankey" in elem_ids  # the go.Sankey Plot the JS bridge hooks
    assert "pz_click" in elem_ids  # the hidden gr.Number double-click sink

    radios = [block for block in demo.blocks.values() if isinstance(block, gr.Radio)]
    assert len(radios) == 1
    labels = _radio_labels(radios[0])
    assert "Images" not in labels  # no counts -> "Images" filtered out, not greyed
    assert "Taxa" in labels


def test_build_app_offers_images_when_counts():
    pytest.importorskip("gradio")
    import gradio as gr

    demo = sa.build_app(rows=ROWS, counts=PER_SOURCE)
    radios = [block for block in demo.blocks.values() if isinstance(block, gr.Radio)]
    assert len(radios) == 1
    labels = _radio_labels(radios[0])
    assert "Images" in labels and "Taxa" in labels
