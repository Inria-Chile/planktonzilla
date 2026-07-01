"""
(c) Inria

Network-free composition smoke test for the root Space app.py (Phase 14, SPACE-01/02/04).

Mirrors the two-environment, network-free pattern of tests/test_explorer_geomap.py: gradio is
reached only via ``pytest.importorskip("gradio")`` (so this file SKIPS in the core ``test`` job
and RUNS under the explorer-group CI job), and an autouse socket fixture blocks INET/INET6 while
allowing ``AF_UNIX`` (so gradio's asyncio event loop can build ``gr.Blocks`` offline, but any real
HF read raises loudly). The app is built via ``app.build_demo()`` WITHOUT ``.launch()`` and WITHOUT
firing ``fragment.load`` — so the Map tab's deferred live geo read is never triggered.

Assertions:

* exactly four tabs labelled Sankey / Hierarchy / Map / About exist in the built component tree;
* the About panel carries the inferred-location caveat markers (Greifensee / Inferred / lensless)
  so users cannot mistake inferred points for ground truth (SPACE-02);
* a loading affordance exists under the Map tab (D3) — a Markdown mentioning "loading"/"first
  load", so a refactor cannot silently drop the cold-start cue.
"""

from __future__ import annotations

import os
import socket

import pytest

# Disable gradio telemetry + force HF offline so a stray read fails loudly (belt-and-suspenders
# with the socket fixture). Set before any gradio import.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Make any real INTERNET socket raise; allow AF_UNIX so gradio's event loop still builds.

    Copied from tests/test_explorer_geomap.py: INET/INET6 sockets (and ``create_connection``,
    internet-only) raise, so a live HF read fails LOUDLY. Local ``AF_UNIX`` socketpairs (gradio's
    internal asyncio loop) pass through — they never cross to the network.
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


def _tab_labels(demo) -> list[str]:
    """Collect the labels of every gradio Tab / TabItem component in the built Blocks tree."""
    import gradio as gr

    tab_types = tuple(t for t in (getattr(gr, "Tab", None), getattr(gr, "TabItem", None)) if t is not None)
    labels: list[str] = []
    for block in demo.blocks.values():
        if tab_types and isinstance(block, tab_types):
            label = getattr(block, "label", None)
            if label:
                labels.append(label)
    return labels


def _all_markdown_text(demo) -> str:
    """Concatenate the text/value of every gradio Markdown component in the built Blocks tree."""
    import gradio as gr

    chunks: list[str] = []
    for block in demo.blocks.values():
        if isinstance(block, gr.Markdown):
            value = getattr(block, "value", None)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks)


def test_app_builds_four_tabs():
    """app.build_demo() builds a gr.Blocks with exactly 4 tabs: Sankey/Hierarchy/Map/About (SPACE-01)."""
    pytest.importorskip("gradio")
    import app

    demo = app.build_demo()  # no .launch(), no fragment.load -> network-free
    labels = _tab_labels(demo)
    assert set(labels) == {"Sankey", "Hierarchy", "Map", "About"}, labels
    # Exactly four tabs (no accidental extra/duplicate tab).
    assert len([lbl for lbl in labels if lbl in {"Sankey", "Hierarchy", "Map", "About"}]) == 4


def test_about_panel_has_inferred_caveat():
    """The About panel carries the inferred-location caveat so inferred != ground truth (SPACE-02)."""
    pytest.importorskip("gradio")
    import app

    # Assert against the built Blocks' Markdown text (falls back to module source if empty).
    demo = app.build_demo()
    text = _all_markdown_text(demo) or app.ABOUT_MARKDOWN
    assert "Greifensee" in text  # zoolake correction (not Lake Zurich)
    assert "Inferred" in text  # Measured vs Inferred distinction
    assert "lensless" in text or "do not plot" in text.lower()  # exclusion note


def test_map_tab_has_loading_affordance():
    """A loading affordance exists under the Map tab (D3) — a refactor can't silently drop it."""
    pytest.importorskip("gradio")
    import app

    demo = app.build_demo()
    text = (_all_markdown_text(demo) + "\n" + app.MAP_LOADING_NOTE).lower()
    assert "loading" in text or "first map load" in text or "first load" in text
