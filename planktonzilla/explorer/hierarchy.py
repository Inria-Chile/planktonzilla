"""
(c) Inria

In-Space interactive Hierarchy view over the Linnaean ranks (Phase 12, HIER-01..04).

This module is the second of the three explorer views. It builds a count-weighted Plotly
``go.Sunburst`` / ``go.Icicle`` FRESH from the Phase 10 canonical
``shapes.build_hierarchy_table`` ({ids, labels, parents, values}) with
``branchvalues="total"`` (D1) and exposes:

* ``make_hierarchy_figure`` — the figure builder (``go.Sunburst`` default ↔ ``go.Icicle``,
  HIER-02) with native drill-down / breadcrumb / zoom (HIER-01), an optional ``level=``
  re-root for search-to-branch zoom (HIER-03), and a ``rank`` ↔ ``root_class`` color toggle
  (HIER-04, default ``rank``).
* ``render`` — a ``gr.Blocks`` fragment: a layout ``gr.Radio`` (Sunburst/Icicle), a search
  ``gr.Textbox`` (zoom-to-branch), a color ``gr.Radio`` (rank/root_class), and a ``gr.Plot``.
* ``main`` — a standalone local Gradio app (``python -m planktonzilla.explorer.hierarchy``).

Design (load-bearing — D4): gradio AND plotly are imported INSIDE function bodies, NEVER at
module scope. The Phase 9 dependency-isolation guard (``tests/test_dependency_isolation.py``)
fails the build on any module-scope viz import under ``planktonzilla/``. The module's pure
helpers (``node_depth``, ``match_node``, ``node_colors``) import cleanly in the CORE env with
gradio/plotly ABSENT; the per-node majority ``root_class`` comes from the pure
``shapes.hierarchy_root_class`` (FULL-distribution argmax — the b92d7d0 Sankey-bug guardrail).
"""

from __future__ import annotations

import polars as pl

from planktonzilla.explorer import shapes
from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_TAXONOMY_CSV_FILENAME,
    TAXONOMY_RANKS,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# Depth/rank palette (copied — NOT imported — from sankey.STAGE_COLORS for visual consistency
# across the two views, per D4; importing the figure builder would be a needless coupling).
RANK_PALETTE = ["#c9191e", "#2b5c8a", "#3f8e6b", "#c77f2e", "#6b5b95", "#3f8e9e", "#a23b72", "#7a8290", "#b0563f"]
# Categorical palette for root_class lineage coloring (mirrors sankey.LINEAGE_PALETTE).
LINEAGE_PALETTE = ["#c9191e", "#2b5c8a", "#3f8e6b", "#c77f2e", "#6b5b95", "#a23b72"]
# Muted gray for the Unknown lineage bucket (mirrors sankey.MUTED).
MUTED = "#aab3bf"
# The lineage "unknown" sentinel — kept in sync with sankey.LINEAGE_UNKNOWN (string literal so
# this pure module never imports the view layer at scope; shapes.hierarchy_root_class emits it).
LINEAGE_UNKNOWN = "Unknown"


# --------------------------------------------------------------------------- #
# Pure helpers — polars / stdlib only. NO gradio / plotly here.
# --------------------------------------------------------------------------- #
def node_depth(node_id: str) -> int:
    """Return a node's depth = number of "/"-separated path segments minus one.

    The hierarchy ids are "/"-joined prefixes (``"animalia/cnidaria"``), so the depth (and
    thus the rank index for color-by-rank) is simply the count of "/" separators: a root
    Kingdom node is depth 0, its Phylum child depth 1, and so on. Pure, no extra data (D4).

    Args:
        node_id: A ``build_hierarchy_table`` node id.

    Returns:
        The node's zero-based depth in the tree.
    """
    return node_id.count("/")


def match_node(table: dict, query: str) -> str | None:
    """Match a free-text taxon ``query`` to a node id for zoom-to-branch (HIER-03, D3).

    Case-insensitive. An EXACT label match wins first (returning the first such node in
    ``ids`` order); otherwise the first node whose label CONTAINS the query is returned. An
    empty/whitespace-only query, or no match at all, yields ``None`` (the caller then shows
    the full chart plus a gentle "not found" note).

    The query is used ONLY for string matching against node labels and to set the Plotly
    ``level=`` re-root — never for eval/exec, shell, or filesystem path interpolation
    (T-12-02).

    Args:
        table: A ``build_hierarchy_table`` result (uses ``ids`` + ``labels``).
        query: The untrusted free-text taxon query.

    Returns:
        The matched node id, or ``None`` when the query is empty / unmatched.
    """
    q = (query or "").strip().lower()
    if not q:
        return None
    ids = table["ids"]
    labels = table["labels"]
    # Exact label match first (case-insensitive), in ids order.
    for node_id, label in zip(ids, labels, strict=True):
        if label.lower() == q:
            return node_id
    # Else first id whose label contains the query.
    for node_id, label in zip(ids, labels, strict=True):
        if q in label.lower():
            return node_id
    return None


def node_colors(table: dict, *, color_by: str = "rank", root_class_by_node: dict | None = None) -> list[str]:
    """Build a per-node hex color list aligned to ``table["ids"]`` (HIER-04, D4).

    ``color_by="rank"`` (default): each node is colored by its DEPTH (``node_depth``) mapped
    into ``RANK_PALETTE`` (wrapping) — trivial and pure, needs no extra data.

    ``color_by="root_class"``: each node is colored by its majority ``root_class`` (from
    ``root_class_by_node``, typically ``shapes.hierarchy_root_class`` — the FULL-distribution
    argmax). Distinct known classes (in first-seen ids order) take ``LINEAGE_PALETTE`` in
    order; the ``Unknown`` sentinel and any node missing from the map map to the muted gray.

    Args:
        table: A ``build_hierarchy_table`` result (uses ``ids``).
        color_by: ``"rank"`` (default) or ``"root_class"``.
        root_class_by_node: ``{node_id: majority_root_class}`` for the ``root_class`` path.

    Returns:
        A list of ``"#rrggbb"`` strings, one per node, in ``table["ids"]`` order.
    """
    ids = table["ids"]
    if color_by == "root_class":
        rcbn = root_class_by_node or {}
        # Assign a stable color per distinct known class, in first-seen ids order.
        class_color: dict[str, str] = {}
        for node_id in ids:
            rc = rcbn.get(node_id, LINEAGE_UNKNOWN)
            if rc != LINEAGE_UNKNOWN and rc not in class_color:
                class_color[rc] = LINEAGE_PALETTE[len(class_color) % len(LINEAGE_PALETTE)]
        return [class_color.get(rcbn.get(node_id, LINEAGE_UNKNOWN), MUTED) for node_id in ids]
    # Default: color by rank/depth.
    return [RANK_PALETTE[node_depth(node_id) % len(RANK_PALETTE)] for node_id in ids]


# --------------------------------------------------------------------------- #
# Lazy figure builder — plotly imported INSIDE the function body (D4).
# --------------------------------------------------------------------------- #
def make_hierarchy_figure(
    table: dict,
    *,
    chart: str = "sunburst",
    color_by: str = "rank",
    level: str | None = None,
    root_class_by_node: dict | None = None,
):
    """Build a count-weighted ``go.Sunburst`` / ``go.Icicle`` from a hierarchy table (HIER-01).

    The trace is a ``go.Sunburst`` when ``chart == "sunburst"`` (default) and a ``go.Icicle``
    when ``chart == "icicle"`` (HIER-02), both built from
    ``table["ids"/"labels"/"parents"/"values"]`` with ``branchvalues="total"`` (D1, SC1 — so
    a parent's wedge equals the sum of its children; the values are already cumulative
    pass-through counts from ``build_hierarchy_table``). Native Plotly drill-down, breadcrumb,
    and zoom come for free (HIER-01). Segment colors come from ``node_colors`` (rank by depth,
    or root_class majority — HIER-04). When ``level`` is a non-empty node id the trace is
    re-rooted to that branch (``level=<id>``) for search-to-branch zoom (HIER-03).

    plotly is imported INSIDE this function (D4) so the module loads in the core env.

    Args:
        table: A ``shapes.build_hierarchy_table`` result.
        chart: ``"sunburst"`` (default) or ``"icicle"``.
        color_by: ``"rank"`` (default) or ``"root_class"`` (needs ``root_class_by_node``).
        level: A matched node id to re-root/zoom to, or ``None``/"" for the full chart.
        root_class_by_node: ``{node_id: majority_root_class}`` for color-by-root_class.

    Returns:
        A ``plotly.graph_objects.Figure`` with one ``go.Sunburst`` or ``go.Icicle`` trace.
    """
    import plotly.graph_objects as go

    colors = node_colors(table, color_by=color_by, root_class_by_node=root_class_by_node)
    trace_kind = go.Icicle if chart == "icicle" else go.Sunburst
    trace = trace_kind(
        ids=table["ids"],
        labels=table["labels"],
        parents=table["parents"],
        values=table["values"],
        branchvalues="total",
        marker={"colors": colors},
        hovertemplate="%{label}<br>%{value} rows<extra></extra>",
    )
    # Re-root to the matched branch for search-to-branch zoom (HIER-03). Empty/None => full chart.
    if level:
        trace.level = level

    fig = go.Figure(data=[trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1a1a1a", "size": 11},
        margin={"l": 8, "r": 8, "t": 30, "b": 8},
    )
    return fig


def build_figure(
    df: pl.DataFrame,
    *,
    chart: str = "sunburst",
    color_by: str = "rank",
    query: str = "",
    ranks: tuple[str, ...] = TAXONOMY_RANKS,
):
    """End-to-end figure: build_hierarchy_table -> (root_class) -> match_node -> figure.

    Convenience wrapper used by ``render``'s callbacks and the tests. Builds the canonical
    hierarchy table (D1); for the ``root_class`` color path it computes the per-node majority
    via ``shapes.hierarchy_root_class`` (FULL-distribution argmax, the b92d7d0 guardrail);
    resolves ``query`` to a zoom ``level`` via ``match_node``; and renders the figure. When
    ``query`` is non-empty but unmatched the full chart is returned alongside a gentle
    "not found" note (HIER-03).

    Args:
        df: The taxonomy frame (typically from ``shapes.load_taxonomy``).
        chart: ``"sunburst"`` (default) or ``"icicle"``.
        color_by: ``"rank"`` (default) or ``"root_class"``.
        query: Optional free-text taxon search (zoom-to-branch).
        ranks: Ordered taxonomic ranks to walk. Defaults to ``TAXONOMY_RANKS``.

    Returns:
        ``(figure, note)`` — the Plotly Figure and a note string ("" unless a non-empty
        ``query`` matched nothing).
    """
    table = shapes.build_hierarchy_table(df, ranks=ranks)
    root_class_by_node = shapes.hierarchy_root_class(df, ranks=ranks) if color_by == "root_class" else None
    level = match_node(table, query)
    note = ""
    if query and query.strip() and level is None:
        note = f"No taxon matching «{query.strip()}» — showing the full hierarchy."
    fig = make_hierarchy_figure(
        table,
        chart=chart,
        color_by=color_by,
        level=level,
        root_class_by_node=root_class_by_node,
    )
    return fig, note


# --------------------------------------------------------------------------- #
# Lazy UI fragment — gradio imported INSIDE the function body (D4).
# --------------------------------------------------------------------------- #
def render(df: pl.DataFrame | None = None):
    """Build the Hierarchy ``gr.Blocks`` fragment (layout + search + color toggles + plot).

    Wires (D1/D2/D3/D4):

    * a ``gr.Radio`` Sunburst↔Icicle layout toggle (default Sunburst, HIER-02),
    * a ``gr.Textbox`` taxon search that zooms to the matched branch (HIER-03),
    * a ``gr.Radio`` color toggle rank↔root_class (default rank, HIER-04),
    * a ``gr.Plot`` output and a ``gr.Markdown`` for the "not found" note.

    gradio is imported INSIDE this function (D4). The frozen taxonomy CSV is loaded lazily
    when ``df`` is not supplied (network-free; pure shapes).

    Args:
        df: Optional pre-loaded taxonomy frame. Loaded from the frozen CSV when ``None``.

    Returns:
        A ``gr.Blocks`` fragment ready to be composed into the Space (Phase 14) or launched
        standalone by ``main``.
    """
    import gradio as gr

    if df is None:
        df = shapes.load_taxonomy(DEFAULT_TAXONOMY_CSV_FILENAME)

    def _update(chart, color_by, query):
        layout = "icicle" if str(chart).lower() == "icicle" else "sunburst"
        return build_figure(df, chart=layout, color_by=str(color_by), query=query or "")

    with gr.Blocks() as fragment:
        gr.Markdown(
            "## Taxonomy Hierarchy\nCount-weighted sunburst / icicle over the Linnaean ranks. "
            "Click a wedge to drill in; search a taxon to zoom to its branch."
        )
        with gr.Row():
            with gr.Column(scale=1):
                chart_in = gr.Radio(["Sunburst", "Icicle"], value="Sunburst", label="Layout")
                search_in = gr.Textbox(label="Search taxon (zoom to branch)", placeholder="e.g. cnidaria")
                color_in = gr.Radio(["rank", "root_class"], value="rank", label="Color by")
                note_out = gr.Markdown("")
            with gr.Column(scale=4):
                plot = gr.Plot(label="Hierarchy")

        controls = [chart_in, color_in, search_in]
        for control in controls:
            control.change(_update, inputs=controls, outputs=[plot, note_out])
        search_in.submit(_update, inputs=controls, outputs=[plot, note_out])

        fragment.load(_update, inputs=controls, outputs=[plot, note_out])

    return fragment


def main(argv: list[str] | None = None) -> int:
    """Launch the standalone local Hierarchy app (``python -m planktonzilla.explorer.hierarchy``).

    Wraps ``render()`` in a ``gr.Blocks`` and ``.launch()``-es it for local development. No
    core console script imports viz at module scope (D4); this entry point runs under the
    explorer group only.

    Args:
        argv: Unused; accepted for CLI-symmetry with the other entry points.

    Returns:
        Process exit code (0 on a clean launch).
    """
    import gradio as gr

    with gr.Blocks(title="planktonzilla — taxonomy Hierarchy") as app:
        render()
    app.launch()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
