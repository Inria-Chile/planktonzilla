"""
(c) Inria

In-Space interactive Geographic Map view of source-dataset sampling sites (Phase 13, MAP-01..03).

This is the third of the three explorer views. It builds a token-free Plotly ``go.Scattergeo``
world map on the built-in NATURAL-EARTH basemap (D2 — NO map tiles, NO runtime network for the
basemap, NO token; NEVER the deprecated ``scatter_mapbox``/``Scattermapbox``) from the
category-graded points produced by the pure ``shapes.aggregate_geo``. It scientifically
distinguishes KNOWN (measured) from INFERRED (high/low) locations with three styled, legended
categories (D3, MAP-02) and exposes:

* ``make_geo_figure`` — the figure builder: ONE ``go.Scattergeo`` trace per category PRESENT
  (Measured / Inferred — high confidence / Inferred — low confidence), one marker per
  (dataset, rounded site), hover = dataset name + sample count (MAP-01); an optional
  ``datasets=`` narrows the plotted markers to one source dataset (MAP-03).
* ``build_figure`` — the end-to-end wrapper: measured coords via the injectable
  ``data_access.load_geo`` seam (network-free in tests) + the committed inferred CSV ->
  ``shapes.aggregate_geo`` (category-graded) -> ``make_geo_figure``.
* ``render`` — a ``gr.Blocks`` fragment: a source-dataset ``gr.Dropdown`` filter (MAP-03) + a
  ``gr.Plot``.
* ``main`` — a standalone local Gradio app (``python -m planktonzilla.explorer.geomap``).

Design (load-bearing — D5): gradio AND plotly are imported INSIDE function bodies, NEVER at
module scope. The Phase 9 dependency-isolation guard (``tests/test_dependency_isolation.py``)
fails the build on any module-scope viz import under ``planktonzilla/``. The module's pure
helpers (``distinct_datasets``, ``filter_points_by_dataset``) import cleanly in the CORE env
with gradio/plotly ABSENT; the category grading lives in the pure ``shapes.aggregate_geo``.

⚠️ LIVE-READ COLD-START RISK (flag for Phase 14, D1): the MEASURED coordinates come from a LIVE
read of the public ``project-oceania/planktonzilla-17M`` via ``data_access.load_geo`` (17M rows,
column-projected to {Latitude, Longitude, dataset}, never images). Reading/aggregating 17M rows
on a free ``cpu-basic`` Space is slow/memory-heavy at cold start (possible timeout). Mitigation:
``aggregate_geo`` collapses to per-(dataset, rounded-coord) points IMMEDIATELY (we never hold or
plot 17M points) and ``data_access`` lru_caches the read for the process lifetime. Phase 14
should surface a loading state / consider hardware (snapshot fallback is NOT built now per D1).
"""

from __future__ import annotations

import polars as pl

from planktonzilla.explorer import data_access, shapes
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# The source-dataset filter column + no-filter sentinel (mirrors sankey/hierarchy — MAP-03).
DATASET_COLUMN = shapes.GEO_DATASET_COL  # "dataset"
ALL_DATASETS = "All"

# Category render order (drives trace order + the legend). Matches shapes.GEO_CATEGORIES.
CATEGORY_ORDER = (shapes.CATEGORY_MEASURED, shapes.CATEGORY_INFERRED_HIGH, shapes.CATEGORY_INFERRED_LOW)

# Friendly legend labels, keyed by the aggregate_geo category values (D3, MAP-02).
CATEGORY_LABELS = {
    shapes.CATEGORY_MEASURED: "Measured",
    shapes.CATEGORY_INFERRED_HIGH: "Inferred — high confidence",
    shapes.CATEGORY_INFERRED_LOW: "Inferred — low confidence",
}

# Per-category marker style — KNOWN vs INFERRED must read at a glance (D3). Measured = filled
# blue circle; inferred = HOLLOW (open) markers in warm colors, high=circle/red, low=diamond/
# orange (mirrors the notebook's hollow-marker convention for inferred sites).
CATEGORY_STYLE = {
    shapes.CATEGORY_MEASURED: {
        "color": "#2b5c8a",
        "symbol": "circle",
        "size": 8,
        "line": {"width": 0.5, "color": "#1a1a1a"},
    },
    shapes.CATEGORY_INFERRED_HIGH: {
        "color": "rgba(0,0,0,0)",  # hollow fill
        "symbol": "circle-open",
        "size": 11,
        "line": {"width": 2.0, "color": "#c0392b"},
    },
    shapes.CATEGORY_INFERRED_LOW: {
        "color": "rgba(0,0,0,0)",  # hollow fill
        "symbol": "diamond-open",
        "size": 12,
        "line": {"width": 2.0, "color": "#e67e22"},
    },
}

# Natural-earth basemap styling (matches the dev notebook look).
LAND_COLOR = "#f3f3f3"
OCEAN_COLOR = "#ffffff"
COASTLINE_COLOR = "#cfcfcf"


# --------------------------------------------------------------------------- #
# Pure helpers — polars / stdlib only. NO gradio / plotly here.
# --------------------------------------------------------------------------- #
def distinct_datasets(points: pl.DataFrame, *, column: str = DATASET_COLUMN) -> list[str]:
    """Return the dataset Dropdown choices: ``"All"`` plus sorted distinct dataset values (MAP-03).

    Args:
        points: A category-graded geo points frame (from ``shapes.aggregate_geo``).
        column: The source-dataset column name. Defaults to ``"dataset"``.

    Returns:
        ``["All", <sorted non-empty distinct dataset values>...]``. If ``column`` is absent
        the result is just ``["All"]``.
    """
    if column not in points.columns:
        return [ALL_DATASETS]
    vals = points.get_column(column).cast(pl.Utf8).str.strip_chars().unique().to_list()
    return [ALL_DATASETS, *sorted(v for v in vals if v)]


def filter_points_by_dataset(points: pl.DataFrame, dataset: str | None, *, column: str = DATASET_COLUMN) -> pl.DataFrame:
    """Narrow the geo points to a single source dataset (MAP-03).

    ``"All"``/``None``/empty (or a missing column) is a no-op that returns the frame
    unchanged. Otherwise rows are kept where ``column`` equals ``dataset`` exactly. The
    input is never mutated. Used ONLY to select rows in a polars frame — never for
    eval/exec, shell, or filesystem path interpolation (T-13-04).

    Args:
        points: A category-graded geo points frame.
        dataset: A dataset value, ``"All"``, ``None``, or "".
        column: The source-dataset column name. Defaults to ``"dataset"``.

    Returns:
        The filtered (or unchanged) frame.
    """
    if not dataset or dataset == ALL_DATASETS or column not in points.columns:
        return points
    return points.filter(pl.col(column).cast(pl.Utf8).str.strip_chars() == dataset)


def _hover_text(datasets: list[str], counts: list[int]) -> list[str]:
    """Build per-marker hover strings: dataset name + sample count (MAP-01)."""
    return [f"{ds}<br>{count} samples" for ds, count in zip(datasets, counts, strict=True)]


# --------------------------------------------------------------------------- #
# Lazy figure builder — plotly imported INSIDE the function body (D5).
# --------------------------------------------------------------------------- #
def make_geo_figure(points: pl.DataFrame, *, datasets: str | list[str] | None = None):
    """Build a token-free ``go.Scattergeo`` natural-earth map of sampling sites (MAP-01/02/03).

    The figure's ONLY traces are ``go.Scattergeo`` (type ``"scattergeo"``) on the built-in
    natural-earth basemap — NEVER ``go.Scattermapbox``/``scatter_mapbox``, no tiles, no token
    (D2). For each category PRESENT in ``points`` (in ``CATEGORY_ORDER``) exactly ONE legended,
    distinctly-styled trace is added (Measured / Inferred — high / Inferred — low, D3): one
    marker per (dataset, rounded site), hover shows the dataset name + sample count (MAP-01).
    Empty/na/no-coord input yields zero markers across all traces (SC4 — ``aggregate_geo``
    already excludes na + no-coord rows). ``datasets`` (a single dataset or a list) narrows the
    plotted markers to that dataset only (MAP-03); ``None``/``"All"`` plots everything.

    plotly is imported INSIDE this function (D5) so the module loads in the core env.

    Args:
        points: A ``shapes.aggregate_geo`` result (carries the ``category`` grade).
        datasets: A single dataset value or list to narrow to, or ``None``/``"All"`` for all.

    Returns:
        A ``plotly.graph_objects.Figure`` whose traces are all ``go.Scattergeo``.
    """
    import plotly.graph_objects as go

    # Narrow by dataset (MAP-03). Accept a single value or a list; "All"/None = no-op.
    selected = points
    if isinstance(datasets, str):
        selected = filter_points_by_dataset(points, datasets)
    elif isinstance(datasets, list) and datasets:
        wanted = [d for d in datasets if d and d != ALL_DATASETS]
        if wanted:
            selected = points.filter(pl.col(DATASET_COLUMN).cast(pl.Utf8).str.strip_chars().is_in(wanted))

    traces = []
    has_category = "category" in selected.columns
    for category in CATEGORY_ORDER:
        sub = selected.filter(pl.col("category") == category) if has_category else selected.clear()
        if sub.height == 0:
            continue  # only emit a trace (and a legend entry) for categories actually present
        style = CATEGORY_STYLE[category]
        traces.append(
            go.Scattergeo(
                lat=sub.get_column(shapes.GEO_LAT_COL).to_list(),
                lon=sub.get_column(shapes.GEO_LON_COL).to_list(),
                mode="markers",
                name=CATEGORY_LABELS[category],
                text=_hover_text(sub.get_column(DATASET_COLUMN).to_list(), sub.get_column("count").to_list()),
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "size": style["size"],
                    "line": style["line"],
                },
                showlegend=True,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor=LAND_COLOR,
        showocean=True,
        oceancolor=OCEAN_COLOR,
        showcoastlines=True,
        coastlinecolor=COASTLINE_COLOR,
        showcountries=False,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.08, "xanchor": "center", "x": 0.5},
        font={"color": "#1a1a1a", "size": 11},
    )
    return fig


def build_figure(
    *,
    repo_id: str | None = None,
    loader=None,
    dataset: str | list[str] | None = None,
):
    """End-to-end figure: measured (seam) + inferred (CSV) -> aggregate_geo -> make_geo_figure.

    Convenience wrapper used by ``render``'s callbacks and the tests. The MEASURED coords are
    read ONLY through the injectable ``data_access.load_geo`` seam (D1) — when ``loader`` is
    given (tests pass a fake) the network is never reached; the inferred locations come from the
    committed local CSV. ``shapes.aggregate_geo`` produces the category-graded points, which are
    optionally narrowed by ``dataset`` (MAP-03) and rendered by ``make_geo_figure``.

    Args:
        repo_id: HF dataset repo id. Defaults to the frozen public planktonzilla repo.
        loader: Optional injected measured loader ``(repo_id) -> pl.DataFrame`` (network-free).
        dataset: Optional source-dataset narrowing for the figure.

    Returns:
        A ``plotly.graph_objects.Figure`` whose traces are all ``go.Scattergeo``.
    """
    measured = data_access.load_geo(repo_id, loader=loader) if repo_id else data_access.load_geo(loader=loader)
    inferred = data_access.inferred_locations()
    points = shapes.aggregate_geo(measured, inferred)
    return make_geo_figure(points, datasets=dataset)


def _points(loader=None) -> pl.DataFrame:
    """Build the category-graded points frame via the seam (measured) + committed inferred CSV."""
    measured = data_access.load_geo(loader=loader)
    inferred = data_access.inferred_locations()
    return shapes.aggregate_geo(measured, inferred)


# --------------------------------------------------------------------------- #
# Lazy UI fragment — gradio imported INSIDE the function body (D5).
# --------------------------------------------------------------------------- #
def render(points: pl.DataFrame | None = None, *, loader=None):
    """Build the Geographic Map ``gr.Blocks`` fragment (dataset filter + plot).

    Wires (D1/D2/D3/D5):

    * a ``gr.Dropdown`` source-dataset filter (default ``"All"``, MAP-03),
    * a ``gr.Plot`` output rendering the token-free natural-earth ``go.Scattergeo`` map.

    gradio is imported INSIDE this function (D5). When ``points`` is not supplied the
    category-graded points are built via the injectable ``data_access.load_geo`` seam (D1) +
    the committed inferred CSV (network-free when ``loader`` is injected).

    Args:
        points: Optional pre-built category-graded points frame. Built via the seam when ``None``.
        loader: Optional injected measured loader ``(repo_id) -> pl.DataFrame`` (network-free).

    Returns:
        A ``gr.Blocks`` fragment ready to be composed into the Space (Phase 14) or launched
        standalone by ``main``.
    """
    import gradio as gr

    if points is None:
        points = _points(loader=loader)

    choices = distinct_datasets(points)

    def _update(dataset):
        return make_geo_figure(points, datasets=dataset)

    with gr.Blocks() as fragment:
        gr.Markdown(
            "## Sampling Locations\nSource-dataset sampling sites on a token-free world map. "
            "Measured locations have per-sample GPS; inferred locations are dataset-level "
            "centroids (hollow markers) graded by confidence."
        )
        with gr.Row():
            with gr.Column(scale=1):
                dataset_in = gr.Dropdown(choices=choices, value=ALL_DATASETS, label="Source dataset filter")
            with gr.Column(scale=4):
                plot = gr.Plot(label="Sampling locations")

        dataset_in.change(_update, inputs=[dataset_in], outputs=[plot])
        fragment.load(_update, inputs=[dataset_in], outputs=[plot])

    return fragment


def main(argv: list[str] | None = None) -> int:
    """Launch the standalone local Geographic Map app (``python -m planktonzilla.explorer.geomap``).

    Wraps ``render()`` in a ``gr.Blocks`` and ``.launch()``-es it for local development. No core
    console script imports viz at module scope (D5); this entry point runs under the explorer
    group only. ⚠️ live cold-start risk applies (D1 — see the module docstring).

    Args:
        argv: Unused; accepted for CLI-symmetry with the other entry points.

    Returns:
        Process exit code (0 on a clean launch).
    """
    import gradio as gr

    with gr.Blocks(title="planktonzilla — sampling locations") as app:
        render()
    app.launch()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
