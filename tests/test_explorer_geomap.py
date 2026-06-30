"""
(c) Inria

Two-environment, network-free tests for the explorer Geographic Map view (Phase 13, MAP-01..03).

Two groups:

* CORE-SAFE (run everywhere, no plotly/gradio): the module imports with viz ABSENT (D5); the
  pure ``shapes.aggregate_geo`` grading/exclusion contract over a FAKE measured frame + the
  committed inferred CSV (na -> 0 rows, zoolake=47.35/8.68 inferred-high, sykezooscan2024
  inferred-low — SC4/SC2); and ``geomap.filter_points_by_dataset``/``distinct_datasets``.
* EXPLORER-GROUP (``pytest.importorskip("plotly")``/``"gradio"`` — SKIP in core, RUN under the
  explorer group CI job): ``make_geo_figure`` traces are ALL ``go.Scattergeo`` (NEVER
  ``scattermapbox``, D2) with one trace per category present (MAP-01/02); na/no-coord-only
  points -> zero markers (SC4); a DeprecationWarning-free build (SC1); the dataset filter
  narrows plotted markers (MAP-03); ``build_figure(loader=fake)`` reaches measured coords ONLY
  via the injected seam (network-blocked, SC4); and a ``render`` smoke.

All tests are network-free: an autouse stdlib socket-block fixture makes any real socket raise
(belt-and-suspenders alongside the injected fake loader); the live HF read is NEVER reached.
"""

from __future__ import annotations

import importlib
import os
import socket
import warnings

import polars as pl
import pytest

from planktonzilla.explorer import data_access, shapes

# Disable gradio telemetry/analytics so render() stays network-free. Set before any gradio
# import in the explorer-group tests (the core tests never import gradio at all).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# A FAKE measured frame mirroring the live-HF casing {Latitude, Longitude, dataset}: two
# datasets, two near-duplicate points for "alpha" (collapse to 1 with count 2) and one for "beta".
FAKE_MEASURED = pl.DataFrame(
    {
        "Latitude": [1.5, 1.5, 40.0],
        "Longitude": [3.5, 3.5, -70.0],
        "dataset": ["alpha", "alpha", "beta"],
    }
)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Make any real INTERNET socket connection raise — enforces network-free without pytest-socket.

    Only INET/INET6 family sockets (and ``create_connection``, which is internet-only) are
    blocked: a live HF read fails LOUDLY here. Local ``AF_UNIX`` socketpairs (which gradio's
    internal asyncio event loop creates when ``gr.Blocks`` is constructed) are allowed through —
    they never cross to the network, so blocking them would only break the offline render smoke.
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


def _graded_points() -> pl.DataFrame:
    """Category-graded points over the FAKE measured frame + the committed inferred CSV."""
    return shapes.aggregate_geo(FAKE_MEASURED, data_access.inferred_locations())


# --------------------------------------------------------------------------- #
# (a) CORE-SAFE: module import + grading/exclusion + filter helpers.
# --------------------------------------------------------------------------- #
def test_geomap_module_imports_in_core():
    """The module imports with NO viz at module scope (D5) — proves the lazy seam holds."""
    mod = importlib.import_module("planktonzilla.explorer.geomap")
    assert hasattr(mod, "make_geo_figure")
    assert hasattr(mod, "build_figure")
    assert hasattr(mod, "render")
    assert hasattr(mod, "main")
    assert hasattr(mod, "distinct_datasets")
    assert hasattr(mod, "filter_points_by_dataset")


def test_aggregate_geo_grading_and_na_exclusion():
    """Category values are graded; confidence=na rows (planktoscope, lensless) yield ZERO rows (SC4)."""
    points = _graded_points()
    assert "category" in points.columns
    assert set(points.get_column("category").to_list()) <= set(shapes.GEO_CATEGORIES)
    # na rows excluded entirely — no real collection site.
    names = points.get_column("dataset").to_list()
    assert "planktoscope" not in names
    assert "lensless" not in names
    # zoolake (high) survives at rounded Greifensee 47.35/8.68 as inferred-high (SC2).
    zoolake = points.filter(pl.col("dataset") == "zoolake")
    assert zoolake.height == 1
    assert zoolake.get_column("Latitude").to_list()[0] == 47.35
    assert zoolake.get_column("Longitude").to_list()[0] == 8.68
    assert zoolake.get_column("category").to_list()[0] == "inferred-high"
    # sykezooscan2024 (low) -> inferred-low style category.
    syke = points.filter(pl.col("dataset") == "sykezooscan2024")
    assert syke.get_column("category").to_list() == ["inferred-low"]
    # The fake measured datasets are present and graded "measured".
    measured_cats = points.filter(pl.col("source") == "measured").get_column("category").to_list()
    assert set(measured_cats) == {"measured"}


def test_filter_points_by_dataset_narrows_and_all_is_noop():
    """filter_points_by_dataset narrows to one dataset; 'All'/None/'' is a no-op (MAP-03)."""
    from planktonzilla.explorer import geomap

    points = _graded_points()
    only_alpha = geomap.filter_points_by_dataset(points, "alpha")
    assert set(only_alpha.get_column("dataset").to_list()) == {"alpha"}
    assert only_alpha.height < points.height
    # No-op cases return the frame unchanged (same height).
    assert geomap.filter_points_by_dataset(points, geomap.ALL_DATASETS).height == points.height
    assert geomap.filter_points_by_dataset(points, None).height == points.height
    assert geomap.filter_points_by_dataset(points, "").height == points.height


def test_distinct_datasets_choices():
    """distinct_datasets returns ['All', *sorted distinct] (MAP-03 dropdown choices)."""
    from planktonzilla.explorer import geomap

    points = _graded_points()
    choices = geomap.distinct_datasets(points)
    assert choices[0] == geomap.ALL_DATASETS
    assert "alpha" in choices and "beta" in choices and "zoolake" in choices
    assert choices[1:] == sorted(choices[1:])


# --------------------------------------------------------------------------- #
# (b) EXPLORER-GROUP: scattergeo (not mapbox) + 3 graded traces + na->0 + filter + smoke.
# --------------------------------------------------------------------------- #
def test_make_geo_figure_is_scattergeo_not_mapbox():
    """Every trace is go.Scattergeo (type 'scattergeo'); NONE is scattermapbox (D2, MAP-01/02)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import geomap

    fig = geomap.make_geo_figure(_graded_points())
    assert fig.data, "figure must have at least one trace"
    types = {trace.type for trace in fig.data}
    assert types == {"scattergeo"}
    assert "scattermapbox" not in types


def test_make_geo_figure_three_graded_traces():
    """measured + inferred-high + inferred-low present -> exactly 3 legended category traces (MAP-02)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import geomap

    fig = geomap.make_geo_figure(_graded_points())
    names = [trace.name for trace in fig.data]
    assert len(fig.data) == 3
    assert set(names) == {
        "Measured",
        "Inferred — high confidence",
        "Inferred — low confidence",
    }
    assert all(trace.showlegend for trace in fig.data)


def test_make_geo_figure_na_and_no_coord_yield_zero_markers():
    """na/no-coord-only inputs -> total marker count across traces == 0 (SC4)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import geomap

    # An inferred-only frame whose every row is na or blank-coord -> aggregate_geo drops all.
    na_inferred = pl.DataFrame(
        {
            "dataset": ["planktoscope", "lensless"],
            "latitude": ["", ""],
            "longitude": ["", ""],
            "confidence": ["na", "na"],
        }
    )
    empty_measured = pl.DataFrame({"Latitude": [], "Longitude": [], "dataset": []})
    points = shapes.aggregate_geo(empty_measured, na_inferred)
    fig = geomap.make_geo_figure(points)
    total_markers = sum(len(trace.lat or ()) for trace in fig.data)
    assert total_markers == 0


def test_make_geo_figure_deprecationwarning_free():
    """Building the figure emits NO DeprecationWarning (SC1 — scatter_geo not deprecated mapbox)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import geomap

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        geomap.make_geo_figure(_graded_points())
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, f"unexpected DeprecationWarning(s): {[str(w.message) for w in deprecations]}"


def test_dataset_filter_narrows_plotted_markers():
    """make_geo_figure(points, datasets='alpha') has fewer total markers than datasets=None (MAP-03)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import geomap

    points = _graded_points()
    all_markers = sum(len(trace.lat or ()) for trace in geomap.make_geo_figure(points).data)
    alpha_markers = sum(len(trace.lat or ()) for trace in geomap.make_geo_figure(points, datasets="alpha").data)
    assert alpha_markers > 0
    assert alpha_markers < all_markers


def test_build_figure_reaches_measured_only_via_injected_seam(monkeypatch):
    """build_figure(loader=fake) returns a scattergeo figure WITHOUT touching the real loader (SC4)."""
    pytest.importorskip("plotly")
    from planktonzilla.explorer import geomap

    def _boom(*args, **kwargs):
        raise AssertionError("real _default_geo_loader must NOT run when a loader is injected")

    monkeypatch.setattr(data_access, "_default_geo_loader", _boom)

    fig = geomap.build_figure(loader=lambda repo_id: FAKE_MEASURED)
    assert fig.data and {trace.type for trace in fig.data} == {"scattergeo"}
    # The injected measured datasets reached the figure (proves the seam fed the figure).
    plotted = set()
    for trace in fig.data:
        for line in trace.text or ():
            plotted.add(line.split("<br>")[0])
    assert {"alpha", "beta"} <= plotted


def test_render_smoke():
    """render(points=...) builds a gr.Blocks fragment (network-free)."""
    pytest.importorskip("plotly")
    gr = pytest.importorskip("gradio")
    from planktonzilla.explorer import geomap

    fragment = geomap.render(_graded_points())
    assert isinstance(fragment, gr.Blocks)
