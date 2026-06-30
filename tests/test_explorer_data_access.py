"""
(c) Inria

Network-free tests for the explorer's single network boundary (FND-06).

The suite must NEVER cross to the network. Two belt-and-suspenders guarantees:

1. Seam injection: ``load_geo(loader=fake)`` returns the fake frame and the real
   ``_default_geo_loader`` (and its lazy huggingface_hub/pyarrow imports) is never
   reached.
2. A stdlib ``autouse`` socket-block fixture (no ``pytest-socket`` dependency —
   deps fence) monkeypatches ``socket.socket`` / ``socket.create_connection`` to
   raise, so any accidental live read fails LOUDLY instead of silently hitting HF.

The tests also pin the projection contract: ``GEO_COLUMNS`` is exactly
``{Latitude, Longitude, dataset}`` and EXCLUDES ``image`` (T-10-01 — never load the
17M-image payload), and the in-process cache on the real path is bypassed by the
injected path (a fake never poisons a real cache entry).
"""

import socket
import sys

import polars as pl
import pytest

from planktonzilla.explorer import data_access

FAKE_GEO_FRAME = pl.DataFrame({"Latitude": [1.5, 2.5], "Longitude": [3.5, 4.5], "dataset": ["alpha", "beta"]})


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Make any real socket connection raise — enforces network-free without pytest-socket."""

    def _no_socket(*args, **kwargs):
        raise RuntimeError("network access is blocked in the explorer test suite")

    monkeypatch.setattr(socket, "socket", _no_socket)
    monkeypatch.setattr(socket, "create_connection", _no_socket)
    yield


def test_injected_loader_returns_fake_without_network(monkeypatch):
    """The injected loader is used; the real loader is never invoked."""

    def _boom(*args, **kwargs):
        raise AssertionError("real _default_geo_loader must NOT be called when a loader is injected")

    monkeypatch.setattr(data_access, "_default_geo_loader", _boom)

    out = data_access.load_geo(loader=lambda repo_id: FAKE_GEO_FRAME)
    assert list(out.columns) == ["Latitude", "Longitude", "dataset"]
    assert out.height == 2


def test_injected_path_imports_no_heavy_deps(monkeypatch):
    """The injected path must not pull in huggingface_hub (lazy seam intact).

    Order-independent: another test in the full suite may legitimately import
    huggingface_hub, so we assert the injected call does not NEWLY import it (and
    that the real loader is never reached, which is where the lazy import lives)
    rather than asserting it is globally absent.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("real _default_geo_loader (the lazy-import site) must NOT run on the injected path")

    monkeypatch.setattr(data_access, "_default_geo_loader", _boom)

    hf_before = "huggingface_hub" in sys.modules
    data_access.load_geo(loader=lambda repo_id: FAKE_GEO_FRAME)
    hf_after = "huggingface_hub" in sys.modules
    # The injected path must not be the thing that imports huggingface_hub.
    assert hf_after == hf_before
    # gradio/plotly must never be present at all (Phase 9 isolation).
    assert "gradio" not in sys.modules
    assert "plotly" not in sys.modules


def test_geo_columns_contract_excludes_image():
    """The projection is exactly {Latitude, Longitude, dataset} and never image."""
    assert tuple(data_access.GEO_COLUMNS) == ("Latitude", "Longitude", "dataset")
    assert "image" not in data_access.GEO_COLUMNS


def test_injected_loader_receives_repo_id():
    """The injected loader is called with the repo_id (default or explicit)."""
    seen = {}

    def _spy(repo_id):
        seen["repo_id"] = repo_id
        return FAKE_GEO_FRAME

    data_access.load_geo(repo_id="some/repo", loader=_spy)
    assert seen["repo_id"] == "some/repo"

    data_access.load_geo(loader=_spy)
    assert seen["repo_id"] == data_access.constants.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID


def test_injected_path_bypasses_cache():
    """Two injected calls with different fakes each return their own frame (no cache poison)."""
    frame_a = pl.DataFrame({"Latitude": [1.0], "Longitude": [1.0], "dataset": ["a"]})
    frame_b = pl.DataFrame({"Latitude": [2.0], "Longitude": [2.0], "dataset": ["b"]})

    out_a = data_access.load_geo(loader=lambda _: frame_a)
    out_b = data_access.load_geo(loader=lambda _: frame_b)
    assert out_a.get_column("dataset").to_list() == ["a"]
    assert out_b.get_column("dataset").to_list() == ["b"]


def test_inferred_locations_reads_committed_csv_no_network():
    """inferred_locations() reads the committed CSV locally (no network)."""
    out = data_access.inferred_locations()
    assert "dataset" in out.columns
    assert "latitude" in out.columns and "longitude" in out.columns
    assert out.height > 0
    # na rows (planktoscope/lensless) are present here as blanks — drop happens in shapes.
    assert "planktoscope" in out.get_column("dataset").to_list()


def test_socket_block_fixture_is_active():
    """Sanity: the autouse socket block makes a raw socket() raise."""
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.socket()
