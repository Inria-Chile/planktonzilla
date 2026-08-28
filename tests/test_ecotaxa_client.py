"""
(c) Inria

Network-free tests for :mod:`planktonzilla.dataset_import.ecotaxa_client`.

Every test drives the client with an injected fake session, so nothing here reaches
ecotaxa.obs-vlfr.fr. What they pin is exactly the behaviour a 2.35-million-object walk
depends on and that a live test could never demonstrate cheaply:

  * the manifest is paged to ``total_ids`` and a SHORT answer is refused rather than
    silently importing an incomplete dataset;
  * a manifest survives the disk round trip, and a stale header is refused rather than
    half-read;
  * a vignette already on disk is not re-fetched (that is what makes a killed run
    resumable), each one lands atomically, and a dead object is COLLECTED rather than
    aborting the other million;
  * a 4xx is not retried, so a permission change surfaces as itself instead of five
    identical timeouts.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import pytest
import requests

from planktonzilla.dataset_import import ecotaxa_client as ec

# One manifest row's worth of `details`, in MANIFEST_FIELD_SPECS order.
ROW_A = [
    1129200000001,
    "tara_pacific_2016_i00oa10_d_hsn_330_tot_1_36",
    5,
    "Harosa",
    33.59,
    -34.78,
    "2016-06-09",
    "09:40:00",
    0.0,
    1.0,
    "63756/4568.jpg",
]
ROW_B = [
    1129200000002,
    "tara_pacific_2016_i00oa10_d_hsn_330_tot_1_430",
    5,
    "Harosa",
    33.59,
    -34.78,
    "2016-06-09",
    "09:40:00",
    0.0,
    1.0,
    "63756/4569.jpg",
]
ROW_C = [
    1129200000003,
    "tara_pacific_2016_i00oa10_d_hsn_330_tot_1_513",
    25828,
    "Copepoda<Multicrustacea",
    None,
    None,
    None,
    None,
    None,
    None,
    "63756/4570.jpg",
]


class _Response:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    """Records every call and replays a scripted list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.posts, self.gets = [], []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append((url, headers, json))
        return self._next()

    def get(self, url, headers=None, timeout=None):
        self.gets.append((url, headers))
        return self._next()

    def _next(self):
        return self._responses.pop(0) if self._responses else _Response(500)


def _page(rows, total):
    return _Response(payload={"details": [list(r) for r in rows], "total_ids": total})


def test_urls_are_the_public_read_endpoints():
    assert ec.project_url(11292) == "https://ecotaxa.obs-vlfr.fr/api/projects/11292"
    assert ec.vault_url("63756/4568.jpg") == "https://ecotaxa.obs-vlfr.fr/vault/63756/4568.jpg"

    url = ec.query_url(11292, window_start=20, window_size=10)
    assert url.startswith("https://ecotaxa.obs-vlfr.fr/api/object_set/11292/query?")
    # Ordering by object id server-side is what makes a re-fetch byte-identical.
    assert "order_field=obj.objid" in url
    assert "window_start=20&window_size=10" in url
    assert ec.MANIFEST_FIELDS in url


def test_manifest_fields_and_columns_stay_aligned():
    """The endpoint returns ``details`` in REQUEST order, so the two lists must match."""
    assert len(ec.MANIFEST_FIELDS.split(",")) == len(ec.MANIFEST_COLUMNS)
    assert ec.MANIFEST_COLUMNS[0] == "objid"
    assert "img.file_name" in ec.MANIFEST_FIELDS and "obj.classif_id" in ec.MANIFEST_FIELDS


def test_fetch_project_manifest_pages_to_total_ids():
    session = _FakeSession([_page([ROW_A, ROW_B], total=3), _page([ROW_C], total=3)])

    rows = ec.fetch_project_manifest(11292, session=session, window_size=2, show_progress=False)

    assert len(rows) == 3
    assert [row["objid"] for row in rows] == [ROW_A[0], ROW_B[0], ROW_C[0]]
    assert rows[0]["display_name"] == "Harosa"
    assert rows[2]["img_file_name"] == "63756/4570.jpg"
    # Two windows, advancing by what was actually received.
    assert "window_start=0" in session.posts[0][0]
    assert "window_start=2" in session.posts[1][0]
    assert session.posts[0][2] == {"filters": {}}


def test_fetch_project_manifest_refuses_a_short_answer():
    """A manifest short of ``total_ids`` would import a silently incomplete dataset."""
    session = _FakeSession([_page([ROW_A], total=3), _page([], total=3)])

    with pytest.raises(ec.EcoTaxaError, match="promised 3 objects but returned 1"):
        ec.fetch_project_manifest(11292, session=session, window_size=1, show_progress=False)


def test_fetch_project_manifest_sends_the_project_user_agent():
    session = _FakeSession([_page([ROW_A], total=1)])
    ec.fetch_project_manifest(11292, session=session, user_agent="planktonzilla/x", show_progress=False)
    assert session.posts[0][1]["User-Agent"] == "planktonzilla/x"


def test_request_retries_a_server_error_then_succeeds():
    session = _FakeSession([_Response(503), _page([ROW_A], total=1)])
    rows = ec.fetch_project_manifest(11292, session=session, retries=3, show_progress=False)
    assert len(rows) == 1
    assert len(session.posts) == 2


def test_request_does_not_retry_a_client_error():
    """403 will not become 200 by waiting; report it once, with its status."""
    session = _FakeSession([_Response(403), _page([ROW_A], total=1)])

    with pytest.raises(ec.EcoTaxaError, match="HTTP 403"):
        ec.fetch_project_manifest(11292, session=session, retries=5, show_progress=False)
    assert len(session.posts) == 1


def test_request_surfaces_a_transport_failure_after_its_retries():
    class _Boom(_FakeSession):
        def post(self, *args, **kwargs):
            self.posts.append(args)
            raise requests.ConnectionError("down")

    session = _Boom([])
    with pytest.raises(ec.EcoTaxaError, match="ConnectionError"):
        ec.fetch_project_manifest(11292, session=session, retries=2, show_progress=False)
    assert len(session.posts) == 2


def test_manifest_round_trips_through_disk(tmp_path):
    session = _FakeSession([_page([ROW_A, ROW_C], total=2)])
    rows = ec.fetch_project_manifest(11292, session=session, show_progress=False)

    path = ec.write_manifest(rows, tmp_path / "ecotaxa_project_11292.tsv")
    back = ec.read_manifest(path)

    assert [row["objid"] for row in back] == [row["objid"] for row in rows]
    assert [row["classif_id"] for row in back] == [5, 25828]
    # Ids keep their type; everything else comes back as text (documented), and a blank
    # column comes back as None rather than "".
    assert back[0]["latitude"] == "33.59"
    assert back[1]["latitude"] is None
    assert back[1]["objdate"] is None


def test_write_manifest_is_atomic(tmp_path):
    """Written via a temp file and renamed, so a kill never leaves a short manifest."""
    path = tmp_path / "nested" / "ecotaxa_project_1.tsv"
    ec.write_manifest([dict(zip(ec.MANIFEST_COLUMNS, ROW_A))], path)

    assert path.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_read_manifest_refuses_a_stale_header(tmp_path):
    path = tmp_path / "ecotaxa_project_1.tsv"
    path.write_text("objid\torig_id\n1\tx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Delete it and re-run"):
        ec.read_manifest(path)


def test_download_vault_images_skips_what_is_already_on_disk(tmp_path):
    """The property that makes an interrupted 2.35-million-image walk resumable."""
    existing = tmp_path / "Harosa" / "1.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already here")

    session = _FakeSession([_Response(content=b"new bytes")])
    jobs = [("https://vault/1.jpg", existing), ("https://vault/2.jpg", tmp_path / "Harosa" / "2.jpg")]

    fetched, skipped, failures = ec.download_vault_images(jobs, session=session, workers=1, show_progress=False)

    assert (fetched, skipped, failures) == (1, 1, [])
    assert existing.read_bytes() == b"already here"
    assert (tmp_path / "Harosa" / "2.jpg").read_bytes() == b"new bytes"
    assert len(session.gets) == 1


def test_download_vault_images_leaves_no_partial_file(tmp_path):
    """Each vignette is renamed into place, and the temp name is invisible to the loader."""
    session = _FakeSession([_Response(content=b"bytes")])
    destination = tmp_path / "Harosa" / "1.jpg"

    ec.download_vault_images([("https://vault/1.jpg", destination)], session=session, workers=1, show_progress=False)

    assert destination.read_bytes() == b"bytes"
    # `resolve_imagefolder_glob` matches `[!._]*`, so a leftover must be dot-prefixed.
    assert not list(tmp_path.rglob("*.part"))


def test_download_vault_images_collects_failures_instead_of_raising(tmp_path):
    """One dead object must not abandon the million good ones; the caller decides."""
    session = _FakeSession([_Response(404), _Response(content=b"ok")])
    jobs = [
        ("https://vault/missing.jpg", tmp_path / "A" / "1.jpg"),
        ("https://vault/present.jpg", tmp_path / "A" / "2.jpg"),
    ]

    fetched, skipped, failures = ec.download_vault_images(jobs, session=session, workers=1, retries=2, show_progress=False)

    assert (fetched, skipped) == (1, 0)
    assert len(failures) == 1 and "HTTP 404" in failures[0]
    assert not (tmp_path / "A" / "1.jpg").exists()


def test_download_vault_images_is_a_noop_for_no_jobs():
    assert ec.download_vault_images([], show_progress=False) == (0, 0, [])
