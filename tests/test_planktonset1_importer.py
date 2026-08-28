"""Regression tests for PlanktonSet-1, whose download endpoint is actively hostile.

`planktonset1.0` failed run after run, and for reasons that were invisible from the logs.
Its `download_uris` is not a stored file but NCEI's Archive Management System *generator*,
which tars the accession on demand. Measured against the live service on 2026-08-27:

  - ~22 s to first byte, then ~0.6 MB/s — about an hour for the whole archive;
  - `Range` IGNORED (a ranged GET answers 200, not 206), so nothing can resume it;
  - `Transfer-Encoding: chunked` with no `Content-Length`, so a short read cannot be
    caught by comparing sizes;
  - a sustained `503` when busy, hours after it had served 57 MB happily.

Meanwhile `datasets` never implemented the `resume_download`/`max_retries` this project
was passing it, and its HEAD-then-GET probing made the server build the tarball two or
three times per attempt. So: an hour-long, unresumable, unverified, triple-built download.

`_fetch_archive_verified` replaces that with one request per attempt, a gzip-framing check
that catches truncation even with no declared size, and retries that actually happen. The
other half of the fix is that the class tree is now LOCATED rather than walked to via a
hard-coded path containing the accession version.
"""

import gzip
import io
import tarfile

import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from hydra.core.global_hydra import GlobalHydra

import planktonzilla.dataset_import.dataset_importer as di


def _importer(tmp_path, extra=()):
    """Instantiate the real planktonset1 config against a temporary data_dir."""
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_planktonset1")
    cfg = hydra.compose(
        config_name="import_dataset",
        overrides=[
            "dataset_import=planktonset1",
            "dataset_import.push_to_hub=False",
            "dataset_import.show_progress=False",
            f"paths.data_dir={tmp_path}",
            f"dataset_import.data_dir={tmp_path}",
            *extra,
        ],
    )
    importer = hydra.utils.instantiate(cfg.dataset_import)
    GlobalHydra.instance().clear()
    return importer


class _Streamed:
    """A streaming requests response. ``declared=None`` omits Content-Length entirely.

    Omitting it is the case that matters: the real endpoint sends chunked with no length,
    which is precisely why a size comparison cannot be the truncation check.
    """

    def __init__(self, chunks, declared=None):
        self._chunks = chunks
        self.headers = {} if declared is None else {"Content-Length": str(declared)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        yield from self._chunks


def _targz_bytes(names=("FINAL_Plankton_Segments_12082014/acantharia_protist/0.jpg",)) -> bytes:
    """A real, well-formed .tar.gz — so the gzip check passes for the right reason."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 4
            tar.addfile(info, io.BytesIO(b"data"))
    return gzip.compress(raw.getvalue())


def test_fetch_archive_verified_accepts_a_whole_archive_and_reuses_it(tmp_path, monkeypatch):
    """One request per attempt, atomic rename, and a second call re-fetches nothing."""
    importer = _importer(tmp_path)
    payload = _targz_bytes()
    calls = []

    def _fake_get(url, **kwargs):
        calls.append((url, kwargs["headers"]["User-Agent"]))
        # Chunked and sizeless, exactly like the real endpoint.
        return _Streamed([payload[:20], payload[20:]])

    monkeypatch.setattr(di.requests, "get", _fake_get)

    target = importer.raw_dir / "0127422.2.3.tar.gz"
    got = importer._fetch_archive_verified("https://example.invalid/a.tar.gz", target)

    assert got.read_bytes() == payload
    assert not list(got.parent.glob("*.part")), "the partial must not survive"
    assert len(calls) == 1, "the DownloadManager's HEAD-then-GET is what we are avoiding"
    assert calls[0][1] == importer.http_user_agent, "identify as this project, as the probe does"

    assert importer._fetch_archive_verified("https://example.invalid/a.tar.gz", target) == got
    assert len(calls) == 1, "an archive already on disk is not re-fetched"


def test_fetch_archive_verified_catches_a_truncated_stream_with_no_content_length(tmp_path, monkeypatch):
    """The failure the endpoint makes undetectable by size, caught by gzip framing.

    A cut-short body is still a *clean* HTTP response, and with no Content-Length there is
    nothing to compare against — so without this the short archive is renamed into place
    and surfaces hours later as a corrupt extraction or a silently short dataset.
    """
    importer = _importer(tmp_path)
    truncated = _targz_bytes()[:40]
    attempts = []

    def _fake_get(url, **kwargs):
        attempts.append(url)
        return _Streamed([truncated])

    monkeypatch.setattr(di.requests, "get", _fake_get)
    monkeypatch.setattr(di.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        importer._fetch_archive_verified("https://example.invalid/a.tar.gz", importer.raw_dir / "0127422.2.3.tar.gz")

    assert len(attempts) == 3, "an unresumable download is retried from the start, not resumed"
    assert not list(importer.raw_dir.glob("*")), "nothing is left that a later run could mistake for the archive"


def test_fetch_archive_verified_retries_a_transient_failure(tmp_path, monkeypatch):
    """503 is what the live endpoint returns when busy, and it clears on its own."""
    importer = _importer(tmp_path)
    payload = _targz_bytes()
    calls = []

    def _fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            raise di.requests.ConnectionError("503 Service Unavailable")
        return _Streamed([payload])

    monkeypatch.setattr(di.requests, "get", _fake_get)
    monkeypatch.setattr(di.time, "sleep", lambda seconds: None)

    got = importer._fetch_archive_verified("https://example.invalid/a.tar.gz", importer.raw_dir / "0127422.2.3.tar.gz")

    assert got.read_bytes() == payload
    assert len(calls) == 3, "the retry loop lives here because datasets' max_retries is inert"


def test_segments_root_is_found_even_when_the_accession_version_changes(tmp_path):
    """The latent silent-empty bug: the old path pinned the accession VERSION.

    It walked `0127422/2.3/data/0-data/FINAL_Plankton_Segments_12082014`. Upstream already
    publishes 1.1, 2.2 and 2.3, so a 2.4 re-release would have made `.glob` match nothing
    — no error, just an empty imagefolder — which is exactly how whoi and SYKE ZooScan
    failed. The tree is located now, so a version bump is survivable.
    """
    importer = _importer(tmp_path)
    extracted = tmp_path / "extracted"
    moved = extracted / "0127422" / "2.9" / "data" / "0-data" / "FINAL_Plankton_Segments_20990101"
    (moved / "acantharia_protist").mkdir(parents=True)
    (moved / "acantharia_protist" / "0.jpg").write_bytes(b"x")
    importer.extracted_dirs = str(extracted)

    assert importer._segments_root() == moved


def test_prepare_imagefolder_copies_the_class_folders_it_locates(tmp_path):
    """End to end over a fake extraction, including the macOS junk the archive carries."""
    importer = _importer(tmp_path)
    extracted = tmp_path / "extracted"
    segments = extracted / "0127422" / "2.3" / "data" / "0-data" / "FINAL_Plankton_Segments_12082014"
    for klass in ("acantharia_protist", "amphipods"):
        (segments / klass).mkdir(parents=True)
        (segments / klass / "0.jpg").write_bytes(b"x")
    (segments / ".DS_Store").write_bytes(b"junk")
    (segments / "._hidden").mkdir()

    importer.extracted_dirs = str(extracted)
    importer.imagefolder_dir.mkdir(parents=True, exist_ok=True)
    importer._prepare_imagefolder()

    copied = sorted(p.name for p in importer.imagefolder_dir.iterdir())
    assert copied == ["acantharia_protist", "amphipods"], "dotfile and macOS entries are skipped"
    assert (importer.imagefolder_dir / "amphipods" / "0.jpg").exists()


def test_giving_up_names_the_static_mirror_as_the_way_out(tmp_path, monkeypatch):
    """When the generator is down the mirror is the only route, so the failure must say so.

    `manual_download_instructions()` deliberately stays silent unless a manual archive is
    already configured AND missing, so on this source it returns "" — silent at exactly the
    moment the fallback is worth knowing. The give-up message therefore carries the hint
    itself, rather than sending the reader back to the AMS page that just failed them.
    """
    importer = _importer(tmp_path)

    assert "ftp-oceans.ncei.noaa.gov" in importer.manual_download_url
    assert "arc0075" in importer.manual_download_url
    assert "FINAL_Plankton_Segments" in importer.manual_download_notes

    monkeypatch.setattr(di.requests, "get", lambda url, **kw: (_ for _ in ()).throw(di.requests.ConnectionError("503")))
    monkeypatch.setattr(di.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError) as excinfo:
        importer._fetch_archive_verified("https://example.invalid/a.tar.gz", importer.raw_dir / "0127422.2.3.tar.gz")

    message = str(excinfo.value)
    assert "ftp-oceans.ncei.noaa.gov" in message, "name the fallback route at the moment it is needed"
    assert "manual_download_local_file_names" in message, "and how to hand the result back"
