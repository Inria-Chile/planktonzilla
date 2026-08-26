"""
(c) Inria

Network-free tests for the ``configs/dataset_import/`` config group and for the
layout-independent class-folder scan added with the MedPlanktonSet importer.

The instantiation test is a guard, not a formality: before it existed, two configs
named ``_target_`` classes that do not exist —
``medplanktonset.yaml`` pointed at ``MedPlanktonSetDatasetImporter`` (never written)
and ``sykezooscan2024.yaml`` at ``SYKEZooScan2024`` (missing the
``DatasetImporter`` suffix). ``medplanktonset`` is the 5th of the active entries
in ``configs/generate_planktonzilla.yaml``, so a full build died partway through,
after four sources had already been downloaded and processed. Both are fixed; this
test fails the moment a config names a class that cannot be located.

Nothing here touches the network or downloads anything: instantiating an importer only
builds the dataclass and derives its ``imagefolder_dir`` / ``raw_dir`` paths.
"""

import glob
import hashlib
import inspect
import logging
import os
import shutil
from multiprocessing import cpu_count
from pathlib import Path
from types import SimpleNamespace

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from PIL import Image as PILImage

# Bound as a module rather than pulling names out of it: several tests monkeypatch its
# attributes, which needs the module object anyway, and mixing `import x` with
# `from x import y` for the same module is the inconsistency the code-quality bot flags.
from planktonzilla.dataset_import import dataset_importer as di
from planktonzilla.planktonzilla_dataset import make_planktonzilla as mk
from planktonzilla.planktonzilla_dataset.generate_planktonzilla import build_overrides as di_build_overrides

# Every config in the group except the abstract base, which has `_target_: null`.
IMPORT_CONFIG_NAMES = sorted(f[:-5] for f in os.listdir(root / "configs" / "dataset_import") if f.endswith(".yaml"))
IMPORT_CONFIG_NAMES = [name for name in IMPORT_CONFIG_NAMES if name != "default"]


def _write_png(path, size=8):
    """Produce a small valid RGB PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (size, size), color=(120, 160, 200)).save(path)


def _make_classes(parent, class_names, n_images=2):
    """Create ``<parent>/<class>/img_<i>.png`` for each class name."""
    for class_name in class_names:
        for i in range(n_images):
            _write_png(parent / class_name / f"img_{i}.png")
    return parent


def test_import_config_names_are_discovered():
    """Sanity-check the parametrization itself, so an empty glob cannot pass vacuously."""
    assert len(IMPORT_CONFIG_NAMES) >= 15
    assert "medplanktonset" in IMPORT_CONFIG_NAMES
    assert "sykezooscan2024" in IMPORT_CONFIG_NAMES
    assert "default" not in IMPORT_CONFIG_NAMES


@pytest.mark.parametrize("config_name", IMPORT_CONFIG_NAMES)
def test_every_import_config_instantiates(config_name, tmp_path):
    """Every dataset_import config resolves to a real, constructible importer class.

    ``push_to_hub=False`` is required: every config in the group sets it true, and
    ``DatasetImporter.__post_init__`` validates that a token is present when it is.
    """
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_import_cfg")
    cfg = hydra.compose(
        config_name="import_dataset",
        overrides=[
            f"dataset_import={config_name}",
            "dataset_import.push_to_hub=False",
            f"dataset_import.data_dir={tmp_path}",
        ],
    )

    importer = hydra.utils.instantiate(cfg.dataset_import)

    GlobalHydra.instance().clear()

    assert isinstance(importer, di.DatasetImporter)
    # imagefolder_dir is namespaced by the lowercased class name, which is what keeps
    # a per-source rebuild from touching another source's folder.
    assert importer.imagefolder_dir == tmp_path / f"{type(importer).__name__.lower()}_imagefolder"


def test_find_class_root_classes_at_root(tmp_path):
    """Class folders directly under the extraction root."""
    _make_classes(tmp_path, ["akashiwo_sanguinea", "centric_diatoms"])
    assert di.find_class_root(tmp_path) == tmp_path


def test_find_class_root_single_wrapper(tmp_path):
    """The common case: the archive wraps everything in one top-level folder."""
    wrapper = tmp_path / "IFCB_images"
    _make_classes(wrapper, ["akashiwo_sanguinea", "centric_diatoms", "chaetoceros_spp"])
    assert di.find_class_root(tmp_path) == wrapper


def test_find_class_root_deeply_nested(tmp_path):
    """A path as deep as SYKE ZooScan 2024's is still found."""
    deep = tmp_path / "0127422" / "2.3" / "data" / "FINAL_Plankton_Segments_12082014"
    _make_classes(deep, ["copepoda", "diatom"])
    assert di.find_class_root(tmp_path) == deep


def test_find_class_root_prefers_the_level_with_most_class_folders(tmp_path):
    """Given a shallow 2-class level and a deeper 5-class level, the deeper one wins."""
    _make_classes(tmp_path / "thumbnails", ["a", "b"])
    real = tmp_path / "images" / "labeled"
    _make_classes(real, ["c", "d", "e", "f", "g"])
    assert di.find_class_root(tmp_path) == real


def test_find_class_root_breaks_ties_toward_the_shallowest(tmp_path):
    """A nested duplicate of the same layout does not displace the shallower one."""
    outer = tmp_path / "outer"
    _make_classes(outer, ["a", "b"])
    _make_classes(outer / "a" / "nested", ["c", "d"])
    assert di.find_class_root(tmp_path) == outer


def test_find_class_root_ignores_dot_directories(tmp_path):
    """macOS/VCS junk directories are not mistaken for class folders."""
    _make_classes(tmp_path / "data", ["real_class_a", "real_class_b"])
    _make_classes(tmp_path / ".hidden", ["x", "y", "z", "w", "v"])
    assert di.find_class_root(tmp_path) == tmp_path / "data"


def test_find_class_root_raises_when_there_are_no_images(tmp_path):
    """An archive of directories with no images is an error, not a silent empty import."""
    (tmp_path / "docs" / "notes").mkdir(parents=True)
    (tmp_path / "docs" / "notes" / "readme.txt").write_text("no images here")

    with pytest.raises(RuntimeError, match="No class folders found"):
        di.find_class_root(tmp_path)


def test_find_class_root_respects_the_depth_cap(tmp_path):
    """Class folders past MAX_CLASS_ROOT_DEPTH are not scanned, so the scan stays bounded."""
    too_deep = tmp_path.joinpath(*[f"level{i}" for i in range(di.MAX_CLASS_ROOT_DEPTH + 2)])
    _make_classes(too_deep, ["a", "b"])

    with pytest.raises(RuntimeError, match="No class folders found"):
        di.find_class_root(tmp_path)


def test_medplanktonset_prepare_imagefolder_normalizes_any_layout(tmp_path):
    """The MedPlanktonSet importer copies class folders out of a wrapped archive.

    Drives the REAL ``_prepare_imagefolder`` against a synthetic extraction tree. The
    class names are real ``medplanktonset`` ``Raw_Labels`` from the taxonomy CSV, so
    the imagefolder this produces is the one the generate pipeline would look up.
    """
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_medplankton")
    cfg = hydra.compose(
        config_name="import_dataset",
        overrides=[
            "dataset_import=medplanktonset",
            "dataset_import.push_to_hub=False",
            f"dataset_import.data_dir={tmp_path}",
            "dataset_import.show_progress=False",
        ],
    )
    importer = hydra.utils.instantiate(cfg.dataset_import)
    GlobalHydra.instance().clear()

    # Synthetic archive: classes under a wrapper, plus an image-free sibling that must
    # not survive as an empty class folder.
    extracted = tmp_path / "extracted"
    classes = ["Akashiwo_sanguinea", "Centric_diatoms", "Chaetoceros_spp"]
    _make_classes(extracted / "IFCB_images", classes, n_images=3)
    (extracted / "IFCB_images" / "metadata").mkdir(parents=True)
    (extracted / "IFCB_images" / "metadata" / "counts.csv").write_text("class,n\n")

    importer.imagefolder_dir.mkdir(parents=True, exist_ok=True)
    importer.extracted_dirs = extracted

    importer._prepare_imagefolder()

    produced = sorted(p.name for p in importer.imagefolder_dir.iterdir() if p.is_dir())
    assert produced == sorted(classes), "image-free 'metadata' folder should not survive"
    for class_name in classes:
        images = sorted((importer.imagefolder_dir / class_name).glob("*.png"))
        assert len(images) == 3


def test_integrity_check_handles_a_split_layout(tmp_path):
    """The opt-in integrity check walks nested layouts instead of crashing on them.

    The flat two-level walk this replaced handed class DIRECTORIES to
    is_valid_image_file on any split layout (train/<class>/<img>), which returns False
    for a directory, and the next line called os.remove on it and raised uncaught.
    """
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_integrity")
    cfg = hydra.compose(
        config_name="import_dataset",
        overrides=[
            "dataset_import=lensless",
            "dataset_import.push_to_hub=False",
            f"dataset_import.data_dir={tmp_path}",
            "dataset_import.show_progress=False",
            "dataset_import.check_image_file_integrity=true",
        ],
    )
    importer = hydra.utils.instantiate(cfg.dataset_import)
    GlobalHydra.instance().clear()

    # A split layout, the shape LenslessDatasetImporter actually produces.
    for split in ("train", "test"):
        _write_png(importer.imagefolder_dir / split / "copepoda" / "img_0.png")
    corrupt = importer.imagefolder_dir / "train" / "copepoda" / "broken.png"
    corrupt.write_text("this is not a PNG")

    # Exercise just the integrity pass, not the whole download/import lifecycle.
    candidates = [p for p in importer.imagefolder_dir.rglob("*") if p.is_file()]
    assert corrupt in candidates

    for path in candidates:
        if not di.is_valid_image_file(path):
            path.unlink()

    assert not corrupt.exists(), "the corrupt file should have been removed"
    for split in ("train", "test"):
        assert (importer.imagefolder_dir / split / "copepoda" / "img_0.png").exists()


def test_is_valid_image_file_rejects_a_directory(tmp_path):
    """A directory is not a valid image — the property the old walk tripped over."""

    (tmp_path / "a_class").mkdir()
    assert di.is_valid_image_file(tmp_path / "a_class") is False


def test_push_to_hub_raises_after_exhausting_retries(tmp_path, monkeypatch):
    """A push that never succeeds raises instead of reporting success.

    Exhausting every retry used to fall through to update_dataset_metadata() and
    return normally, so the card was refreshed for a dataset that was never uploaded.
    """
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_push_fail")
    cfg = hydra.compose(
        config_name="import_dataset",
        overrides=[
            "dataset_import=lensless",
            "dataset_import.push_to_hub=False",
            f"dataset_import.data_dir={tmp_path}",
        ],
    )
    importer = hydra.utils.instantiate(cfg.dataset_import)
    GlobalHydra.instance().clear()

    class _AlwaysFails:
        def __bool__(self):
            return True

        def push_to_hub(self, *args, **kwargs):
            raise ConnectionError("hub unreachable")

    importer.push_to_hub = True
    importer.hf_dataset = _AlwaysFails()
    importer.push_to_hub_retries = 2

    metadata_calls = []
    monkeypatch.setattr(type(importer), "update_dataset_metadata", lambda self: metadata_calls.append(1))

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        importer._push_to_hub()

    assert metadata_calls == [], "the dataset card must not be refreshed for a failed push"


# --- Manual downloads and the Fairdata resolver ------------------------------------


def _importer(config_name, tmp_path, extra=()):
    """Instantiate one dataset_import config against a temporary data_dir.

    ``paths.data_dir`` is overridden as well as ``dataset_import.data_dir``: manual
    archive paths interpolate ``${paths.data_dir}``, so overriding only the importer's
    own data_dir leaves them pointing at the REAL repository ``data/`` — which a test
    that writes one would then pollute.
    """
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_manual_dl")
    cfg = hydra.compose(
        config_name="import_dataset",
        overrides=[
            f"dataset_import={config_name}",
            "dataset_import.push_to_hub=False",
            f"paths.data_dir={tmp_path}",
            f"dataset_import.data_dir={tmp_path}",
            *extra,
        ],
    )
    importer = hydra.utils.instantiate(cfg.dataset_import)
    GlobalHydra.instance().clear()

    for path in importer.manual_download_paths():
        assert str(path).startswith(str(tmp_path)), f"test would touch a real path: {path}"

    return importer


def test_zoolake_is_not_actually_a_manual_download(tmp_path):
    """zoolake has a direct download URL and no manual override.

    Pins the finding behind the documentation fix: it was listed for years as needing a
    hand-downloaded .zip, but nothing in its config forces that.
    """
    importer = _importer("zoolake", tmp_path)

    assert importer.download_uris, "zoolake should have a direct download URL"
    assert importer.manual_download_local_file_names is None
    assert importer.missing_manual_downloads() == []


def test_jedi_defaults_to_the_direct_download(tmp_path):
    """JEDI downloads automatically now; its manual override used to shadow the URL.

    Checked against the live host on 2026-08-01: that URL serves a zip whose first entry
    is CPICS_Validated/20141001-07.zip, the nested layout this importer expects. There is
    no anti-bot protection to work around.
    """
    importer = _importer("jedi_oceans_cpics", tmp_path)

    assert importer.download_uris, "the direct URL is declared"
    assert importer.manual_download_local_file_names is None, "and is no longer shadowed"
    assert importer.missing_manual_downloads() == []


def test_a_manual_override_shadows_a_direct_url(tmp_path):
    """Setting a manual archive takes precedence over download_uris.

    Exercised through an explicit override rather than a config default, so this pins the
    MECHANISM and cannot break again just because a source flips to automatic.
    """
    archive = tmp_path / "manual_downloads" / "CPICS_Validated.zip"
    importer = _importer(
        "jedi_oceans_cpics",
        tmp_path,
        extra=[f"dataset_import.manual_download_local_file_names={archive}"],
    )

    assert importer.download_uris, "the direct URL is still declared"
    assert importer.manual_download_local_file_names, "but the manual archive wins"
    assert importer.missing_manual_downloads() == [archive]


def test_missing_manual_download_reports_instructions_not_a_stack_trace(tmp_path):
    """A missing hand-downloaded archive names the file and where to get it.

    The override is explicit so this never falls through to the direct URL and starts a
    real download inside the test suite.
    """
    archive = tmp_path / "manual_downloads" / "CPICS_Validated.zip"
    importer = _importer(
        "jedi_oceans_cpics",
        tmp_path,
        extra=[f"dataset_import.manual_download_local_file_names={archive}"],
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        importer._download_and_extract()

    message = str(excinfo.value)
    assert "CPICS_Validated.zip" in message, "the wanted file must be named"
    assert "dbarchive.biosciencedbc.jp" in message, "and where to get it"
    assert "re-run" in message


def test_present_manual_download_is_not_reported_missing(tmp_path):
    """Once the archive is in place, the preflight is satisfied."""
    archive = tmp_path / "manual_downloads" / "CPICS_Validated.zip"
    importer = _importer(
        "jedi_oceans_cpics",
        tmp_path,
        extra=[f"dataset_import.manual_download_local_file_names={archive}"],
    )

    expected = importer.manual_download_paths()[0]
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"not really a zip, but it exists")

    assert importer.missing_manual_downloads() == []
    assert importer.manual_download_instructions() == ""


def test_source_with_no_url_and_no_manual_file_fails_clearly(tmp_path):
    """Neither a URL nor a manual archive is an explicit error, not an obscure one."""
    importer = _importer("sykezooscan2024", tmp_path, extra=["dataset_import.fairdata_pid=null"])

    with pytest.raises(ValueError, match="nothing to fetch"):
        importer._download_and_extract()


def test_sykezooscan2024_is_configured_for_fairdata(tmp_path):
    """SYKE ZooScan 2024 resolves through Fairdata, with a manual route documented."""
    importer = _importer("sykezooscan2024", tmp_path)

    assert importer.fairdata_pid == "6fa42787-9772-41a5-a6fc-0dde489ed908"
    assert "etsin.fairdata.fi" in importer.manual_download_url

    # Pinned to the exact value, not just "https://": the importer passes this field
    # explicitly, so a value that differs from resolve_fairdata_download_url's own
    # default silently OVERRIDES the verified endpoint. It did — the field shipped as
    # https://download.fairdata.fi, which answers 404 to every path — and the loose
    # assertion this replaces is what let that through.
    assert importer.fairdata_api_base == "https://etsin.fairdata.fi/api/download"
    assert importer.fairdata_api_base == inspect.signature(di.resolve_fairdata_download_url).parameters["api_base"].default


class _FakeResponse:
    def __init__(self, payload, ok=True, status_code=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


# The exact body the live service returned for the SYKE ZooScan 2024 dataset on
# 2026-08-01, captured from GET /api/download/requests?cr_id=<pid>. Kept verbatim so
# these tests fail if the resolver stops handling the real shape.
LIVE_REQUESTS_RESPONSE = {
    "checksum": "sha256:0a309fa8774b467de491115d40ab0cc95c7be75d0b7925ce4ddf8d513418710c",
    "dataset": "6fa42787-9772-41a5-a6fc-0dde489ed908",
    "generated": "2026-07-07T06:19:46Z",
    "initiated": "2026-07-07T06:19:28Z",
    "package": "6fa42787-9772-41a5-a6fc-0dde489ed908_nhungsie.zip",
    "size": 79363785,
    "status": "SUCCESS",
}

# ...and the body POST /api/download/authorize returned for it (token truncated).
LIVE_AUTHORIZE_RESPONSE = {"url": "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.trunc"}

# The resolver's default api_base, asserted on so a silent change of host is caught.
_DEFAULT_BASE = "https://etsin.fairdata.fi/api/download"


class _FakeFairdata:
    """Scripted stand-in for the Fairdata Download API, speaking the verified contract."""

    def __init__(self, statuses, authorize=None, fail_post=False):
        self._statuses = list(statuses)
        self._authorize = authorize if authorize is not None else LIVE_AUTHORIZE_RESPONSE
        self._fail_post = fail_post
        self.posts = []
        self.gets = []

    def get(self, url, params=None, timeout=None):
        self.gets.append((url, params))
        return _FakeResponse(self._statuses.pop(0) if self._statuses else {})

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if self._fail_post:
            return _FakeResponse({}, ok=False, status_code=500)
        if url.endswith("/authorize"):
            return _FakeResponse(self._authorize)
        return _FakeResponse({"status": "accepted"})


def test_fairdata_reuses_an_already_generated_package():
    """A ready package is downloaded straight away, without requesting a new one."""

    api = _FakeFairdata(statuses=[LIVE_REQUESTS_RESPONSE])

    url = di.resolve_fairdata_download_url("pid-1", session=api, sleep=lambda s: None)

    # The service hands back a complete single-use URL; it must be returned as-is.
    assert url == LIVE_AUTHORIZE_RESPONSE["url"]
    assert not any(u.endswith("/requests") for u, _ in api.posts), "should not request a new package"

    # The parameter is cr_id: the service rejects `dataset` as an unknown field.
    assert api.gets[0][1] == {"cr_id": "pid-1"}
    assert api.posts[0][1] == {"cr_id": "pid-1", "package": LIVE_REQUESTS_RESPONSE["package"]}


def test_fairdata_requests_then_polls_until_ready():
    """No ready package -> request one, poll until it appears."""

    pending = {**LIVE_REQUESTS_RESPONSE, "status": "PENDING"}
    api = _FakeFairdata(statuses=[{}, pending, LIVE_REQUESTS_RESPONSE])

    url = di.resolve_fairdata_download_url("pid-2", session=api, sleep=lambda s: None)

    assert url == LIVE_AUTHORIZE_RESPONSE["url"]
    assert api.posts[0] == (f"{_DEFAULT_BASE}/requests", {"cr_id": "pid-2"}), "should request generation"


def test_fairdata_gives_up_with_the_manual_fallback():
    """A package that never becomes ready raises, naming the manual route."""

    api = _FakeFairdata(statuses=[{}] * 10)

    with pytest.raises(di.FairdataResolutionError, match="manual_download_local_file_names"):
        di.resolve_fairdata_download_url(
            "pid-3",
            session=api,
            sleep=lambda s: None,
            poll_attempts=3,
            source_url="https://etsin.fairdata.fi/dataset/pid-3",
        )


def test_fairdata_unrecognised_shape_does_not_guess():
    """An unfamiliar response shape raises rather than inventing a package name."""

    # Plausible-looking, but no recognisable package entry.
    api = _FakeFairdata(statuses=[{"data": {"files": ["a.zip", "b.zip"]}}] * 6)

    with pytest.raises(di.FairdataResolutionError):
        di.resolve_fairdata_download_url("pid-4", session=api, sleep=lambda s: None, poll_attempts=2)


def test_fairdata_http_error_is_actionable():
    """A non-OK response says which step failed and what to do instead."""

    class _Failing:
        def get(self, url, params=None, timeout=None):
            return _FakeResponse({}, ok=False, status_code=503)

    with pytest.raises(di.FairdataResolutionError, match="HTTP 503"):
        di.resolve_fairdata_download_url("pid-5", session=_Failing(), sleep=lambda s: None)


def test_fairdata_missing_url_is_an_error():
    """Authorization without a url is a failure, not a broken URL built from parts."""

    api = _FakeFairdata(statuses=[LIVE_REQUESTS_RESPONSE], authorize={"detail": "denied"})

    with pytest.raises(di.FairdataResolutionError, match="no download url"):
        di.resolve_fairdata_download_url("pid-6", session=api, sleep=lambda s: None)


def test_fairdata_network_failure_is_wrapped():
    """A transport error becomes a FairdataResolutionError with the fallback."""
    import requests as _requests

    class _Offline:
        def get(self, url, params=None, timeout=None):
            raise _requests.ConnectionError("dns failure")

    with pytest.raises(di.FairdataResolutionError, match="Could not reach"):
        di.resolve_fairdata_download_url("pid-7", session=_Offline(), sleep=lambda s: None)


def test_syke_importer_fetches_the_package_once_then_delegates(tmp_path, monkeypatch):
    """The single-use URL is downloaded HERE, then handed to the manual-archive path.

    Assigning it to ``download_uris`` instead — which is what this used to do — gives it
    to DownloadManager, and that requests the URL TWICE: ``datasets`` calls
    ``fsspec_head`` (which falls back to a ranged GET when the server discloses no size,
    as this one does) and then ``fsspec_get``. Measured against the live service on
    2026-08-04, a completed GET consumes the token and the next one is 401, so the real
    download failed with UNAUTHORIZED after resolving perfectly.
    """
    importer = _importer("sykezooscan2024", tmp_path)
    monkeypatch.setattr(di, "resolve_fairdata_download_url", lambda pid, **kw: f"https://example.invalid/{pid}.zip")

    package = tmp_path / "package.zip"
    fetched = []
    monkeypatch.setattr(
        di.SYKEZooScan2024DatasetImporter,
        "_fetch_single_use",
        lambda self, url: (fetched.append(url), package)[1],
    )

    delegated = {}
    monkeypatch.setattr(
        di.DatasetImporter,
        "_download_and_extract",
        lambda self: delegated.setdefault("manual", self.manual_download_local_file_names),
    )

    importer._download_and_extract()

    assert fetched == ["https://example.invalid/6fa42787-9772-41a5-a6fc-0dde489ed908.zip"], "fetched exactly once"
    assert not importer.download_uris, "the single-use URL must never reach DownloadManager"

    # A bare string, not a one-element list: DownloadManager.extract MIRRORS the
    # structure it is given, so a list makes extracted_dirs a list and every
    # _prepare_imagefolder dies on Path(self.extracted_dirs). Cost an import to learn.
    assert delegated["manual"] == str(package)
    assert isinstance(delegated["manual"], str)


class _FakeStreamedResponse:
    """A streaming requests response: context manager, chunks, and a declared length."""

    def __init__(self, chunks, declared=None):
        self._chunks = chunks
        total = sum(len(chunk) for chunk in chunks)
        self.headers = {"Content-Length": str(declared if declared is not None else total)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        yield from self._chunks


def test_fetch_single_use_writes_atomically_and_reuses_what_it_wrote(tmp_path, monkeypatch):
    """A .part file is renamed only on success, and a second call re-fetches nothing.

    The token is spent by the first completed GET, so a retry cannot resume — which
    makes "is the file on disk complete?" the only safe question to ask afterwards.
    """
    importer = _importer("sykezooscan2024", tmp_path)
    calls = []

    def _fake_get(url, **kwargs):
        calls.append((url, kwargs["headers"]["User-Agent"]))
        return _FakeStreamedResponse([b"PK\x03\x04", b"payload"])

    monkeypatch.setattr(di.requests, "get", _fake_get)

    package = importer._fetch_single_use("https://example.invalid/pkg.zip")

    assert package.read_bytes() == b"PK\x03\x04payload"
    assert package.name == f"{importer.fairdata_pid}.zip"
    assert not list(package.parent.glob("*.part")), "the partial file must not survive"
    assert calls[0][1] == importer.http_user_agent, "the package is fetched as this project"

    # Already on disk: no second request, no second token spent.
    assert importer._fetch_single_use("https://example.invalid/pkg.zip") == package
    assert len(calls) == 1


def test_fetch_single_use_refuses_a_truncated_package(tmp_path, monkeypatch):
    """A short read is an error now, not a corrupt archive that fails hours later."""
    importer = _importer("sykezooscan2024", tmp_path)

    monkeypatch.setattr(
        di.requests,
        "get",
        lambda url, **kwargs: _FakeStreamedResponse([b"only-this"], declared=99999),
    )

    with pytest.raises(RuntimeError, match="single-use"):
        importer._fetch_single_use("https://example.invalid/pkg.zip")

    assert not list(importer.raw_dir.glob("*")), "nothing partial is left to be mistaken for the package"


def test_syke_manual_override_skips_the_resolver(tmp_path, monkeypatch):
    """Pointing at a hand-downloaded archive bypasses the Fairdata flow entirely."""
    archive = tmp_path / "SYKE.zip"
    archive.write_bytes(b"zip")

    importer = _importer(
        "sykezooscan2024",
        tmp_path,
        extra=[f"dataset_import.manual_download_local_file_names={archive}"],
    )

    def _boom(*a, **k):
        raise AssertionError("resolver must not run when a manual archive is given")

    monkeypatch.setattr(di, "resolve_fairdata_download_url", _boom)
    monkeypatch.setattr(di.DatasetImporter, "_download_and_extract", lambda self: None)

    importer._download_and_extract()


def test_syke_prepare_imagefolder_unwraps_the_nested_archive(tmp_path):
    """SYKE's package is a zip inside a zip; the class folders are three levels down.

    Reproduces the REAL archive layout, captured from the live download on 2026-08-01:

        <package>.zip
          SYKE-plankton_ZooScan_2024/readme.md
          SYKE-plankton_ZooScan_2024/SYKE-plankton_ZooScan_2024.zip
            SYKE-plankton_ZooScan_2024/images/SYKE-plankton_ZooScan_2024/<class>/*.png
            SYKE-plankton_ZooScan_2024/class_splits/…
            __MACOSX/…

    Before this was fixed, _prepare_imagefolder globbed
    "0127422/2.3/data/FINAL_Plankton_Segments_12082014" — PlanktonSet1's NOAA accession
    path, which does not exist in this archive — so the loop iterated nothing and
    produced an EMPTY imagefolder without erroring.
    """
    import zipfile

    importer = _importer("sykezooscan2024", tmp_path)

    classes = ["Bivalvia", "Copepoda_nauplius", "Synchaeta_sp"]
    staging = tmp_path / "staging"
    inner_root = staging / "SYKE-plankton_ZooScan_2024" / "images" / "SYKE-plankton_ZooScan_2024"
    _make_classes(inner_root, classes, n_images=2)
    (staging / "SYKE-plankton_ZooScan_2024" / "class_splits").mkdir(parents=True, exist_ok=True)
    (staging / "SYKE-plankton_ZooScan_2024" / "class_splits" / "split.txt").write_text("a\n")
    # macOS junk, which must not become a class folder.
    junk = staging / "__MACOSX" / "SYKE-plankton_ZooScan_2024"
    junk.mkdir(parents=True, exist_ok=True)
    (junk / "._readme.md").write_bytes(b"\x00")

    inner_zip = tmp_path / "SYKE-plankton_ZooScan_2024.zip"
    with zipfile.ZipFile(inner_zip, "w") as z:
        for path in staging.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(staging))

    # The extraction dir as the download manager leaves it: the OUTER zip unpacked.
    extracted = tmp_path / "extracted" / "SYKE-plankton_ZooScan_2024"
    extracted.mkdir(parents=True, exist_ok=True)
    (extracted / "readme.md").write_text("# SYKE ZooScan 2024\n")
    shutil.copy(inner_zip, extracted / "SYKE-plankton_ZooScan_2024.zip")

    importer.extracted_dirs = tmp_path / "extracted"
    importer.imagefolder_dir.mkdir(parents=True, exist_ok=True)

    importer._prepare_imagefolder()

    produced = sorted(p.name for p in importer.imagefolder_dir.iterdir() if p.is_dir())
    assert produced == sorted(classes), "expected exactly the class folders, no wrappers or junk"
    for class_name in classes:
        assert len(list((importer.imagefolder_dir / class_name).glob("*.png"))) == 2


def test_taxonomy_csv_still_matches_the_recorded_syke_class_names():
    """The CSV's 20 sykezooscan2024 labels still equal the class names last seen in the archive.

    SCOPE: this pins the CSV against a RECORDED snapshot of the archive's class folders,
    captured from the live download on 2026-08-01. It cannot detect a change to the
    archive itself — CI has no network — so it catches CSV edits that would break the
    taxonomy lookup for this source, not upstream re-releases. Re-run the importer
    against a fresh download to revalidate the other direction.
    """
    import polars as pl

    # Captured from the live archive on 2026-08-01.
    archive_classes = {
        "Bivalvia",
        "Bivalvia_multiple",
        "Bosmina_sp",
        "Bubbles",
        "Ceriodaphnia_sp",
        "Copepoda_calanoida",
        "Copepoda_cyclopoida",
        "Copepoda_nauplius",
        "Daphnia_sp",
        "Eggs",
        "Evadne_sp",
        "Fibers_etc",
        "Fish_eggs",
        "Gastropoda",
        "Harpacticoida",
        "Mysis_sp",
        "Podon_sp",
        "Polychaeta",
        "Sessilia",
        "Synchaeta_sp",
    }

    df = pl.read_csv(root / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv")
    csv_labels = set(df.filter(pl.col("Dataset") == "sykezooscan2024")["Raw_Labels"].to_list())

    assert archive_classes == csv_labels
    assert len(csv_labels) == 20


# --- Download pre-flight: what a source would fetch, and whether it could ------------


class _FakeHTTPResponse:
    """Minimal stand-in for a ``requests.Response``.

    Separate from ``_FakeResponse`` above, which models the Fairdata JSON API and takes
    ``ok`` explicitly: here ``ok`` is DERIVED from the status code, exactly as requests
    derives it, because the probe branches on both and a double that let them disagree
    would prove nothing about the real thing.
    """

    def __init__(self, status_code=200, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.ok = 200 <= status_code < 400
        self._payload = payload
        self.closed = False

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def close(self):
        self.closed = True


class _FakeHTTP:
    """Scripted HTTP double that records every call, so the REQUESTS made are assertable.

    A ``post`` is defined only to fail: nothing in the pre-flight may POST, because on
    Fairdata that starts a packaging job on someone else's infrastructure.
    """

    def __init__(self, head=None, get=None):
        self._head = head
        self._get = get
        self.heads = []
        self.gets = []

    def _answer(self, scripted):
        if isinstance(scripted, Exception):
            raise scripted
        return scripted

    def head(self, url, headers=None, allow_redirects=None, timeout=None):
        self.heads.append((url, allow_redirects, timeout, headers))
        return self._answer(self._head)

    def get(self, url, headers=None, params=None, stream=None, allow_redirects=None, timeout=None):
        self.gets.append((url, headers, params, allow_redirects))
        return self._answer(self._get)

    def post(self, *args, **kwargs):
        raise AssertionError("a pre-flight must never POST")


def test_download_targets_normalises_a_bare_string_and_a_list(tmp_path):
    """14 of the 16 sources declare download_uris as a string; whoi declares nine.

    Iterating the string form directly would probe one URL per CHARACTER, so the
    coercion is the difference between a report and nonsense.
    """
    zoolake = _importer("zoolake", tmp_path).download_targets()
    whoi = _importer("whoi-plankton", tmp_path).download_targets()

    assert len(zoolake) == 1, "a bare string is one URL, not one per character"
    assert zoolake[0][0] == "url"
    assert zoolake[0][1].endswith("/data.zip")

    assert len(whoi) == 9
    assert {kind for kind, _ in whoi} == {"url"}


def test_download_targets_manual_archive_shadows_the_url(tmp_path):
    """A hand-downloaded archive short-circuits download_uris, so the URL is NOT fetched.

    Mirrors ``_download_and_extract``'s precedence: reporting the URL here would describe
    a download the run never performs.
    """
    archive = tmp_path / "manual_downloads" / "CPICS_Validated.zip"
    importer = _importer(
        "jedi_oceans_cpics",
        tmp_path,
        extra=[f"dataset_import.manual_download_local_file_names={archive}"],
    )

    assert importer.download_targets() == [("file", str(archive))]


def test_download_targets_lensless_is_the_bundled_zip(tmp_path):
    """Lensless ships inside the package: nothing to download, something to check."""
    targets = _importer("lensless", tmp_path).download_targets()

    assert len(targets) == 1
    kind, location = targets[0]
    assert kind == "bundled"
    assert location.endswith("lensless_dataset.zip")
    assert di.Path(location).exists(), "the bundled archive must ship with the package"

    (result,) = _importer("lensless", tmp_path).probe_downloads()
    assert result.ok and result.size, "the bundled zip is readable and its size is known"


def test_download_targets_syke_reports_fairdata_not_its_empty_url(tmp_path):
    """sykezooscan2024 has download_uris="" and resolves through the packaging API."""
    assert _importer("sykezooscan2024", tmp_path).download_targets() == [("fairdata", "6fa42787-9772-41a5-a6fc-0dde489ed908")]

    archive = tmp_path / "SYKE.zip"
    importer = _importer(
        "sykezooscan2024",
        tmp_path,
        extra=[f"dataset_import.manual_download_local_file_names={archive}"],
    )
    assert importer.download_targets() == [("file", str(archive))], "a manual archive still wins"


def test_download_targets_global_uvp5_includes_the_objects_metadata(tmp_path):
    """global_uvp5 fetches a SECOND zip from _prepare_imagefolder, not from the lifecycle.

    A pre-flight reading only download_uris would call it a one-URL source and pass a
    build that then dies on the metadata it never checked.
    """
    targets = _importer("global_uvp5net", tmp_path).download_targets()

    assert len(targets) == 2
    assert targets[1] == ("url", di.GlobalUVP5NetDatasetImporter.OBJECTS_URL)


def test_probe_downloads_reports_a_source_that_declares_no_way_to_get_its_data(tmp_path):
    """No URL and no archive is a config error, surfaced before anything is fetched."""
    importer = _importer("sykezooscan2024", tmp_path, extra=["dataset_import.fairdata_pid=null"])

    assert importer.download_targets() == []
    (result,) = importer.probe_downloads()
    assert not result.ok
    assert "nothing to fetch" in result.detail


def test_probe_url_accepts_a_plain_head(tmp_path):
    """A HEAD that answers 200 is enough; the size and type are reported from it."""
    session = _FakeHTTP(head=_FakeHTTPResponse(headers={"Content-Type": "application/zip", "Content-Length": "492398129"}))

    result = di.probe_url("https://example.invalid/data.zip", session=session, timeout=7)

    assert result.ok
    assert result.size == 492398129
    assert "application/zip" in result.detail
    assert session.gets == [], "a working HEAD needs no second request"
    assert session.heads[0][1] is True, "redirects must be followed — requests.head does not by default"
    assert session.heads[0][2] == 7


def test_probe_url_falls_back_to_a_ranged_get_when_head_is_refused(tmp_path):
    """Several of these hosts refuse HEAD and serve the file to a ranged GET.

    Reporting them as unreachable would send someone hunting a download that works, so
    the refusal costs one extra one-byte request instead of a wrong verdict.
    """
    session = _FakeHTTP(
        head=_FakeHTTPResponse(status_code=405),
        get=_FakeHTTPResponse(
            status_code=206,
            headers={"Content-Type": "application/zip", "Content-Length": "1", "Content-Range": "bytes 0-0/79363785"},
        ),
    )

    result = di.probe_url("https://example.invalid/data.zip", session=session)

    assert result.ok
    # Content-Length on a 206 is the length of the RANGE; the total is after the slash.
    assert result.size == 79363785
    assert session.gets[0][1] == {"Range": "bytes=0-0"}


def test_probe_url_reports_an_html_body_as_a_warning_not_a_pass(tmp_path):
    """A 200 that is HTML is a login wall or a moved-dataset notice, not an archive."""
    session = _FakeHTTP(head=_FakeHTTPResponse(headers={"Content-Type": "text/html; charset=utf-8"}))

    result = di.probe_url("https://example.invalid/data.zip", session=session)

    assert result.ok, "only a human can tell an interstitial from a real page"
    assert "HTML" in result.warning


def test_probe_url_reports_a_dead_url_and_a_dead_network(tmp_path):
    """A 404 and a transport failure are verdicts, not exceptions."""
    import requests as _requests

    gone = _FakeHTTP(head=_FakeHTTPResponse(status_code=404), get=_FakeHTTPResponse(status_code=404))
    assert not di.probe_url("https://example.invalid/gone.zip", session=gone).ok

    offline = _FakeHTTP(head=_requests.ConnectionError("dns failure"), get=_requests.ConnectionError("dns failure"))
    result = di.probe_url("https://example.invalid/gone.zip", session=offline)
    assert not result.ok
    assert "ConnectionError" in result.detail


def test_probe_url_falls_back_when_head_raises_not_only_when_it_refuses(tmp_path):
    """Measured against the live hosts on 2026-08-03: some close the connection on a HEAD.

    Taking a HEAD that RAISES as the verdict reported 10 of the 22 bundled archives as
    broken, when the ranged GET reached every one of them.
    """
    import requests as _requests

    session = _FakeHTTP(
        head=_requests.ConnectionError("Remote end closed connection without response"),
        get=_FakeHTTPResponse(status_code=206, headers={"Content-Range": "bytes 0-0/10"}),
    )

    result = di.probe_url("https://example.invalid/data.zip", session=session)

    assert result.ok, "a HEAD that hangs up is not evidence about the file"
    assert session.gets, "the ranged GET must still be tried"
    assert result.size == 10


def test_probe_url_says_what_a_refused_client_probably_means(tmp_path):
    """A host refusing THIS client is not a missing file, and the fix is different.

    Not worked around with a spoofed User-Agent on purpose: the real download runs
    through datasets' DownloadManager, which is no more browser-like than this probe, so
    a spoof would turn a run that WILL fail into a pre-flight that says it is fine.
    """
    blocked = _FakeHTTP(head=_FakeHTTPResponse(status_code=403), get=_FakeHTTPResponse(status_code=403))

    result = di.probe_url("https://example.invalid/data.zip", session=blocked)

    assert not result.ok
    assert "manual_download_local_file_names" in result.detail


def test_probe_local_file_catches_a_truncated_archive(tmp_path):
    """An interrupted download passes an existence check and fails hours later.

    Opening the central directory is what tells the two apart, and it costs nothing.
    """
    absent = di.probe_local_file(tmp_path / "nope.zip")
    assert not absent.ok and "not on disk" in absent.detail

    truncated = tmp_path / "half.zip"
    truncated.write_bytes(b"PK\x03\x04 and then the connection dropped")
    result = di.probe_local_file(truncated)
    assert not result.ok
    assert "NOT a readable zip" in result.detail


def test_probe_fairdata_reports_a_ready_package_without_asking_for_one(tmp_path):
    """Only the read-only step of the resolver's flow runs: GET, never POST.

    The POST asks Fairdata to build a package — a side effect on someone else's
    infrastructure that can occupy it for minutes. _FakeHTTP.post raises if reached.
    """
    session = _FakeHTTP(get=_FakeHTTPResponse(payload=LIVE_REQUESTS_RESPONSE))

    result = di.probe_fairdata_package("pid-1", api_base="https://api.invalid", session=session)

    assert result.ok
    assert result.size == LIVE_REQUESTS_RESPONSE["size"]
    assert session.gets[0][0] == "https://api.invalid/requests"
    assert session.gets[0][2] == {"cr_id": "pid-1"}


def test_probe_fairdata_treats_a_200_without_a_package_as_reachable(tmp_path):
    """No package yet is the normal state, not a failure: a real run would request one."""
    result = di.probe_fairdata_package(
        "pid-2", api_base="https://api.invalid", session=_FakeHTTP(get=_FakeHTTPResponse(payload={}))
    )

    assert result.ok
    assert "package" in result.warning


def test_probe_fairdata_does_not_excuse_a_404(tmp_path):
    """A 404 is where a real run STOPS, so the probe must stop there too.

    resolve_fairdata_download_url's `_json` raises for any non-OK response, so reading a
    404 as "no package generated yet" passed a source that could not be downloaded at
    all: probed on 2026-08-04, the base URL this importer shipped with answered nginx's
    404 page for every path, while the Etsin proxy returned the package JSON.
    """
    result = di.probe_fairdata_package(
        "pid-9", api_base="https://api.invalid", session=_FakeHTTP(get=_FakeHTTPResponse(status_code=404))
    )

    assert not result.ok
    assert "fairdata_api_base" in result.detail, "the likeliest cause must be named"


def test_probe_fairdata_reports_a_sick_service(tmp_path):
    """A 503 or a non-JSON body is a blocking failure that names the endpoint."""
    down = di.probe_fairdata_package(
        "pid-3", api_base="https://api.invalid", session=_FakeHTTP(get=_FakeHTTPResponse(status_code=503))
    )
    assert not down.ok and "503" in down.detail

    garbled = di.probe_fairdata_package(
        "pid-4",
        api_base="https://api.invalid",
        session=_FakeHTTP(get=_FakeHTTPResponse(payload=ValueError("not json"))),
    )
    assert not garbled.ok and "non-JSON" in garbled.detail


def test_the_probe_identifies_itself_exactly_as_the_download_does(tmp_path):
    """A probe that identified itself differently would answer a different question.

    Not hypothetical: whoi's host drops the connection for python-requests', aiohttp's and
    datasets' User-Agents while serving the archive to one naming this project, so the
    probe and the downloader disagreeing here means the pre-flight is worthless on exactly
    the source that needs it.
    """
    importer = _importer("zoolake", tmp_path)
    session = _FakeHTTP(head=_FakeHTTPResponse(headers={"Content-Type": "application/zip"}))

    importer.probe_downloads(session=session)

    sent = session.heads[0][3]["User-Agent"]
    assert sent == importer.http_user_agent
    assert sent == importer.storage_options()["client_kwargs"]["headers"]["User-Agent"], (
        "the probe and the real download must send the same identity"
    )


def test_the_default_user_agent_names_the_project_and_where_to_complain(tmp_path):
    """Identification, not impersonation: no browser string, and a reachable contact."""
    importer = _importer("zoolake", tmp_path)

    assert importer.http_user_agent.startswith("planktonzilla/")
    assert "github.com/Inria-Chile/planktonzilla" in importer.http_user_agent
    for impersonation in ("Mozilla", "Chrome", "Safari", "curl"):
        assert impersonation not in importer.http_user_agent


def test_an_explicit_user_agent_overrides_the_default(tmp_path):
    """One override changes what BOTH the download and the probe send."""
    importer = _importer("zoolake", tmp_path, extra=["dataset_import.http_user_agent=my-mirror/2.0"])

    assert importer.http_user_agent == "my-mirror/2.0"
    assert importer.storage_options()["client_kwargs"]["headers"] == {"User-Agent": "my-mirror/2.0"}


def test_storage_options_carry_the_timeout_and_the_identity(tmp_path):
    """The User-Agent's only route into a real download is client_kwargs.headers.

    datasets 5.0 builds a user-agent header in get_from_cache and then calls
    fsspec_head/fsspec_get without it, so DownloadConfig.user_agent would change nothing
    on the wire. This is the seam that does.
    """
    importer = _importer("zoolake", tmp_path)
    client_kwargs = importer.storage_options()["client_kwargs"]

    assert client_kwargs["timeout"].total == importer.http_timeout
    assert client_kwargs["headers"]["User-Agent"] == importer.http_user_agent


# --- resolve_imagefolder_glob: the fallback that decides what every source loads -------
#
# KI-16 froze the split probe in the build path at the repository root, so the
# no-explicit-splits fallback is the ONLY branch `import_and_redefine_source` ever takes.
# It used to be a hard-coded depth-2 glob that could not see a split layout, which made
# `lensless` and `zoolake` — both active registry entries — raise on load.


def test_resolve_imagefolder_glob_picks_the_flat_depth_first(tmp_path):
    """A flat `<class>/<image>` source resolves the exact pattern it always did."""
    _make_classes(tmp_path, ["copepoda", "diatom"])

    assert di.resolve_imagefolder_glob(tmp_path) == str(tmp_path / "*" / "[!._]*")


def test_resolve_imagefolder_glob_reaches_a_split_layout(tmp_path):
    """`<split>/<class>/<image>` (lensless, zoolake) resolves one level deeper."""
    for split in ("train", "test"):
        _make_classes(tmp_path / split, ["copepoda"])

    assert di.resolve_imagefolder_glob(tmp_path) == str(tmp_path / "*" / "*" / "[!._]*")


def test_resolve_imagefolder_glob_ignores_zoolake_style_split_names(tmp_path):
    """ZooLake names its splits train_split/val_split/test_split — no alias matches them.

    The resolver keys on DEPTH, not on the split's name, which is what lets that layout
    load at all.
    """
    for split in ("train_split", "val_split", "test_split"):
        _make_classes(tmp_path / split, ["copepoda"])

    assert di.resolve_imagefolder_glob(tmp_path) == str(tmp_path / "*" / "*" / "[!._]*")


def test_resolve_imagefolder_glob_is_not_recursive(tmp_path):
    """A stray image at the imagefolder root must not drag the match to a mixed depth.

    `imagefolder` only emits a `label` column when the matched files sit at a uniform
    depth; a `**` glob would pick the stray up, silently drop `label`, and make
    `_taxonomy_row`'s `class_names[example["label"]]` raise. A fixed depth cannot.
    """
    _make_classes(tmp_path, ["copepoda"])
    _write_png(tmp_path / "loose_at_root.png")

    pattern = di.resolve_imagefolder_glob(tmp_path)

    assert pattern == str(tmp_path / "*" / "[!._]*")
    assert str(tmp_path / "loose_at_root.png") not in glob.glob(pattern)


def test_resolve_imagefolder_glob_warns_but_still_returns_the_flat_pattern(tmp_path, caplog):
    """An empty imagefolder must keep failing where it always did, not raise here.

    The resolver is output-preserving on purpose: the shallowest pattern is the string
    this call site has always produced, so an empty imagefolder still surfaces from the
    loader. Raising instead would invent a failure mode for every caller that never
    resolves the pattern against a real filesystem — which is what the Hydra tests that
    monkeypatch ``load_dataset`` do.
    """
    (tmp_path / "empty_class").mkdir()

    with caplog.at_level(logging.WARNING):
        pattern = di.resolve_imagefolder_glob(tmp_path)

    assert pattern == str(tmp_path / "*" / "[!._]*")
    assert "No image files found" in caplog.text


class TestNumProcIsOverridable:
    """``num_proc`` must be settable with the plain override form, not only with ``+``.

    It was a dataclass field that no config declared, and Hydra's struct mode rejects
    setting a key a config never declared::

        pz_planktonzilla import_overrides=[dataset_import.num_proc=1]
        -> ConfigAttributeError: Key 'num_proc' is not in struct

    which reads like the field does not exist rather than like it was undeclared. It is
    the knob you reach for when a download misbehaves — datasets' map_nested spawns a
    process pool once num_proc > 1 and there are two or more URLs — so needing the
    add-form to reach it is exactly backwards.
    """

    def _importer(self, tmp_path, extra):
        overrides = di_build_overrides(str(tmp_path), "zoolake", True, [], refresh="reuse", import_overrides=extra)
        hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_num_proc")
        try:
            cfg = hydra.compose(config_name="import_dataset", overrides=overrides)
            return hydra.utils.instantiate(cfg.dataset_import)
        finally:
            GlobalHydra.instance().clear()

    def test_the_plain_override_form_works(self, tmp_path):
        assert self._importer(tmp_path, ["dataset_import.num_proc=1"]).num_proc == 1

    def test_null_resolves_to_the_cpu_count(self, tmp_path):
        """The config stays machine-independent; __post_init__ supplies the real value."""
        assert self._importer(tmp_path, []).num_proc == cpu_count()

    @pytest.mark.parametrize("value", [0, -1])
    def test_falsy_and_sentinel_values_are_preserved(self, tmp_path, value):
        """`or cpu_count()` would eat both; map_nested gives -1 its own meaning."""
        assert self._importer(tmp_path, [f"dataset_import.num_proc={value}"]).num_proc == value


# --- sidecar protocol: what the redefine step needs on EVERY run ---------------------------


def _frepj_sidecar_setup(monkeypatch, tmp_path, extra=()):
    """A frepj importer over tmp_path with a synthetic manifest (fixture bytes, real md5s)."""
    from planktonzilla.dataset_import.frepj_importer import FREPJDatasetImporter
    from planktonzilla.planktonzilla_dataset import frepj_tables

    fixtures = root / "tests" / "fixtures" / "frepj"
    manifest = []
    for name, fixture in (
        ("Table_S1.csv", "table_s1_sample.csv"),
        ("Table_S3.csv", "table_s3_sample.csv"),
        ("Table_S4.csv", "table_s4_sample.csv"),
    ):
        data = (fixtures / fixture).read_bytes()
        manifest.append(
            {
                "name": name,
                "file_id": 0,
                "url": f"https://example.invalid/{name}",
                "md5": hashlib.md5(data).hexdigest(),
                "size": len(data),
                "_bytes": data,
            }
        )
    monkeypatch.setattr(FREPJDatasetImporter, "SIDECAR_MANIFEST", manifest)
    monkeypatch.setattr(FREPJDatasetImporter, "CROSSWALK_PATH", fixtures / "frepj_crosswalk_sample.csv")

    def _boom(*args, **kwargs):
        raise AssertionError("DownloadManager must not be constructed")

    monkeypatch.setattr(frepj_tables, "DownloadManager", _boom)
    return _importer("frepj", tmp_path, extra=list(extra)), manifest, frepj_tables


def _seed(importer, manifest, names=None):
    importer.sidecar_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        if names is None or entry["name"] in names:
            (importer.sidecar_dir / entry["name"]).write_bytes(entry["_bytes"])


def test_download_targets_frepj_appends_the_sidecar_tables_and_the_crosswalk(tmp_path):
    """frepj probes five targets: the archive, the three md5-pinned tables, the committed crosswalk."""
    from planktonzilla.dataset_import import frepj_layout
    from planktonzilla.planktonzilla_dataset import frepj_tables

    importer = _importer("frepj", tmp_path)
    targets = importer.download_targets()

    assert targets == [
        ("url", frepj_layout.DOWNLOAD_URL),
        *[("url", entry["url"]) for entry in frepj_tables.FREPJ_TABLE_MANIFEST],
        ("bundled", str(frepj_tables.DEFAULT_CROSSWALK_PATH)),
    ]
    assert Path(targets[-1][1]).exists(), "the crosswalk is committed"
    # The archive lifecycle is untouched: the bare string keeps extracted_dirs a single path.
    assert importer._downloadable_uris() == frepj_layout.DOWNLOAD_URL


# The sources that DO declare sidecars: frepj (its md5-pinned geodata tables) and the four
# Tara Pacific deposits (their per-object EcoTaxa manifests). Everything else is archive-only.
SIDECAR_CONFIG_NAMES = {"frepj", *(name for name in IMPORT_CONFIG_NAMES if name.startswith("tara_pacific_"))}


@pytest.mark.parametrize("name", [n for n in IMPORT_CONFIG_NAMES if n not in SIDECAR_CONFIG_NAMES])
def test_sources_without_sidecars_are_unchanged(name, tmp_path):
    """The fifteen archive-only sources: no sidecars, and download_targets() exactly as before."""
    importer = _importer(name, tmp_path)
    assert importer.sidecar_targets() == []
    assert importer.missing_sidecars() == []
    assert importer.ensure_sidecars() == {}
    assert all(kind in {"url", "file", "bundled", "fairdata"} for kind, _ in importer.download_targets())
    assert not any(str(location).endswith("frepj_site_crosswalk.csv") for _, location in importer.download_targets())


def test_frepj_sidecar_dir_is_the_run_data_dir_not_the_repo(tmp_path):
    """Tables live beside the imagefolder under the RUN's data_dir; the CLI default shares the name."""
    from planktonzilla.planktonzilla_dataset import frepj_tables

    importer = _importer("frepj", tmp_path)
    assert importer.sidecar_dir == Path(importer.data_dir) / frepj_tables.TABLES_DIRNAME
    assert importer.sidecar_dir.parent == importer.imagefolder_dir.parent
    assert frepj_tables.DEFAULT_TABLES_DIR.name == frepj_tables.TABLES_DIRNAME


def test_frepj_missing_sidecars_is_md5_aware_and_free(monkeypatch, tmp_path):
    """Absent -> all three; seeded -> none; one drifted -> that one; never a download."""
    importer, manifest, _ = _frepj_sidecar_setup(monkeypatch, tmp_path)
    assert [p.name for p in importer.missing_sidecars()] == ["Table_S1.csv", "Table_S3.csv", "Table_S4.csv"]
    assert all(p.parent == importer.sidecar_dir for p in importer.missing_sidecars())

    _seed(importer, manifest)
    assert importer.missing_sidecars() == []

    (importer.sidecar_dir / "Table_S4.csv").write_bytes(b"drifted")
    assert [p.name for p in importer.missing_sidecars()] == ["Table_S4.csv"]


def test_frepj_ensure_sidecars_skips_the_download_when_all_verified(monkeypatch, tmp_path):
    """Verified tables + the crosswalk come back as {name: path}; the boom manager is never built."""
    importer, manifest, _ = _frepj_sidecar_setup(monkeypatch, tmp_path)
    _seed(importer, manifest)

    sidecars = importer.ensure_sidecars()

    assert set(sidecars) == {"Table_S1.csv", "Table_S3.csv", "Table_S4.csv", "frepj_site_crosswalk.csv"}
    assert sidecars["Table_S3.csv"] == importer.sidecar_dir / "Table_S3.csv"
    assert Path(sidecars["frepj_site_crosswalk.csv"]).exists()


def test_frepj_ensure_sidecars_fetches_only_the_misses_with_the_importers_download_config(monkeypatch, tmp_path):
    """One miss -> one download, through a config carrying the importer's User-Agent and force_download."""
    importer, manifest, frepj_tables = _frepj_sidecar_setup(monkeypatch, tmp_path)
    _seed(importer, manifest, names={"Table_S1.csv", "Table_S3.csv"})
    seen = {}

    class _Manager:
        def __init__(self, *args, **kwargs):
            seen["config"] = kwargs["download_config"]
            seen["downloads"] = []

        def download(self, url):
            seen["downloads"].append(url)
            entry = next(e for e in manifest if e["url"] == url)
            served = tmp_path / "served.bin"
            served.write_bytes(entry["_bytes"])
            return str(served)

    monkeypatch.setattr(frepj_tables, "DownloadManager", _Manager)

    sidecars = importer.ensure_sidecars()

    assert seen["downloads"] == ["https://example.invalid/Table_S4.csv"]
    config = seen["config"]
    assert config.force_download is True
    assert config.num_proc == 1
    assert Path(config.cache_dir) == importer.sidecar_dir / ".download_cache", "blobs never land among the pinned CSVs"
    assert config.storage_options["client_kwargs"]["headers"]["User-Agent"] == importer.http_user_agent
    assert (importer.sidecar_dir / "Table_S4.csv").read_bytes() == manifest[2]["_bytes"]
    assert set(sidecars) == {"Table_S1.csv", "Table_S3.csv", "Table_S4.csv", "frepj_site_crosswalk.csv"}


def test_frepj_ensure_sidecars_names_the_remedy_on_failure(monkeypatch, tmp_path):
    """A dead host becomes a RuntimeError carrying every URL, md5, the directory and the probe command."""
    importer, manifest, frepj_tables = _frepj_sidecar_setup(monkeypatch, tmp_path)

    class _Dead:
        def __init__(self, *args, **kwargs):
            pass

        def download(self, url):
            raise ConnectionError("host unreachable")

    monkeypatch.setattr(frepj_tables, "DownloadManager", _Dead)

    with pytest.raises(RuntimeError) as excinfo:
        importer.ensure_sidecars()
    message = str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    for entry in manifest:
        assert entry["url"] in message and entry["md5"] in message
    assert str(importer.sidecar_dir) in message
    assert "check_downloads=needed" in message


def test_frepj_ensure_sidecars_refuses_to_run_without_the_committed_crosswalk(monkeypatch, tmp_path):
    """No fetch can repair a missing committed file: fail first, name `git checkout`."""
    from planktonzilla.dataset_import.frepj_importer import FREPJDatasetImporter

    importer, manifest, _ = _frepj_sidecar_setup(monkeypatch, tmp_path)
    _seed(importer, manifest)
    monkeypatch.setattr(FREPJDatasetImporter, "CROSSWALK_PATH", tmp_path / "gone" / "frepj_site_crosswalk.csv")

    with pytest.raises(FileNotFoundError, match=r"frepj_site_crosswalk\.csv.*git checkout"):
        importer.ensure_sidecars()


def test_preflight_reports_the_real_frepj_importers_sidecars(monkeypatch, tmp_path):
    """The generic sidecars:<name> Check over the REAL importer (no stub): missing -> fetcher; verified -> not."""
    importer, manifest, _ = _frepj_sidecar_setup(monkeypatch, tmp_path)
    entry = {"name": "frepj", "import_name": "frepj", "cleanup": False, "redefiner": "frepj"}
    # A built imagefolder, so that only the sidecars decide whether the source fetches.
    (importer.imagefolder_dir / "a_class").mkdir(parents=True)
    (importer.imagefolder_dir / "a_class" / "img.png").write_bytes(b"x")

    checks, fetch_names = mk.report_source_state([(entry, importer)], SimpleNamespace(refresh="reuse"))
    assert fetch_names == ["frepj"]
    (check,) = [c for c in checks if c.name == "sidecars:frepj"]
    assert check.ok and "Table_S1.csv" in check.detail and str(importer.sidecar_dir) in check.detail

    _seed(importer, manifest)
    checks, fetch_names = mk.report_source_state([(entry, importer)], SimpleNamespace(refresh="reuse"))
    assert fetch_names == []
    (check,) = [c for c in checks if c.name == "sidecars:frepj"]
    assert check.ok and check.detail == "3 fetched sidecar target(s) satisfied, 1 bundled"


def test_frepj_verified_sidecars_survive_refresh_redownload(monkeypatch, tmp_path):
    """force_download (what refresh=redownload sets) never re-fetches a table that carries its pin."""
    importer, manifest, _ = _frepj_sidecar_setup(monkeypatch, tmp_path, extra=["dataset_import.force_download=True"])
    assert importer.force_download is True
    _seed(importer, manifest)
    sidecars = importer.ensure_sidecars()  # the boom DownloadManager proves nothing is fetched
    assert set(sidecars) == {"Table_S1.csv", "Table_S3.csv", "Table_S4.csv", "frepj_site_crosswalk.csv"}


def test_probe_downloads_covers_sidecars_even_when_download_targets_is_overridden(monkeypatch, tmp_path):
    """Lensless overrides download_targets() without super(); a sidecar it declared is still probed."""
    importer = _importer("lensless", tmp_path)
    bundled = tmp_path / "declared_sidecar.csv"
    bundled.write_text("x")
    monkeypatch.setattr(importer, "sidecar_targets", lambda: [("bundled", str(bundled))])

    assert ("bundled", str(bundled)) not in importer.download_targets(), "the override drops it"
    results = importer.probe_downloads(timeout=1)
    assert any(result.location == str(bundled) and result.ok for result in results), "probe_downloads guarantees it"
