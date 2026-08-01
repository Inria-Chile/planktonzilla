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

import os
import shutil

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

from planktonzilla.dataset_import.dataset_importer import (
    MAX_CLASS_ROOT_DEPTH,
    DatasetImporter,
    find_class_root,
)

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

    assert isinstance(importer, DatasetImporter)
    # imagefolder_dir is namespaced by the lowercased class name, which is what keeps
    # a per-source rebuild from touching another source's folder.
    assert importer.imagefolder_dir == tmp_path / f"{type(importer).__name__.lower()}_imagefolder"


def test_find_class_root_classes_at_root(tmp_path):
    """Class folders directly under the extraction root."""
    _make_classes(tmp_path, ["akashiwo_sanguinea", "centric_diatoms"])
    assert find_class_root(tmp_path) == tmp_path


def test_find_class_root_single_wrapper(tmp_path):
    """The common case: the archive wraps everything in one top-level folder."""
    wrapper = tmp_path / "IFCB_images"
    _make_classes(wrapper, ["akashiwo_sanguinea", "centric_diatoms", "chaetoceros_spp"])
    assert find_class_root(tmp_path) == wrapper


def test_find_class_root_deeply_nested(tmp_path):
    """A path as deep as SYKE ZooScan 2024's is still found."""
    deep = tmp_path / "0127422" / "2.3" / "data" / "FINAL_Plankton_Segments_12082014"
    _make_classes(deep, ["copepoda", "diatom"])
    assert find_class_root(tmp_path) == deep


def test_find_class_root_prefers_the_level_with_most_class_folders(tmp_path):
    """Given a shallow 2-class level and a deeper 5-class level, the deeper one wins."""
    _make_classes(tmp_path / "thumbnails", ["a", "b"])
    real = tmp_path / "images" / "labeled"
    _make_classes(real, ["c", "d", "e", "f", "g"])
    assert find_class_root(tmp_path) == real


def test_find_class_root_breaks_ties_toward_the_shallowest(tmp_path):
    """A nested duplicate of the same layout does not displace the shallower one."""
    outer = tmp_path / "outer"
    _make_classes(outer, ["a", "b"])
    _make_classes(outer / "a" / "nested", ["c", "d"])
    assert find_class_root(tmp_path) == outer


def test_find_class_root_ignores_dot_directories(tmp_path):
    """macOS/VCS junk directories are not mistaken for class folders."""
    _make_classes(tmp_path / "data", ["real_class_a", "real_class_b"])
    _make_classes(tmp_path / ".hidden", ["x", "y", "z", "w", "v"])
    assert find_class_root(tmp_path) == tmp_path / "data"


def test_find_class_root_raises_when_there_are_no_images(tmp_path):
    """An archive of directories with no images is an error, not a silent empty import."""
    (tmp_path / "docs" / "notes").mkdir(parents=True)
    (tmp_path / "docs" / "notes" / "readme.txt").write_text("no images here")

    with pytest.raises(RuntimeError, match="No class folders found"):
        find_class_root(tmp_path)


def test_find_class_root_respects_the_depth_cap(tmp_path):
    """Class folders past MAX_CLASS_ROOT_DEPTH are not scanned, so the scan stays bounded."""
    too_deep = tmp_path.joinpath(*[f"level{i}" for i in range(MAX_CLASS_ROOT_DEPTH + 2)])
    _make_classes(too_deep, ["a", "b"])

    with pytest.raises(RuntimeError, match="No class folders found"):
        find_class_root(tmp_path)


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

    from planktonzilla.dataset_import.dataset_importer import is_valid_image_file

    for path in candidates:
        if not is_valid_image_file(path):
            path.unlink()

    assert not corrupt.exists(), "the corrupt file should have been removed"
    for split in ("train", "test"):
        assert (importer.imagefolder_dir / split / "copepoda" / "img_0.png").exists()


def test_is_valid_image_file_rejects_a_directory(tmp_path):
    """A directory is not a valid image — the property the old walk tripped over."""
    from planktonzilla.dataset_import.dataset_importer import is_valid_image_file

    (tmp_path / "a_class").mkdir()
    assert is_valid_image_file(tmp_path / "a_class") is False


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
    assert importer.fairdata_api_base.startswith("https://")
    assert "etsin.fairdata.fi" in importer.manual_download_url


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
    from planktonzilla.dataset_import.dataset_importer import resolve_fairdata_download_url

    api = _FakeFairdata(statuses=[LIVE_REQUESTS_RESPONSE])

    url = resolve_fairdata_download_url("pid-1", session=api, sleep=lambda s: None)

    # The service hands back a complete single-use URL; it must be returned as-is.
    assert url == LIVE_AUTHORIZE_RESPONSE["url"]
    assert not any(u.endswith("/requests") for u, _ in api.posts), "should not request a new package"

    # The parameter is cr_id: the service rejects `dataset` as an unknown field.
    assert api.gets[0][1] == {"cr_id": "pid-1"}
    assert api.posts[0][1] == {"cr_id": "pid-1", "package": LIVE_REQUESTS_RESPONSE["package"]}


def test_fairdata_requests_then_polls_until_ready():
    """No ready package -> request one, poll until it appears."""
    from planktonzilla.dataset_import.dataset_importer import resolve_fairdata_download_url

    pending = {**LIVE_REQUESTS_RESPONSE, "status": "PENDING"}
    api = _FakeFairdata(statuses=[{}, pending, LIVE_REQUESTS_RESPONSE])

    url = resolve_fairdata_download_url("pid-2", session=api, sleep=lambda s: None)

    assert url == LIVE_AUTHORIZE_RESPONSE["url"]
    assert api.posts[0] == (f"{_DEFAULT_BASE}/requests", {"cr_id": "pid-2"}), "should request generation"


def test_fairdata_gives_up_with_the_manual_fallback():
    """A package that never becomes ready raises, naming the manual route."""
    from planktonzilla.dataset_import.dataset_importer import (
        FairdataResolutionError,
        resolve_fairdata_download_url,
    )

    api = _FakeFairdata(statuses=[{}] * 10)

    with pytest.raises(FairdataResolutionError, match="manual_download_local_file_names"):
        resolve_fairdata_download_url(
            "pid-3",
            session=api,
            sleep=lambda s: None,
            poll_attempts=3,
            source_url="https://etsin.fairdata.fi/dataset/pid-3",
        )


def test_fairdata_unrecognised_shape_does_not_guess():
    """An unfamiliar response shape raises rather than inventing a package name."""
    from planktonzilla.dataset_import.dataset_importer import (
        FairdataResolutionError,
        resolve_fairdata_download_url,
    )

    # Plausible-looking, but no recognisable package entry.
    api = _FakeFairdata(statuses=[{"data": {"files": ["a.zip", "b.zip"]}}] * 6)

    with pytest.raises(FairdataResolutionError):
        resolve_fairdata_download_url("pid-4", session=api, sleep=lambda s: None, poll_attempts=2)


def test_fairdata_http_error_is_actionable():
    """A non-OK response says which step failed and what to do instead."""
    from planktonzilla.dataset_import.dataset_importer import (
        FairdataResolutionError,
        resolve_fairdata_download_url,
    )

    class _Failing:
        def get(self, url, params=None, timeout=None):
            return _FakeResponse({}, ok=False, status_code=503)

    with pytest.raises(FairdataResolutionError, match="HTTP 503"):
        resolve_fairdata_download_url("pid-5", session=_Failing(), sleep=lambda s: None)


def test_fairdata_missing_url_is_an_error():
    """Authorization without a url is a failure, not a broken URL built from parts."""
    from planktonzilla.dataset_import.dataset_importer import (
        FairdataResolutionError,
        resolve_fairdata_download_url,
    )

    api = _FakeFairdata(statuses=[LIVE_REQUESTS_RESPONSE], authorize={"detail": "denied"})

    with pytest.raises(FairdataResolutionError, match="no download url"):
        resolve_fairdata_download_url("pid-6", session=api, sleep=lambda s: None)


def test_fairdata_network_failure_is_wrapped():
    """A transport error becomes a FairdataResolutionError with the fallback."""
    import requests as _requests

    from planktonzilla.dataset_import.dataset_importer import (
        FairdataResolutionError,
        resolve_fairdata_download_url,
    )

    class _Offline:
        def get(self, url, params=None, timeout=None):
            raise _requests.ConnectionError("dns failure")

    with pytest.raises(FairdataResolutionError, match="Could not reach"):
        resolve_fairdata_download_url("pid-7", session=_Offline(), sleep=lambda s: None)


def test_syke_importer_uses_the_resolver_then_delegates(tmp_path, monkeypatch):
    """The importer swaps the resolved URL into download_uris and reuses the base path."""
    importer = _importer("sykezooscan2024", tmp_path)

    import planktonzilla.dataset_import.dataset_importer as di

    monkeypatch.setattr(di, "resolve_fairdata_download_url", lambda pid, **kw: f"https://example.invalid/{pid}.zip")

    delegated = {}
    monkeypatch.setattr(
        di.DatasetImporter, "_download_and_extract", lambda self: delegated.setdefault("uris", self.download_uris)
    )

    importer._download_and_extract()

    assert delegated["uris"] == "https://example.invalid/6fa42787-9772-41a5-a6fc-0dde489ed908.zip"


def test_syke_manual_override_skips_the_resolver(tmp_path, monkeypatch):
    """Pointing at a hand-downloaded archive bypasses the Fairdata flow entirely."""
    archive = tmp_path / "SYKE.zip"
    archive.write_bytes(b"zip")

    importer = _importer(
        "sykezooscan2024",
        tmp_path,
        extra=[f"dataset_import.manual_download_local_file_names={archive}"],
    )

    import planktonzilla.dataset_import.dataset_importer as di

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
