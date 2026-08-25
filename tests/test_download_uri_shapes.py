"""
(c) Inria

Pin the shape of what reaches ``DownloadManager.download``.

The bug these guard against: Hydra hands a YAML list over as an OmegaConf ``ListConfig``,
which is not a ``list`` subclass, so ``datasets.map_nested`` treats the whole container as
a single item and stringifies it to ``"['https://a', 'https://b']"``. That string has no
URL scheme, so ``_download_single`` takes it for a relative path and joins it onto
``base_path`` — and with ``base_path`` a ``PosixPath``, urlparse fails with
``'PosixPath' object has no attribute 'decode'``, an error naming neither the URL nor the
source. Only whoi declares a list, so every other source worked and this stayed hidden.
"""

import hydra
import pytest
from hydra import initialize_config_dir
from omegaconf import ListConfig

from planktonzilla.dataset_import.dataset_importer import _as_uri_list
from planktonzilla.planktonzilla_dataset.generate_planktonzilla import build_overrides

CONFIG_DIR = "/Users/luismarti/test/planktonzilla/configs"


def _importer(stem, tmp_path):
    overrides = build_overrides(str(tmp_path), stem, True, [], refresh="reuse", import_overrides=[])
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = hydra.compose(config_name="import_dataset", overrides=overrides)
        return hydra.utils.instantiate(cfg.dataset_import)


def test_a_listconfig_is_not_a_list():
    """The premise of the whole bug. If omegaconf ever changes this, the rest is moot."""
    assert not isinstance(ListConfig(["a", "b"]), list)


class TestDownloadableUris:
    def test_a_multi_uri_source_yields_a_plain_list(self, tmp_path):
        """whoi's nine release archives must arrive as a real list, not a ListConfig."""
        importer = _importer("whoi-plankton", tmp_path)
        assert isinstance(importer.download_uris, ListConfig), "fixture no longer covers the ListConfig case"

        uris = importer._downloadable_uris()

        assert type(uris) is list
        assert len(uris) == 9
        assert all(type(uri) is str and uri.startswith("https://") for uri in uris)

    def test_a_single_uri_source_is_passed_through_as_a_bare_string(self, tmp_path):
        """NOT wrapped in a list: extract() mirrors the structure it is given, and almost
        every _prepare_imagefolder opens with Path(self.extracted_dirs)."""
        importer = _importer("zoolake", tmp_path)

        uris = importer._downloadable_uris()

        assert type(uris) is str
        assert uris.startswith("https://")

    @pytest.mark.parametrize("stem", ["whoi-plankton", "zoolake", "planktoscope", "jedi_oceans_cpics"])
    def test_the_download_path_and_the_probe_path_agree(self, stem, tmp_path):
        """The asymmetry that let the pre-flight pass while the import crashed.

        probe_downloads has always normalised via _as_uri_list; the real download did not.
        A source whose two paths disagree is one whose `check_downloads` verdict is not
        evidence about whether the import works.
        """
        importer = _importer(stem, tmp_path)

        downloaded = importer._downloadable_uris()
        as_list = [downloaded] if isinstance(downloaded, str) else downloaded

        assert as_list == _as_uri_list(importer.download_uris)
        assert as_list == [location for kind, location in importer.download_targets() if kind == "url"]


class TestDownloadManagerArguments:
    """What ``_download_and_extract`` actually hands the DownloadManager, without downloading."""

    def _capture(self, stem, tmp_path, monkeypatch):
        from planktonzilla.dataset_import import dataset_importer as module

        seen = {}

        class RecordingDownloadManager:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def download(self, uris):
                seen["downloaded"] = uris
                return "/fake/archive"

            def extract(self, paths):
                return paths

        monkeypatch.setattr(module, "DownloadManager", RecordingDownloadManager)
        importer = _importer(stem, tmp_path)
        importer._download_and_extract()
        return seen

    def test_base_path_is_a_str_not_a_path(self, tmp_path, monkeypatch):
        """datasets calls urlparse() on base_path, and urlparse rejects a PosixPath."""
        seen = self._capture("zoolake", tmp_path, monkeypatch)

        assert type(seen["base_path"]) is str

    def test_whoi_receives_all_nine_urls_as_a_plain_list(self, tmp_path, monkeypatch):
        """The regression itself: a ListConfig here is silently collapsed into one string."""
        seen = self._capture("whoi-plankton", tmp_path, monkeypatch)

        assert type(seen["downloaded"]) is list
        assert len(seen["downloaded"]) == 9

    def test_a_single_uri_source_still_receives_a_bare_string(self, tmp_path, monkeypatch):
        seen = self._capture("zoolake", tmp_path, monkeypatch)

        assert type(seen["downloaded"]) is str

    @pytest.mark.parametrize("stem", ["whoi-plankton", "zoolake"])
    def test_downloads_never_use_a_process_pool(self, stem, tmp_path, monkeypatch):
        """num_proc=1 keeps map_nested out of multiprocessing.Pool.

        Two independent reasons, either sufficient. The pool FAILS: whoi's nine archives
        died partway through under 12 processes and completed under one. And it HIDES the
        reason: a worker's exception is pickled back to the parent, aiohttp errors carry
        CIMultiDictProxy headers which cannot be pickled, so Pool reports the pickling
        failure instead of the HTTP one.
        """
        seen = self._capture(stem, tmp_path, monkeypatch)

        assert seen["download_config"].num_proc == 1

    def test_the_importers_own_num_proc_is_left_alone(self, tmp_path, monkeypatch):
        """Only the DOWNLOAD is serialised; imagefolder prep and the push still scale."""
        from multiprocessing import cpu_count

        self._capture("zoolake", tmp_path, monkeypatch)
        importer = _importer("zoolake", tmp_path)

        assert importer.num_proc == cpu_count()


def test_an_aiohttp_error_cannot_cross_a_process_boundary():
    """The mechanism behind the masking, pinned so the reasoning above stays checkable."""
    import pickle

    import aiohttp
    import multidict

    headers = multidict.CIMultiDictProxy(multidict.CIMultiDict({"content-type": "text/html"}))
    error = aiohttp.ClientResponseError(
        request_info=aiohttp.RequestInfo(url="http://x", method="GET", headers=headers, real_url="http://x"),
        history=(),
        status=503,
        message="Service Unavailable",
        headers=headers,
    )

    with pytest.raises(TypeError, match="pickle"):
        pickle.dumps(error)
