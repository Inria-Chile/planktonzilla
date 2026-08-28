"""Regression tests for preparing an imagefolder into a data_dir that never held it.

A real from-scratch ``pz_planktonzilla`` build (fresh machine, empty ``data`` directory)
died at «whoi» with::

    FileNotFoundError: .../whoiplanktondatasetimporter_imagefolder/2014

because WHOI's ``_prepare_imagefolder`` creates each class dir with a bare
``mkdir(exist_ok=True)`` — no ``parents=True`` — silently assuming the imagefolder ROOT
already exists. Every other importer creates the root as a side effect (``copytree`` and
``mkdir(parents=True)`` both create missing parents), which is why the assumption held on
any data_dir that had ever completed a run and only a truly from-scratch build hit it.

``import_dataset`` now guarantees the root exists before invoking the subclass hook, so
the guarantee is tested on the base class and reproduced end-to-end on WHOI.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from pathlib import Path

from PIL import Image

import planktonzilla.dataset_import.dataset_importer as dataset_importer


def _write_rgb_png(path: Path, color=(90, 130, 170), size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path, "PNG")


def test_whoi_prepares_into_a_data_dir_that_never_held_it(tmp_path, monkeypatch):
    """The observed crash: a from-scratch build must not assume the imagefolder root."""
    imp = dataset_importer.WHOIPlanktonDatasetImporter(
        data_dir=tmp_path,
        hf_dataset_name="whoi",
        push_to_hub=False,
        show_progress=False,
    )

    # One extracted release holding one class dir, as _download_and_extract leaves them.
    _write_rgb_png(imp.raw_dir / "2014" / "Ciliate" / "img_0.png")
    _write_rgb_png(imp.raw_dir / "2014" / "Ciliate" / "img_1.png")
    monkeypatch.setattr(
        dataset_importer.WHOIPlanktonDatasetImporter,
        "_download_and_extract",
        lambda self: setattr(self, "extracted_dirs", ["2014"]),
    )
    # Stub the HF loader: no network or dataset processing, the copy is what matters.
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    assert not imp.imagefolder_dir.exists()  # the precondition that used to crash

    imp.import_dataset()

    copied = sorted(path.name for path in (imp.imagefolder_dir / "Ciliate").glob("*.png"))
    assert copied == ["img_0.png", "img_1.png"]
    assert imp.hf_dataset == "FAKE_DATASET"


def test_whoi_finds_classes_nested_under_a_release_year_wrapper(tmp_path, monkeypatch):
    """The observed crash: a real ``pz_planktonzilla sources=[whoi]`` build downloaded all
    nine release archives, extracted them, then died with
    ``ValueError: Instruction "train" corresponds to no data!`` because each archive
    unpacks to a year-named wrapper directory ABOVE the class dirs
    (``<release>/2006/Ciliate/*.png``), not the class dirs directly
    (``<release>/Ciliate/*.png``) that ``_prepare_imagefolder`` assumed. It copied zero
    files and left empty ``2006``..``2014`` folders in the imagefolder.
    """
    imp = dataset_importer.WHOIPlanktonDatasetImporter(
        data_dir=tmp_path,
        hf_dataset_name="whoi",
        push_to_hub=False,
        show_progress=False,
    )

    _write_rgb_png(imp.raw_dir / "release_2006" / "2006" / "Ciliate" / "img_0.png")
    _write_rgb_png(imp.raw_dir / "release_2006" / "2006" / "Chaetoceros" / "img_1.png")
    monkeypatch.setattr(
        dataset_importer.WHOIPlanktonDatasetImporter,
        "_download_and_extract",
        lambda self: setattr(self, "extracted_dirs", ["release_2006"]),
    )
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    imp.import_dataset()

    assert sorted(path.name for path in (imp.imagefolder_dir / "Ciliate").glob("*.png")) == ["img_0.png"]
    assert sorted(path.name for path in (imp.imagefolder_dir / "Chaetoceros").glob("*.png")) == ["img_1.png"]
    # No leftover empty year-wrapper directory at the imagefolder root.
    assert not (imp.imagefolder_dir / "2006").exists()


def test_import_dataset_refuses_an_imagefolder_a_hook_left_empty(tmp_path, monkeypatch):
    """The bug CLASS behind both WHOI failures, caught once for every importer.

    Every _prepare_imagefolder walks a path it believes the archive has, and Path.glob on
    a path that does not exist returns nothing rather than raising — so a layout that
    shifted by one directory yields an EMPTY imagefolder and no error. That has bitten
    SYKE ZooScan (KI-22), then WHOI. Unguarded, the run continues to load_dataset and dies
    with `Instruction "train" corresponds to no data!`, naming neither the source nor the
    cause; this must fail immediately instead, naming both.
    """
    import pytest

    class _MissesEverything(dataset_importer.DatasetImporter):
        def _download_and_extract(self):
            self.extracted_dirs = "somewhere"

        def _prepare_imagefolder(self):
            # The real shape of the bug: a glob over a path that is not there.
            for stray in (self.raw_dir / "not-the-real-layout").glob("*"):
                copied = stray  # pragma: no cover - the point is that this never runs

    imp = _MissesEverything(
        data_dir=tmp_path,
        hf_dataset_name="misses",
        push_to_hub=False,
        show_progress=False,
    )
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    with pytest.raises(RuntimeError) as excinfo:
        imp.import_dataset()

    message = str(excinfo.value)
    assert "_MissesEverything._prepare_imagefolder" in message, "name the hook that produced nothing"
    assert str(imp.imagefolder_dir) in message


def test_import_dataset_accepts_an_imagefolder_that_holds_images(tmp_path, monkeypatch):
    """The guard must not fire on a hook that did its job — including a split layout."""

    class _NestsTwoDeep(dataset_importer.DatasetImporter):
        def _download_and_extract(self):
            self.extracted_dirs = "somewhere"

        def _prepare_imagefolder(self):
            _write_rgb_png(self.imagefolder_dir / "train" / "classA" / "0.png")

    imp = _NestsTwoDeep(
        data_dir=tmp_path,
        hf_dataset_name="nested",
        push_to_hub=False,
        show_progress=False,
    )
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    imp.import_dataset()

    assert imp.hf_dataset == "FAKE_DATASET"


def test_import_dataset_creates_the_imagefolder_root_before_the_subclass_hook(tmp_path, monkeypatch):
    """The guarantee is the BASE class's, so every _prepare_imagefolder may rely on it."""
    seen = {}

    class _ProbeImporter(dataset_importer.DatasetImporter):
        def _download_and_extract(self):
            self.extracted_dirs = "unused"

        def _prepare_imagefolder(self):
            seen["root_existed"] = self.imagefolder_dir.is_dir()
            _write_rgb_png(self.imagefolder_dir / "classA" / "0.png")

    imp = _ProbeImporter(
        data_dir=tmp_path,
        hf_dataset_name="probe",
        push_to_hub=False,
        show_progress=False,
    )
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    imp.import_dataset()

    assert seen == {"root_existed": True}
