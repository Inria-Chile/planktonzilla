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
