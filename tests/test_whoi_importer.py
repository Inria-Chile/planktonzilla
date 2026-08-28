"""Regression tests for WHOI's imagefolder preparation on a from-scratch machine.

A real from-scratch ``pz_planktonzilla`` build surfaced two WHOI defects in sequence:

1. ``FileNotFoundError: .../whoiplanktondatasetimporter_imagefolder/2014`` — the
   per-class ``mkdir(exist_ok=True)`` (no ``parents=True``) silently assumed the
   imagefolder ROOT existed. Every other importer creates the root as a side effect
   (``copytree`` and ``mkdir(parents=True)`` both create missing parents), which is why
   the assumption held on any data_dir that had ever completed a run.
   ``import_dataset`` now guarantees the root before invoking the subclass hook.

2. With the root guaranteed, the same run produced an imagefolder of EMPTY year dirs
   and zero images: each WHOI archive wraps its class folders in a year directory
   (``2014/<class>/*.png`` — verified against the live bitstreams' central
   directories), and the copy loop iterated the extraction root's immediate children,
   treating ``2014`` itself as a class. The NEXT run then reused the hollow tree
   ("Using existing imagefolder with 9 categories") and died in the HF loader with
   ``Instruction "train" corresponds to no data!``. Three fixes, each tested here:
   WHOI locates the class root per release (``find_class_root``), a preparation that
   copies zero files raises at the point of failure, and a hollow imagefolder no
   longer counts as complete — so an already-poisoned data_dir heals on re-run.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from pathlib import Path

import pytest
from PIL import Image

import planktonzilla.dataset_import.dataset_importer as dataset_importer


def _write_rgb_png(path: Path, color=(90, 130, 170), size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path, "PNG")


def _whoi(tmp_path):
    return dataset_importer.WHOIPlanktonDatasetImporter(
        data_dir=tmp_path,
        hf_dataset_name="whoi",
        push_to_hub=False,
        show_progress=False,
    )


def _stub_pipeline(monkeypatch, importer, extracted_dirs):
    monkeypatch.setattr(
        type(importer),
        "_download_and_extract",
        lambda self: setattr(self, "extracted_dirs", extracted_dirs),
    )
    # Stub the HF loader: no network or dataset processing, the copy is what matters.
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")


def test_whoi_prepares_into_a_data_dir_that_never_held_it(tmp_path, monkeypatch):
    """The observed crashes end-to-end: fresh data_dir, and the REAL archive layout —
    class folders wrapped in a year directory, not sitting at the extraction root."""
    imp = _whoi(tmp_path)
    _write_rgb_png(imp.raw_dir / "release_2014" / "2014" / "Ciliate" / "img_0.png")
    _write_rgb_png(imp.raw_dir / "release_2014" / "2014" / "Ciliate" / "img_1.png")
    _write_rgb_png(imp.raw_dir / "release_2014" / "2014" / "detritus" / "img_2.png")
    _stub_pipeline(monkeypatch, imp, ["release_2014"])

    assert not imp.imagefolder_dir.exists()  # the precondition that used to crash

    imp.import_dataset()

    assert sorted(path.name for path in (imp.imagefolder_dir / "Ciliate").glob("*.png")) == ["img_0.png", "img_1.png"]
    assert [path.name for path in (imp.imagefolder_dir / "detritus").glob("*.png")] == ["img_2.png"]
    assert not (imp.imagefolder_dir / "2014").exists()  # the year wrapper is NOT a class
    assert imp.hf_dataset == "FAKE_DATASET"


def test_whoi_copes_with_a_release_without_the_year_wrapper(tmp_path, monkeypatch):
    """find_class_root also accepts classes at the extraction root, so a re-release
    that DROPS the wrapper would not break the importer either."""
    imp = _whoi(tmp_path)
    _write_rgb_png(imp.raw_dir / "release_flat" / "Ciliate" / "img_0.png")
    _stub_pipeline(monkeypatch, imp, ["release_flat"])

    imp.import_dataset()

    assert [path.name for path in (imp.imagefolder_dir / "Ciliate").glob("*.png")] == ["img_0.png"]


def test_import_dataset_accepts_an_imagefolder_that_holds_images(tmp_path, monkeypatch):
    """The zero-images guard must not fire on a hook that did its job.

    The split layout specifically: LenslessDatasetImporter (train/ + test/) and
    ZooLakeDatasetImporter nest images one level deeper than the flat sources, so a guard
    that only looked one level down would reject them.
    """

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
    """The root guarantee is the BASE class's, so every _prepare_imagefolder may rely on it."""
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


def test_a_preparation_that_copies_nothing_raises_at_the_point_of_failure(tmp_path, monkeypatch):
    """Zero images is never a valid result of preparation: without this the run died
    later inside the HF loader, and the hollow tree poisoned every later run."""

    class _HollowImporter(dataset_importer.DatasetImporter):
        def _download_and_extract(self):
            self.extracted_dirs = "unused"

        def _prepare_imagefolder(self):
            (self.imagefolder_dir / "2014").mkdir()  # a class dir, but nothing in it

    imp = _HollowImporter(
        data_dir=tmp_path,
        hf_dataset_name="hollow",
        push_to_hub=False,
        show_progress=False,
    )
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    with pytest.raises(RuntimeError) as excinfo:
        imp.import_dataset()

    message = str(excinfo.value)
    assert "no image files" in message
    assert "_HollowImporter._prepare_imagefolder" in message, "name the hook that produced nothing"
    assert str(imp.imagefolder_dir) in message, "and where to look"


def test_a_preparation_that_copies_only_junk_also_raises(tmp_path, monkeypatch):
    """Files alone are not enough — copytree_filtered can carry a stray .DS_Store across.

    A tree of nothing but junk is as broken as an empty one, and would pass the cheaper
    "any file" test that imagefolder_is_complete() uses on the reuse path.
    """

    class _JunkOnlyImporter(dataset_importer.DatasetImporter):
        def _download_and_extract(self):
            self.extracted_dirs = "unused"

        def _prepare_imagefolder(self):
            (self.imagefolder_dir / "Ciliate").mkdir()
            (self.imagefolder_dir / "Ciliate" / ".DS_Store").write_bytes(b"junk")

    imp = _JunkOnlyImporter(
        data_dir=tmp_path,
        hf_dataset_name="junk",
        push_to_hub=False,
        show_progress=False,
    )
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    with pytest.raises(RuntimeError, match="no image files"):
        imp.import_dataset()


def test_a_hollow_imagefolder_is_not_complete(tmp_path):
    """Empty class dirs — what a broken preparation leaves behind — must trigger a
    rebuild on the next run, not be reused as "9 categories" of nothing."""
    imp = _whoi(tmp_path)

    assert imp.imagefolder_is_complete() is False  # missing entirely

    for year in ("2006", "2014"):
        (imp.imagefolder_dir / year).mkdir(parents=True)
    assert imp.imagefolder_is_complete() is False  # non-empty, but not one file

    _write_rgb_png(imp.imagefolder_dir / "Ciliate" / "img_0.png")
    assert imp.imagefolder_is_complete() is True
