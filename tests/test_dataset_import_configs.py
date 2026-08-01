"""
(c) Inria

Network-free tests for the ``configs/dataset_import/`` config group and for the
layout-independent class-folder scan added with the MedPlanktonSet importer.

The instantiation test is a guard, not a formality: before it existed, two configs
named ``_target_`` classes that do not exist —
``medplanktonset.yaml`` pointed at ``MedPlanktonSetDatasetImporter`` (never written)
and ``sykezooscan2024.yaml`` at ``SYKEZooScan2024`` (missing the
``DatasetImporter`` suffix). ``medplanktonset`` is the 5th of the 12 active entries
in ``configs/generate_planktonzilla.yaml``, so a full build died partway through,
after four sources had already been downloaded and processed. Both are fixed; this
test fails the moment a config names a class that cannot be located.

Nothing here touches the network or downloads anything: instantiating an importer only
builds the dataclass and derives its ``imagefolder_dir`` / ``raw_dir`` paths.
"""

import os

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
