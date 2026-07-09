"""
(c) Inria

Network-free tests for :class:`FREPJDatasetImporter`.

These tests are offline BY CONSTRUCTION: every fixture is a synthetic
``Image.new(...)`` tree built under pytest's ``tmp_path`` — there is NO download, NO
figshare call, and NO HuggingFace Hub access. They pin the importer's observable
contract: both magnifications flatten into ONE class dir per taxon with collision-safe
``40_``/``100_`` prefixes, non-RGB images are normalized by DECODED CONTENT (not by the
``.jpg`` extension), the layout is deterministic across rebuilds, corrupt files are left
in place for the base class's ``check_image_file_integrity`` filter, and the import
config mirrors the frozen ``frepj_layout`` constants.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import yaml
from PIL import Image

import planktonzilla.dataset_import.dataset_importer as dataset_importer
from planktonzilla.dataset_import import frepj_layout
from planktonzilla.dataset_import.dataset_importer import is_valid_image_file
from planktonzilla.dataset_import.frepj_importer import FREPJDatasetImporter

# Real comma-preserving FREPJ Class,Order,Family,Genus,Species class-dir names.
TAXON_A = "Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina longirostris"
TAXON_B = "Copepoda,Calanoida,Diaptomidae,Eodiaptomus,Eodiaptomus japonicus"


def _write_rgb_jpg(path, color=(90, 130, 170), size=8):
    """Write a tiny valid 3-channel RGB JPEG (mirrors test_frepj_layout._write_jpg)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path, format="JPEG", quality=90)


def _write_png_as_jpg(path, size=8):
    """Write the exact content/extension mismatch the Phase-15 scan found: RGBA content
    saved as PNG format under a ``.jpg`` filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (size, size), (10, 20, 30, 40)).save(path, format="PNG")


def _build_extracted(base):
    """Build a synthetic ``<base>/extracted/plankton_images/{images_40,images_100}`` tree.

    Layout: ``TAXON_A`` has ``0.jpg`` under BOTH magnifications (shared-basename
    collision); ``TAXON_B`` lives only under ``images_40`` with a genuine RGB ``1.jpg``
    and a non-RGB ``5.jpg`` outlier (RGBA/PNG content wearing a ``.jpg`` name).
    """
    extracted = base / "extracted"
    archive_root = extracted / frepj_layout.ARCHIVE_ROOT
    images_40 = archive_root / "images_40"
    images_100 = archive_root / "images_100"

    _write_rgb_jpg(images_40 / TAXON_A / "0.jpg")
    _write_rgb_jpg(images_100 / TAXON_A / "0.jpg")
    _write_rgb_jpg(images_40 / TAXON_B / "1.jpg")
    _write_png_as_jpg(images_40 / TAXON_B / "5.jpg")

    return extracted


def _make_importer(base):
    """Build the fixture under ``base``, run ``_prepare_imagefolder``, return the importer."""
    extracted = _build_extracted(base)
    imp = FREPJDatasetImporter(data_dir=base, hf_dataset_name="frepj")
    imp.extracted_dirs = str(extracted)
    imp._prepare_imagefolder()
    return imp


def _layout(imagefolder_dir):
    """Sorted list of ``(class_dir_name, file_name)`` relative pairs in the imagefolder."""
    return sorted(
        (class_dir.name, f.name) for class_dir in imagefolder_dir.iterdir() if class_dir.is_dir() for f in class_dir.iterdir()
    )


def test_importer_merges_both_magnifications_collision_safe(tmp_path):
    """IMP-01/IMP-04: both magnifications flatten into ONE class per taxon; the shared
    ``0.jpg`` basename survives via the ``40_``/``100_`` prefixes, no silent overwrite."""
    imp = _make_importer(tmp_path)
    imagefolder_dir = imp.imagefolder_dir

    class_dirs = sorted(p.name for p in imagefolder_dir.iterdir() if p.is_dir())
    assert class_dirs == sorted([TAXON_A, TAXON_B])  # one class per taxon, not doubled

    # The cross-magnification basename collision was resolved; both files survived.
    assert (imagefolder_dir / TAXON_A / "40_0.jpg").exists()
    assert (imagefolder_dir / TAXON_A / "100_0.jpg").exists()

    # The comma survived verbatim; no per-magnification sibling classes were created.
    assert "," in (imagefolder_dir / TAXON_A).name
    assert not (imagefolder_dir / f"{TAXON_A}@40x").exists()
    assert not (imagefolder_dir / f"{TAXON_A}@100x").exists()


def test_non_rgb_jpg_normalized_by_content(tmp_path):
    """IMP-03: a ``.jpg``-named RGBA/PNG file is normalized to a genuine RGB JPEG because
    the decision is made by DECODED CONTENT, never by the extension."""
    imp = _make_importer(tmp_path)
    imagefolder_dir = imp.imagefolder_dir

    with Image.open(imagefolder_dir / TAXON_B / "40_5.jpg") as img:
        assert img.mode == "RGB"
        assert img.format == "JPEG"

    # A genuinely-RGB neighbor is left as a valid RGB image too.
    with Image.open(imagefolder_dir / TAXON_B / "40_1.jpg") as img:
        assert img.mode == "RGB"


def test_layout_is_deterministic(tmp_path):
    """IMP-03: the SAME fixture built into two independent importers yields byte-identical
    class layouts (sorted enumeration in both the merge seam and the normalization pass)."""
    imp1 = _make_importer(tmp_path / "run1")
    imp2 = _make_importer(tmp_path / "run2")

    assert _layout(imp1.imagefolder_dir) == _layout(imp2.imagefolder_dir)


def test_corrupt_image_is_left_for_integrity_filter(tmp_path):
    """IMP-03: ``_prepare_imagefolder`` never raises on a corrupt file; it leaves it in
    place so the base class's ``is_valid_image_file`` integrity filter can drop it."""
    extracted = _build_extracted(tmp_path)
    bad = extracted / frepj_layout.ARCHIVE_ROOT / "images_40" / TAXON_B / "bad.jpg"
    bad.write_bytes(b"not a real image")

    imp = FREPJDatasetImporter(data_dir=tmp_path, hf_dataset_name="frepj")
    imp.extracted_dirs = str(extracted)
    imp._prepare_imagefolder()  # must NOT raise on the corrupt file

    imagefolder_dir = imp.imagefolder_dir
    # The corrupt file was copied+kept but the integrity gate would reject it.
    assert (imagefolder_dir / TAXON_B / "40_bad.jpg").exists()
    assert is_valid_image_file(imagefolder_dir / TAXON_B / "40_bad.jpg") is False
    # A valid neighbor still passes the integrity gate.
    assert is_valid_image_file(imagefolder_dir / TAXON_B / "40_1.jpg") is True


def test_integrity_removal_loop_removes_corrupt_end_to_end(tmp_path, monkeypatch):
    """IMP-03 wiring: ``import_dataset()``'s ``check_image_file_integrity`` branch actually
    removes a corrupt file end-to-end, network-free (HF loader stubbed, no push)."""
    imp = FREPJDatasetImporter(
        data_dir=tmp_path,
        hf_dataset_name="frepj",
        check_image_file_integrity=True,
        force_imagefolder_preparation=False,
        push_to_hub=False,
        show_progress=False,
    )

    # Pre-populate the imagefolder so import_dataset() skips download/extract/prepare and
    # falls straight into the integrity-removal branch.
    taxon_dir = imp.imagefolder_dir / TAXON_B
    taxon_dir.mkdir(parents=True)
    corrupt = taxon_dir / "40_bad.jpg"
    corrupt.write_bytes(b"not a real image")
    good = taxon_dir / "40_1.jpg"
    _write_rgb_jpg(good)

    # Stub the HF loader so no dataset processing or network happens after removal.
    monkeypatch.setattr(dataset_importer, "load_dataset", lambda *args, **kwargs: "FAKE_DATASET")

    imp.import_dataset()

    assert not corrupt.exists()  # base removal loop dropped the corrupt file
    assert good.exists()  # the valid neighbor survived
    assert imp.hf_dataset == "FAKE_DATASET"  # removal ran before the (stubbed) load


def test_frepj_config_contract():
    """IMP-02 guard: the import config targets the new importer with the direct figshare
    URL, CC BY 4.0, and citations that equal the ``frepj_layout`` constants (normalized)."""
    cfg = yaml.safe_load((root / "configs" / "dataset_import" / "frepj.yaml").read_text())

    assert cfg["_target_"] == "planktonzilla.dataset_import.frepj_importer.FREPJDatasetImporter"
    assert cfg["download_uris"] == "https://ndownloader.figshare.com/files/48928918"
    assert cfg["license"] == "cc-by-4.0"
    assert cfg["push_to_hub"] is False
    assert cfg["check_image_file_integrity"] is True
    assert "manual_download_local_file_names" not in cfg
    assert "Otake" in cfg["citation_apa"]
    assert frepj_layout.PAPER_DOI in cfg["citation_bibtex"]

    # Strengthening: citations equal the frepj_layout constants by normalized whitespace.
    def _norm(text):
        return " ".join(text.split())

    assert _norm(cfg["citation_apa"]) == _norm(frepj_layout.CITATION_APA)
    assert _norm(cfg["citation_bibtex"]) == _norm(frepj_layout.CITATION_BIBTEX)
