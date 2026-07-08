"""
(c) Inria

Network-free tests pinning the frozen Phase-15 FREPJ acquisition inputs.

These four tests close ACQ-02/03/04/05 as committed, independently-verifiable
contracts. They are offline BY CONSTRUCTION: every assertion reads only in-repo
fixtures under ``tests/fixtures/frepj/`` and the importable constants in
``planktonzilla.dataset_import.frepj_layout`` — no HTTP, no download, no live
figshare call. The live 963 MB archive is handled separately in Plan 15-02.

Zero behavioral drift: these tests PIN the recorded reconnaissance facts, they do
not "fix" them.
"""

import json
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from PIL import Image

from planktonzilla.dataset_import import frepj_layout
from planktonzilla.dataset_import.dataset_importer import is_valid_image_file

FIXTURES = Path(__file__).parent / "fixtures" / "frepj"

EXPECTED_TSV_HEADER = ("class_dir", "in_40x", "in_100x", "n_40x", "n_100x", "n_total")


def _write_jpg(path, size=8):
    """Produce a tiny valid 3-channel RGB JPEG for the synthetic merge fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(90, 130, 170)).save(path, format="JPEG", quality=90)


def test_class_dir_list_frozen():
    """ACQ-02: the committed TSV holds exactly 229 comma-preserving class-dir rows."""
    tsv_path = FIXTURES / "frepj_class_dirs.tsv"
    lines = tsv_path.read_text().splitlines()

    # Header must be the expected 6 tab-separated columns.
    header = tuple(lines[0].split("\t"))
    assert header == EXPECTED_TSV_HEADER

    data_rows = lines[1:]
    assert len(data_rows) == 229

    # Constant, fixture row count, and requirement value all agree.
    assert frepj_layout.N_CLASS_DIRS == 229 == len(data_rows)

    # Every data row keeps exactly 6 tab-delimited fields — proving the commas inside
    # class_dir were NOT turned into extra columns or stripped (they are legitimate
    # Class,Order,Family,Genus,Species tuples; T-15-02).
    class_dirs = []
    for row in data_rows:
        fields = row.split("\t")
        assert len(fields) == 6, f"row has {len(fields)} fields, expected 6: {row!r}"
        class_dirs.append(fields[0])

    comma_rows = [cd for cd in class_dirs if "," in cd]
    assert comma_rows, "expected at least one comma-containing class_dir; commas may have been stripped"


def test_sample_images_are_rgb_jpeg():
    """ACQ-03: the committed fixture is a decodable 3-channel baseline RGB JPEG."""
    jpg_path = FIXTURES / "sample_rgb.jpg"

    with Image.open(jpg_path) as img:
        assert img.format == frepj_layout.EXPECTED_IMAGE_FORMAT == "JPEG"
        assert img.mode == frepj_layout.EXPECTED_IMAGE_MODE == "RGB"

    # The existing corrupt-image gate must accept it (T-15-04).
    assert is_valid_image_file(str(jpg_path)) is True


def test_license_and_citation():
    """ACQ-04: recorded license/citation/DOI constants mirror the figshare manifest."""
    manifest = json.loads((FIXTURES / "frepj_figshare_manifest.json").read_text())

    # License faithfully mirrored (T-15-05).
    assert manifest["license"]["name"] == frepj_layout.LICENSE_NAME == "CC BY 4.0"
    assert frepj_layout.LICENSE == "cc-by-4.0"

    # The archive md5 constant byte-matches the manifest record (T-15-01).
    archive = next(f for f in manifest["files"] if f["name"] == frepj_layout.ARCHIVE_FILENAME)
    assert archive["md5"] == frepj_layout.ARCHIVE_MD5

    # Both DOIs mirror the manifest.
    assert frepj_layout.DATA_DOI == manifest["data_doi"]
    assert frepj_layout.PAPER_DOI == manifest["paper_doi"]

    # The recorded APA citation records the first author and the paper DOI.
    assert "Otake" in frepj_layout.CITATION_APA
    assert frepj_layout.PAPER_DOI in frepj_layout.CITATION_APA


def test_merge_collision_safe(tmp_path):
    """ACQ-05: both magnifications flatten into ONE comma-named class, collision-safe."""
    taxon = "Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina longirostris"

    # Same basename 0.jpg present under BOTH magnification roots for the same taxon.
    images_40 = tmp_path / "images_40"
    images_100 = tmp_path / "images_100"
    _write_jpg(images_40 / taxon / "0.jpg")
    _write_jpg(images_100 / taxon / "0.jpg")

    dest = tmp_path / "dest"
    copied = frepj_layout.merge_two_magnification_roots(images_40, images_100, dest)

    # Both source files survived the shared-basename collision.
    assert copied == 2

    # Exactly ONE class dir — the taxon is a single class, not doubled per magnification.
    class_dirs = sorted(p.name for p in dest.iterdir() if p.is_dir())
    assert class_dirs == [taxon]

    # The comma in the taxon name survived verbatim (never sanitized; T-15-02).
    merged_taxon = dest / taxon
    assert "," in merged_taxon.name

    # Both magnification-prefixed files exist; neither overwrote the other.
    assert (merged_taxon / "40_0.jpg").exists()
    assert (merged_taxon / "100_0.jpg").exists()

    # No per-magnification suffixed sibling classes were created.
    assert not (dest / f"{taxon}@40x").exists()
    assert not (dest / f"{taxon}@100x").exists()
