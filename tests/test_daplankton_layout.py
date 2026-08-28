"""
(c) Inria

Network-free tests pinning the recorded DAPlankton acquisition facts and the
five-domain merge helper.

Offline BY CONSTRUCTION: every assertion reads only the committed fixture under
``tests/fixtures/daplankton/`` and the importable constants in
``planktonzilla.dataset_import.daplankton_layout``. No HTTP, no download, no live
Fairdata call — the live 2.74 GB archive was enumerated once, on 2026-08-27, and what
that enumeration found is what these tests PIN. They do not "fix" the recorded facts.

The merge tests build synthetic domain roots rather than touching the real archive, so
they exercise the two properties that actually matter: that one taxon imaged in several
subset/instrument roots collapses to ONE class dir, and that the per-domain filename
prefix is what keeps that collapse from losing images to basename collisions. That
collision is not hypothetical — image basenames restart their counter in every
class-domain folder, so ``Aphanizomenon_flosaquae00001`` genuinely exists five times.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import csv
from pathlib import Path

import pytest
from PIL import Image

from planktonzilla.dataset_import import daplankton_layout as dl

FIXTURES = Path(__file__).parent / "fixtures" / "daplankton"
CLASS_DIRS_TSV = FIXTURES / "daplankton_class_dirs.tsv"

EXPECTED_TSV_HEADER = (
    "class_dir",
    "in_lab",
    "in_sea",
    "n_lab_cs",
    "n_lab_fc",
    "n_lab_ifcb",
    "n_sea_cs",
    "n_sea_ifcb",
    "n_total",
)


def _rows():
    with CLASS_DIRS_TSV.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_image(path, suffix=".png", size=8):
    """Produce a tiny valid image for the synthetic merge fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if suffix.lower() == ".jpg" else "PNG"
    Image.new("RGB", (size, size), color=(90, 130, 170)).save(path, format=fmt)


def test_class_dir_list_frozen():
    """The fixture is the authoritative 44-class contract, and it is self-consistent."""
    assert tuple(_rows()[0].keys()) == EXPECTED_TSV_HEADER

    rows = _rows()
    assert len(rows) == dl.N_CLASS_DIRS == 44
    names = [row["class_dir"] for row in rows]
    assert len(names) == len(set(names)), "duplicate class_dir in the frozen fixture"

    # Per-row arithmetic: n_total is the sum of the five domain cells.
    for row in rows:
        cells = [int(row[col]) for col in EXPECTED_TSV_HEADER[3:8]]
        assert sum(cells) == int(row["n_total"]), f"{row['class_dir']}: cells do not sum to n_total"


def test_recorded_totals_match_the_frozen_fixture():
    """The scalar constants are not an independent claim — they must equal the fixture."""
    rows = _rows()

    lab = sum(int(r["n_lab_cs"]) + int(r["n_lab_fc"]) + int(r["n_lab_ifcb"]) for r in rows)
    sea = sum(int(r["n_sea_cs"]) + int(r["n_sea_ifcb"]) for r in rows)

    assert lab == dl.N_LAB_IMAGES == 47471
    assert sea == dl.N_SEA_IMAGES == 64453
    assert lab + sea == dl.N_IMAGES == 111924

    assert sum(1 for r in rows if r["in_lab"] == "1") == dl.N_LAB_CLASSES == 15
    assert sum(1 for r in rows if r["in_sea"] == "1") == dl.N_SEA_CLASSES == 31


def test_the_two_taxa_imaged_in_both_subsets_are_why_44_not_46():
    """15 + 31 = 46 label slots, but 44 class dirs. The difference is these two, exactly."""
    both = sorted(r["class_dir"] for r in _rows() if r["in_lab"] == "1" and r["in_sea"] == "1")

    assert both == ["Aphanizomenon_flosaquae", "Pseudopedinella_sp"]
    assert dl.N_LAB_CLASSES + dl.N_SEA_CLASSES - len(both) == dl.N_CLASS_DIRS


def test_sea_has_no_flowcam_root():
    """FlowCam imaged the cultures only — five domains, not six. A sixth would be a bug."""
    assert len(dl.DOMAIN_PREFIXES) == 5

    domains = {domain for domain, _prefix in dl.DOMAIN_PREFIXES}
    assert ("DAPlankton_sea", "FC") not in domains
    assert domains == {
        ("DAPlankton_lab", "CS"),
        ("DAPlankton_lab", "FC"),
        ("DAPlankton_lab", "IFCB"),
        ("DAPlankton_sea", "CS"),
        ("DAPlankton_sea", "IFCB"),
    }

    # And the fixture agrees: no class carries sea/FC images, because the column does not exist.
    assert all(int(r["n_lab_fc"]) == 0 for r in _rows() if r["in_lab"] == "0")


def test_domain_prefixes_are_unique_and_non_empty():
    """A shared or empty prefix would silently reintroduce the collisions it exists to prevent."""
    prefixes = [prefix for _domain, prefix in dl.DOMAIN_PREFIXES]

    assert len(prefixes) == len(set(prefixes))
    assert all(prefix and prefix.endswith("_") for prefix in prefixes)


def test_license_and_deposit_identity():
    """The transcribed deposit facts, pinned so an edit to one place fails loudly."""
    assert dl.LICENSE == "cc-by-4.0"
    assert dl.LICENSE_URL == "https://creativecommons.org/licenses/by/4.0/"

    # The Etsin dataset UUID, NOT the DOI: the DOI form 404s against the download service.
    assert dl.FAIRDATA_PID == "a53a55a9-a591-404a-a372-d657d7efb89f"
    assert dl.FAIRDATA_PID in dl.SOURCE_URL
    assert dl.DATA_DOI not in dl.FAIRDATA_PID

    assert dl.PAPER_ARXIV_ID == "2402.05615"
    assert "Batrakhanov" in dl.CITATION_APA
    assert "Batrakhanov" in dl.CITATION_BIBTEX
    assert dl.DATA_DOI in dl.CITATION_APA


def test_both_image_suffixes_are_accepted():
    """CytoSense ships JPEG and IFCB/FlowCam ship PNG; globbing one drops an instrument."""
    assert set(dl.IMAGE_SUFFIXES) == {".jpg", ".png"}


def _build_domains(root_dir, layout):
    """Materialize ``{(subset, instrument): {class: [filenames]}}`` under ``root_dir``."""
    for (subset, instrument), classes in layout.items():
        for class_name, filenames in classes.items():
            for filename in filenames:
                _write_image(root_dir / subset / instrument / class_name / filename, Path(filename).suffix)


def test_merge_collapses_one_taxon_across_domains_into_one_class(tmp_path):
    """The whole point: a taxon imaged in several domains is ONE class, not five."""
    archive_root = tmp_path / "DAPlankton"
    _build_domains(
        archive_root,
        {
            ("DAPlankton_lab", "CS"): {"Aphanizomenon_flosaquae": ["Aphanizomenon_flosaquae00001.jpg"]},
            ("DAPlankton_lab", "FC"): {"Aphanizomenon_flosaquae": ["Aphanizomenon_flosaquae00001.png"]},
            ("DAPlankton_lab", "IFCB"): {"Aphanizomenon_flosaquae": ["Aphanizomenon_flosaquae00001.png"]},
            ("DAPlankton_sea", "CS"): {"Aphanizomenon_flosaquae": ["Aphanizomenon_flosaquae00001.jpg"]},
            ("DAPlankton_sea", "IFCB"): {"Aphanizomenon_flosaquae": ["Aphanizomenon_flosaquae00001.png"]},
        },
    )
    dest = tmp_path / "imagefolder"

    copied = dl.merge_domain_roots(archive_root, dest)

    # One class dir...
    assert sorted(p.name for p in dest.iterdir()) == ["Aphanizomenon_flosaquae"]
    # ...holding all five images, because the prefix disambiguated an identical basename
    # that would otherwise have collided four times over.
    assert copied == 5
    assert sorted(p.name for p in (dest / "Aphanizomenon_flosaquae").iterdir()) == [
        "lab_cs_Aphanizomenon_flosaquae00001.jpg",
        "lab_fc_Aphanizomenon_flosaquae00001.png",
        "lab_ifcb_Aphanizomenon_flosaquae00001.png",
        "sea_cs_Aphanizomenon_flosaquae00001.jpg",
        "sea_ifcb_Aphanizomenon_flosaquae00001.png",
    ]


def test_merge_keeps_both_extensions_and_skips_non_images(tmp_path):
    """JPEG and PNG both survive; a readme or a stray .txt does not become an example."""
    archive_root = tmp_path / "DAPlankton"
    _build_domains(
        archive_root,
        {
            ("DAPlankton_sea", "CS"): {"Ciliata": ["Ciliata00001.jpg"]},
            ("DAPlankton_sea", "IFCB"): {"Ciliata": ["Ciliata00001.png"]},
        },
    )
    (archive_root / "DAPlankton_sea" / "IFCB" / "Ciliata" / "notes.txt").write_text("not an image")
    dest = tmp_path / "imagefolder"

    copied = dl.merge_domain_roots(archive_root, dest)

    assert copied == 2
    assert sorted(p.name for p in (dest / "Ciliata").iterdir()) == ["sea_cs_Ciliata00001.jpg", "sea_ifcb_Ciliata00001.png"]


def test_merge_skips_an_absent_domain_root(tmp_path):
    """``DAPlankton_sea/FC`` legitimately does not exist; that is not an error."""
    archive_root = tmp_path / "DAPlankton"
    _build_domains(archive_root, {("DAPlankton_lab", "CS"): {"Diatoma_tenuis": ["Diatoma_tenuis00001.jpg"]}})
    dest = tmp_path / "imagefolder"

    assert dl.merge_domain_roots(archive_root, dest) == 1


def test_merge_refuses_to_overwrite(tmp_path):
    """No file is ever silently clobbered — the guard is explicit, not incidental."""
    archive_root = tmp_path / "DAPlankton"
    _build_domains(archive_root, {("DAPlankton_lab", "CS"): {"Ciliata": ["Ciliata00001.jpg"]}})
    dest = tmp_path / "imagefolder"

    assert dl.merge_domain_roots(archive_root, dest) == 1
    # A second run copies nothing new and leaves the existing file alone.
    assert dl.merge_domain_roots(archive_root, dest) == 0
    assert len(list((dest / "Ciliata").iterdir())) == 1


def test_a_rebuild_does_not_emit_a_warning_per_already_present_file(tmp_path, caplog):
    """Skipping an existing file is the NORMAL state of a rebuild, not a warning-worthy event.

    ``refresh=rebuild`` re-enters ``_prepare_imagefolder`` over a populated imagefolder —
    only ``redownload`` clears it first — so every one of the source's images is skipped.
    At full scale that is 111,924 skips, and one WARNING record each buries every other
    line the run emits. One summary line at INFO, and the detail at DEBUG.
    """
    archive_root = tmp_path / "DAPlankton"
    _build_domains(
        archive_root,
        {
            ("DAPlankton_lab", "CS"): {"Ciliata": ["Ciliata00001.jpg", "Ciliata00002.jpg"]},
            ("DAPlankton_sea", "IFCB"): {"Ciliata": ["Ciliata00001.png"]},
        },
    )
    dest = tmp_path / "imagefolder"
    assert dl.merge_domain_roots(archive_root, dest) == 3

    caplog.clear()
    with caplog.at_level("DEBUG"):
        assert dl.merge_domain_roots(archive_root, dest) == 0

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert warnings == [], f"a rebuild emitted {len(warnings)} warning(s)"
    assert any("Left 3 already-present" in record.message for record in caplog.records)
    # And the tree is still complete — the skip is correct behaviour, not a loss.
    assert len(list((dest / "Ciliata").iterdir())) == 3


def test_find_archive_root_handles_the_double_nesting(tmp_path):
    """A real extraction lands the subsets two DAPlankton levels down; that must still resolve."""
    real = tmp_path / "DAPlankton" / "DAPlankton"
    _build_domains(real, {("DAPlankton_lab", "CS"): {"Ciliata": ["Ciliata00001.jpg"]}})

    assert dl.find_archive_root(tmp_path) == real


def test_find_archive_root_prefers_the_shallowest_match(tmp_path):
    """A nested duplicate of the layout cannot displace the real, shallower one."""
    outer = tmp_path / "DAPlankton"
    _build_domains(outer, {("DAPlankton_lab", "CS"): {"Ciliata": ["Ciliata00001.jpg"]}})
    _build_domains(outer / "DAPlankton_lab" / "CS" / "nested", {("DAPlankton_sea", "CS"): {"Ciliata": ["x.jpg"]}})

    assert dl.find_archive_root(tmp_path) == outer


def test_find_archive_root_says_what_went_wrong(tmp_path):
    """The likely cause — the second unwrap did not happen — is named in the message."""
    (tmp_path / "DAPlankton").mkdir()

    with pytest.raises(FileNotFoundError, match="doubly nested"):
        dl.find_archive_root(tmp_path)
