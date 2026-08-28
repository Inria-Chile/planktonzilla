"""
(c) Inria

Network-free tests for the Tara Pacific acquisition/layout seam
(:mod:`planktonzilla.dataset_import.tara_pacific_layout`).

They pin the facts an import depends on and that nothing else re-derives: the seven
EcoTaxa project ids grouped into four sources, the frozen
``ecotaxa_taxon_id -> class dir`` map that keeps ``Raw_Labels`` stable while EcoTaxa
renames taxa upstream, the CC BY 4.0 slug transcribed into four importer configs, and the
``<objid>.jpg`` naming the redefiner reads back out of ``original_path``.

Offline BY CONSTRUCTION: the only files read are the committed TSV beside the module and
the four committed YAMLs. No HTTP, no downloads.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import csv
from collections import Counter

import pytest
import yaml

from planktonzilla.dataset_import import tara_pacific_layout as layout

_IMPORT_CONFIG_DIR = root / "configs" / "dataset_import"

# The counts published by each SEANOE deposit's abstract and observed on EcoTaxa, kept
# here as an INDEPENDENT transcription of what the module records. If the module is edited,
# this copy makes the test fail rather than agree with the edit.
_EXPECTED = {
    "tara_pacific_bongo": {"projects": (11370, 11369), "doi": "10.17882/102694", "classes": 137},
    "tara_pacific_decknet": {"projects": (11353, 11341), "doi": "10.17882/102697", "classes": 132},
    "tara_pacific_hsn": {"projects": (11292,), "doi": "10.17882/102336", "classes": 159},
    "tara_pacific_manta": {"projects": (1344, 1345), "doi": "10.17882/102537", "classes": 172},
}


def test_sources_match_the_four_seanoe_deposits():
    """Four sources, their EcoTaxa projects and their DOIs, as read from SEANOE."""
    assert set(layout.SOURCES) == set(_EXPECTED)
    for name, expected in _EXPECTED.items():
        assert layout.SOURCES[name]["projects"] == expected["projects"]
        assert layout.SOURCES[name]["deposit_doi"] == expected["doi"]
        assert layout.deposit_url(name) == f"https://doi.org/{expected['doi']}"


def test_every_ecotaxa_project_belongs_to_exactly_one_source():
    """A project claimed twice would import the same images under two ``dataset`` values."""
    assert len(layout.ALL_PROJECTS) == len(set(layout.ALL_PROJECTS)) == 7
    assert set(layout.ALL_PROJECTS) == {p for source in _EXPECTED.values() for p in source["projects"]}


def test_ecotaxa_project_urls_are_the_public_pages():
    urls = layout.ecotaxa_project_urls("tara_pacific_manta")
    assert urls == ("https://ecotaxa.obs-vlfr.fr/prj/1344", "https://ecotaxa.obs-vlfr.fr/prj/1345")


@pytest.mark.parametrize("source_name", sorted(_EXPECTED))
def test_class_map_covers_each_source_with_the_expected_class_count(source_name):
    """The frozen map has one entry per (source, taxon), and the class dirs are unique."""
    class_map = layout.load_class_map(source_name)
    assert len(class_map) == _EXPECTED[source_name]["classes"]
    # One taxon must not map to two class dirs, and two taxa must not share one: either
    # would make Raw_Labels ambiguous for the taxonomy join.
    assert len(set(class_map.values())) == len(class_map)
    assert layout.class_dirs(source_name) == tuple(sorted(class_map.values()))


def test_class_map_file_is_well_formed():
    """Exactly three columns, no blanks, integer taxon ids, sorted within each source."""
    with open(layout.CLASSES_TSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert tuple(reader.fieldnames) == layout.CLASSES_TSV_COLUMNS
        rows = list(reader)

    assert len(rows) == sum(source["classes"] for source in _EXPECTED.values())
    assert {row["dataset"] for row in rows} == set(_EXPECTED)
    for row in rows:
        assert row["class_dir"].strip(), row
        assert row["ecotaxa_taxon_id"].isdigit(), row

    # Written in source order, and by taxon id within a source, so a regeneration diffs
    # cleanly instead of reshuffling 600 lines.
    assert [row["dataset"] for row in rows] == sorted((row["dataset"] for row in rows), key=list(layout.SOURCES).index)
    for source_name in _EXPECTED:
        ids = [int(row["ecotaxa_taxon_id"]) for row in rows if row["dataset"] == source_name]
        assert ids == sorted(ids)


def test_unknown_source_raises_rather_than_importing_nothing():
    with pytest.raises(KeyError, match="no-such-source"):
        layout.load_class_map("no-such-source")


def test_image_file_name_round_trips_the_object_id():
    """``<objid>.jpg`` is the only link from an imagefolder path back to EcoTaxa."""
    assert layout.image_file_name(1129200000001) == "1129200000001.jpg"
    assert layout.object_id_from_file_name("1129200000001.jpg") == "1129200000001"
    assert layout.object_id_from_file_name("/Copepoda<Multicrustacea/1129200000001.jpg") == "1129200000001"


@pytest.mark.parametrize("name", ["", "not-a-number.jpg", "1129200000001.png", "40_123.jpg", "thumb.jpeg"])
def test_object_id_from_file_name_never_raises_on_a_foreign_name(name):
    """``original_path`` is untrusted input: an unparseable name yields "" mid-build."""
    assert layout.object_id_from_file_name(name) in ("", "1129200000001")


def test_reconcile_display_names_reports_renames_without_rewriting():
    """A renamed taxon is reported; the committed spelling is what the caller keeps."""
    class_map = {5: "Harosa", 25828: "Copepoda<Multicrustacea"}
    rows = [
        {"classif_id": 5, "display_name": "Harosa"},
        {"classif_id": 25828, "display_name": "Copepoda<Maxillopoda"},
        {"classif_id": 99999, "display_name": "Brand New Taxon"},
    ]
    renamed, unknown = layout.reconcile_display_names(rows, class_map)

    assert renamed == {25828: "Copepoda<Maxillopoda"}
    assert unknown == {99999: "Brand New Taxon"}
    # The map itself is untouched — the class dir keeps the frozen spelling.
    assert class_map[25828] == "Copepoda<Multicrustacea"


def test_reconcile_display_names_tolerates_missing_fields():
    """An unclassified object (null classif_id) is neither a rename nor an unknown taxon."""
    renamed, unknown = layout.reconcile_display_names([{"classif_id": None, "display_name": None}], {5: "Harosa"})
    assert renamed == {} and unknown == {}


def test_class_counts_only_counts_rows_the_map_can_name():
    rows = [{"classif_id": 5}, {"classif_id": 5}, {"classif_id": 42}]
    assert layout.class_counts(rows, {5: "Harosa"}) == Counter({"Harosa": 2})


@pytest.mark.parametrize("source_name", sorted(_EXPECTED))
def test_config_transcribes_the_layout_constants(source_name):
    """Each importer config repeats the licence, the DOI and the project ids; pin them."""
    config = yaml.safe_load((_IMPORT_CONFIG_DIR / f"{source_name}.yaml").read_text())

    assert config["license"] == layout.LICENSE
    assert config["source_url"] == layout.deposit_url(source_name)
    assert tuple(config["ecotaxa_projects"]) == layout.SOURCES[source_name]["projects"]
    # No archive: an empty download_uris is what routes the source through the sidecars.
    assert config["download_uris"] == ""
    assert config["push_to_hub"] is False
    assert config["check_image_file_integrity"] is True


def test_license_url_is_the_cc_by_deed():
    assert layout.LICENSE == "cc-by-4.0"
    assert layout.LICENSE_URL == "https://creativecommons.org/licenses/by/4.0/"
    assert layout.PAPER_DOI == "10.5194/essd-17-2761-2025"
