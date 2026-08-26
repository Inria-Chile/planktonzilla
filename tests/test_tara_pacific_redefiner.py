"""
(c) Inria

Network-free end-to-end test for :class:`TaraPacificRedefiner`.

Drives the REAL ``redefine`` path over a synthetic imagefolder plus a synthetic EcoTaxa
manifest, and pins the reason this redefiner exists at all: the four Tara Pacific sources
come from EcoTaxa, but :class:`EcoTaxaRedefiner` would issue one
``GET /api/object/{objid}`` per image — 2.35 MILLION requests across them — to re-learn
what the importer's manifest already states. This one reads the manifest and makes ZERO.

It also pins the shape of what reaches the consolidated schema: ``ObjID`` from the file
name, lat/lon/depth from the manifest, ``timestamp`` as ``objdate`` refined by ``objtime``,
and ``custom_metadata`` carrying the two facts only this source has (``orig_id`` — the
station/tow key back into the SEANOE deposit — and the EcoTaxa project id).

Offline by construction: the imagefolder loader is forced offline via the HF flags, and
``requests.get`` / ``requests.Session.get`` are monkeypatched to raise if the redefine path
ever touches the network.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import json

import datasets
import huggingface_hub.constants
import pytest
from datasets import Value
from PIL import Image as PILImage

from planktonzilla.dataset_import import ecotaxa_client
from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp

DATASET = "tara_pacific_hsn"
CLASS_DIR = "Copepoda<Multicrustacea"

# objid -> (class dir, orig_id, lat, lon, depth_min, depth_max, objdate, objtime).
# 1129200000004 is deliberately ABSENT from the manifest: an image whose object the
# manifest does not know must degrade to ObjID-only, never raise mid-build.
_MANIFEST_ROWS = {
    1129200000001: (
        CLASS_DIR,
        "tara_pacific_2016_i00oa10_d_hsn_330_tot_1_36",
        "33.5987835",
        "-34.781233",
        "0.0",
        "1.0",
        "2016-06-09",
        "09:40:00",
    ),
    1129200000002: (
        CLASS_DIR,
        "tara_pacific_2016_i00oa10_d_hsn_330_tot_1_430",
        "12.5",
        "-140.25",
        "0.0",
        "1.0",
        "2016-06-10",
        "",
    ),
    1129200000003: (CLASS_DIR, "tara_pacific_2017_i19oa142_d_hsn_330_tot_2_7", "", "", "", "", "", ""),
}
_ORPHAN_OBJID = 1129200000004
# A second class dir, so the imagefolder loader emits a `label` column (it does not for a
# single-class tree) — and so the taxonomy join is exercised on more than one label.
_OTHER_CLASS_DIR = "Harosa"


def _write_jpg(path, size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (size, size), (70, 110, 150)).save(path, format="JPEG", quality=90)


def _write_taxonomy_csv(path):
    """A tiny taxonomy CSV ``_build_lookup`` can resolve for this source's one class."""
    header = (
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
    )
    rows = [
        f"{DATASET},{CLASS_DIR},animalia,arthropoda,copepoda,,,,,copepoda,True,living,full_body,Q133602,,1080.0,6854.0,",
        f"{DATASET},{_OTHER_CLASS_DIR},chromista,,,,,,,chromista,True,living,full_body,Q862296,,7.0,,",
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _write_manifest(path):
    rows = []
    for objid, (_, orig_id, lat, lon, dmin, dmax, date, time) in _MANIFEST_ROWS.items():
        row = dict.fromkeys(ecotaxa_client.MANIFEST_COLUMNS)
        row.update(
            objid=objid,
            orig_id=orig_id,
            classif_id=25828,
            display_name=CLASS_DIR,
            latitude=lat,
            longitude=lon,
            objdate=date,
            objtime=time,
            depth_min=dmin,
            depth_max=dmax,
            img_file_name=f"a/{objid}.jpg",
        )
        rows.append(row)
    return ecotaxa_client.write_manifest(rows, path)


def test_tara_pacific_redefiner_joins_the_manifest_offline(monkeypatch, tmp_path):
    """The whole redefine runs from local files: manifest in, consolidated schema out."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True, raising=False)

    def _no_network(*args, **kwargs):
        raise AssertionError("network called")

    # EcoTaxaRedefiner would call exactly these, once per image. This one must not.
    monkeypatch.setattr(gp.requests, "get", _no_network)
    monkeypatch.setattr(gp.requests.Session, "get", _no_network)
    monkeypatch.setattr(gp, "num_proc", 1)

    manifest = _write_manifest(tmp_path / "manifests" / "ecotaxa_project_11292.tsv")

    imagefolder = tmp_path / "imagefolder"
    for objid in _MANIFEST_ROWS:
        _write_jpg(imagefolder / CLASS_DIR / f"{objid}.jpg")
    _write_jpg(imagefolder / _OTHER_CLASS_DIR / f"{_ORPHAN_OBJID}.jpg")

    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)

    dataset = datasets.load_dataset("imagefolder", data_files={"train": str(imagefolder / "*/*.jpg")})

    redefiner = gp.TaraPacificRedefiner(csv_taxonomies_path=str(csv_path))
    redefiner.attach_sidecars({manifest.name: manifest, "tara_pacific_classes.tsv": tmp_path / "classes.tsv"})

    result = redefiner.redefine(hf_dataset=dataset, dataset_name=DATASET, num_proc=1)

    assert len(result) == len(_MANIFEST_ROWS) + 1
    # No column of its own: everything rides in the consolidated schema.
    assert set(result.column_names) == set(constants.CONSOLIDATED_COLUMNS)
    assert result.features[constants.CUSTOM_METADATA_COL] == Value("string")

    columns = ("ObjID", "Latitude", "Longitude", "Depth_min", "Depth_max", "timestamp", "custom_metadata", "dataset", "Class")
    paths = result["original_path"]
    records = {paths[i].split("/")[-1]: {col: result[col][i] for col in columns} for i in range(len(result))}

    first = records["1129200000001.jpg"]
    assert first["dataset"] == DATASET
    assert first["ObjID"] == "1129200000001"
    assert first["Latitude"] == pytest.approx(33.5987835, abs=1e-4)
    assert first["Longitude"] == pytest.approx(-34.781233, abs=1e-4)
    assert first["Depth_min"] == pytest.approx(0.0)
    assert first["Depth_max"] == pytest.approx(1.0)
    # objdate refined by objtime into one ISO timestamp.
    assert first["timestamp"] == "2016-06-09T09:40:00"
    # The taxonomy join still happened, keyed on the class dir.
    assert first["Class"] == "copepoda"
    # The two facts only this source has, sorted by key, nothing else.
    assert json.loads(first["custom_metadata"]) == {
        "ecotaxa_project": "11292",
        "orig_id": "tara_pacific_2016_i00oa10_d_hsn_330_tot_1_36",
    }

    # A row with a date but no time keeps the bare date rather than inventing midnight.
    assert records["1129200000002.jpg"]["timestamp"] == "2016-06-10"

    # A row EcoTaxa recorded without coordinates, depth or date: nulls, never zeros.
    third = records["1129200000003.jpg"]
    assert third["Latitude"] is None and third["Longitude"] is None
    assert third["Depth_min"] is None and third["Depth_max"] is None
    assert third["timestamp"] is None
    assert third["ObjID"] == "1129200000003"

    # An image the manifest does not know: ObjID only, and no raise.
    orphan = records[f"{_ORPHAN_OBJID}.jpg"]
    assert orphan["ObjID"] == str(_ORPHAN_OBJID)
    assert orphan["Latitude"] is None
    assert orphan["custom_metadata"] == constants.EMPTY_CUSTOM_METADATA


def test_redefiner_is_lazy_and_names_the_importer_when_no_manifest_is_attached(tmp_path):
    """Constructing costs nothing (the pipeline builds every redefiner up front); the
    first read fails with the remedy, and this class NEVER downloads."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)

    redefiner = gp.TaraPacificRedefiner(csv_taxonomies_path=str(csv_path))
    redefiner.attach_sidecars({"tara_pacific_classes.tsv": tmp_path / "classes.tsv"})

    with pytest.raises(FileNotFoundError) as excinfo:
        redefiner._load_manifests()

    message = str(excinfo.value)
    assert "tara_pacific_manifests" in message
    assert "NEVER downloads" in message


@pytest.mark.parametrize(
    ("object_id", "expected"),
    [
        ("1129200000001", "11292"),
        ("134500000001", "1345"),
        ("134400000001", "1344"),
        ("1134100000001", "11341"),
        # Not one of the seven projects, too short, or not a number at all: no project
        # rather than a wrong one.
        ("999900000001", ""),
        ("123", ""),
        ("", ""),
    ],
)
def test_project_of_decodes_only_the_seven_known_projects(object_id, expected):
    assert gp.TaraPacificRedefiner._project_of(object_id) == expected


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"objdate": "2016-06-09", "objtime": "09:40:00"}, "2016-06-09T09:40:00"),
        ({"objdate": "2016-06-09", "objtime": ""}, "2016-06-09"),
        ({"objdate": "2016-06-09", "objtime": None}, "2016-06-09"),
        # A malformed time is dropped, not guessed at: the date alone is still true.
        ({"objdate": "2016-06-09", "objtime": "not-a-time"}, "2016-06-09"),
        ({"objdate": "", "objtime": "09:40:00"}, None),
        ({"objdate": None, "objtime": None}, None),
    ],
)
def test_iso_timestamp_refines_the_date_without_inventing_one(row, expected):
    assert gp.TaraPacificRedefiner._iso_timestamp(row) == expected


def test_attach_sidecars_is_a_no_op_on_the_other_redefiners(tmp_path):
    """The protocol is opt-in: an archive-only source's redefiner ignores what it is given."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)

    for cls in (gp.NoMetadataRedefiner, gp.EcoTaxaRedefiner, gp.WHOIRedefiner):
        assert cls(csv_taxonomies_path=str(csv_path)).attach_sidecars({"x": tmp_path}) is None


def test_the_registry_key_resolves_to_this_class():
    assert gp.REDEFINERS["tara_pacific"] is gp.TaraPacificRedefiner
