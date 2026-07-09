"""
(c) Inria

Network-free end-to-end test for the FREPJ geodata redefiner (GEO-03).

Drives the REAL ``FrepjRedefiner.redefine`` path over a synthetic imagefolder plus
SYNTHETIC Table_S3/S4 + crosswalk fixtures — the real 8.5 MB tables are NEVER touched.
It pins the per-image join (parse ``40_``/``100_`` filename -> per-image index ->
committed crosswalk) and proves an unresolved site token degrades to null
``Latitude``/``Longitude`` (carrying ``site``/``date``/``magnification``) WITHOUT raising.

Offline by construction: FREPJ uses only local CSVs, and the imagefolder loader is
forced offline via the HF flags. We additionally monkeypatch ``requests.get`` /
``requests.Session.get`` to raise if the redefine path ever hits the network.

Zero behavioral drift: this PINS current behavior, it never "fixes" it.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import datasets
import huggingface_hub.constants
import pytest
from PIL import Image as PILImage

from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp

# --- Synthetic sidecar tables ---------------------------------------------------------
# Headers mirror the real Table_S3/S4 (trailing-space "Order "/"Others " columns); the
# Table_S4 rows carry the source's stray trailing commas to exercise the ragged-line path.
# IDs match the imagefolder filenames below. Table_S3 has TWO rows for ID 1 (a losing
# "duplicatelosersite" first, the winning "akigawadam" last) to pin last-write-wins.
_TABLE_S3 = (
    "ID,Class,Order ,Family,Genus,Species,Others ,Sampling site,Sampling date\n"
    "1,Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina longirostris,,duplicatelosersite,1999.01.01\n"
    "1,Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina longirostris,,akigawadam,2018.03.15\n"
    "3,Branchiopoda,Diplostraca,Daphniidae,Daphnia,Daphnia galeata,,biwako,2019.07.01\n"
    "9,Branchiopoda,Diplostraca,Chydoridae,Chydorus,Chydorus sphaericus,,unresolvedtoken,2020.01.01\n"
)
_TABLE_S4 = (
    "ID,Class,Order ,Family,Genus,Species,Others ,Sampling site,Sampling date\n"
    "2,Hexanauplia,Cyclopoida,Cyclopidae,Cyclops,Cyclops sp.,,hajidamu,2020.05.10,,,\n"
)

# The committed synthetic crosswalk maps the three resolvable tokens to known coords and
# OMITS "unresolvedtoken" (and "duplicatelosersite") — so those tokens resolve to null.
_CROSSWALK_FIXTURE = root / "tests" / "fixtures" / "frepj" / "frepj_crosswalk_sample.csv"

# filename -> (magnification, site, date, latitude, longitude). None lat/lon == unresolved.
_EXPECTED = {
    "40_1.jpg": ("40", "akigawadam", "2018.03.15", 35.42534327, 137.4318704),
    "100_2.jpg": ("100", "hajidamu", "2020.05.10", 34.6, 132.5),
    "40_3.jpg": ("40", "biwako", "2019.07.01", 35.25, 136.05),
    "40_9.jpg": ("40", "unresolvedtoken", "2020.01.01", None, None),
}


def _write_jpg(path):
    """Write a small valid RGB JPEG so the imagefolder loader reads a real image."""
    PILImage.new("RGB", (8, 8), color=(120, 160, 200)).save(path, "JPEG")


def _write_taxonomy_csv(path):
    """Write a tiny taxonomy CSV that ``_build_lookup`` can resolve for the FREPJ source."""
    header = (
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
    )
    row = "frepj,copepoda,Animalia,Arthropoda,,,,,,Copepoda,True,zoo,,Q3386609,274;1231,135336.0,6854.0,"
    with open(path, "w") as f:
        f.write(header + "\n" + row + "\n")


def test_frepj_redefiner_offline_attaches_geodata_and_nulls_unresolved(monkeypatch, tmp_path):
    """Offline FREPJ redefine attaches lat/lon/date/mag/site and nulls the unresolved token."""
    # Offline by construction; these guards PROVE it. HF offline env + captured flags stop
    # the imagefolder loader from resolving its builder via a Hub HEAD, and the monkeypatched
    # requests raise if the FREPJ redefine path ever touches the network.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True, raising=False)

    def _no_network(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(gp.requests, "get", _no_network)
    monkeypatch.setattr(gp.requests.Session, "get", _no_network)

    # Determinism: the module-level global read by _serialize/_flatten. Set via
    # monkeypatch so it auto-reverts and never leaks into later tests (e.g.
    # test_gen_planktonzilla_hydra pins gp.num_proc == default_num_proc()).
    monkeypatch.setattr(gp, "num_proc", 1)

    # Synthetic sidecar tables under a throwaway tables_dir (never the real 8.5 MB files).
    tables_dir = tmp_path / "frepj_tables"
    tables_dir.mkdir()
    (tables_dir / "Table_S3.csv").write_text(_TABLE_S3)
    (tables_dir / "Table_S4.csv").write_text(_TABLE_S4)

    # Imagefolder: 3 resolvable images + 1 whose Table_S3 token has no crosswalk entry.
    imagefolder = tmp_path / "frepj_imagefolder"
    layout = {"copepoda": ["40_1.jpg", "100_2.jpg"], "cladocera": ["40_3.jpg", "40_9.jpg"]}
    for class_name, filenames in layout.items():
        class_dir = imagefolder / class_name
        class_dir.mkdir(parents=True)
        for filename in filenames:
            _write_jpg(class_dir / filename)

    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)

    dataset = datasets.load_dataset("imagefolder", data_files={"train": str(imagefolder / "*/*.jpg")})

    redefiner = gp.FrepjRedefiner(
        csv_taxonomies_path=str(csv_path),
        crosswalk_path=str(_CROSSWALK_FIXTURE),
        tables_dir=str(tables_dir),
    )
    # Must NOT raise even though "40_9.jpg" carries an unresolved site token.
    result = redefiner.redefine(hf_dataset=dataset, dataset_name="frepj", num_proc=1)

    assert len(result) == len(_EXPECTED)

    # Additive FREPJ columns AND the existing metadata columns are all present.
    for col in ("magnification", "site", "date", "Latitude", "Longitude", "Depth_max", "Depth_min", "timestamp", "ObjID"):
        assert col in result.column_names

    # Index rows by filename (column access avoids decoding the image column).
    paths = result["original_path"]
    records = {
        paths[i].split("/")[-1]: {
            "magnification": result["magnification"][i],
            "site": result["site"][i],
            "date": result["date"][i],
            "Latitude": result["Latitude"][i],
            "Longitude": result["Longitude"][i],
            "Depth_max": result["Depth_max"][i],
            "Depth_min": result["Depth_min"][i],
            "timestamp": result["timestamp"][i],
            "ObjID": result["ObjID"][i],
            "dataset": result["dataset"][i],
        }
        for i in range(len(result))
    }
    assert set(records) == set(_EXPECTED)

    for filename, (magnification, site, date, latitude, longitude) in _EXPECTED.items():
        rec = records[filename]
        assert rec["dataset"] == "frepj"
        # magnification matches the filename prefix; site is the raw token; date is verbatim.
        assert rec["magnification"] == magnification
        assert rec["site"] == site
        assert rec["date"] == date

        if latitude is None:
            # Unresolved token: null lat/lon, but site/date/magnification still carried.
            assert rec["Latitude"] is None
            assert rec["Longitude"] is None
        else:
            assert rec["Latitude"] == pytest.approx(latitude, abs=1e-3)
            assert rec["Longitude"] == pytest.approx(longitude, abs=1e-3)

        # No Depth for FREPJ (surface net tows); FREPJ sets neither ObjID nor Timestamp.
        assert rec["Depth_max"] is None
        assert rec["Depth_min"] is None
        assert rec["ObjID"] is None
        assert rec["timestamp"] is None
