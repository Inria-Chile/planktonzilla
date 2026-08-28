"""
(c) Inria

Network-free end-to-end test for the FREPJ geodata redefiner (GEO-03).

Drives the REAL ``FrepjRedefiner.redefine`` path over a synthetic imagefolder plus
SYNTHETIC Table_S1/S3/S4 + crosswalk fixtures — the real 8.5 MB tables are NEVER touched.
It pins the per-image join (parse ``40_``/``100_`` filename -> per-image index ->
committed crosswalk), the normalization of the hand-typed upstream sampling date into
``timestamp`` (KI-26: fixed rules, Table_S1 disambiguation of a three-digit day, null
rather than a guess), the ``custom_metadata`` JSON object carrying ``magnification`` /
``site`` (FREPJ has no column of its own), and proves an unresolved site token degrades
to null ``Latitude``/``Longitude`` WITHOUT raising.

Offline by construction: FREPJ uses only local CSVs, and the imagefolder loader is
forced offline via the HF flags. We additionally monkeypatch ``requests.get`` /
``requests.Session.get`` to raise if the redefine path ever hits the network.
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

from planktonzilla.planktonzilla_dataset import constants, frepj_tables
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp

# --- Synthetic sidecar tables ---------------------------------------------------------
# Table_S1: one row per (site, sampling date). Lake Biwa has BOTH 2021-11-01 and
# 2021-11-15 (so a three-digit day naming both stays null) plus a month-only row that
# contributes nothing. Site names match the `resolved_site` column of the crosswalk
# fixture, which is how a token reaches its Table_S1 dates.
_TABLE_S1 = (
    "site,North latitude,East latitude,date\n"
    "Akigawa Dam,35.42534327,137.4318704,2018.03.15\n"
    "Lake Biwa,35.25,136.05,2021.11.01\n"
    "Lake Biwa,35.25,136.05,2021.11.15\n"
    "Lake Biwa,35.25,136.05,2019.07\n"
    "Haji Dam,34.6,132.5,2020.05.10\n"
)
# Headers mirror the real Table_S3/S4 (trailing-space "Order "/"Others " columns); the
# Table_S4 rows carry the source's stray trailing commas to exercise the ragged-line path.
# IDs match the imagefolder filenames below. Table_S3 has TWO rows for ID 1 (a losing
# "duplicatelosersite" first, the winning "akigawadam" last) to pin last-write-wins. The
# "Sampling date" values cover the KI-26 families: clean, three-digit day resolvable via
# Table_S1 (ID 5), three-digit day ambiguous in Table_S1 (ID 7), bare site token (ID 8),
# dots omitted (ID 9) and a YYMMDD+site prefix (ID 4).
_TABLE_S3 = (
    "ID,Class,Order ,Family,Genus,Species,Others ,Sampling site,Sampling date\n"
    "1,Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina longirostris,,duplicatelosersite,1999.01.01\n"
    "1,Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina longirostris,,akigawadam,2018.03.15\n"
    "3,Branchiopoda,Diplostraca,Daphniidae,Daphnia,Daphnia galeata,,biwako,2019.07.01\n"
    "5,Branchiopoda,Diplostraca,Daphniidae,Daphnia,Daphnia galeata,,biwako,2021.11.011\n"
    "7,Branchiopoda,Diplostraca,Daphniidae,Daphnia,Daphnia galeata,,biwako,2021.11.015\n"
    "8,Branchiopoda,Diplostraca,Bosminidae,Bosmina,Bosmina longirostris,,akigawadam,akanko1\n"
    "9,Branchiopoda,Diplostraca,Chydoridae,Chydorus,Chydorus sphaericus,,unresolvedtoken,20200101\n"
)
_TABLE_S4 = (
    "ID,Class,Order ,Family,Genus,Species,Others ,Sampling site,Sampling date\n"
    "2,Hexanauplia,Cyclopoida,Cyclopidae,Cyclops,Cyclops sp.,,hajidamu,2020.05.10,,,\n"
    "4,Hexanauplia,Cyclopoida,Cyclopidae,Cyclops,Cyclops sp.,,hajidamu,230815hajidamu,,,\n"
)

# The committed synthetic crosswalk maps the three resolvable tokens to known coords and
# OMITS "unresolvedtoken" (and "duplicatelosersite") — so those tokens resolve to null.
_CROSSWALK_FIXTURE = root / "tests" / "fixtures" / "frepj" / "frepj_crosswalk_sample.csv"

# filename -> (magnification, site, timestamp, latitude, longitude).
# None lat/lon == unresolved token; None timestamp == a date the build refuses to guess.
_EXPECTED = {
    "40_1.jpg": ("40", "akigawadam", "2018-03-15", 35.42534327, 137.4318704),
    "100_2.jpg": ("100", "hajidamu", "2020-05-10", 34.6, 132.5),
    "40_3.jpg": ("40", "biwako", "2019-07-01", 35.25, 136.05),
    "100_4.jpg": ("100", "hajidamu", "2023-08-15", 34.6, 132.5),
    "40_5.jpg": ("40", "biwako", "2021-11-01", 35.25, 136.05),
    "40_7.jpg": ("40", "biwako", None, 35.25, 136.05),
    "40_8.jpg": ("40", "akigawadam", None, 35.42534327, 137.4318704),
    "40_9.jpg": ("40", "unresolvedtoken", "2020-01-01", None, None),
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
    """Offline FREPJ redefine attaches lat/lon, a normalized timestamp and custom_metadata."""
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
    # The redefine path reads local files only: constructing a DownloadManager here is a bug.
    monkeypatch.setattr(frepj_tables, "DownloadManager", _no_network)

    # Determinism: the module-level global read by _serialize/_flatten. Set via
    # monkeypatch so it auto-reverts and never leaks into later tests (e.g.
    # test_gen_planktonzilla_hydra pins gp.num_proc == default_num_proc()).
    monkeypatch.setattr(gp, "num_proc", 1)

    # Synthetic sidecar tables under a throwaway tables_dir (never the real 8.5 MB files).
    tables_dir = tmp_path / "frepj_tables"
    tables_dir.mkdir()
    (tables_dir / "Table_S1.csv").write_text(_TABLE_S1)
    (tables_dir / "Table_S3.csv").write_text(_TABLE_S3)
    (tables_dir / "Table_S4.csv").write_text(_TABLE_S4)

    # Imagefolder: 7 resolvable images + 1 whose Table_S3 token has no crosswalk entry.
    imagefolder = tmp_path / "frepj_imagefolder"
    layout = {
        "copepoda": ["40_1.jpg", "100_2.jpg", "100_4.jpg", "40_8.jpg"],
        "cladocera": ["40_3.jpg", "40_5.jpg", "40_7.jpg", "40_9.jpg"],
    }
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

    # FREPJ adds NO column of its own: everything rides in the consolidated schema.
    assert set(result.column_names) == set(constants.CONSOLIDATED_COLUMNS)
    for col in ("magnification", "site", "date", "metadata"):
        assert col not in result.column_names
    assert result.features[constants.CUSTOM_METADATA_COL] == Value("string")
    assert result.features["timestamp"] == Value("string")

    # Index rows by filename (column access avoids decoding the image column).
    paths = result["original_path"]
    columns = ("custom_metadata", "timestamp", "Latitude", "Longitude", "Depth_max", "Depth_min", "ObjID", "dataset")
    records = {paths[i].split("/")[-1]: {col: result[col][i] for col in columns} for i in range(len(result))}
    assert set(records) == set(_EXPECTED)

    for filename, (magnification, site, timestamp, latitude, longitude) in _EXPECTED.items():
        rec = records[filename]
        assert rec["dataset"] == "frepj"
        # magnification matches the filename prefix; site is the raw token — both in the
        # JSON object, sorted by key, nothing else.
        assert json.loads(rec["custom_metadata"]) == {"magnification": magnification, "site": site}
        assert rec["custom_metadata"] == f'{{"magnification": "{magnification}", "site": "{site}"}}'
        # The upstream sampling date normalized to ISO, or null when it cannot be read.
        assert rec["timestamp"] == timestamp

        if latitude is None:
            # Unresolved token: null lat/lon, but site/magnification/timestamp still carried.
            assert rec["Latitude"] is None
            assert rec["Longitude"] is None
        else:
            assert rec["Latitude"] == pytest.approx(latitude, abs=1e-3)
            assert rec["Longitude"] == pytest.approx(longitude, abs=1e-3)

        # No Depth for FREPJ (surface net tows); FREPJ sets no ObjID.
        assert rec["Depth_max"] is None
        assert rec["Depth_min"] is None
        assert rec["ObjID"] is None


def test_frepj_redefiner_is_lazy_and_names_the_importer_on_a_missing_table(tmp_path):
    """Construction needs no table (the pipeline builds every redefiner up front); the first
    read does, and its error points at the importer that fetches them."""
    tables_dir = tmp_path / "frepj_tables"
    tables_dir.mkdir()
    (tables_dir / "Table_S3.csv").write_text(_TABLE_S3)
    (tables_dir / "Table_S4.csv").write_text(_TABLE_S4)
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)

    redefiner = gp.FrepjRedefiner(
        csv_taxonomies_path=str(csv_path), crosswalk_path=str(_CROSSWALK_FIXTURE), tables_dir=str(tables_dir)
    )
    assert redefiner._tables_loaded is False

    with pytest.raises(FileNotFoundError, match=r"Table_S1\.csv.*ensure_sidecars"):
        redefiner._load_tables()


def test_frepj_redefiner_names_the_checkout_remedy_for_a_missing_crosswalk(tmp_path):
    """Construction reads nothing (in the pipeline the importer is the gate); the first read of
    an absent crosswalk says it is committed and how to restore it."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)
    redefiner = gp.FrepjRedefiner(
        csv_taxonomies_path=str(csv_path),
        crosswalk_path=str(tmp_path / "frepj_site_crosswalk.csv"),
        tables_dir=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match=r"frepj_site_crosswalk\.csv.*git checkout"):
        redefiner._load_tables()


def test_attached_sidecars_win_over_tables_dir(tmp_path):
    """What the importer hands over through attach_sidecars is read, not tables_dir."""
    empty = tmp_path / "empty_tables"
    empty.mkdir()
    attached = tmp_path / "attached"
    attached.mkdir()
    (attached / "Table_S1.csv").write_text(_TABLE_S1)
    (attached / "Table_S3.csv").write_text(_TABLE_S3)
    (attached / "Table_S4.csv").write_text(_TABLE_S4)
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)

    redefiner = gp.FrepjRedefiner(
        csv_taxonomies_path=str(csv_path), crosswalk_path=str(_CROSSWALK_FIXTURE), tables_dir=str(empty)
    )
    redefiner.attach_sidecars(
        {
            "Table_S1.csv": attached / "Table_S1.csv",
            "Table_S3.csv": attached / "Table_S3.csv",
            "Table_S4.csv": attached / "Table_S4.csv",
            "frepj_site_crosswalk.csv": _CROSSWALK_FIXTURE,
        }
    )
    redefiner._load_tables()

    assert redefiner.site_index[(40, "5")] == ("biwako", "2021.11.011")
    assert redefiner.crosswalk_sites["biwako"] == "Lake Biwa"
    assert redefiner.site_dates["Lake Biwa"] == {"2021-11-01", "2021-11-15"}
    # Attaching again REPLACES the mapping and invalidates the cache: with nothing attached
    # the empty tables_dir is what gets read, and it has nothing.
    redefiner.attach_sidecars({})
    assert redefiner._tables_loaded is False
    with pytest.raises(FileNotFoundError, match=r"Table_S1\.csv"):
        redefiner._load_tables()


def test_attach_sidecars_is_a_no_op_on_the_other_redefiners(tmp_path):
    """The sixteen archive-only sources' redefiners accept the seam's call and keep nothing."""
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(csv_path)
    for klass in (gp.NoMetadataRedefiner, gp.WHOIRedefiner, gp.EcoTaxaRedefiner, gp.JediRedefiner):
        redefiner = klass(csv_taxonomies_path=str(csv_path))
        assert redefiner.attach_sidecars({"x": "y"}) is None
        assert not hasattr(redefiner, "_sidecars")
