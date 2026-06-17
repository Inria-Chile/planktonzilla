"""Phase 5 golden-output baseline for ``planktonzilla.dataset_generation``.

These tests PIN the CURRENT behavior of the dataset_generation pure logic so the
Phase 6 refactors have a safety net. They are network-free: every NCBI / Wikidata
/ WHOI / EcoTaxa call is mocked. Where the current behavior is surprising it is
pinned anyway, with a ``# pins current behavior:`` comment.

This file is strictly additive (no source under ``planktonzilla/`` is modified).
"""

import io
import math
import os
import tarfile
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    Value,
)
from PIL import Image as PILImage

# ──────────────────────────────────────────────────────────────────────────────
# TEST-01 — import smoke test + shared constants
# ──────────────────────────────────────────────────────────────────────────────


def test_package_imports_and_all_exports():
    """`import planktonzilla.dataset_generation` works and exposes its __all__."""
    import planktonzilla.dataset_generation as dg

    # A couple of representative __all__ exports must exist as attributes.
    assert "export_to_tar_shards" in dg.__all__
    assert "load_unique_taxa" in dg.__all__
    assert "search_wikidata_taxon" in dg.__all__
    assert callable(dg.export_to_tar_shards)
    assert callable(dg.load_unique_taxa)
    assert callable(dg.search_wikidata_taxon)


def test_all_seven_modules_import():
    """All 7 dataset_generation modules import cleanly.

    This is the test that would have caught the original broken ``__init__``.
    """
    import importlib

    modules = [
        "constants",
        "extract_cox",
        "extract_taxon_ids",
        "gen_planktonzilla",
        "gen_planktonzilla_only_plankton",
        "save_planktonzilla_for_clip",
        "update_planktonzilla",
    ]
    for name in modules:
        mod = importlib.import_module(f"planktonzilla.dataset_generation.{name}")
        assert mod is not None


def test_constants_exact_values_and_types():
    """Pin the shared constants in constants.py."""
    from planktonzilla.dataset_generation import constants as c

    assert c.REPO_ID == "project-oceania/planktonzilla-17M"
    assert c.TAXONOMY_RANKS == ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
    assert isinstance(c.TAXONOMY_RANKS, tuple)
    assert len(c.TAXONOMY_RANKS) == 7
    assert c.EXTRA_COLS == ("proposed_label", "plankton", "root_class", "qualifier")
    assert c.ID_STR_COLS == ("wikidata_ID", "ecotaxa_ID")
    assert c.ID_NUM_COLS == ("aphia_ID", "NCBI_ID", "BOLD_ID")


def test_default_num_proc_matches_formula():
    """default_num_proc() == max(1, (os.cpu_count() or 1)//2)."""
    from planktonzilla.dataset_generation import constants as c

    expected = max(1, (os.cpu_count() or 1) // 2)
    assert c.default_num_proc() == expected
    assert c.default_num_proc() >= 1


# ──────────────────────────────────────────────────────────────────────────────
# TEST-02 — network-free pure logic
# ──────────────────────────────────────────────────────────────────────────────


def test_build_query_expand_to_children_true():
    """Pin the exact expanded (Organism:exp) Entrez query string."""
    from planktonzilla.dataset_generation import extract_cox as ec

    expected = (
        '(txid124140[Organism:exp]) AND ("COI"[All Fields] OR "CO1"[All Fields] OR '
        '"COX1"[All Fields] OR "COXI"[All Fields] OR '
        '"cytochrome c oxidase subunit I"[All Fields] OR '
        '"cytochrome c oxidase subunit 1"[All Fields] OR '
        '"cytochrome oxidase subunit I"[All Fields])'
    )
    assert ec.build_query(124140, expand_to_children=True) == expected
    # The txid + scope appear exactly as the code builds them.
    assert "txid124140[Organism:exp]" in ec.build_query(124140, expand_to_children=True)
    # Every COX term is present.
    for term in ec.COX_TERMS:
        assert f'"{term}"[All Fields]' in ec.build_query(124140, expand_to_children=True)


def test_build_query_expand_to_children_false():
    """Pin the exact non-expanded (Organism:noexp) Entrez query string."""
    from planktonzilla.dataset_generation import extract_cox as ec

    expected = (
        '(txid124140[Organism:noexp]) AND ("COI"[All Fields] OR "CO1"[All Fields] OR '
        '"COX1"[All Fields] OR "COXI"[All Fields] OR '
        '"cytochrome c oxidase subunit I"[All Fields] OR '
        '"cytochrome c oxidase subunit 1"[All Fields] OR '
        '"cytochrome oxidase subunit I"[All Fields])'
    )
    assert ec.build_query(124140, expand_to_children=False) == expected
    assert "txid124140[Organism:noexp]" in ec.build_query(124140, expand_to_children=False)


def test_extract_property_present_missing_malformed():
    """Pin _extract_property: present -> value, missing -> None, malformed -> None."""
    from planktonzilla.dataset_generation import extract_taxon_ids as eti

    claims = {"P685": [{"mainsnak": {"datavalue": {"value": "12345"}}}]}
    assert eti._extract_property(claims, "P685") == "12345"
    # Missing property -> None.
    assert eti._extract_property(claims, "P999") is None
    # Malformed claim ({prop: [{}]}) -> None (KeyError swallowed).
    assert eti._extract_property({"P850": [{}]}, "P850") is None


def test_load_unique_taxa_dedup_and_filter():
    """Pin load_unique_taxa: lowercases, dedups, drops all-empty rows."""
    from planktonzilla.dataset_generation import extract_taxon_ids as eti

    header = eti.SEP.join(eti.COLS)
    lines = [
        header,
        "Animalia,Arthropoda,,,,,",  # dup of the lowercase row below after to_lowercase
        "animalia,arthropoda,,,,,",
        "Animalia,Cnidaria,,,,,",
        ",,,,,,",  # all empty -> filtered out
    ]
    content = "\n".join(lines) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        df = eti.load_unique_taxa(path, limit=None)
    finally:
        os.unlink(path)

    # pins current behavior: lowercasing collapses Animalia/animalia + Arthropoda/arthropoda
    # into one row, and the all-empty row is dropped -> 2 unique taxa.
    assert df.height == 2
    assert df.columns == eti.COLS
    phyla = sorted(df["Phylum"].to_list())
    assert phyla == ["arthropoda", "cnidaria"]
    assert all(k == "animalia" for k in df["Kingdom"].to_list())


def test_load_unique_taxa_limit_applied_before_dedup():
    """Pin load_unique_taxa: limit slices the raw rows (df[:limit]) before dedup."""
    from planktonzilla.dataset_generation import extract_taxon_ids as eti

    header = eti.SEP.join(eti.COLS)
    lines = [header, "A,P1,,,,,", "B,P2,,,,,", "C,P3,,,,,"]
    content = "\n".join(lines) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        df = eti.load_unique_taxa(path, limit=2)
    finally:
        os.unlink(path)

    assert df.height == 2


def test_build_sync_dict_structure():
    """Pin build_sync_dict: (Dataset, Raw_Labels) keying + the value conversions."""
    from planktonzilla.dataset_generation import update_planktonzilla as up

    cols = ["Dataset", "Raw_Labels", *up.SYNC_COLS]
    row = {
        "Dataset": "ds1",
        "Raw_Labels": "copepod",
        "Kingdom": "Animalia",
        "Phylum": "Arthropoda",
        "Class": "",
        "Order": "",
        "Family": "",
        "Genus": "",
        "Species": "",
        "proposed_label": "Copepoda",
        "plankton": "True",
        "root_class": "zoo",
        "qualifier": "",
        "wikidata_ID": "Q3386609",
        "ecotaxa_ID": "274;1231",
        "aphia_ID": "135336.0",
        "NCBI_ID": "6854.0",
        "BOLD_ID": "",
    }
    header = ",".join(cols)
    data_line = ",".join(str(row[c]) for c in cols)
    content = header + "\n" + data_line + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        sync = up.build_sync_dict(path)
    finally:
        os.unlink(path)

    # Keyed by (Dataset, Raw_Labels).
    assert list(sync.keys()) == [("ds1", "copepod")]
    entry = sync[("ds1", "copepod")]

    # NUMERIC_ID_COLS: float "135336.0"/"6854.0" -> string without decimals.
    # pins current behavior: stored as str, not int, because the column is saved as string.
    assert entry["aphia_ID"] == "135336"
    assert entry["NCBI_ID"] == "6854"

    # String IDs kept as-is (semicolon-joined strings stay intact).
    assert entry["wikidata_ID"] == "Q3386609"
    assert entry["ecotaxa_ID"] == "274;1231"

    # to_null: blank strings + NaN become None (null).
    assert entry["Class"] is None
    assert entry["qualifier"] is None
    assert entry["BOLD_ID"] is None

    # Non-empty taxonomy text preserved.
    assert entry["Kingdom"] == "Animalia"
    assert entry["Phylum"] == "Arthropoda"
    assert entry["proposed_label"] == "Copepoda"

    # pins current behavior: pandas parses the "True" column as numpy bool; to_null
    # leaves the plankton boolean untouched.
    assert entry["plankton"] == True  # noqa: E712


def test_stratified_split_determinism_and_sizes():
    """Pin stratified split: deterministic for a fixed seed, fixed split sizes."""
    from planktonzilla.dataset_generation import gen_planktonzilla_only_plankton as gp

    # dataset d1: class 0 x 10 (splittable), class 1 x 2 (< MIN_CLASS_FREQ -> all to train).
    n0, n1 = 10, 2
    labels = [0] * n0 + [1] * n1
    feats = Features({"label": ClassLabel(names=["classA", "classB"]), "dataset": Value("string")})
    ds = Dataset.from_dict({"label": labels, "dataset": ["d1"] * (n0 + n1)}, features=feats)

    tr1, va1, te1 = gp.stratified_split_by_dataset(ds, num_proc=1, seed=42)
    tr2, va2, te2 = gp.stratified_split_by_dataset(ds, num_proc=1, seed=42)

    # Same seed -> identical split sizes and identical label multisets across calls.
    assert (len(tr1), len(va1), len(te1)) == (len(tr2), len(va2), len(te2))
    assert sorted(tr1["label"]) == sorted(tr2["label"])
    assert sorted(va1["label"]) == sorted(va2["label"])
    assert sorted(te1["label"]) == sorted(te2["label"])

    # Fixed sizes for this fixed input:
    #   remaining (class 0) n=10 -> first cut test_size=int(10*0.4)=4 -> train=6, val_test=4
    #   second cut test_size=int(10*0.2)=2 -> test=2, val=2
    #   minority (class 1, count 2) all 2 -> train -> train=8
    assert len(tr1) == 8
    assert len(va1) == 2
    assert len(te1) == 2

    # pins current behavior: the minority class (< MIN_CLASS_FREQ) is sent whole to train,
    # so it never appears in val/test.
    assert gp.MIN_CLASS_FREQ == 5
    assert set(va1["label"]) == {0}
    assert set(te1["label"]) == {0}
    assert sorted(tr1["label"]) == [0, 0, 0, 0, 0, 0, 1, 1]


def test_stratified_split_minority_only_goes_all_to_train():
    """Pin: a dataset whose every class is < MIN_CLASS_FREQ goes entirely to train."""
    from planktonzilla.dataset_generation import gen_planktonzilla_only_plankton as gp

    # Every class has 2 examples (< MIN_CLASS_FREQ=5) -> nothing left to split -> all train.
    labels = [0, 0, 1, 1]
    feats = Features({"label": ClassLabel(names=["a", "b"]), "dataset": Value("string")})
    ds = Dataset.from_dict({"label": labels, "dataset": ["d1"] * 4}, features=feats)

    train, val, test = gp.stratified_split_by_dataset(ds, num_proc=1, seed=42)
    assert len(train) == 4
    # pins current behavior: with no remaining (non-minority) rows, val/test concatenate
    # empty lists -> None.
    assert val is None
    assert test is None


def test_build_only_plankton_filter_and_label_encoding():
    """Pin build_only_plankton: plankton+Kingdom filter, tax_label join, label encoding.

    The Image-feature cast is exercised end-to-end with a synthetic RGB dataset
    (the cast is practical here, so no part is skipped).
    """
    from planktonzilla.dataset_generation import gen_planktonzilla_only_plankton as gp

    imgs = [PILImage.new("RGB", (4, 4), (i * 40, 0, 0)) for i in range(6)]
    data = {
        "image": imgs,
        "plankton": [True, True, True, False, True, True],
        "Kingdom": ["Animalia", "Animalia", "Animalia", "Animalia", "", "Animalia"],
        "Phylum": ["Arthropoda", "Arthropoda", "Cnidaria", "X", "Y", "Arthropoda"],
        "Class": [""] * 6,
        "Order": [""] * 6,
        "Family": [""] * 6,
        "Genus": [""] * 6,
        "Species": [""] * 6,
        "dataset": ["d1", "d1", "d2", "d3", "d4", "d1"],
    }
    ds = Dataset.from_dict(data).cast_column("image", Image())

    out = gp.build_only_plankton(ds, num_proc=1)

    # Filter: plankton is True AND Kingdom != "". Row 3 (plankton False) and row 4
    # (empty Kingdom) are dropped -> 4 rows survive.
    assert len(out) == 4

    # Only the training columns remain, in cast order.
    assert out.column_names == ["image", "label", "dataset"]

    # tax_label is the space-joined non-empty ranks; encoded as a sorted ClassLabel.
    assert out.features["label"].names == ["Animalia Arthropoda", "Animalia Cnidaria"]
    # Rows 0,1,5 -> "Animalia Arthropoda" (class 0); row 2 -> "Animalia Cnidaria" (class 1).
    assert out["label"] == [0, 0, 1, 0]
    assert out["dataset"] == ["d1", "d1", "d2", "d1"]

    # Image survived the cast and decodes to a PIL image.
    assert isinstance(out[0]["image"], PILImage.Image)


def test_export_to_tar_shards_members_and_sharding():
    """Pin export_to_tar_shards: member names, JPEG/RGB image, .txt content, sharding."""
    from planktonzilla.dataset_generation import save_planktonzilla_for_clip as sp

    imgs = [PILImage.new("RGB", (4, 4), (i * 50, 10, 20)) for i in range(3)]
    feats = Features({"image": Image(), "label": ClassLabel(names=["copepoda", "diatom"])})
    ds = Dataset.from_dict({"image": imgs, "label": [0, 1, 0]}, features=feats)
    dd = DatasetDict({"train": ds})

    with tempfile.TemporaryDirectory() as out_dir:
        sp.export_to_tar_shards(dd, output_dir=out_dir, shard_size=2)

        train_dir = os.path.join(out_dir, "train")
        shards = sorted(os.listdir(train_dir))

        # shard_size=2 over 3 examples -> 2 shards (zero-padded 5-digit names).
        assert shards == ["shard_00000.tar", "shard_00001.tar"]

        with tarfile.open(os.path.join(train_dir, "shard_00000.tar")) as tar:
            names = tar.getnames()
            # pins current behavior: the label member is named image_{i}.txt (NOT text_{i}),
            # intentionally sharing the image basename so WebDataset groups them.
            assert names == ["image_0.jpg", "image_0.txt", "image_1.jpg", "image_1.txt"]

            # .txt content is the class name.
            assert tar.extractfile("image_0.txt").read().decode("utf-8") == "copepoda"
            assert tar.extractfile("image_1.txt").read().decode("utf-8") == "diatom"

            # Image is re-encoded as RGB JPEG.
            img0 = PILImage.open(io.BytesIO(tar.extractfile("image_0.jpg").read()))
            assert img0.format == "JPEG"
            assert img0.mode == "RGB"

        with tarfile.open(os.path.join(train_dir, "shard_00001.tar")) as tar:
            # pins current behavior: members use the shard-relative index, so the 3rd
            # example (absolute index 2) becomes image_0 in the second shard, not image_2.
            assert tar.getnames() == ["image_0.jpg", "image_0.txt"]
            assert tar.extractfile("image_0.txt").read().decode("utf-8") == "copepoda"


# ──────────────────────────────────────────────────────────────────────────────
# TEST-03 — fetchers via mocks (no live calls)
# ──────────────────────────────────────────────────────────────────────────────


def _whoi_response(ok=True, json_data=None, text=""):
    r = MagicMock()
    r.ok = ok
    r.json.return_value = json_data
    r.text = text
    return r


def test_retrieve_whoi_metadata_happy_path():
    """Pin retrieve_whoi_metadata: parsed JSON + .hdr Temp/Humidity."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    api_json = {"lat": 41.5, "lng": -70.6, "depth": 4.0, "timestamp_iso": "2018-05-03T12:34:56Z"}
    # Header line "Temp Humidity" -> 2 columns; values "22.5,55.0" -> 2 comma-split values.
    hdr_text = 'junk\n"Temp Humidity"\n22.5,55.0\n'

    session = MagicMock()

    def side(url, timeout=10):
        if url.endswith(".hdr"):
            return _whoi_response(ok=True, text=hdr_text)
        return _whoi_response(ok=True, json_data=api_json)

    session.get.side_effect = side

    info = gp.retrieve_whoi_metadata("B123", session=session)

    assert info["Latitude"] == 41.5
    assert info["Longitude"] == -70.6
    assert info["Depth"] == 4.0
    # Timestamp keeps only the date part.
    assert info["Timestamp"] == "2018-05-03"
    # Temp/Humidity come from the .hdr parser, cast to float.
    assert info["Temperature"] == 22.5
    assert info["Humidity"] == 55.0
    assert info["BinID"] == "B123"
    # All float fields are real floats.
    for k in ("Latitude", "Longitude", "Depth", "Temperature", "Humidity"):
        assert isinstance(info[k], float)


def test_retrieve_whoi_metadata_failure_returns_all_nan():
    """Pin retrieve_whoi_metadata failure path: session.get raises -> all-NaN/None info."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    session = MagicMock()
    session.get.side_effect = RuntimeError("boom")

    info = gp.retrieve_whoi_metadata("Bxx", session=session)

    for k in ("Latitude", "Longitude", "Depth", "Temperature", "Humidity"):
        assert isinstance(info[k], float) and math.isnan(info[k])
    assert info["Timestamp"] is None
    # pins current behavior: the additive warning does not change the returned dict;
    # BinID is still set from the argument.
    assert info["BinID"] == "Bxx"


def test_retrieve_ecotaxa_metadata_happy_path():
    """Pin retrieve_ecotaxa_metadata: parsed depth/lat/lon/objdate."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    data = {
        "depth_max": 50.0,
        "depth_min": 5.0,
        "latitude": -33.0,
        "longitude": -71.5,
        "objdate": "2019-07-01",
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    session = MagicMock()
    session.get.return_value = resp

    info = gp.retrieve_ecotaxa_metadata(987, session=session)

    assert info["Depth_max"] == 50.0
    assert info["Depth_min"] == 5.0
    assert info["Latitude"] == -33.0
    assert info["Longitude"] == -71.5
    assert info["Timestamp"] == "2019-07-01"
    assert info["ObjID"] == "987"


def test_retrieve_ecotaxa_metadata_non_200_returns_nan():
    """Pin retrieve_ecotaxa_metadata: non-200 -> NaN-filled dict (early return)."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    resp = MagicMock()
    resp.status_code = 404
    session = MagicMock()
    session.get.return_value = resp

    info = gp.retrieve_ecotaxa_metadata(111, session=session)

    for k in ("Depth_max", "Depth_min", "Latitude", "Longitude"):
        assert isinstance(info[k], float) and math.isnan(info[k])
    assert info["Timestamp"] is None
    assert info["ObjID"] == "111"


@pytest.fixture(autouse=True)
def _clear_wikidata_cache():
    """Reset the module-level search cache between tests (TEST-03 fetchers)."""
    from planktonzilla.dataset_generation import extract_taxon_ids as eti

    eti._SEARCH_CACHE.clear()
    yield
    eti._SEARCH_CACHE.clear()


def test_search_wikidata_taxon_biological_hit_and_caching():
    """Pin search_wikidata_taxon: biological hit -> dict; second call uses the cache."""
    from planktonzilla.dataset_generation import extract_taxon_ids as eti

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"search": [{"id": "Q123", "label": "Copepoda", "description": "taxon of crustaceans"}]}
    mock_get = MagicMock(return_value=resp)
    with patch.object(eti.session, "get", mock_get):
        result = eti.search_wikidata_taxon("Copepoda")

        assert result == {
            "qid": "Q123",
            "label": "Copepoda",
            "description": "taxon of crustaceans",
            "url": "https://www.wikidata.org/wiki/Q123",
        }
        assert mock_get.call_count == 1

        # Second call is served from _SEARCH_CACHE: no extra request.
        cached = eti.search_wikidata_taxon("Copepoda")
        assert cached == result
        assert mock_get.call_count == 1


def test_search_wikidata_taxon_non_biological_returns_none():
    """Pin search_wikidata_taxon: a non-biological description -> None."""
    from planktonzilla.dataset_generation import extract_taxon_ids as eti

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"search": [{"id": "Q999", "label": "X", "description": "a music album"}]}
    with patch.object(eti.session, "get", MagicMock(return_value=resp)):
        assert eti.search_wikidata_taxon("AlbumThing") is None


def test_search_wikidata_taxon_429_then_200_retry():
    """Pin search_wikidata_taxon: HTTP 429 -> sleep -> retry -> 200 hit."""
    from planktonzilla.dataset_generation import extract_taxon_ids as eti

    r429 = MagicMock()
    r429.status_code = 429
    r200 = MagicMock()
    r200.status_code = 200
    r200.json.return_value = {"search": [{"id": "Q5", "label": "Fishy", "description": "species of fish"}]}

    mock_get = MagicMock(side_effect=[r429, r200])
    # Patch time.sleep so the 2s retry backoff does not slow the test.
    with patch.object(eti.session, "get", mock_get), patch.object(eti.time, "sleep"):
        result = eti.search_wikidata_taxon("Fishy")

    assert result == {
        "qid": "Q5",
        "label": "Fishy",
        "description": "species of fish",
        "url": "https://www.wikidata.org/wiki/Q5",
    }
    # pins current behavior: the 429 branch recurses, so two requests are made total.
    assert mock_get.call_count == 2


# ──────────────────────────────────────────────────────────────────────────────
# TEST-04 — REFACTOR-04 characterization tests (Phase 6 Wave B)
#
# These PIN the current output of gen_planktonzilla.RedefineDataset.redefine(),
# the _add_metadata serialization for all four redefiners, the datasets_configs
# override table in main(), and the extract_cox single-ID path. They are written
# against the CURRENT (un-refactored) code and must stay green through the
# god-function decomposition. All are network-free / mocked.
# ──────────────────────────────────────────────────────────────────────────────

import json as _json

import orjson as _orjson


def _write_taxonomy_csv(path, dataset_name, raw_label):
    """Write a tiny taxonomy CSV that _build_lookup can resolve.

    Columns mirror the real planktonzilla_taxonomy.csv: Dataset, Raw_Labels plus
    the lookup columns (taxonomy ranks, extras, str IDs and numeric IDs).
    """
    header = (
        "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
        "proposed_label,plankton,root_class,qualifier,"
        "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
    )
    # aphia_ID / NCBI_ID / BOLD_ID are numeric in the CSV (float) -> text w/o decimals.
    row = f"{dataset_name},{raw_label},Animalia,Arthropoda,,,,,,Copepoda,True,zoo,,Q3386609,274;1231,135336.0,6854.0,"
    with open(path, "w") as f:
        f.write(header + "\n" + row + "\n")


def _make_split(tmpdir, dataset_name, raw_label, png_subpath):
    """Build a 1-row datasets.Dataset with a ClassLabel `label` + Image pointing at a real PNG.

    png_subpath has >= 3 path components so chunks[-3:] vs chunks[-2:] is exercised.
    """
    png_path = os.path.join(tmpdir, png_subpath)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    PILImage.new("RGB", (4, 4), (10, 20, 30)).save(png_path)

    feats = Features({"label": ClassLabel(names=[raw_label]), "image": Image()})
    ds = Dataset.from_dict({"label": [0], "image": [png_path]}, features=feats)
    return ds


def test_redefine_nometadata_two_splits_pins_output():
    """Pin NoMetadataRedefiner.redefine() output for a 2-split DatasetDict.

    Pins: column set, dataset/original_label/original_path values (original_path
    uses the last-3 path components when n_splits>=2), the taxonomy join values,
    and that metadata is flattened away.
    """
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    dataset_name = "isiisnet"
    raw_label = "copepod"

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "taxo.csv")
        _write_taxonomy_csv(csv_path, dataset_name, raw_label)

        # >= 3 path components so last-3 vs last-2 slicing is observable.
        train = _make_split(tmp, dataset_name, raw_label, "imgs/train/copepod/a.png")
        val = _make_split(tmp, dataset_name, raw_label, "imgs/val/copepod/b.png")
        dd = DatasetDict({"train": train, "validation": val})

        redefiner = gp.NoMetadataRedefiner(csv_taxonomies_path=csv_path)
        out = redefiner.redefine(hf_dataset=dd, dataset_name=dataset_name, num_proc=1)

    # Two splits concatenated -> 2 rows.
    assert len(out) == 2

    cols = set(out.column_names)
    # The structural / join columns the pipeline always produces.
    for c in [
        "image",
        "dataset",
        "original_label",
        "original_path",
        "Kingdom",
        "Phylum",
        "proposed_label",
        "plankton",
        "root_class",
        "qualifier",
        "wikidata_ID",
        "ecotaxa_ID",
        "aphia_ID",
        "NCBI_ID",
        "ObjID",
        "timestamp",
        "Latitude",
        "Longitude",
        "Depth_max",
        "Depth_min",
    ]:
        assert c in cols, f"missing column {c}"
    # metadata is flattened away; the raw label column is dropped.
    assert "metadata" not in cols
    assert "label" not in cols

    rows = out.to_list()
    for r in rows:
        assert r["dataset"] == dataset_name
        assert r["original_label"] == raw_label
        # Taxonomy join values.
        assert r["Kingdom"] == "Animalia"
        assert r["Phylum"] == "Arthropoda"
        assert r["proposed_label"] == "Copepoda"
        assert r["root_class"] == "zoo"
        assert r["wikidata_ID"] == "Q3386609"
        assert r["ecotaxa_ID"] == "274;1231"
        # Numeric IDs come back as strings without decimals.
        assert r["aphia_ID"] == "135336"
        assert r["NCBI_ID"] == "6854"
        # No metadata -> all metadata-derived columns are null.
        assert r["ObjID"] is None
        assert r["timestamp"] is None
        assert r["Latitude"] is None

    # pins current behavior: with n_splits>=2 original_path keeps the LAST 3 components
    # (leading slash + 3 components = "/<split>/<class>/<file>").
    paths = sorted(r["original_path"] for r in rows)
    assert paths == ["/train/copepod/a.png", "/val/copepod/b.png"]


def test_redefine_nometadata_one_split_pins_short_path():
    """Pin NoMetadataRedefiner.redefine() for a 1-split DatasetDict.

    pins current behavior: with n_splits==1 original_path keeps the LAST 2 path
    components ("/<class>/<file>"), not 3.
    """
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    dataset_name = "lensless"
    raw_label = "diatom"

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "taxo.csv")
        _write_taxonomy_csv(csv_path, dataset_name, raw_label)

        train = _make_split(tmp, dataset_name, raw_label, "imgs/train/diatom/c.png")
        dd = DatasetDict({"train": train})

        redefiner = gp.NoMetadataRedefiner(csv_taxonomies_path=csv_path)
        out = redefiner.redefine(hf_dataset=dd, dataset_name=dataset_name, num_proc=1)

    assert len(out) == 1
    row = out.to_list()[0]
    # Last 2 components only (no split component).
    assert row["original_path"] == "/diatom/c.png"
    assert row["dataset"] == dataset_name
    assert row["original_label"] == raw_label


def test_redefine_unknown_label_uses_null_default():
    """Pin the taxonomy-miss default: lookup.get((name,label), {col: None}) -> nulls."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "taxo.csv")
        # CSV resolves a DIFFERENT (dataset, label) than what the split carries.
        _write_taxonomy_csv(csv_path, "other_ds", "other_label")

        train = _make_split(tmp, "lensless", "unknownlabel", "imgs/train/unknownlabel/x.png")
        dd = DatasetDict({"train": train})

        redefiner = gp.NoMetadataRedefiner(csv_taxonomies_path=csv_path)
        out = redefiner.redefine(hf_dataset=dd, dataset_name="lensless", num_proc=1)

    row = out.to_list()[0]
    # Unknown (dataset, label) -> every lookup column is None.
    for c in ["Kingdom", "Phylum", "proposed_label", "root_class", "wikidata_ID", "aphia_ID"]:
        assert row[c] is None
    # But the structural columns are still set.
    assert row["dataset"] == "lensless"
    assert row["original_label"] == "unknownlabel"


# ── _add_metadata serialization pins (the block to be extracted) ──────────────


def _redefiner_add_metadata_ds(redefiner_cls, csv_path, n_rows=2):
    """Build a tiny Dataset with an `original_path` column ready for `_add_metadata`.

    Returns (redefiner, ds) so the test can run _add_metadata and pin the column.
    """
    paths = [f"/train/cls/img_{i}.jpg" for i in range(n_rows)]
    ds = Dataset.from_dict({"original_path": paths})
    redefiner = redefiner_cls(csv_taxonomies_path=csv_path)
    return redefiner, ds


def test_add_metadata_nometadata_serializes_empty_json():
    """Pin NoMetadataRedefiner._add_metadata: metadata col is json.dumps({}) as Value(string)."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "taxo.csv")
        _write_taxonomy_csv(csv_path, "lensless", "diatom")
        redefiner, ds = _redefiner_add_metadata_ds(gp.NoMetadataRedefiner, csv_path)
        out = redefiner._add_metadata(ds)

    assert out.features["metadata"] == Value("string")
    vals = out["metadata"]
    assert all(v == _json.dumps({}) for v in vals)
    assert all(_orjson.loads(v) == {} for v in vals)


def test_add_metadata_jedi_serializes_fixed_dict():
    """Pin JediRedefiner._add_metadata: metadata col is json.dumps(fixed dict) as Value(string)."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "taxo.csv")
        _write_taxonomy_csv(csv_path, "jedi_oceans_cpics", "x")
        redefiner, ds = _redefiner_add_metadata_ds(gp.JediRedefiner, csv_path)
        out = redefiner._add_metadata(ds)

    assert out.features["metadata"] == Value("string")
    expected = _json.dumps(
        {
            "Latitude": "34.682718",
            "Longitude": "139.444779",
            "Depth_min": "20",
            "Depth_max": "20",
        }
    )
    vals = out["metadata"]
    assert all(v == expected for v in vals)


def test_add_metadata_ecotaxa_serializes_mocked_metadata():
    """Pin EcoTaxaRedefiner._add_metadata with retrieve_ecotaxa_metadata mocked (no network)."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    fake = {
        "Depth_max": 50.0,
        "Depth_min": 5.0,
        "Latitude": -33.0,
        "Longitude": -71.5,
        "Timestamp": "2019-07-01",
        "ObjID": "987",
    }

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "taxo.csv")
        _write_taxonomy_csv(csv_path, "flowcamnet", "cls")
        redefiner, ds = _redefiner_add_metadata_ds(gp.EcoTaxaRedefiner, csv_path)
        with patch.object(gp, "retrieve_ecotaxa_metadata", return_value=dict(fake)):
            out = redefiner._add_metadata(ds)

    assert out.features["metadata"] == Value("string")
    # normalize() stringifies every non-None value before json.dumps.
    expected = _json.dumps({str(k): str(v) for k, v in fake.items() if v is not None})
    vals = out["metadata"]
    assert all(v == expected for v in vals)


def test_add_metadata_whoi_serializes_mocked_metadata():
    """Pin WHOIRedefiner._add_metadata with retrieve_whoi_metadata mocked (no network)."""
    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    fake = {
        "Latitude": 41.5,
        "Longitude": -70.6,
        "Depth": 4.0,
        "Temperature": 22.5,
        "Humidity": 55.0,
        "Timestamp": "2018-05-03",
        "BinID": "B123",
    }

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "taxo.csv")
        _write_taxonomy_csv(csv_path, "whoi", "cls")
        # original_path of the form <bin>_<suffix>.jpg so extract_bin_id yields a bin id.
        paths = ["/train/cls/B123_00001.jpg", "/train/cls/B123_00002.jpg"]
        ds = Dataset.from_dict({"original_path": paths})
        redefiner = gp.WHOIRedefiner(csv_taxonomies_path=csv_path)
        with patch.object(gp, "retrieve_whoi_metadata", return_value=dict(fake)):
            out = redefiner._add_metadata(ds)

    assert out.features["metadata"] == Value("string")
    assert "bin_id" not in out.column_names
    expected = _json.dumps({str(k): str(v) for k, v in fake.items() if v is not None})
    vals = out["metadata"]
    assert all(v == expected for v in vals)


# ── datasets_configs override table pin ──────────────────────────────────────


def _expected_overrides(data_root, dataset_import, cleanup):
    return [
        f"dataset_import={dataset_import}",
        f"dataset_import.cleanup_after_processing={cleanup}",
        "dataset_import.push_to_hub=False",
        f"dataset_import.data_dir={data_root}",
    ]


def test_datasets_configs_structure_pins(monkeypatch, tmp_path):
    """Pin the datasets_configs override blocks built inside gen_planktonzilla.main().

    Drives main() with a real tiny CSV and mocks the whole per-dataset loop body
    (hydra.compose / instantiate / load_dataset / redefiner.redefine) so the loop
    runs to completion. Captures, in order, the exact `overrides` list passed to
    hydra.compose and the redefiner type bound to each dataset_name, then asserts
    they match the expected (name, import_name, cleanup, redefiner) table EXACTLY.
    This pins the literal the decomposition must reproduce byte for byte.
    """
    import sys

    from planktonzilla.dataset_generation import gen_planktonzilla as gp

    # Real CSV so the redefiner constructors (run while building datasets_configs)
    # succeed; --taxo-csv routes every redefiner here.
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "x", "y")

    # hydra.initialize is a context manager; keep a no-op CM so the `with` works.
    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(gp.hydra, "initialize", lambda *a, **k: _NullCM())

    captured_overrides = []  # one entry per dataset, in iteration order.

    def _fake_compose(*args, **kwargs):
        captured_overrides.append(list(kwargs["overrides"]))
        return MagicMock()

    monkeypatch.setattr(gp.hydra, "compose", _fake_compose)

    # Make the loop body a no-op: a mock importer whose imagefolder is "present"
    # and empty so no real import / load happens, and a redefine() that returns a
    # trivial dataset so concatenate/clean/save still work.
    importer = MagicMock()
    importer.imagefolder_dir = tmp_path  # a Path; exists, force has_content below
    monkeypatch.setattr(gp.hydra.utils, "instantiate", lambda *a, **k: importer)
    monkeypatch.setattr(gp.os, "listdir", lambda p: ["dummy_category"])
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: MagicMock())

    # Record (dataset_name -> redefiner type) and short-circuit the heavy work.
    captured_redefiners = {}
    tiny = Dataset.from_dict({"x": [1]})

    def _fake_redefine(self, hf_dataset, dataset_name, num_proc):
        captured_redefiners[dataset_name] = type(self)
        return tiny

    monkeypatch.setattr(gp.RedefineDataset, "redefine", _fake_redefine)
    # Skip corrupt-cleaning and disk save (both would choke on the tiny dataset).
    monkeypatch.setattr(gp, "clean_corrupt_examples_optimized", lambda ds, **k: ds)
    monkeypatch.setattr(gp.Dataset, "save_to_disk", lambda self, path: None)

    monkeypatch.setattr(sys, "argv", ["gen_planktonzilla.py", "--taxo-csv", str(csv_path)])

    gp.main()

    data_root = (gp.root / "data").resolve()

    # (dataset_name, import_name, cleanup, redefiner_type) — the table the literal
    # encodes. Order matters: datasets_configs is iterated in insertion order.
    expected_table = [
        ("isiisnet", "isiisnet", "True", gp.NoMetadataRedefiner),
        ("whoi", "whoi-plankton", "True", gp.WHOIRedefiner),
        ("flowcamnet", "flowcamnet", "True", gp.EcoTaxaRedefiner),
        ("lensless", "lensless", "True", gp.NoMetadataRedefiner),
        ("medplanktonset", "medplanktonset", "True", gp.NoMetadataRedefiner),
        ("uvp6net", "uvp6net", "True", gp.EcoTaxaRedefiner),
        ("zoocamnet", "zoocamnet", "True", gp.NoMetadataRedefiner),
        ("zooscan", "zooscannet", "True", gp.EcoTaxaRedefiner),
        ("planktonset1.0", "planktonset1", "False", gp.NoMetadataRedefiner),
        ("syke_ifcb_2022", "syke_ifcb_2022", "False", gp.NoMetadataRedefiner),
        ("planktoscope", "planktoscope", "False", gp.EcoTaxaRedefiner),
        ("global_uvp5", "global_uvp5net", "False", gp.EcoTaxaRedefiner),
    ]

    # Exactly these 12 active datasets, in this order (commented ones excluded).
    assert list(captured_redefiners.keys()) == [t[0] for t in expected_table]
    assert len(captured_overrides) == len(expected_table)

    for (name, import_name, cleanup, rtype), overrides in zip(expected_table, captured_overrides):
        # Override block reproduced byte for byte.
        assert overrides == _expected_overrides(data_root, import_name, cleanup)
        # Redefiner type bound to this dataset is exactly the expected class.
        assert captured_redefiners[name] is rtype


# ── extract_cox single-ID path pin ───────────────────────────────────────────


def test_extract_cox_single_id_path(monkeypatch, tmp_path):
    """Pin extract_cox.main() single --ncbi_id branch with get_cox_sequences/save_fasta mocked."""
    import sys

    from planktonzilla.dataset_generation import extract_cox as ec

    out_dir = tmp_path / "single_out"

    fake_records = [MagicMock(), MagicMock()]  # 2 fake SeqRecords -> records truthy
    get_seqs = MagicMock(return_value=fake_records)
    save = MagicMock()
    cfg = MagicMock()

    monkeypatch.setattr(ec, "get_cox_sequences", get_seqs)
    monkeypatch.setattr(ec, "save_fasta", save)
    monkeypatch.setattr(ec, "configure_entrez", cfg)

    argv = [
        "extract_cox.py",
        "--ncbi_id",
        "124140",
        "--out_dir_s",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    ec.main()

    # Entrez configured exactly once with the resolved email.
    cfg.assert_called_once()
    # get_cox_sequences called with the id, expand_to_children True (no --noexp),
    # max_results default MAX_SEQS_PER_SPECIES.
    get_seqs.assert_called_once_with(
        "124140",
        expand_to_children=True,
        max_results=ec.MAX_SEQS_PER_SPECIES,
    )
    # Output dir created and save_fasta called with the <ncbi_id>.fasta path.
    assert out_dir.exists()
    save.assert_called_once()
    saved_records, saved_path = save.call_args.args
    assert saved_records is fake_records
    assert str(saved_path) == str(out_dir / "124140.fasta")


def test_extract_cox_single_id_no_records_skips_save(monkeypatch, tmp_path):
    """Pin: single-ID path with empty records does NOT call save_fasta but still mkdirs."""
    import sys

    from planktonzilla.dataset_generation import extract_cox as ec

    out_dir = tmp_path / "single_out_empty"

    monkeypatch.setattr(ec, "get_cox_sequences", MagicMock(return_value=[]))
    save = MagicMock()
    monkeypatch.setattr(ec, "save_fasta", save)
    monkeypatch.setattr(ec, "configure_entrez", MagicMock())

    argv = ["extract_cox.py", "--ncbi_id", "999", "--out_dir_s", str(out_dir)]
    monkeypatch.setattr(sys, "argv", argv)

    ec.main()

    assert out_dir.exists()
    save.assert_not_called()
