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
