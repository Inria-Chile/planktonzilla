"""
(c) Inria

Network-free regression test for the taxonomy-map sparse-null cast crash.

Drives the REAL ``RedefineDataset.redefine`` map path (the ``ds.map`` that assigns
taxonomy/ID columns) over a synthetic in-memory dataset engineered to reproduce the
build-time failure ``TypeError: Couldn't cast array of type string to null``.

Mechanism being pinned: without an explicit ``features=`` schema, ``datasets`` lets
the per-shard ``ArrowWriter`` INFER column types from the first write batch. A
SPARSE-NULL source (like FREPJ: many blank external IDs, null Species/Genus) can
hand a shard a first ``writer_batch_size`` (1000) batch whose taxonomy/ID column is
entirely ``None`` -> inferred as pyarrow ``null``; a later string batch in that same
shard then fails to cast, surfacing through ``iflatmap_unordered`` under
``num_proc>1``. We stage exactly that: 2100 rows with ``num_proc=2`` makes the
contiguous rank-0 shard span rows 0..1049, whose FIRST batch (rows 0..999) is
all-null and SECOND batch (rows 1000..1049) is string.

Without the fix this test ERRORS at ``redefine`` (the crash). With the explicit
``features=`` schema the map never infers ``null``, so it succeeds and the resulting
schema/values are correct. Offline by construction (no imagefolder loader, no Hub);
we additionally monkeypatch ``requests`` to PROVE the redefine path never hits the
network.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import datasets
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp

# writer_batch_size default is 1000; with num_proc=2 the rank-0 shard must exceed it
# so its first batch is all-null and a later batch is string (the crash trigger).
_N_NULL = 1000  # rows for the class ABSENT from the CSV -> all-None taxonomy (first shard/batch)
_N_STR = 1100  # rows for the class PRESENT in the CSV -> string taxonomy (later batch)

_UNKNOWN = "unknownclass"  # label 0 -> not in the CSV -> _taxonomy_row all-None default
_KNOWN = "knownclass"  # label 1 -> full CSV taxonomy row (strings)

_CSV_HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,root_class,qualifier,"
    "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
)
# aphia_ID/NCBI_ID/BOLD_ID are numeric in the CSV (float) -> decimal-free strings downstream.
_CSV_ROW = f"src,{_KNOWN},Animalia,Arthropoda,Hexanauplia,Calanoida,,Calanus,,Copepoda,True,zoo,,Q3386609,274,135336.0,6854.0,"


def test_sparse_null_taxonomy_map_does_not_crash_under_multiprocessing(monkeypatch, tmp_path):
    """The taxonomy ds.map survives an all-null-first-batch shard and pins the output."""
    # Offline guards: the redefine path here touches no network, and these PROVE it.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True, raising=False)

    def _no_network(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(gp.requests, "get", _no_network)
    monkeypatch.setattr(gp.requests.Session, "get", _no_network)

    # Determinism for the module-global num_proc read by _serialize/_flatten metadata maps
    # (the taxonomy map itself uses the redefine() num_proc arg below).
    monkeypatch.setattr(gp, "num_proc", 1)

    # One tiny valid PNG referenced by every row keeps the fixture cheap while still giving
    # _taxonomy_row a real example["image"]["path"] to read.
    png = tmp_path / "one.png"
    PILImage.new("RGB", (4, 4), color=(1, 2, 3)).save(png)

    # unknown FIRST so rows 0..999 (all-None taxonomy) form the start of the rank-0 shard.
    class_names = [_UNKNOWN, _KNOWN]
    paths = [str(png)] * (_N_NULL + _N_STR)
    labels = [0] * _N_NULL + [1] * _N_STR
    features = Features({"image": Image(), "label": ClassLabel(names=class_names)})
    hf = DatasetDict({"train": Dataset.from_dict({"image": paths, "label": labels}, features=features)})

    csv_path = tmp_path / "taxo.csv"
    csv_path.write_text(_CSV_HEADER + "\n" + _CSV_ROW + "\n")

    redefiner = gp.NoMetadataRedefiner(csv_taxonomies_path=str(csv_path))

    # Without the fix this raises TypeError: Couldn't cast array of type string to null.
    ds = redefiner.redefine(hf_dataset=hf, dataset_name="src", num_proc=2)

    assert len(ds) == _N_NULL + _N_STR

    # Schema matches what _cast_scalar_types produces: taxonomy/ID cols are string,
    # plankton is bool (string at the map stage, cast to bool downstream).
    assert ds.features["Kingdom"] == Value("string")
    assert ds.features["Genus"] == Value("string")
    assert ds.features["NCBI_ID"] == Value("string")
    assert ds.features["plankton"] == Value("bool")

    # Column access avoids decoding the image column.
    original_label = ds["original_label"]
    kingdom = ds["Kingdom"]
    genus = ds["Genus"]
    ncbi = ds["NCBI_ID"]
    proposed = ds["proposed_label"]
    plankton = ds["plankton"]

    assert all(name == "src" for name in ds["dataset"])
    assert original_label.count(_UNKNOWN) == _N_NULL
    assert original_label.count(_KNOWN) == _N_STR

    for i, name in enumerate(original_label):
        if name == _UNKNOWN:
            # Class absent from the CSV -> every lookup column resolves to None.
            assert kingdom[i] is None
            assert genus[i] is None
            assert ncbi[i] is None
            assert proposed[i] is None
            assert plankton[i] is None
        else:
            # Class present in the CSV -> taxonomy/IDs/extras come through unchanged.
            assert kingdom[i] == "Animalia"
            assert genus[i] == "Calanus"
            assert ncbi[i] == "6854"  # 6854.0 -> decimal-free string
            assert proposed[i] == "Copepoda"
            assert plankton[i] is True
