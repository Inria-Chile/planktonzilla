"""
(c) Inria

Network-free single-source (lensless) end-to-end test for
``planktonzilla.planktonzilla_dataset.generate_planktonzilla``.

This drives the REAL imagefolder import + redefine path against ONLY the
``lensless`` source dataset and asserts on the reloaded consolidated dataset. It
is a fast, deterministic, OFFLINE regression guard complementing the structural /
override-pin tests in ``tests/test_gen_planktonzilla_hydra.py``.

Zero behavioral drift: this PINS current behavior, it never "fixes" it. The whole
flow is offline by construction — lensless uses ``redefiner: none``
(``NoMetadataRedefiner``), which never touches the network, and the imagefolder is
pre-created so ``dataset_importer.import_dataset()`` is never invoked. We
additionally monkeypatch ``requests.get`` / ``requests.Session.get`` to PROVE it.
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
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image as PILImage

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset import generate_planktonzilla as gp


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


def _write_png(path, size=8):
    """Produce a small valid RGB PNG so the imagefolder loader reads real images."""
    PILImage.new("RGB", (size, size), color=(120, 160, 200)).save(path)


def test_lensless_only_e2e_generation_is_offline_and_pins_behavior(monkeypatch, tmp_path):
    """Run the REAL lensless-only gen pipeline offline and pin the output rows."""
    # (finding 1+9) Offline by construction; these guards PROVE it. HF offline env
    # plus monkeypatched requests that raise if the redefine path ever hits network.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    # The env vars above are read by datasets/huggingface_hub at IMPORT time, which
    # already happened, so flip the captured offline flags directly. This stops the
    # imagefolder loader from resolving its builder via a Hub HEAD (it uses the local
    # packaged builder instead) — making the run genuinely network-free, not just
    # network-with-fallback. monkeypatch.setattr auto-reverts after the test.
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True, raising=False)

    def _no_network(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(gp.requests, "get", _no_network)
    monkeypatch.setattr(gp.requests.Session, "get", _no_network)

    # (finding 7) Determinism: the module-level global read by _serialize/_flatten.
    gp.num_proc = 1

    # (finding 2+3) Pre-create the imagefolder so import_dataset() is NEVER called.
    # Name matches DatasetImporter.__post_init__: data_dir / "<classname>_imagefolder".
    imagefolder = tmp_path / "lenslessdatasetimporter_imagefolder"
    # Lookup key is ("lensless", <class_folder_name>): "copepoda" matches the CSV row
    # (non-null taxonomy); "diatom" has no CSV row (all-None taxonomy default).
    class_pngs = {"copepoda": 3, "diatom": 2}
    total_png_count = sum(class_pngs.values())
    for class_name, n in class_pngs.items():
        class_dir = imagefolder / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _write_png(class_dir / f"img_{i}.png")
    expected_classes = set(class_pngs)

    # CSV row only for the "copepoda" class -> exercises a non-null taxonomy row.
    csv_path = tmp_path / "taxo.csv"
    _write_taxonomy_csv(str(csv_path), "lensless", "copepoda")

    # (finding 3+8) Compose the REAL config. data_dir=tmp_path makes imagefolder_dir
    # resolve under tmp_path; the output saves to tmp_path/planktonzilla-17M.
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base="1.3", job_name="test_gen_lensless_e2e")
    cfg = hydra.compose(
        config_name="generate_planktonzilla",
        overrides=[f"taxonomy_csv_path={csv_path}", f"data_dir={tmp_path}", "num_proc=1"],
    )

    # (finding 6) Restrict to lensless only, keeping the real entry shape.
    OmegaConf.set_struct(cfg, False)
    cfg.datasets = [d for d in cfg.datasets if d["name"] == "lensless"]
    assert len(cfg.datasets) == 1
    entry = cfg.datasets[0]
    assert entry["name"] == "lensless"
    assert entry["import_name"] == "lensless"
    assert entry["cleanup"] is True
    assert entry["redefiner"] == "none"

    # (finding 2+4+10) Drive the real pipeline: real import_dataset compose +
    # instantiate (push_to_hub forced False -> no token needed), load_dataset
    # imagefolder, NoMetadataRedefiner.redefine, the real
    # clean_corrupt_examples_optimized (all valid PNGs kept), then save_to_disk.
    # main is the @hydra.main entry point; call the body directly via cfg-passthrough
    # (the former _run seam was merged into main).
    gp.main(cfg)
    GlobalHydra.instance().clear()

    # (finding 8) Reload + assert against the consolidated dataset on disk.
    output_path = tmp_path / constants.DEFAULT_PLANKTONZILLA_DATASET_NAME
    ds = datasets.load_from_disk(str(output_path))
    if isinstance(ds, datasets.DatasetDict):
        ds = ds["train"]

    assert len(ds) == total_png_count

    rows = [dict(row) for row in ds]

    # Every row comes from the single lensless source.
    assert all(row["dataset"] == "lensless" for row in rows)

    # original_label values are exactly the created class folder names.
    assert {row["original_label"] for row in rows} == expected_classes

    # (finding 5) Single train split => short_path is the last two chunks, so
    # original_path ends with "<class>/<file>.png" (e.g. "/copepoda/img_0.png").
    for row in rows:
        suffix = row["original_path"].rsplit("/", 2)[-2:]
        assert len(suffix) == 2
        assert suffix[0] == row["original_label"]
        assert suffix[1].endswith(".png")

    # The CSV-matched class ("copepoda") gets taxonomy + extras from the CSV.
    copepoda_rows = [row for row in rows if row["original_label"] == "copepoda"]
    assert len(copepoda_rows) == class_pngs["copepoda"]
    for row in copepoda_rows:
        assert row["proposed_label"] == "Copepoda"
        assert row["plankton"] is True
        assert row["Phylum"] == "Arthropoda"
        assert row["Kingdom"] == "Animalia"

    # The unmatched class ("diatom") resolves to all-None taxonomy/extras.
    diatom_rows = [row for row in rows if row["original_label"] == "diatom"]
    assert len(diatom_rows) == class_pngs["diatom"]
    for row in diatom_rows:
        assert row["proposed_label"] is None
        assert row["Phylum"] is None
        assert row["plankton"] is None

    # (finding 1) NoMetadataRedefiner emits {} -> flattened metadata columns exist
    # and are None for ALL lensless rows.
    metadata_cols = ["Latitude", "Longitude", "Depth_max", "Depth_min", "timestamp", "ObjID"]
    for col in metadata_cols:
        assert col in ds.column_names
        assert all(row[col] is None for row in rows)
