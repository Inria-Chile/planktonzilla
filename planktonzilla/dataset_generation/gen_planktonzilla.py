"""Build the full planktonzilla dataset from scratch.

For each source dataset it builds the imagefolder with Hydra, assigns the
taxonomy and external IDs from the taxonomy CSV, fetches the metadata through the
APIs (latitude, longitude, depth, temperature, humidity and date) and, at the
end, concatenates everything, drops the corrupt examples and saves the result to
disk.

Prerequisites:

  - Taxonomy CSV with the taxonomy and external ID columns
    (wikidata_ID, ecotaxa_ID, aphia_ID, NCBI_ID, BOLD_ID), indexed by
    (Dataset, Raw_Labels).

  - Some datasets have anti-bot protection on their download, so you have to
    download the .zip by hand and pass its path in the Hydra overrides. Until that
    path is given they stay commented out below:
      * Zoolake: https://opendata.eawag.ch/dataset/.../download/data.zip
      * SYKE ZooScan 2024: https://etsin.fairdata.fi/dataset/.../data
      * JEDI CPICS: https://dbarchive.biosciencedbc.jp/data/jedisystem-oceansdb/LATEST/CPICS_Validated.zip

  - Internet access for the WHOI and EcoTaxa APIs. EcoTaxa objects in private
    projects do not return metadata and stay null.
"""

import concurrent.futures
import json
import os
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import hydra
import numpy as np
import orjson
import polars as pl
import pyrootutils
import requests
from datasets import (
    Image,
    Value,
    concatenate_datasets,
    load_dataset,
)
from joblib import Parallel, delayed
from tqdm import tqdm

from planktonzilla.utils.logger import get_pylogger

root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

logger = get_pylogger(__name__)
num_proc = int(cpu_count() / 2)


# Cleaning up corrupt examples
def clean_corrupt_examples_optimized(dataset, batch_size=1000, n_jobs=-1):
    """Drop the corrupt examples, reading in batches to go fast."""
    total = len(dataset)

    def process_batch(start):
        end = min(start + batch_size, total)
        batch = range(start, end)
        try:
            # If the whole batch reads without error, all of them are fine.
            _ = dataset[start:end]
            return list(batch)
        except Exception:
            # If the batch fails, we check row by row and drop the corrupt ones.
            valid = []
            for i in batch:
                try:
                    _ = dataset[i]
                    valid.append(i)
                except Exception:
                    continue
            return valid

    starts = range(0, total, batch_size)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(s) for s in tqdm(starts, desc="Checking integrity")
    )
    good = [i for batch in results for i in batch]

    print(f"Original: {total} -> clean: {len(good)} (removed {total - len(good)})")
    return dataset.select(good)


# Fetching metadata through the APIs
def retrieve_whoi_metadata(bin_id, session=None):
    """Get lat/lon, depth, temperature, humidity and date from a WHOI bin."""
    api_url = f"https://ifcb-data.whoi.edu/api/bin/{bin_id}"
    hdr_url = f"https://ifcb-data.whoi.edu/mvco/{bin_id}.hdr"

    requester = session or requests

    info = {
        "Latitude": np.nan,
        "Longitude": np.nan,
        "Depth": np.nan,
        "Temperature": np.nan,
        "Humidity": np.nan,
        "Timestamp": None,
        "BinID": str(bin_id),
    }

    try:
        # JSON metadata: coordinates, depth and date of the bin.
        r = requester.get(api_url, timeout=10)
        if r.ok:
            data = r.json()
            info["Latitude"] = data.get("lat")
            info["Longitude"] = data.get("lng")
            info["Depth"] = data.get("depth")
            ts = data.get("timestamp_iso")
            # We keep only the date (YYYY-MM-DD).
            info["Timestamp"] = ts.split("T")[0] if ts else None

        # Metadata in the .hdr file: temperature and humidity.
        r = requester.get(hdr_url, timeout=10)
        if r.ok:
            lines = r.text.splitlines()
            for idx, line in enumerate(lines):
                if "Temp Humidity" in line and idx + 1 < len(lines):
                    headers = line.replace('"', "").split()
                    values = lines[idx + 1].replace('"', "").split(",")
                    if len(values) < len(headers):
                        values = lines[idx + 1].split()
                    mapping = dict(zip(headers, values))
                    info["Temperature"] = mapping.get("Temp")
                    info["Humidity"] = mapping.get("Humidity")
                    break

        # Numeric cast for the fields that need it.
        for k in ("Latitude", "Longitude", "Depth", "Temperature", "Humidity"):
            v = info[k]
            info[k] = float(v) if v not in (None, "", np.nan) else np.nan

    except Exception:
        pass

    return info


def retrieve_ecotaxa_metadata(obj_id, session=None):
    """Get depth, lat/lon and date from an EcoTaxa object."""
    api_url = f"https://ecotaxa.obs-vlfr.fr/api/object/{obj_id}"

    info = {
        "Depth_max": np.nan,
        "Depth_min": np.nan,
        "Latitude": np.nan,
        "Longitude": np.nan,
        "Timestamp": None,
        "ObjID": str(obj_id),
    }

    requester = session or requests

    try:
        response = requester.get(api_url, timeout=10)
        if response.status_code != 200:
            return info

        data = response.json()

        for src, dst in [
            ("depth_max", "Depth_max"),
            ("depth_min", "Depth_min"),
            ("latitude", "Latitude"),
            ("longitude", "Longitude"),
        ]:
            val = data.get(src)
            info[dst] = float(val) if val is not None else np.nan

        # objdate already comes as a date (YYYY-MM-DD).
        info["Timestamp"] = data.get("objdate")

    except (requests.RequestException, ValueError, TypeError):
        pass

    return info


# Assigning taxonomy, IDs and metadata
class RedefineDataset:
    """Base class to assign taxonomy, external IDs and metadata to a dataset."""

    TAXONOMY_COLS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
    EXTRA_COLS = ("proposed_label", "plankton", "root_class", "qualifier")
    ID_STR_COLS = ("wikidata_ID", "ecotaxa_ID")        # already text in the CSV
    ID_NUM_COLS = ("aphia_ID", "NCBI_ID", "BOLD_ID")   # come as numbers -> text without decimals

    def __init__(self, csv_taxonomies_path):
        # Columns pulled from the CSV, indexed by (Dataset, Raw_Labels).
        self.lookup_cols = [
            *self.TAXONOMY_COLS, *self.EXTRA_COLS, *self.ID_STR_COLS, *self.ID_NUM_COLS,
        ]
        self.lookup = self._build_lookup(csv_taxonomies_path)

        # Columns flattened out of the metadata JSON.
        self.metadata_cols_final = [
            "Latitude", "Humidity", "Temperature", "Longitude",
            "ObjID", "Depth_max", "Depth_min", "timestamp",
        ]

    @staticmethod
    def _norm(v):
        """Empty or blank strings become None; everything else is left as is."""
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    def _build_lookup(self, csv_path):
        df = pl.read_csv(csv_path)

        # Numeric IDs are stored as text without decimals (135336.0 -> "135336").
        for c in self.ID_NUM_COLS:
            if c in df.columns:
                df = df.with_columns(
                    pl.col(c).cast(pl.Int64, strict=False).cast(pl.Utf8).alias(c)
                )

        present = [c for c in self.lookup_cols if c in df.columns]
        keys = zip(df["Dataset"].to_list(), df["Raw_Labels"].to_list())
        rows = df.select(present).to_dicts()

        lookup = {}
        for key, row in zip(keys, rows):
            lookup[key] = {col: self._norm(row.get(col)) for col in self.lookup_cols}
        return lookup

    def _add_metadata(self, ds):
        """Attach the metadata as a JSON string. Defined by the subclasses."""
        raise NotImplementedError()

    def _flatten_metadata(self, ds):
        """Turn the metadata JSON into separate columns."""

        def extract(example):
            try:
                md = orjson.loads(example["metadata"]) if example["metadata"] else {}
            except Exception:
                md = {}

            for col in self.metadata_cols_final:
                example[col] = None

            # ObjID for EcoTaxa, BinID for WHOI.
            obj = md.get("ObjID") if md.get("ObjID") is not None else md.get("BinID")
            example["ObjID"] = str(obj) if obj not in (None, "") else None

            # WHOI gives a single depth; EcoTaxa gives a range.
            depth = md.get("Depth")
            if depth not in (None, ""):
                example["Depth_max"] = np.float32(depth)
                example["Depth_min"] = np.float32(depth)
            else:
                d_max = md.get("Depth_max")
                d_min = md.get("Depth_min")
                example["Depth_max"] = np.float32(d_max) if d_max not in (None, "") else None
                example["Depth_min"] = np.float32(d_min) if d_min not in (None, "") else None

            for col in ["Latitude", "Humidity", "Temperature", "Longitude"]:
                v = md.get(col)
                example[col] = np.float32(v) if v not in (None, "") else None

            ts = md.get("Timestamp")
            example["timestamp"] = ts if ts not in (None, "") else None

            return example

        ds = ds.map(extract, desc="Flattening metadata", num_proc=num_proc)
        return ds.remove_columns("metadata")

    def _cast_scalar_types(self, ds):
        """Set consistent types so all datasets concatenate without conflicts."""
        features = ds.features.copy()

        string_cols = [
            *self.TAXONOMY_COLS,
            "proposed_label", "root_class", "qualifier",
            "dataset", "original_label", "original_path", "ObjID", "timestamp",
            *self.ID_STR_COLS,
            *self.ID_NUM_COLS,
        ]
        for c in string_cols:
            if c in features:
                features[c] = Value("string")

        if "plankton" in features:
            features["plankton"] = Value("bool")

        for c in ["Latitude", "Longitude", "Temperature", "Humidity", "Depth_max", "Depth_min"]:
            if c in features:
                features[c] = Value("float32")

        return ds.cast(features)

    def redefine(self, hf_dataset, dataset_name, num_proc):
        """Assign taxonomy, IDs and metadata to every split and concatenate them."""
        parts = []
        n_splits = len(hf_dataset)

        for split in hf_dataset.keys():
            ds = hf_dataset[split]
            class_names = ds.features["label"].names
            ds = ds.cast_column("image", Image(decode=False))

            def process_row(example):
                label_str = class_names[example["label"]]
                full_path = example["image"]["path"]

                chunks = full_path.split(os.sep)
                short_path = (
                    "/" + "/".join(chunks[-3:])
                    if n_splits >= 2
                    else "/" + "/".join(chunks[-2:])
                )

                tax = self.lookup.get(
                    (dataset_name, label_str),
                    {col: None for col in self.lookup_cols},
                )

                return {
                    "dataset": dataset_name,
                    "original_label": label_str,
                    "original_path": short_path,
                    **tax,
                }

            print(f"Processing split {split}...")
            ds = ds.map(process_row, desc="Mapping taxonomy", num_proc=num_proc)

            ds = self._add_metadata(ds)
            ds = self._flatten_metadata(ds)

            if "label" in ds.column_names:
                ds = ds.remove_columns("label")

            ds = ds.cast_column("image", Image(decode=True))
            ds = self._cast_scalar_types(ds)

            parts.append(ds)

        return concatenate_datasets(parts)


class EcoTaxaRedefiner(RedefineDataset):
    """EcoTaxa datasets (flowcamnet, uvp6net, zooscan, etc.)."""

    def _add_metadata(self, ds):
        ids = [path.split("/")[-1].split(".")[0] for path in ds["original_path"]]

        with requests.Session() as session:
            func = partial(retrieve_ecotaxa_metadata, session=session)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_proc) as executor:
                raw = list(
                    tqdm(executor.map(func, ids), total=len(ids), desc="Metadata EcoTaxa")
                )

        def normalize(md):
            if not md:
                return {}
            return {str(k): str(v) for k, v in md.items() if v is not None}

        metadata = [normalize(r) for r in raw]
        ds = ds.add_column("metadata", metadata)

        ds = ds.map(
            lambda ex: {"metadata": json.dumps(ex["metadata"])},
            desc="Serializing metadata",
            num_proc=num_proc,
        )

        features = ds.features.copy()
        features["metadata"] = Value("string")
        return ds.cast(features)


class NoMetadataRedefiner(RedefineDataset):
    """Datasets without external metadata (lensless, medplanktonset, zoolake, etc.)."""

    def _add_metadata(self, ds):
        ds = ds.add_column("metadata", [{}] * len(ds))

        ds = ds.map(
            lambda ex: {"metadata": json.dumps(ex["metadata"])},
            desc="Serializing metadata",
            num_proc=num_proc,
        )

        features = ds.features.copy()
        features["metadata"] = Value("string")
        return ds.cast(features)


class WHOIRedefiner(RedefineDataset):
    """WHOI dataset: the metadata is queried by bin_id."""

    def _add_metadata(self, ds):
        def extract_bin_id(example):
            fname = example["original_path"].split("/")[-1]
            parts = fname.split(".")[0].split("_")[:-1]
            return {"bin_id": "_".join(parts)}

        ds = ds.map(extract_bin_id, desc="Extracting WHOI bin_id")

        bin_ids = np.unique(ds["bin_id"])
        print(f"{len(bin_ids)} unique bin_ids")

        # A bin groups many images, so we query once per bin.
        bin_lookup = {}
        with requests.Session() as session:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_proc) as executor:
                futures = {
                    executor.submit(retrieve_whoi_metadata, bin_id, session): bin_id
                    for bin_id in bin_ids
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Metadata WHOI",
                ):
                    bin_id = futures[future]
                    try:
                        raw = future.result()
                        bin_lookup[bin_id] = {
                            str(k): str(v) for k, v in raw.items() if v is not None
                        }
                    except Exception:
                        bin_lookup[bin_id] = {}

        ds = ds.map(
            lambda ex: {"metadata": bin_lookup.get(ex["bin_id"], {})},
            desc="Attaching WHOI metadata",
        )
        ds = ds.remove_columns("bin_id")

        ds = ds.map(
            lambda ex: {"metadata": json.dumps(ex["metadata"])},
            desc="Serializing metadata",
            num_proc=num_proc,
        )

        features = ds.features.copy()
        features["metadata"] = Value("string")
        return ds.cast(features)


class JediRedefiner(RedefineDataset):
    """JEDI Oceans dataset: fixed metadata for all the examples."""

    def __init__(self, csv_taxonomies_path):
        super().__init__(csv_taxonomies_path)
        self.metadata = {
            "Latitude": "34.682718",
            "Longitude": "139.444779",
            "Depth_min": "20",
            "Depth_max": "20",
        }

    def _add_metadata(self, ds):
        ds = ds.add_column("metadata", [self.metadata] * len(ds))

        ds = ds.map(
            lambda ex: {"metadata": json.dumps(ex["metadata"])},
            desc="Serializing metadata",
            num_proc=num_proc,
        )

        features = ds.features.copy()
        features["metadata"] = Value("string")
        return ds.cast(features)

def main():

    DATA_ROOT = (root / "data").resolve()
    taxo_csv_path = str(DATA_ROOT / "planktonzilla_taxonomy_v20.csv")

    datasets_configs = {
        "isiisnet": {
            "overrides": [
                "dataset_import=isiisnet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "whoi": {
            "overrides": [
                "dataset_import=whoi-plankton",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": WHOIRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "flowcamnet": {
            "overrides": [
                "dataset_import=flowcamnet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        # JEDI Oceans requires downloading CPICS_Validated.zip by hand.
        # "jedi_oceans_cpics": {
        #     "overrides": [
        #         "dataset_import=jedi",
        #         "dataset_import.cleanup_after_processing=True",
        #         "dataset_import.push_to_hub=False",
        #         f"dataset_import.data_dir={DATA_ROOT}",
        #         f"dataset_import.manual_download_local_file_names={DATA_ROOT / 'CPICS_Validated.zip'}",
        #     ],
        #     "redefiner": JediRedefiner(csv_taxonomies_path=taxo_csv_path),
        # },
        "lensless": {
            "overrides": [
                "dataset_import=lensless",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "medplanktonset": {
            "overrides": [
                "dataset_import=medplanktonset",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        # SYKE ZooScan 2024 requires downloading its .zip by hand.
        # "sykezooscan2024": {
        #     "overrides": [
        #         "dataset_import=sykezooscan2024",
        #         "dataset_import.cleanup_after_processing=True",
        #         "dataset_import.push_to_hub=False",
        #         f"dataset_import.data_dir={DATA_ROOT}",
        #         f"dataset_import.manual_download_local_file_names={DATA_ROOT / 'SYKE-plankton_ZooScan_2024.zip'}",
        #     ],
        #     "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        # },
        "uvp6net": {
            "overrides": [
                "dataset_import=uvp6net",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "zoocamnet": {
            "overrides": [
                "dataset_import=zoocamnet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        # Zoolake requires downloading data.zip by hand.
        # "zoolake": {
        #     "overrides": [
        #         "dataset_import=zoolake",
        #         "dataset_import.cleanup_after_processing=True",
        #         "dataset_import.push_to_hub=False",
        #         f"dataset_import.data_dir={DATA_ROOT}",
        #         f"dataset_import.manual_download_local_file_names={DATA_ROOT / 'zoolake_data.zip'}",
        #     ],
        #     "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        # },
        "zooscan": {
            "overrides": [
                "dataset_import=zooscannet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "planktonset1.0": {
            "overrides": [
                "dataset_import=planktonset1",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "syke_ifcb_2022": {
            "overrides": [
                "dataset_import=syke_ifcb_2022",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "planktoscope": {
            "overrides": [
                "dataset_import=planktoscope",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
        "global_uvp5": {
            "overrides": [
                "dataset_import=global_uvp5net",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
    }

    parts = []

    with hydra.initialize(version_base="1.3", config_path="../configs"):
        for dataset_name, ds_cfg in datasets_configs.items():
            print(f"\n=== Dataset: {dataset_name} ===")

            cfg = hydra.compose(config_name="import_dataset", overrides=ds_cfg["overrides"])

            dataset_importer = hydra.utils.instantiate(cfg.dataset_import)
            imagefolder_dir = Path(dataset_importer.imagefolder_dir)

            # Reuse the imagefolder if it already exists; otherwise build it.
            has_content = imagefolder_dir.exists() and bool(os.listdir(imagefolder_dir))
            if has_content:
                num_items = len(os.listdir(imagefolder_dir))
                print(f"Using existing imagefolder with {num_items} categories in {imagefolder_dir}")
            else:
                print("Building imagefolder from the raw data...")
                dataset_importer.import_dataset()

            # Resolve the files for each split (accepts the val/validation alias).
            split_aliases = {
                "train": ["train"],
                "validation": ["validation", "val"],
                "test": ["test"],
            }
            data_files = {}
            for canonical_split, aliases in split_aliases.items():
                for alias in aliases:
                    split_path = root / alias
                    if split_path.exists():
                        data_files[canonical_split] = str(split_path / "*/[!._]*")
                        break

            # No explicit splits: take everything as train.
            if not data_files:
                data_files = {"train": str(dataset_importer.imagefolder_dir / "*/*[!._]*")}

            print("Loading dataset with the imagefolder loader...")
            dataset = load_dataset("imagefolder", data_files=data_files)

            print("Assigning taxonomy, IDs and metadata...")
            dataset = ds_cfg["redefiner"].redefine(
                hf_dataset=dataset,
                dataset_name=dataset_name,
                num_proc=num_proc,
            )

            parts.append(dataset)

    ds = concatenate_datasets(parts)

    # With the full dataset ready, we drop the examples whose image is corrupt.
    ds = clean_corrupt_examples_optimized(ds, batch_size=1000, n_jobs=-1)

    output_path = DATA_ROOT / "planktonzilla_17M"
    print(f"Saving dataset to {output_path}")
    ds.save_to_disk(output_path)

    print("\nProcess completed")


if __name__ == "__main__":
    main()
