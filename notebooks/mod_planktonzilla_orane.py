# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:50:45 2026

@author: equil

MODIFICATION FROM PLANKTONZILLA_FULL DATASET DIRECTLY
"""

# Manipulation fichiers et processus
import os # accès au système de fichiers (dossiers, chemins, etc.)
import sys # accès à l’environnement Python (arguments, chemins Python)
import shutil # copier / déplacer / supprimer des dossiers
import subprocess # exécuter des commandes système

from pathlib import Path # gestion moderne des chemins (remplace souvent os.path)
from shutil import rmtree # fonction qui supprime un dossier entier récursivement
import numpy as np
import requests # Permet d'interroger des API scientifiques (EcoTaxa, WHOI)
import time
import json
import orjson

# Configuration Hydra
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import pyrootutils # permet de trouver automatiquement la racine du projet
import polars as pl # dataframe rapide (plus rapide que pandas)
from tqdm import tqdm # barre de progression
import concurrent.futures # exécution parallèle (threads)

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk
) # librairie Hugging Face Datasets sert à manipuler datasets ML, gérer images

from planktonzilla.utils.logger import get_pylogger # crée un logger (INFO ERROR WARRNING)
from planktonzilla.dataset_import.dataset_importer import (
    DatasetImporter, # télécharge et prépare les datasets
    is_dir_empty, # vérifie si dossier vide
    is_valid_image_file, # vérifie si une image est corrompue"
)




from multiprocessing import cpu_count
num_proc = min(cpu_count(), 32)


root = pyrootutils.setup_root(
    search_from=".",  
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
) # pour trouver la racine du repo, ajouter les chemins python, charger .env

logger = get_pylogger(__name__) # crée un loger __name__ qui correspond au modèle actuel


# ============= GENERATING HF DATASETS WITH METADATA ============= #


def cast_metadata_json(ds): # "metadata" : dict → string JSON => "metadata": '{"lat": 10, "lon": 20}'
    # 1. Convert dict → JSON string
    def to_json(example): 
        return {"metadata": json.dumps(example["metadata"])} 

    ds = ds.map(to_json, desc="Serializing metadata")

    # 2. Cast feature
    features = ds.features.copy()
    features["metadata"] = Value("string")
    return ds.cast(features)


#%%



class ProcessDataset:
    def __init__(self):
        self.meta_cols = ['Depth_max', 'Depth_min', 'Depth', 'Latitude', 'Longitude', 'ObjID', 'BinID', 'Humidity', 'Temperature', 'Date', 'Time'] #  salinity, timestamp
        self.taxonomy_cols = ['image', 'dataset', 'original_label', 'original_path', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'proposed_label', 'plankton', 'living', 'metadata']
        self.all_cols = self.taxonomy_cols + self.meta_cols

        self.ecotaxadatasets = ['flowcamnet', 'uvp6net', 'zooscan']

    def retrieve_ecotaxa_metadata(obj_id, session=None):
        if obj_id is None:
            return {}
        api_url = f"https://ecotaxa.obs-vlfr.fr/api/object/{obj_id}"

        info = {
            "Depth_max": np.nan,
            "Depth_min": np.nan,
            "Latitude": np.nan,
            "Longitude": np.nan,
            "ObjID": str(obj_id),
            "Date": np.nan,
            "Time": np.nan,
        } # initialise metadata

        requester = session if session else requests

        try:
            response = requester.get(api_url, timeout=10) # recupere metadata
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

        except (requests.RequestException, ValueError, TypeError):
            pass

        return info
    
    def retrieve_whoi_metadata(bin_id, session=None):
        api_url = f"https://ifcb-data.whoi.edu/api/bin/{bin_id}"
        hdr_url = f"https://ifcb-data.whoi.edu/mvco/{bin_id}.hdr"

        requester = session or requests

        info = {
            "Latitude": np.nan,
            "Longitude": np.nan,
            "Depth": np.nan,
            "Temperature": np.nan,
            "Humidity": np.nan,
            "BinID": str(bin_id),
        }

        try:
            # ---------- JSON metadata ----------
            r = requester.get(api_url, timeout=10)
            if r.ok:
                data = r.json()
                info["Latitude"] = data.get("lat")
                info["Longitude"] = data.get("lng")
                info["Depth"] = data.get("depth")

            # ---------- HDR metadata ----------
            r = requester.get(hdr_url, timeout=10)
            if r.ok:
                lines = r.text.splitlines()

                for idx, line in enumerate(lines):
                    if "Temp Humidity" in line and idx + 1 < len(lines):
                        headers = line.replace('"', '').split()
                        values = lines[idx + 1].replace('"', '').split(",")

                        if len(values) < len(headers):
                            values = lines[idx + 1].split()

                        mapping = dict(zip(headers, values))
                        info["Temperature"] = mapping.get("Temp")
                        info["Humidity"] = mapping.get("Humidity")
                        break

            # ---------- Fast float cast ----------
            for k in ("Latitude", "Longitude", "Depth", "Temperature", "Humidity"):
                v = info[k]
                info[k] = float(v) if v not in (None, "", np.nan) else np.nan

        except Exception:
            pass

        return info

    def _add_metadata(self, ds):
        ecotaxa_indices = []
        ecotaxa_ids = []
        whoi_indices = []
        whoi_ids = []
        for i, (obj_id, bin_id) in enumerate(zip(ds["ObjID"], ds["BinID"])):
            # Ecotaxa case
            if obj_id not in (None, ""):
                ecotaxa_indices.append(i)
                ecotaxa_ids.append(obj_id)
                # Whoi case
            elif bin_id not in (None, ""):
                whoi_indices.append(i)
                whoi_ids.append(bin_id)
        
        # ecotaxa_ids = list(set(ecotaxa_ids))
        # whoi_ids = list(set(whoi_ids))

        from functools import partial
        import concurrent.futures
        
        # Ecotaxa API
        ecotaxa_lookup = {}
        with requests.Session() as session:
            func = partial(retrieve_ecotaxa_metadata, session=session)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor: # before : 32
                results = list(tqdm(executor.map(func, ecotaxa_ids), total=len(ecotaxa_ids)))
        for obj_id, md in zip(ecotaxa_ids, results):
            ecotaxa_lookup[obj_id] = md

        # WHOI API
        whoi_lookup = {}
        with requests.Session() as session:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.retrieve_COXid_metadata, bin_id, session): bin_id
                    for bin_id in whoi_ids
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="WHOI"
                ):
                    bin_id = futures[future]
                    try:
                        whoi_lookup[bin_id] = future.result()
                    except Exception:
                        whoi_lookup[bin_id] = {}



        def normalize_metadata(md: dict | None) -> dict:
            if not md:
                return {}
            return {str(k): str(v) for k, v in md.items() if v is not None}
            
        existing_metadata = ds["metadata"]

        metadata = [
            dict(m) if isinstance(m, dict) else {}
            for m in existing_metadata
        ]

        # Merge EcoTaxa
        for idx, obj_id in zip(ecotaxa_indices, ecotaxa_ids):
            md = ecotaxa_lookup.get(obj_id)
            if md:
                metadata[idx].update(normalize_metadata(md))

        # Merge WHOI
        for idx, bin_id in zip(whoi_indices, whoi_ids):
            md = whoi_lookup.get(bin_id)
            if md:
                metadata[idx].update(normalize_metadata(md))

        ds = ds.remove_columns("metadata")
        ds = ds.add_column("metadata", metadata)

        return cast_metadata_json(ds)
    
    def retrieve_COXid_metadata(self, processed_ds):
        raise NotImplementedError()
    


    def redefine_dataset(self, ds , num_proc): # transforms dataset
        
        # critical optimization
        ds = ds.cast_column("image", Image(decode=False)) # doesnt decode images

        #add more metadata (date, time)
        ds = self._add_metadata(ds)

        def process_row(example):
            md = orjson.loads(example["metadata"])
            
            # extract metadata into columns
            for i in self.meta_cols :
                val = md.get(i)
                example[i]= val if val not in (None, "") else None
            
            return example
            '''
            "qualifier":"",
            "cox_gene_id ": "",
            "ecotaxa_id": "",
            "wikipedia_id": "",
            '''

        processed_ds = ds.map(process_row, desc="Columns mapping",num_proc=num_proc,keep_in_memory=True, load_from_cache_file=False,)
        processed_ds = processed_ds.remove_columns("metadata") # remove_columns=["plankton", "living"]

        processed_ds = processed_ds.cast_column("image", Image(decode=True))
        return processed_ds
    

# ============= GENERATING HF DATASETS WITH METADATA ============= #


def main():

    # DATA_ROOT = Path("planktonzilla/notebooks").resolve()
    #base_path = Path.home() / "group_storage_sophia/saguilera/Labels"

    #file1 = base_path / "eKOI_taxonomy_labels.parquet"
    #file2 = base_path / "MetaCOXI_taxonomy_labels.parquet"

    #df = pl.read_parquet(file1)
    #print(df.head())

    dataset = load_dataset(
        "project-oceania/planktonzilla_full",
        split="train",
        # streaming=True # creates IterableDataset != normal dataset so num_proc is not supported
    )
    

    ds = dataset.take(1000) #shuffle().select(range(1000)) 
    
    # Process dataset 
    processor = ProcessDataset()
    modified_ds = processor.redefine_dataset(ds, num_proc=num_proc)
    
    for example in modified_ds.take(2) :
        print(example)


    # modified_ds.save_to_disk(DATA_ROOT / "planktonzilla_full_modified")
    
if __name__ == "__main__":
    main()