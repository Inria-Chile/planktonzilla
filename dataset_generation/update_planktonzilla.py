import math
import os

import pandas as pd
from datasets import Value, load_dataset

# Configuracion
REPO_ID = "project-oceania/planktonzilla-17M"

# Copia en disco del espacio compartido, que luego lee retrieve_timestamp.py.
OUTPUT_DIR = "/home/acontreras/group_storage_rennes/acontreras/planktonzilla_17M_updated"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "data", "planktonzilla_taxonomy_v20.csv")

# Columnas de taxonomia que se re-sincronizan.
TAXO_COLS = [
    "Kingdom", "Phylum", "Class", "Order", "Family",
    "Genus", "Species", "proposed_label", "plankton",
    "root_class", "qualifier",
]

# Columnas de IDs de bases de datos externas. Todas se guardan como string.
STR_ID_COLS = ["wikidata_ID", "ecotaxa_ID"]            # ya vienen como string en el CSV
NUMERIC_ID_COLS = ["aphia_ID", "NCBI_ID", "BOLD_ID"]   # vienen como float en el CSV -> string sin decimales
ID_COLS = STR_ID_COLS + NUMERIC_ID_COLS

# Todas las columnas a actualizar. Ya existen en el dataset.
SYNC_COLS = TAXO_COLS + ID_COLS


def build_sync_dict(csv_path):
    """Carga el CSV y arma el diccionario (Dataset, Raw_Labels) -> valores a actualizar."""
    print("Cargando CSV y preparando diccionario...")
    df = pd.read_csv(csv_path, sep=",")

    # wikidata_ID / ecotaxa_ID: string tal cual (ej. "Q3386609" o "274;1231;15123").
    for c in STR_ID_COLS:
        df[c] = df[c].apply(lambda v: str(v) if pd.notna(v) else None)

    # aphia/NCBI/BOLD: el CSV los lee como float (135336.0); los pasamos a string
    # sin decimales ("135336"), no a int, porque la columna se guarda como string.
    for c in NUMERIC_ID_COLS:
        df[c] = df[c].apply(lambda v: str(int(v)) if pd.notna(v) else None)

    rows = df.set_index(["Dataset", "Raw_Labels"])[SYNC_COLS].to_dict("index")

    # Vacios -> None (null): tanto NaN (float) como cadenas en blanco. Se hace sobre
    # el dict de Python porque a nivel DataFrame pandas reconvierte los None a NaN.
    # El booleano plankton no se ve afectado.
    def to_null(v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    return {
        key: {col: to_null(val) for col, val in row.items()}
        for key, row in rows.items()
    }


def sync_columns(ds, sync_dict):
    """Actualiza los valores de las columnas ya existentes a partir del CSV."""
    # Todas las columnas de IDs quedan como string.
    new_features = ds.features.copy()
    for c in ID_COLS:
        new_features[c] = Value("string")

    # Columnas de texto que no deben quedar con cadenas vacias: taxonomia (sin el
    # booleano plankton) mas los IDs.
    text_cols = [c for c in TAXO_COLS if c != "plankton"] + ID_COLS

    def actualizar_ejemplo(example):
        key = (example["dataset"], example["original_label"])
        updates = sync_dict.get(key)

        if updates is not None:
            for col in SYNC_COLS:
                example[col] = updates[col]
        else:
            # Sin match en el CSV: dejamos la taxonomia como esta y nulamos los IDs.
            for col in ID_COLS:
                example[col] = None

        # Cualquier cadena vacia o en blanco pasa a None (null, no "" ni nan).
        for col in text_cols:
            v = example[col]
            if isinstance(v, str) and v.strip() == "":
                example[col] = None

        return example

    print("Actualizando columnas...")
    return ds.map(
        actualizar_ejemplo,
        num_proc= int(os.cpu_count()/2),
        features=new_features,
        desc="Re-sincronizando taxonomia e IDs externos",
    )


def main():
    print(f"Cargando dataset {REPO_ID}...")
    ds = load_dataset(REPO_ID, split="train")

    sync_dict = build_sync_dict(CSV_PATH)
    dataset_final = sync_columns(ds, sync_dict)

    print(f"Guardando dataset en disco ({OUTPUT_DIR})...")
    dataset_final.save_to_disk(OUTPUT_DIR)

    # print(f"Subiendo dataset al Hub ({REPO_ID})...")
    # dataset_final.push_to_hub(
    #     REPO_ID,
    #     split="train",
    #     commit_message="update: re-sync columns",
    # )

    print("\n¡Proceso finalizado!")


if __name__ == "__main__":
    main()
