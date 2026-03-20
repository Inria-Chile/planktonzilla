from datasets import ClassLabel, Dataset, Sequence, concatenate_datasets, load_dataset, Features, Image, Value, DatasetDict, load_from_disk
from joblib import Parallel, delayed
from tqdm import tqdm

from datasets import concatenate_datasets
import numpy as np
import os


def clean_corrupt_examples_optimized(dataset, batch_size=100, n_jobs=-1):
    """
    Elimina ejemplos corruptos usando lectura por lotes para maximizar velocidad.
    """
    total_size = len(dataset)
    
    # Función que procesa un bloque de índices
    def process_batch(start_idx):
        end_idx = min(start_idx + batch_size, total_size)
        batch_range = range(start_idx, end_idx)
        
        try:
            # --- 1. Intento Optimista ---
            # Intentamos acceder al bloque entero. 
            # Si HF puede leer este slice sin error, todos están sanos.
            _ = dataset[start_idx:end_idx] 
            return list(batch_range) # Retornamos todos los índices del lote
            
        except Exception:
            # --- 2. Fallback Pesimista ---
            # Si el bloque falla, iteramos uno por uno SOLO en este bloque
            valid_indices = []
            for i in batch_range:
                try:
                    _ = dataset[i]
                    valid_indices.append(i)
                except Exception:
                    # Aquí encontramos el archivo corrupto y simplemente lo ignoramos
                    pass
            return valid_indices

    # Generamos los puntos de inicio para los batches
    starts = range(0, total_size, batch_size)

    # Paralelizamos por BATCH, no por fila (reduciendo overhead masivamente)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(s) for s in tqdm(starts, desc="Verificando integridad")
    )

    # Aplanamos la lista de listas resultante
    good_indices = [idx for batch in results for idx in batch]

    print(f"Original: {total_size} -> Limpio: {len(good_indices)}")
    print(f"Eliminados: {total_size - len(good_indices)} ejemplos corruptos.")

    return dataset.select(good_indices)


def build_ood(ds, num_proc=1):

    # 1️⃣ eliminar ejemplos corruptos
    ds = clean_corrupt_examples_optimized(ds, batch_size=1000, n_jobs=-1)

    # 2️⃣ filtrar solo los que NO son plankton
    ds = ds.filter(lambda x: x["plankton"] == False, num_proc=num_proc)

    # 3️⃣ obtener clases únicas desde original_label
    unique_labels = sorted(set(ds["original_label"]))
    class_label = ClassLabel(names=unique_labels)

    # 4️⃣ codificar original_label → label
    def encode_label(example):
        return {
            "label": class_label.str2int(example["original_label"])
        }

    ds = ds.map(encode_label, num_proc=num_proc)

    # 5️⃣ dejar solo columnas necesarias
    ds = ds.remove_columns(
        [c for c in ds.column_names if c not in ["image", "label", "dataset"]]
    )

    # 6️⃣ redefinir features
    new_features = Features({
        "image": ds.features["image"],
        "label": class_label,
        "dataset": Value("string")
    })

    ds = ds.cast(new_features)

    return ds


def main():

    num_proc = min(32, os.cpu_count())

    dataset = load_dataset("project-oceania/planktonzilla_full", split="train")

    dataset = build_ood(
        dataset,
        num_proc=num_proc
    )

    dataset = DatasetDict({
        "train": dataset,
    })

    dataset.save_to_disk(f"/lustre/fsn1/projects/rech/tec/uod68bo/data/planktonzilla_ood")
    
    print("DONE")
    
if __name__ == "__main__":
    main()
