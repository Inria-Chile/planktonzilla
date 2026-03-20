from datasets import ClassLabel, Dataset, Sequence, concatenate_datasets, load_dataset, Features, Image, Value, DatasetDict, load_from_disk
from joblib import Parallel, delayed
from tqdm import tqdm

from datasets import concatenate_datasets
import numpy as np
import os

import os
import io
import tarfile
from io import BytesIO
from PIL import Image
from tqdm import tqdm 

def export_to_tar_shards(dataset_dict, output_dir="data", shard_size=1_000):
    """
    Exporta un DatasetDict a shards .tar para entrenamiento tipo CLIP/WebDataset,
    asegurando que todas las imágenes se guarden en formato RGB.
    """
    os.makedirs(output_dir, exist_ok=True)


    for split_name, dataset in dataset_dict.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        total_samples = len(dataset)
        n_shards = (total_samples + shard_size - 1) // shard_size
        
        taxo_classes = dataset.features["label"].names

        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, total_samples)

            shard_path = os.path.join(split_dir, f"shard_{shard_idx:05d}.tar")
            shard_indices = range(start, end)
            
            with tarfile.open(shard_path, "w") as tar:
                # Iterar sobre los índices absolutos del dataset
                for i_abs in tqdm(shard_indices, desc=f"{split_name} shard {shard_idx}"):
                    example = dataset[i_abs]
                    i = i_abs - start # Índice relativo dentro del shard

                    # --- 1. Imagen (key: image_{i}.jpg) ---
                    img = example["image"]
                    if not isinstance(img, Image.Image):
                        raise ValueError(f"El campo 'image' en el índice {i_abs} no es un objeto PIL.Image")

                    # *** CAMBIO CLAVE: Convertir la imagen a RGB antes de guardar ***
                    # Esto maneja imágenes en escala de grises (L) o con paleta (P), o RGBA
                    img_rgb = img.convert('RGB')
                    
                    # Guardar la imagen en un buffer de bytes
                    img_bytes = io.BytesIO()
                    img_rgb.save(img_bytes, format="PNG") 
                    img_bytes.seek(0)

                    # Crear el TarInfo y añadir el archivo de imagen
                    img_info = tarfile.TarInfo(name=f"image_{i}.png")
                    img_info.size = len(img_bytes.getbuffer())
                    tar.addfile(img_info, img_bytes)

                    # --- 2. Etiqueta/Texto (key: text_{i}.txt) ---
                    label_str = str(taxo_classes[example["label"]])
                    label_bytes = io.BytesIO(label_str.encode("utf-8"))
                    
                    # Crear el TarInfo y añadir el archivo de texto
                    label_info = tarfile.TarInfo(name=f"image_{i}.txt")
                    label_info.size = len(label_bytes.getbuffer())
                    tar.addfile(label_info, label_bytes)


def main():
    dataset = load_from_disk("/lustre/fsn1/projects/rech/tec/uod68bo/data/planktonzilla_only_plankton2")
    # del dataset["test"]
    export_to_tar_shards(dataset, output_dir="/lustre/fsn1/projects/rech/tec/uod68bo/data/shards")
    
    print("DONE")
if __name__ == "__main__":
    main()
