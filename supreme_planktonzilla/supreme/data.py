"""
Constructores de DataLoader para el método SUPREME.

Orquesta la carga de datasets HuggingFace mediante load_dataset_from_config,
la construcción del wrapper CLIPDataset y la creación de los DataLoaders
para el conjunto de soporte (few-shot) y los conjuntos de evaluación.
"""

import json
from typing import Optional

from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import CLIPDataset, DatasetConfig, few_shot_subset, load_dataset_from_config
from utils.logger import ExperimentLogger
from utils.transforms import default_train_transform, default_val_transform


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloader(
    dataset: CLIPDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Fábrica genérica de DataLoader.

    Args:
        dataset     : CLIPDataset de PyTorch a envolver.
        batch_size  : Tamaño del lote.
        shuffle     : Si se mezclan los datos en cada época.
        num_workers : Número de procesos de carga paralela.
        pin_memory  : Si se usa memoria fijada para transferencias GPU más rápidas.

    Returns:
        DataLoader: Cargador de datos listo para iterar.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_support_loader(
    dataset_cfg: DatasetConfig,
    shots: int,
    batch_size: int,
    image_size: int = 224,
    mean: tuple = (0.48145466, 0.4578275,  0.40821073),
    std:  tuple = (0.26862954, 0.26130258, 0.27577711),
    seed: int = 42,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[DataLoader, list[str]]:
    """
    Construye un DataLoader para el conjunto de soporte few-shot.

    Carga el dataset según la configuración de dataset_cfg, aplica la
    selección few-shot y retorna el DataLoader junto con los nombres de
    clase del dataset completo.

    Args:
        dataset_cfg : Configuración del dataset (source, split, samples_per_class).
        shots       : Número máximo de imágenes por clase a seleccionar.
                      Si es None, se usa el dataset completo sin submuestreo.
        batch_size  : Tamaño del lote.
        image_size  : Tamaño de la imagen cuadrada de entrada al modelo.
        mean        : Media de normalización por canal (R, G, B).
        std         : Desviación estándar de normalización por canal (R, G, B).
        seed        : Semilla para la selección reproducible del subconjunto.
        num_workers : Número de procesos de carga paralela.
        transform   : Transformación personalizada; si es None usa la de entrenamiento.
        logger      : Logger para registrar tiempos y mensajes de carga.

    Returns:
        tuple[DataLoader, list[str]]: Cargador del soporte y lista de nombres de clase.
    """
    hf_ds = load_dataset_from_config(
        dataset_cfg.source,
        split=dataset_cfg.split,
        samples_per_class=dataset_cfg.samples_per_class,
        logger=logger,
    )
    full_ds = CLIPDataset(hf_ds, transform=transform or default_train_transform(image_size, mean, std))

    if shots == None:
        loader = get_dataloader(full_ds, batch_size=batch_size,shuffle=False,num_workers=num_workers)
        return loader,full_ds.class_names
    else:
        if dataset_cfg.idx_file is not None:
            with open(dataset_cfg.idx_file, "r") as f:
                indices = json.load(f)["indices"]
            support_ds = CLIPDataset(full_ds.dataset.select(indices), full_ds.transform)
            if logger:
                logger.info(
                    f"Conjunto de soporte: {len(support_ds)} muestras "
                    f"(índices cargados desde '{dataset_cfg.idx_file}')"
                )
        else:
            
            support_ds = few_shot_subset(full_ds, shots=shots, seed=seed)
            if logger:
                logger.info(
                    f"Conjunto de soporte: {len(support_ds)} muestras "
                    f"({shots}-shot × {len(full_ds.class_names)} clases)"
                )
    
        loader = get_dataloader(
            support_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return loader, full_ds.class_names


def build_eval_loader(
    dataset_cfg: DatasetConfig,
    batch_size: int,
    image_size: int = 224,
    mean: tuple = (0.48145466, 0.4578275,  0.40821073),
    std:  tuple = (0.26862954, 0.26130258, 0.27577711),
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[DataLoader, list[str]]:
    """
    Construye un DataLoader para evaluación (validación ID o conjunto OOD).

    Args:
        dataset_cfg : Configuración del dataset (source, split, samples_per_class).
        batch_size  : Tamaño del lote.
        image_size  : Tamaño de la imagen cuadrada de entrada al modelo.
        mean        : Media de normalización por canal (R, G, B).
        std         : Desviación estándar de normalización por canal (R, G, B).
        num_workers : Número de procesos de carga paralela.
        transform   : Transformación personalizada; si es None usa la de validación.
        logger      : Logger para registrar tiempos y mensajes de carga.

    Returns:
        tuple[DataLoader, list[str]]: Cargador de evaluación y lista de nombres de clase.
    """
    hf_ds = load_dataset_from_config(
        dataset_cfg.source,
        split=dataset_cfg.split,
        samples_per_class=dataset_cfg.samples_per_class,
        logger=logger,
    )
    ds = CLIPDataset(hf_ds, transform=transform or default_val_transform(image_size, mean, std))
    loader = get_dataloader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return loader, ds.class_names
