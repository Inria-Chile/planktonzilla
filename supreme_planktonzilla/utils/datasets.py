"""
Utilidades de carga de datasets en formato HuggingFace y wrapper PyTorch.

Proporciona una interfaz unificada para cargar datasets desde el Hub de
HuggingFace o desde disco local, y un Dataset de PyTorch compatible con
los DataLoaders del framework.
"""

import os
import random
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, DatasetDict

from utils.logger import ExperimentLogger


@dataclass
class DatasetConfig:
    """
    Configuración de un dataset individual para carga y submuestreo.

    Encapsula todos los parámetros necesarios para cargar un dataset desde
    HuggingFace Hub o desde disco local, incluyendo el split a usar y el
    número opcional de muestras por clase.

    Args:
        source (str): Identificador del dataset en HuggingFace Hub o ruta local.
        split (str): Split a cargar ('train', 'validation', 'test'). Por defecto 'train'.
        samples_per_class (int, optional): Número máximo de muestras por clase.
            Si es None, se conservan todas las muestras disponibles.
        idx_file (str, optional): Ruta a un archivo JSON con índices predefinidos
            para el subconjunto de soporte. Si se especifica, tiene prioridad sobre
            la selección few-shot aleatoria.
    """

    source: str = ""
    split: str = "train"
    samples_per_class: Optional[int] = None
    idx_file: Optional[str] = None


def load_dataset_from_config(
    source: str,
    split: str = "train",
    samples_per_class: Optional[int] = None,
    logger: Optional[ExperimentLogger] = None,
):
    """
    Carga un dataset desde HuggingFace Hub o desde una ruta local con
    submuestreo balanceado opcional por clase.

    Intenta primero cargar con load_dataset (Hub o formato HF local con splits).
    Si falla, intenta load_from_disk (dataset guardado en disco sin splits).

    Args:
        source (str): Identificador del dataset en HuggingFace Hub o ruta local.
        split (str): Split a cargar ('train', 'validation', 'test'). Por defecto 'train'.
        samples_per_class (int, optional): Número máximo de muestras por clase.
            Si es None, se conservan todas las muestras disponibles.
        logger (ExperimentLogger, optional): Logger para registrar tiempos y mensajes.

    Returns:
        Dataset | None: Dataset de HuggingFace cargado y filtrado, o None si falla.
    """
    timer_key = f"loading_{os.path.basename(source.rstrip('/'))}"
    if logger:
        logger.start_timer(timer_key)

    ds = None
    try:
        ds = load_dataset(source, split=split, num_proc=4)
    except Exception:
        try:
            result = load_from_disk(source)
            ds = result[split] if isinstance(result, DatasetDict) else result
        except Exception:
            traceback.print_exc()
            if logger:
                logger.error(f"No se pudo cargar el dataset desde '{source}'.")
            return None

    if samples_per_class is not None:
        label_counts: dict = {}
        indices_to_keep = []
        for idx, label in enumerate(ds["label"]):
            if label_counts.get(label, 0) < samples_per_class:
                indices_to_keep.append(idx)
                label_counts[label] = label_counts.get(label, 0) + 1
        ds = ds.select(indices_to_keep)

    if logger:
        logger.end_timer(timer_key)
        feat = ds.features.get("label")
        num_classes = (
            feat.num_classes if hasattr(feat, "num_classes") else len(set(ds["label"]))
        )
        logger.info(
            f"Dataset '{source}' | split='{split}' | "
            f"{len(ds):,} muestras | {num_classes} clases"
        )

    return ds


class CLIPDataset(Dataset):
    """
    Wrapper de PyTorch sobre un dataset HuggingFace para su uso con DataLoaders.

    Espera que el dataset HF tenga las columnas 'image' (PIL Image decodificada)
    y 'label' (entero o ClassLabel). Aplica la transformación indicada a cada
    imagen antes de retornarla.

    Args:
        hf_dataset : Dataset de HuggingFace (Dataset, no DatasetDict).
        transform  : Transformación de torchvision a aplicar a cada imagen.
    """

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Retorna el número total de muestras del dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Retorna el par (imagen transformada, etiqueta) del índice indicado.

        Args:
            idx (int): Índice de la muestra.

        Returns:
            tuple[Tensor, int]: Imagen como tensor y etiqueta entera.
        """
        item = self.dataset[idx]
        image = item["image"].convert("RGB")   # PIL Image decodificada por HuggingFace
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> list[str]:
        """
        Retorna la lista de nombres de clase del dataset.

        Si el dataset tiene un ClassLabel en la columna 'label', usa sus
        nombres. En caso contrario, usa los valores únicos de la columna.

        Returns:
            list[str]: Nombres de las clases en orden de su índice.
        """
        feat = self.dataset.features.get("label")
        if feat is not None and hasattr(feat, "names"):
            return feat.names
        return [str(i) for i in sorted(set(self.dataset["label"]))]


def few_shot_subset(
    dataset: CLIPDataset,
    shots: int,
    seed: int = 42,
) -> CLIPDataset:
    """
    Retorna un nuevo CLIPDataset con exactamente `shots` imágenes por clase.

    Si una clase tiene menos imágenes que `shots`, se conservan todas las
    disponibles. La selección es reproducible mediante la semilla indicada.

    Args:
        dataset (CLIPDataset): Dataset completo del que extraer el subconjunto.
        shots (int): Número máximo de imágenes a seleccionar por clase.
        seed (int): Semilla para la selección aleatoria reproducible.

    Returns:
        CLIPDataset: Subconjunto few-shot con la misma transformación que el original.
    """
    rng = random.Random(seed)
    indices_per_class: dict[int, list[int]] = {}

    for idx, label in enumerate(dataset.dataset["label"]):
        indices_per_class.setdefault(label, []).append(idx)

    selected: list[int] = []
    for label in sorted(indices_per_class):
        pool = indices_per_class[label]
        k = min(shots, len(pool))
        selected.extend(rng.sample(pool, k))

    hf_subset = dataset.dataset.select(selected)
    return CLIPDataset(hf_subset, dataset.transform)
