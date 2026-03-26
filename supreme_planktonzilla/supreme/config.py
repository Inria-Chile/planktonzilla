"""
Configuración del método SUPREME.

Define el dataclass Config con todos los hiperparámetros y la configuración
de datasets necesaria para entrenamiento y evaluación. Soporta carga desde
archivo YAML mediante el método de clase from_yaml().

Cada dataset se define como un DatasetConfig con source, split y
samples_per_class, lo que permite mezclar fuentes del HuggingFace Hub con
rutas locales en la misma configuración.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from utils.datasets import DatasetConfig


@dataclass
class Config:
    """
    Contenedor de hiperparámetros y configuración de datasets para SUPREME.

    Los valores por defecto corresponden a la configuración original del paper.
    Para sobreescribir parámetros, usar from_yaml() con un archivo YAML
    personalizado.
    """

    # Backbone
    clip_model: str = "hf-hub:imageomics/bioclip-2"
    clip_pretrained: Optional[str] = None
    embed_dim: int = 768       # Dimensión de salida del codificador ViT-L/14 (imagen y texto)
    n_lm: int = 768            # Dimensión de embedding de tokens del codificador de texto
    img_size: int = 224        # Tamaño de la imagen cuadrada de entrada al modelo
    normalize_mean: list = field(default_factory=lambda: [0.48145466, 0.4578275,  0.40821073])
    normalize_std:  list = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])

    # BPG
    context_length: int = 16   # L: número de tokens de contexto aprendibles

    # Entrenamiento
    shots: int = 16            # k imágenes por clase en el conjunto de soporte
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.002
    lr_bias: float = 0.0001    # lr reducido para mu y sigma del sesgo gaussiano
    momentum: float = 0.9
    trials: int = 3

    # Pesos de la función de pérdida
    alpha: float = 0.005       # Peso para l_inter + l_intra
    beta: float = 0.1          # Peso para l_bias
    tau: float = 0.01          # Temperatura CLIP

    # Misceláneos
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    text_encode_chunk_size: int = 256   # Máximo de secuencias por chunk en el codificador de texto

    # Datasets
    id_train: DatasetConfig = field(default_factory=DatasetConfig)
    id_val: DatasetConfig = field(default_factory=DatasetConfig)
    id_test: DatasetConfig = field(default_factory=DatasetConfig)
    ood_test: List[DatasetConfig] = field(default_factory=list)

    # Evaluación
    score: str = "gmp"              # Función de puntuación: gmp | mmp | mcm_txt | mcm_img
    mean_img_prototypes: bool = False  # True → promediar K shots → (C, D); False → mantener (C, K, D)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Carga la configuración desde un archivo YAML.

        El YAML se organiza en secciones (clip, train, misc, datasets,
        evaluation) que se aplanan automáticamente a los campos del dataclass.
        ood_test puede ser un único bloque o una lista de bloques.

        Solo se aplican las claves presentes en el YAML; los parámetros
        ausentes conservan sus valores por defecto del dataclass.

        Args:
            path (str): Ruta al archivo YAML de configuración.

        Returns:
            Config: Instancia con los parámetros cargados.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Aplanar secciones anidadas → dict plano equivalente al dataclass
        section_map = {
            "clip":       {"model": "clip_model", "pretrained": "clip_pretrained",
                           "embed_dim": "embed_dim", "n_lm": "n_lm", "tau": "tau",
                           "img_size": "img_size", "normalize_mean": "normalize_mean",
                           "normalize_std": "normalize_std"},
            "train":      None,   # todas las claves pasan tal cual
            "misc":       None,
            "evaluation": None,
        }
        data: dict = {}
        for section, key_map in section_map.items():
            block = raw.pop(section, {}) or {}
            if key_map is None:
                data.update(block)
            else:
                for yaml_key, field_key in key_map.items():
                    if yaml_key in block:
                        data[field_key] = block[yaml_key]

        # Datasets viven en su propia sección
        datasets_block = raw.pop("datasets", {}) or {}
        data.update(datasets_block)

        # Claves de nivel raíz (compatibilidad hacia atrás con YAMLs planos)
        data.update(raw)

        ds_fields = {f.name for f in DatasetConfig.__dataclass_fields__.values()}

        # Convertir bloques de dataset individuales a DatasetConfig
        for key in ("id_train", "id_val", "id_test"):
            if key in data and isinstance(data[key], dict):
                data[key] = DatasetConfig(
                    **{k: v for k, v in data[key].items() if k in ds_fields}
                )

        # ood_test puede ser un único bloque o lista de bloques
        if "ood_test" in data:
            raw = data["ood_test"]
            if isinstance(raw, dict):
                data["ood_test"] = [
                    DatasetConfig(**{k: v for k, v in raw.items() if k in ds_fields})
                ]
            elif isinstance(raw, list):
                data["ood_test"] = [
                    DatasetConfig(**{k: v for k, v in entry.items() if k in ds_fields})
                    for entry in raw
                ]

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
