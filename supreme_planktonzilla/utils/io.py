"""
Utilidades de entrada/salida compartidas entre los métodos del framework.

Incluye carga de configuraciones YAML y guardado de resultados en formato JSON.
"""

import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict:
    """
    Carga un archivo YAML y lo retorna como diccionario.

    Parámetros
    ----------
    path : Ruta al archivo YAML.

    Retorna
    -------
    dict : Contenido del archivo YAML.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str) -> None:
    """
    Guarda un objeto serializable como archivo JSON, creando los directorios
    necesarios si no existen.

    Parámetros
    ----------
    data : Objeto a serializar (dict, list, etc.).
    path : Ruta de destino del archivo JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def config_stem(config_path: str) -> str:
    """
    Extrae el nombre base (sin extensión) de un archivo de configuración.

    Se usa para nombrar las carpetas de modelos y resultados de manera
    consistente con el archivo YAML que generó el experimento.

    Parámetros
    ----------
    config_path : Ruta al archivo YAML de configuración.

    Retorna
    -------
    str : Nombre base del archivo sin extensión.
    """
    return Path(config_path).stem
