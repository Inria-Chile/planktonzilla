"""
Prueba de ejecución Hydra + Submitit (Jean Zay)
"""

import os
import socket
import datetime
import pyrootutils
from omegaconf import DictConfig
import hydra

# --- Configuración raíz del proyecto ---
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig):
    print("\n" + "AAAAAAAAA")
    print("\n" + "AAAAAAAAA")


if __name__ == "__main__":
    main()
