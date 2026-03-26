"""
SUPREME: Multi-modal Prototypes and Image Bias Estimation for Few-Shot OOD Detection.

Expone las funciones principales de entrenamiento y evaluación para que sean
invocadas desde main.py de manera uniforme con el resto de métodos del framework.
"""

from .train import run as run_train
from .evaluate import run as run_evaluate

__all__ = ["run_train", "run_evaluate"]
