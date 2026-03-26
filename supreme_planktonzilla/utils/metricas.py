"""
Métricas de evaluación para detección de distribuciones fuera de muestra (OOD).

Convención de puntuaciones: mayor valor = más dentro de la distribución (in-distribution).
Todas las funciones operan sobre arrays de NumPy.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def fpr_at_tpr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tpr_threshold: float = 0.95,
) -> float:
    """
    Calcula la Tasa de Falsos Positivos (FPR) cuando la Tasa de Verdaderos
    Positivos (TPR) alcanza el umbral indicado (por defecto 95%).

    Parámetros
    ----------
    id_scores     : Puntuaciones de confianza para muestras in-distribution.
    ood_scores    : Puntuaciones de confianza para muestras OOD.
    tpr_threshold : Nivel de TPR deseado (valor entre 0 y 1).

    Retorna
    -------
    float : FPR en el umbral de TPR especificado.
    """
    threshold = np.percentile(id_scores, (1 - tpr_threshold) * 100)
    fpr = (ood_scores >= threshold).mean()
    return float(fpr)


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Calcula el Área Bajo la Curva ROC (AUROC) para separar muestras ID de OOD.

    Parámetros
    ----------
    id_scores  : Puntuaciones de confianza para muestras in-distribution.
    ood_scores : Puntuaciones de confianza para muestras OOD.

    Retorna
    -------
    float : Valor AUROC entre 0 y 1.
    """
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))
