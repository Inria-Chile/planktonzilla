"""Tests for utils/metricas.py — fpr_at_tpr, auroc."""

import numpy as np
import pytest

from utils.metricas import auroc, fpr_at_tpr


# ── fpr_at_tpr ────────────────────────────────────────────────────────────────

def test_fpr_at_tpr_perfect_separation():
    """When ID scores are all higher than OOD, FPR should be 0."""
    id_scores = np.array([0.8, 0.9, 0.85, 0.95, 0.88])
    ood_scores = np.array([0.1, 0.2, 0.15, 0.05, 0.12])
    fpr = fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95)
    assert fpr == 0.0


def test_fpr_at_tpr_complete_overlap():
    """When ID and OOD are identical distributions, FPR ~= 1 - tpr_threshold."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, size=1000)
    fpr = fpr_at_tpr(scores, scores, tpr_threshold=0.95)
    # Threshold is set at 5th percentile of ID, so ~95% of OOD exceeds it
    assert abs(fpr - 0.95) < 0.05


def test_fpr_at_tpr_returns_float():
    id_scores = np.linspace(0.5, 1.0, 100)
    ood_scores = np.linspace(0.0, 0.5, 100)
    result = fpr_at_tpr(id_scores, ood_scores)
    assert isinstance(result, float)


def test_fpr_at_tpr_in_range():
    rng = np.random.default_rng(0)
    id_scores = rng.normal(0.7, 0.1, 200)
    ood_scores = rng.normal(0.3, 0.1, 200)
    fpr = fpr_at_tpr(id_scores, ood_scores)
    assert 0.0 <= fpr <= 1.0


# ── auroc ─────────────────────────────────────────────────────────────────────

def test_auroc_perfect_separation():
    """Perfect separation should yield AUROC = 1.0."""
    id_scores = np.array([0.9, 0.85, 0.95, 0.8])
    ood_scores = np.array([0.1, 0.2, 0.15, 0.05])
    assert auroc(id_scores, ood_scores) == pytest.approx(1.0)


def test_auroc_random_chance():
    """Same distribution should yield AUROC ~= 0.5."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, size=2000)
    half = len(scores) // 2
    result = auroc(scores[:half], scores[half:])
    assert abs(result - 0.5) < 0.05


def test_auroc_returns_float():
    id_scores = np.linspace(0.5, 1.0, 50)
    ood_scores = np.linspace(0.0, 0.5, 50)
    result = auroc(id_scores, ood_scores)
    assert isinstance(result, float)


def test_auroc_in_range():
    rng = np.random.default_rng(7)
    id_scores = rng.normal(0.7, 0.15, 300)
    ood_scores = rng.normal(0.3, 0.15, 300)
    result = auroc(id_scores, ood_scores)
    assert 0.0 <= result <= 1.0
