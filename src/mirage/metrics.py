"""Métriques d'évaluation. Primaire = R²_OOS (pré-enregistrée)."""
from __future__ import annotations

import numpy as np


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def r2_oos(y, p, baseline=None) -> float:
    """R² out-of-sample vs une baseline (par défaut : zero-forecast).

    R²_OOS = 1 - SSE(modèle) / SSE(baseline).
    > 0  => le modèle bat la baseline 'prédire 0'.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    baseline = np.zeros_like(y) if baseline is None else np.asarray(baseline, float)
    sse_m = np.sum((y - p) ** 2)
    sse_b = np.sum((y - baseline) ** 2)
    if sse_b <= 0:
        return float("nan")
    return float(1.0 - sse_m / sse_b)


def directional_accuracy(y, p) -> float:
    """Taux de bon signe (on ignore les cibles nulles)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(p[m]) == np.sign(y[m])))
