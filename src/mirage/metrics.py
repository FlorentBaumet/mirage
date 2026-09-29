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


def r2_per_dim(Y, pred, base) -> np.ndarray:
    """R²_OOS colonne par colonne, contre une baseline `base` non constante.

    Utilisé pour l'évaluation par dimension du vecteur d'état, où la baseline
    appropriée diffère selon la dimension : 0 (random walk) pour un rendement,
    « no-change » pour un niveau (spread, imbalance...).
    """
    Y, pred, base = np.asarray(Y, float), np.asarray(pred, float), np.asarray(base, float)
    sse_m = np.sum((Y - pred) ** 2, axis=0)
    sse_b = np.sum((Y - base) ** 2, axis=0)
    return 1.0 - sse_m / np.where(sse_b == 0, np.nan, sse_b)


def directional_accuracy(y, p) -> float:
    """Taux de bon signe (on ignore les cibles nulles)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(p[m]) == np.sign(y[m])))
