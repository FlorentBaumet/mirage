"""Baselines obligatoires à battre OUT-OF-SAMPLE.

- zero        : prédire un rendement de 0 (random walk / 'prix inchangé').
- persistence : prédire r_{t+1} ~ r_t (= feature ret_lag_1, en valeurs BRUTES).
- linear      : Ridge sur les features (le 'trivial mais réel' — si le modèle ne
                le bat pas, il n'apporte rien).

NB : 'persistence' est traité à part dans eval.py car il réutilise une feature
brute (ret_lag_1) comme prédiction ; il ne doit donc PAS voir les features
standardisées.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class ZeroForecast:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class LinearBaseline:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
