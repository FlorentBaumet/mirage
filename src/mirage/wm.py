"""World model d'état (Phase 1) : (fenêtre d'états passés) -> état suivant.

Interface uniforme : tous les modèles prennent des fenêtres APLATIES brutes
`X` de forme (m, lookback*d) et renvoient l'état suivant brut (m, d). La
standardisation (entrée + cible) est gérée DANS chaque modèle appris, fit sur le
train uniquement -> le rollout autorégressif reste simple et sans fuite.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def make_supervised(S: np.ndarray, lookback: int):
    """(n, d) -> X (m, lookback*d), Y (m, d), pos (m,), d  avec m = n - lookback.

    X[k] = états [k : k+lookback] aplatis ; Y[k] = état k+lookback (la cible).
    """
    n, d = S.shape
    m = n - lookback
    idx = np.arange(lookback)[None, :] + np.arange(m)[:, None]  # (m, lookback)
    X = S[idx].reshape(m, lookback * d)
    Y = S[lookback:]
    pos = np.arange(lookback, n)
    return X, Y, pos, d


# --- baselines -------------------------------------------------------------
class PersistenceWM:
    """État suivant = dernier état observé (martingale)."""

    def fit(self, X, Y):
        self.d = Y.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X)[:, -self.d:]


class MeanWM:
    """État suivant = moyenne (train) de chaque dimension."""

    def fit(self, X, Y):
        self.m = np.asarray(Y).mean(0)
        return self

    def predict(self, X):
        return np.tile(self.m, (len(X), 1))


# --- modèles appris (scaler interne, fit train-only) -----------------------
class _Scaled:
    def _fit_x(self, X):
        X = np.asarray(X, float)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0

    def _sx(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd


class LinearWM(_Scaled):
    """Ridge multi-sorties (VAR linéaire sur la fenêtre)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, Y):
        self._fit_x(X)
        self.model = Ridge(alpha=self.alpha).fit(self._sx(X), np.asarray(Y, float))
        return self

    def predict(self, X):
        return self.model.predict(self._sx(X))


class MLPWM(_Scaled):
    """MLP multi-sorties avec normalisation de la cible (anti-divergence)."""

    def __init__(self, hidden=(64, 32), max_iter=300, seed=0, alpha=1e-3):
        self.kw = dict(hidden_layer_sizes=tuple(hidden), max_iter=max_iter,
                       random_state=seed, alpha=alpha, early_stopping=True,
                       n_iter_no_change=15, validation_fraction=0.15)

    def fit(self, X, Y):
        self._fit_x(X)
        Y = np.asarray(Y, float)
        self.ymu = Y.mean(0)
        self.ysd = Y.std(0)
        self.ysd[self.ysd == 0] = 1.0
        self.model = MLPRegressor(**self.kw).fit(self._sx(X), (Y - self.ymu) / self.ysd)
        return self

    def predict(self, X):
        return self.model.predict(self._sx(X)) * self.ysd + self.ymu


def make_model(name: str, cfg: dict):
    if name == "mlp":
        return MLPWM(hidden=tuple(cfg.get("hidden", [64, 32])),
                     max_iter=int(cfg.get("max_iter", 300)),
                     seed=int(cfg.get("seed", 0)),
                     alpha=float(cfg.get("alpha", 1e-3)))
    if name == "linear":
        return LinearWM()
    raise ValueError(f"world model inconnu : {name}")


def rollout(model, win: np.ndarray, horizon: int) -> np.ndarray:
    """Rollout autorégressif BATCHÉ.

    win : (m, lookback, d) états bruts. Renvoie (m, horizon, d) : à chaque pas on
    prédit l'état suivant et on le re-injecte (on glisse la fenêtre).
    """
    win = np.asarray(win, float)
    m, lookback, d = win.shape
    cur = win.copy()
    out = np.empty((m, horizon, d))
    for h in range(horizon):
        nxt = model.predict(cur.reshape(m, lookback * d))  # (m, d) brut
        out[:, h, :] = nxt
        cur = np.concatenate([cur[:, 1:, :], nxt[:, None, :]], axis=1)
    return out
