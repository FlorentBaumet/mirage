"""Features CAUSALES sur klines OHLCV (pas de carnet) + cible rendement next-bar.

Convention identique à features.py : r_t = Δlog(close) connu à t ; target_t = r_{t+1}
(ou somme sur `horizon`) ; persist = rendement passé sur `horizon` (baseline momentum).
Toutes les fenêtres glissantes incluent t (info passée), jamais le futur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def build_features(bars: pd.DataFrame, lags=(1, 2, 3, 5, 10), vol_window: int = 30,
                   horizon: int = 1):
    c = bars["close"]
    clog = np.log(c)
    r = clog.diff()                                  # r_t (connu à t)

    f = pd.DataFrame(index=bars.index)
    for k in lags:
        f[f"ret_lag_{k}"] = r.shift(k - 1)           # ret_lag_1 = r_t
    f["rvol"] = r.rolling(vol_window).std()          # volatilité réalisée (causale)
    f["range_rel"] = (bars["high"] - bars["low"]) / c
    # imbalance d'agresseurs (order-flow) : part achat - part vente, dans [-1, 1]
    f["taker_imb"] = (2.0 * bars["taker_buy_base"] - bars["volume"]) / (bars["volume"] + EPS)
    f["vol_ratio"] = bars["volume"] / (bars["volume"].rolling(vol_window).mean() + EPS)
    f["trades_z"] = bars["trades"] / (bars["trades"].rolling(vol_window).mean() + EPS)

    f["target"] = clog.shift(-horizon) - clog        # rendement futur sur h barres
    f["__persist"] = clog - clog.shift(horizon)      # rendement passé sur h (baseline)
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    y = f.pop("target")
    persist = f.pop("__persist")
    return f, y, persist
