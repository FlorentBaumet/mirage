"""Vecteur d'ÉTAT compact pour le world model (Phase 1).

État (5 dims), borné/stationnaire et *déroulable* (on peut le prédire ET le
re-fournir en entrée pour un rollout autorégressif) :
  ret        : Δ log(mid)            (dynamique de prix ; cumulé => trajectoire du mid)
  spread_rel : (ask1 - bid1) / mid
  imb1       : imbalance L1          (bid_size1 - ask_size1) / somme
  depth_imb  : imbalance de profondeur agrégée (L1..L10)
  micro_dev  : (micro_price - mid) / mid

Tout est causal (état à t = info <= t). La cible du world model = état à t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
STATE_COLS = ["ret", "spread_rel", "imb1", "depth_imb", "micro_dev"]
RET_IDX = 0  # position de 'ret' dans STATE_COLS (utilisé pour le rollout du prix)


def build_state(bars: pd.DataFrame, levels: int = 10):
    """Renvoie (S, mid) : S = DataFrame des états (STATE_COLS), mid aligné."""
    a1, b1 = bars["ask_price_1"], bars["bid_price_1"]
    mid = (a1 + b1) / 2.0
    mlog = np.log(mid)
    as1, bs1 = bars["ask_size_1"], bars["bid_size_1"]
    micro = (a1 * bs1 + b1 * as1) / (as1 + bs1 + EPS)
    bid_depth = sum(bars[f"bid_size_{i}"] for i in range(1, levels + 1))
    ask_depth = sum(bars[f"ask_size_{i}"] for i in range(1, levels + 1))

    S = pd.DataFrame(index=bars.index)
    S["ret"] = mlog.diff()
    S["spread_rel"] = (a1 - b1) / mid
    S["imb1"] = (bs1 - as1) / (bs1 + as1 + EPS)
    S["depth_imb"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + EPS)
    S["micro_dev"] = (micro - mid) / mid

    S = S[STATE_COLS].dropna()
    return S, mid.loc[S.index]
