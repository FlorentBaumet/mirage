"""Construction des features CAUSALES + cible (rendement next-step).

Règle d'or (anti-lookahead, I3) : toute feature à la barre t n'utilise QUE de
l'information disponible jusqu'à t inclus. La cible est le rendement futur t -> t+1.

Conventions de rendement :
  r_t      = log(mid_t) - log(mid_{t-1})   (réalisé, connu à t)
  target_t = r_{t+1}                       (ce qu'on prédit)  = r.shift(-1)
  ret_lag_k = r_{t-k+1}                     (passé, connu à t) = r.shift(k-1)
              -> ret_lag_1 = r_t  (sert de baseline 'persistence')
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _ofi_best_level(b: pd.DataFrame) -> pd.Series:
    """Order-Flow Imbalance au meilleur niveau (Cont et al.), causal (t et t-1)."""
    bp, bs = b["bid_price_1"], b["bid_size_1"]
    ap, asz = b["ask_price_1"], b["ask_size_1"]
    bp1, bs1 = bp.shift(1), bs.shift(1)
    ap1, as1 = ap.shift(1), asz.shift(1)

    e_b = np.where(bp > bp1, bs, np.where(bp == bp1, bs - bs1, -bs1))
    e_a = np.where(ap < ap1, -asz, np.where(ap == ap1, asz - as1, as1))
    return pd.Series(e_b - e_a, index=b.index)


def build_features(
    bars: pd.DataFrame,
    levels: int = 10,
    lags: tuple[int, ...] = (1, 2, 3, 5, 10),
    use_ofi: bool = True,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Renvoie (X, y, persist) alignés, NaN initiaux/finaux supprimés.

    y       = target_t = rendement log du mid sur les `horizon` barres suivantes
              = log(mid_{t+h}) - log(mid_t).
    persist = baseline momentum naïve = return passé sur `horizon` barres
              = log(mid_t) - log(mid_{t-h})  (causal, connu à t).
              Pour horizon=1, persist == ret_lag_1 == r_t.
    """
    b = bars
    mid = (b["ask_price_1"] + b["bid_price_1"]) / 2.0
    mlog = np.log(mid)
    r = mlog.diff()  # r_t, réalisé à t

    feats = pd.DataFrame(index=b.index)
    feats["rel_spread"] = (b["ask_price_1"] - b["bid_price_1"]) / mid

    a1, b1 = b["ask_price_1"], b["bid_price_1"]
    as1, bs1 = b["ask_size_1"], b["bid_size_1"]
    micro = (a1 * bs1 + b1 * as1) / (as1 + bs1 + EPS)
    feats["micro_price_dev"] = (micro - mid) / mid

    for i in range(1, levels + 1):
        asz, bsz = b[f"ask_size_{i}"], b[f"bid_size_{i}"]
        feats[f"imb_{i}"] = (bsz - asz) / (bsz + asz + EPS)

    bid_depth = sum(b[f"bid_size_{i}"] for i in range(1, levels + 1))
    ask_depth = sum(b[f"ask_size_{i}"] for i in range(1, levels + 1))
    feats["depth_imb"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + EPS)

    for k in lags:
        feats[f"ret_lag_{k}"] = r.shift(k - 1)  # ret_lag_1 = r_t (causal)

    if use_ofi:
        feats["ofi"] = _ofi_best_level(b)

    feats["target"] = mlog.shift(-horizon) - mlog        # rendement futur sur h barres
    feats["__persist"] = mlog - mlog.shift(horizon)      # rendement passé sur h barres (baseline)
    feats = feats.dropna()
    y = feats.pop("target")
    persist = feats.pop("__persist")
    return feats, y, persist
