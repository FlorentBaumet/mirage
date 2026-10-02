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
# Dimensions de profondeur (Phase 1e) : elles décrivent le carnet HORS meilleur niveau.
# Ajoutées APRÈS les cinq de base et AVANT l'OFI, pour que RET_IDX reste 0 et que les
# cinq premières dimensions gardent leur sens exact.
DEEP_COLS = ["imb_deep", "slope_asym"]
RET_IDX = 0  # position de 'ret' dans STATE_COLS (utilisé pour le rollout du prix)
OFI_COL = "ofi"


def build_state(bars: pd.DataFrame, levels: int = 10, ofi: pd.Series | None = None,
                deep: bool = False):
    """Renvoie (S, mid) : S = DataFrame des états, mid aligné.

    Si `ofi` (Series alignée sur l'index des barres) est fournie, elle est AJOUTÉE EN
    DERNIÈRE POSITION. Conséquence voulue : RET_IDX reste 0 et les cinq premières
    dimensions gardent leur sens exact, donc tout l'aval (export .npz, bootstrap,
    position = signe de la prédiction de ret) est inchangé.

    Si `deep` est vrai, DEUX colonnes de profondeur (DEEP_COLS) sont insérées après les
    cinq de base et avant l'OFI : `imb_deep` (imbalance des niveaux 2..levels) et
    `slope_asym` (asymétrie de l'empilement au contact). Elles n'introduisent JAMAIS de
    NaN (garde EPS + fillna(0.0)) : le nombre de lignes reste IDENTIQUE à celui du bras
    de base, condition de l'appariement.
    """
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

    cols = STATE_COLS
    if deep:
        bid_deep = sum(bars[f"bid_size_{i}"] for i in range(2, levels + 1))
        ask_deep = sum(bars[f"ask_size_{i}"] for i in range(2, levels + 1))
        S["imb_deep"] = ((bid_deep - ask_deep) / (bid_deep + ask_deep + EPS)).fillna(0.0)
        S["slope_asym"] = ((bs1 / (bid_depth + EPS)) - (as1 / (ask_depth + EPS))).fillna(0.0)
        cols = cols + DEEP_COLS
    if ofi is not None:
        # Une seconde sans evenement touchant le meilleur niveau a un flux NUL, pas une
        # donnee manquante : fillna(0) conserve l'echantillon, donc la comparabilite
        # exacte avec le bras de base (meme nombre de lignes, memes cibles).
        S[OFI_COL] = ofi.reindex(S.index).fillna(0.0).to_numpy()
        cols = cols + [OFI_COL]

    S = S[cols].dropna()
    return S, mid.loc[S.index]
