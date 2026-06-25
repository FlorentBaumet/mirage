"""Overlay d'impact MÉCANISTE (Phase 1b) — action-conditioning sur LOBSTER.

Partie MESURABLE (ancrée dans le vrai carnet) :
  un ordre au marché de taille q « mange » les niveaux du carnet -> on calcule
  exactement le prix d'exécution (VWAP), le slippage, et le saut de mid immédiat.

Partie MODÉLISÉE (pas de contrefactuel sur l'historique -> assumptions documentées) :
  dynamique permanente + temporaire après le trade (décroissance). Voir
  `impact_decay_path`. Sa vraie validation = ABIDES (Stage 2).

Convention : ordre d'ACHAT (side=+1) -> on consomme les ASKS. Vente symétrique.
Tailles, prix, slippage en relatif (fraction du mid) ; *1e4 = bp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def book_arrays(bars: pd.DataFrame, levels: int = 10):
    """Extrait (ask_prices, ask_sizes, bid_prices, bid_sizes, mid) en arrays."""
    ap = bars[[f"ask_price_{i}" for i in range(1, levels + 1)]].to_numpy(float)
    az = bars[[f"ask_size_{i}" for i in range(1, levels + 1)]].to_numpy(float)
    bp = bars[[f"bid_price_{i}" for i in range(1, levels + 1)]].to_numpy(float)
    bz = bars[[f"bid_size_{i}" for i in range(1, levels + 1)]].to_numpy(float)
    mid = (ap[:, 0] + bp[:, 0]) / 2.0
    return ap, az, bp, bz, mid


def _sweep(prices: np.ndarray, sizes: np.ndarray, opp_best: np.ndarray,
           mid: np.ndarray, q: float, side: int):
    """Marche le carnet pour un ordre de q (vectorisé sur les lignes).

    prices/sizes : côté consommé (asks pour un achat), du meilleur au pire.
    opp_best     : meilleur prix du côté opposé (bid1 pour un achat).
    Renvoie (slippage_rel, mid_impact_rel) ; NaN si q > profondeur visible.
    """
    n, L = prices.shape
    ar = np.arange(n)
    cum = np.cumsum(sizes, axis=1)
    cost_cum = np.cumsum(sizes * prices, axis=1)
    reached = cum >= q
    filled = reached.any(axis=1)
    lvl = np.argmax(reached, axis=1)              # premier niveau atteignant q

    prev = lvl - 1
    cost_before = np.where(prev >= 0, cost_cum[ar, np.clip(prev, 0, L - 1)], 0.0)
    size_before = np.where(prev >= 0, cum[ar, np.clip(prev, 0, L - 1)], 0.0)
    partial = q - size_before
    price_lvl = prices[ar, lvl]
    vwap = (cost_before + partial * price_lvl) / q
    slip = side * (vwap - mid) / mid              # >0 = coût

    rem = sizes[ar, lvl] - partial                # reste au niveau partiellement mangé
    new_best = np.where(rem > 0, price_lvl, prices[ar, np.clip(lvl + 1, 0, L - 1)])
    mid_after = (new_best + opp_best) / 2.0
    impact = side * (mid_after - mid) / mid       # >0 = mid poussé dans le sens de l'ordre

    slip = np.where(filled, slip, np.nan)
    impact = np.where(filled, impact, np.nan)
    return slip, impact, float(np.mean(filled))


def execution_cost(bars: pd.DataFrame, q: float, side: int = 1, levels: int = 10):
    """Slippage + impact immédiat pour un ordre de q actions, à chaque barre."""
    ap, az, bp, bz, mid = book_arrays(bars, levels)
    if side > 0:                                  # achat : consomme les asks
        return _sweep(ap, az, bp[:, 0], mid, q, +1)
    return _sweep(bp, bz, ap[:, 0], mid, q, -1)   # vente : consomme les bids


def cost_curve(bars: pd.DataFrame, ks=(0.25, 0.5, 1, 2, 4, 8), side: int = 1,
               levels: int = 10) -> pd.DataFrame:
    """Courbe coût vs taille, q = k × (taille moyenne du meilleur niveau)."""
    ap, az, bp, bz, mid = book_arrays(bars, levels)
    ref = np.nanmean(az[:, 0] if side > 0 else bz[:, 0])  # taille L1 de référence
    rows = []
    for k in ks:
        q = float(k) * ref
        slip, impact, fill = execution_cost(bars, q, side, levels)
        rows.append({
            "k_x_L1": k, "q_shares": round(q, 1),
            "slippage_bp": round(float(np.nanmean(slip)) * 1e4, 3),
            "mid_impact_bp": round(float(np.nanmean(impact)) * 1e4, 3),
            "fill_rate": round(fill, 3),
        })
    return pd.DataFrame(rows)


def impact_decay_path(immediate: float, horizon: int,
                      permanent_frac: float = 0.3, decay: float = 0.7) -> np.ndarray:
    """ASSUMPTION (non testable sur historique) : trajectoire de l'impact sur le mid
    après un trade. Part PERMANENTE (reste) + part TEMPORAIRE (décroît géométriquement).

        impact_h = immediate * [ permanent_frac + (1 - permanent_frac) * decay**h ]

    permanent_frac et decay sont des hypothèses -> à calibrer/valider en ABIDES.
    """
    h = np.arange(horizon)
    return immediate * (permanent_frac + (1.0 - permanent_frac) * decay ** h)
