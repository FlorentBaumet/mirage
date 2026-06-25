"""Sanity checks du moteur d'impact mécaniste (Phase 1b)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mirage.impact import execution_cost


def _book():
    # carnet jouet L2 : mid = 100.00, demi-spread = 1 bp
    return pd.DataFrame([{
        "ask_price_1": 100.01, "ask_size_1": 100, "bid_price_1": 99.99, "bid_size_1": 100,
        "ask_price_2": 100.02, "ask_size_2": 200, "bid_price_2": 99.98, "bid_size_2": 200,
    }])


def test_tiny_order_pays_half_spread_no_impact():
    slip, impact, fill = execution_cost(_book(), q=10, side=1, levels=2)
    assert abs(slip[0] - 1e-4) < 1e-9      # exécuté au meilleur ask = demi-spread (1 bp)
    assert abs(impact[0]) < 1e-12          # L1 pas vidé -> mid inchangé
    assert fill == 1.0


def test_eating_a_level_moves_mid():
    slip, impact, _ = execution_cost(_book(), q=100, side=1, levels=2)
    assert impact[0] > 0                   # L1 vidé -> meilleur ask remonte -> mid poussé
    assert abs(slip[0] - 1e-4) < 1e-9      # rempli au L1 -> slippage = demi-spread


def test_slippage_monotonic_in_size():
    s_small = execution_cost(_book(), 10, 1, 2)[0][0]
    s_big = execution_cost(_book(), 150, 1, 2)[0][0]   # mange L1 + une partie de L2
    assert s_big > s_small


def test_oversized_order_not_filled():
    slip, impact, fill = execution_cost(_book(), q=400, side=1, levels=2)  # > 300 visibles
    assert np.isnan(slip[0]) and fill == 0.0
