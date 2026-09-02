"""Tests des règles de frontière de journée (multi-jours carnet crypto)."""
from __future__ import annotations

import numpy as np

from mirage.backtest import intraday_starts, position_changes


def test_position_changes_single_day():
    posn = np.array([1, 1, -1, -1, 0])
    days = np.zeros(5)
    # entrée (1), maintien (0), retournement (2), maintien (0), sortie (1)
    assert list(position_changes(posn, days)) == [1, 0, 2, 0, 1]


def test_position_changes_resets_each_day():
    posn = np.array([1, 1, -1, -1])
    days = np.array([0, 0, 1, 1])       # nouveau jour à l'indice 2
    d = position_changes(posn, days)
    # à la frontière on repart à plat : entrer en -1 coûte 1, pas 2 (faux retournement)
    assert list(d) == [1, 0, 1, 0]


def test_intraday_starts_drops_overnight_windows():
    days = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    starts = np.arange(0, 5)
    kept = intraday_starts(starts, days, horizon=3)
    # start=2 -> couvre 2,3,4 : chevauche la nuit -> exclu ; start=3 -> 3,4,5 -> exclu
    assert list(kept) == [0, 1, 4]


def test_intraday_starts_empty():
    assert len(intraday_starts(np.array([], dtype=int), np.zeros(5), 3)) == 0
