"""Helpers de backtest conscients des frontières de jours.

Quand on concatène plusieurs journées, deux pièges silencieux fausseraient le verdict :
  1. compter un « retournement » de position entre la dernière barre du jour J et la
     première du jour J+1 (alors qu'on repart à plat chaque jour) ;
  2. laisser un rollout démarrer en fin de journée et « prédire » par-dessus la nuit.
Ces deux fonctions isolent ces règles pour qu'elles soient testables.
"""
from __future__ import annotations

import numpy as np


def position_changes(posn: np.ndarray, days: np.ndarray) -> np.ndarray:
    """|Δposition| avec remise à plat au début de chaque journée.

    posn : positions (typiquement -1/0/+1) ; days : identifiant de journée par barre.
    À la première barre d'un jour, on entre depuis une position nulle -> le coût est
    celui d'une entrée simple, pas d'un retournement.
    """
    posn = np.asarray(posn, float)
    days = np.asarray(days)
    prev = np.concatenate([[0.0], posn[:-1]])
    day_start = np.concatenate([[True], days[1:] != days[:-1]])
    prev[day_start] = 0.0
    return np.abs(posn - prev)


def intraday_starts(starts: np.ndarray, days: np.ndarray, horizon: int) -> np.ndarray:
    """Ne garde que les points de départ dont tout l'horizon tient dans la même journée."""
    starts = np.asarray(starts)
    if len(starts) == 0:
        return starts
    return starts[days[starts] == days[starts + horizon - 1]]
