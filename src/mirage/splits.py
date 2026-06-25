"""Walk-forward purgé avec embargo — la colonne vertébrale (I3).

On n'utilise JAMAIS de shuffle. Train = passé, test = futur. Entre la fin du
train et le début du test on laisse un trou (`embargo`) >= à la fenêtre de
lookback des features, pour tuer la fuite aux frontières (purged walk-forward,
esprit López de Prado).
"""
from __future__ import annotations

import numpy as np


def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    embargo: int = 10,
    min_train_frac: float = 0.4,
    scheme: str = "expanding",
):
    """Génère (train_idx, test_idx) sur indices [0, n).

    - scheme='expanding' : train = [0, train_end)  (tout le passé).
    - scheme='rolling'   : train = fenêtre de taille fixe (= taille du 1er train).
    - train_end = test_start - embargo  (purge).
    """
    if not 0 < min_train_frac < 1:
        raise ValueError("min_train_frac doit être dans (0, 1).")
    start = int(n * min_train_frac)
    if start < 1 or start >= n:
        raise ValueError(f"min_train_frac invalide pour n={n}.")

    edges = np.linspace(start, n, n_folds + 1, dtype=int)
    window = start  # taille fenêtre rolling

    for k in range(n_folds):
        test_start, test_end = int(edges[k]), int(edges[k + 1])
        train_end = test_start - embargo
        if train_end <= 0 or test_end <= test_start:
            continue
        train_start = 0 if scheme == "expanding" else max(0, train_end - window)
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx
