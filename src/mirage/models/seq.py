"""World models 'séquence simple' pour la Phase 0.

- MLPModel : sklearn (défaut, zéro dépendance lourde). Les features encodent déjà
  les lags, donc un MLP tabulaire est un modèle de séquence simple et honnête.
- GRUModel : petit GRU (torch, optionnel). Construit des séquences glissantes par
  bloc. Import paresseux : ne casse rien si torch n'est pas installé.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class MLPModel:
    """MLP avec NORMALISATION DE LA CIBLE.

    Indispensable ici : les rendements à 1 s sont ~O(1e-4). Sans standardiser y,
    un MLP (ReLU + Adam) diverge et sort des prédictions énormes (R²_OOS très
    négatif). On entraîne sur (y - mu_y)/sd_y puis on inverse à la prédiction.
    mu_y/sd_y sont estimés sur le TRAIN seulement (causal).
    """

    def __init__(self, hidden=(64, 32), max_iter: int = 500, seed: int = 0,
                 alpha: float = 1e-3):
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden),
            max_iter=max_iter,
            random_state=seed,
            alpha=alpha,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15,  # val = fin du train (causal car train ordonné)
        )
        self.y_mu = 0.0
        self.y_sd = 1.0

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.y_mu = float(y.mean())
        self.y_sd = float(y.std()) or 1.0
        self.model.fit(X, (y - self.y_mu) / self.y_sd)
        return self

    def predict(self, X):
        return self.model.predict(X) * self.y_sd + self.y_mu


class GRUModel:
    """Petit GRU (torch). Optionnel : `pip install -e ".[torch]"`."""

    def __init__(self, seq_len: int = 16, hidden: int = 32, epochs: int = 15,
                 lr: float = 1e-3, batch: int = 256, seed: int = 0):
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed
        self.net = None
        self.y_mu = 0.0
        self.y_sd = 1.0

    @staticmethod
    def _lazy_torch():
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'GRU nécessite torch : pip install -e ".[torch]"'
            ) from e
        import torch
        return torch

    def _make_seqs(self, X: np.ndarray) -> np.ndarray:
        """(n, f) -> (n, seq_len, f). Padding gauche par répétition de la 1re ligne.

        Causal : la séquence du point t n'utilise que les lignes <= t.
        """
        n, f = X.shape
        pad = np.repeat(X[:1], self.seq_len - 1, axis=0)
        Xp = np.vstack([pad, X])
        idx = np.arange(self.seq_len)[None, :] + np.arange(n)[:, None]
        return Xp[idx]  # (n, seq_len, f)

    def fit(self, X, y):
        torch = self._lazy_torch()
        import torch.nn as nn

        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype=float)
        self.y_mu = float(y.mean())
        self.y_sd = float(y.std()) or 1.0
        y = ((y - self.y_mu) / self.y_sd).astype("float32").reshape(-1, 1)  # cible normalisée
        seqs = torch.from_numpy(self._make_seqs(X))
        yt = torch.from_numpy(y)
        f = X.shape[1]

        class _Net(nn.Module):
            def __init__(self, f, h):
                super().__init__()
                self.gru = nn.GRU(f, h, batch_first=True)
                self.head = nn.Linear(h, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :])

        self.net = _Net(f, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = seqs.shape[0]
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                bidx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(seqs[bidx]), yt[bidx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        torch = self._lazy_torch()
        X = np.asarray(X, dtype="float32")
        seqs = torch.from_numpy(self._make_seqs(X))
        self.net.eval()
        with torch.no_grad():
            pred = self.net(seqs).numpy().ravel()
        return pred * self.y_sd + self.y_mu  # inverse de la normalisation cible
