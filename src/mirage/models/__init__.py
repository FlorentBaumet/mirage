"""Fabrique d'estimateurs (baselines + world model)."""
from __future__ import annotations

from ..baselines import LinearBaseline, ZeroForecast
from .seq import GRUModel, MLPModel


def make_estimators(model_cfg: dict, baselines: list[str]):
    """Renvoie (dict d'estimateurs standardisés, nom du world model focal).

    'persistence' n'est PAS inclus ici (géré à part dans eval, en features brutes).
    """
    est: dict = {}
    if "zero" in baselines:
        est["zero"] = ZeroForecast()
    if "linear" in baselines:
        est["linear"] = LinearBaseline()

    name = model_cfg.get("name", "mlp")
    if name == "mlp":
        est["mlp"] = MLPModel(
            hidden=tuple(model_cfg.get("hidden", [64, 32])),
            max_iter=int(model_cfg.get("max_iter", 500)),
            seed=int(model_cfg.get("seed", 0)),
            alpha=float(model_cfg.get("alpha", 1e-3)),
        )
    elif name == "ridge":
        est["ridge"] = LinearBaseline()
    elif name == "gru":
        est["gru"] = GRUModel(
            hidden=int(model_cfg.get("hidden_gru", 32)),
            seq_len=int(model_cfg.get("seq_len", 16)),
            epochs=int(model_cfg.get("epochs", 15)),
            seed=int(model_cfg.get("seed", 0)),
        )
    else:
        raise ValueError(f"model.name inconnu : {name}")

    return est, name
