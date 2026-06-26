"""Éval Phase 0 CRYPTO : walk-forward purgé sur des mois de klines, multi-symboles.

Même colonne vertébrale que l'éval LOBSTER (splits purgés, scaler train-only,
baselines, R²_OOS pré-enregistré) — mais sur des MOIS de données, donc le
walk-forward est inter-périodes (robustesse temporelle réelle, pas un seul jour).

    python -m mirage.crypto_eval --config configs/phase0_crypto.yaml --out experiments
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .crypto_features import build_features
from .data.crypto import load_symbol
from .eval import load_config, standardize
from .metrics import directional_accuracy, mae, r2_oos, rmse
from .models import make_estimators
from .splits import walk_forward_splits

warnings.filterwarnings("ignore", category=ConvergenceWarning)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except Exception:  # pragma: no cover
    MPL = False


def _score(symbol, fold, name, y, pred) -> dict:
    return dict(symbol=symbol, fold=fold, estimator=name,
               r2_oos=r2_oos(y, pred), rmse=rmse(y, pred), mae=mae(y, pred),
               dir_acc=directional_accuracy(y, pred),
               gross_bp=float(np.mean(np.sign(pred) * y)) * 1e4)


def run(cfg):
    sp = cfg["split"]
    horizon = int(cfg["target"]["horizon"])
    vol_window = int(cfg["features"]["vol_window"])
    embargo = max(int(sp["embargo"]), horizon, vol_window)

    rows, meta = [], []
    for sym in cfg["data"]["symbols"]:
        bars = load_symbol(sym, cfg["data"]["interval"], cfg["data"]["raw_dir"])
        X, y, persist = build_features(bars, tuple(cfg["features"]["lags"]),
                                       vol_window, horizon)
        n = len(X)
        folds = list(walk_forward_splits(n, sp["n_folds"], embargo,
                                         sp["min_train_frac"], sp["scheme"]))
        meta.append(dict(symbol=sym, bars=n, folds=len(folds),
                         span=f"{X.index[0].date()}..{X.index[-1].date()}"))
        for fi, (tr, te) in enumerate(folds):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()
            if sp.get("standardize", True):
                Xtr, Xte = standardize(Xtr, Xte)
            Xtr_a, Xte_a = Xtr.to_numpy(), Xte.to_numpy()

            est, _ = make_estimators(cfg["model"], cfg["baselines"])
            for name, model in est.items():
                model.fit(Xtr_a, ytr)
                rows.append(_score(sym, fi, name, yte, model.predict(Xte_a)))
            if "persistence" in cfg["baselines"]:
                rows.append(_score(sym, fi, "persistence", yte,
                                   persist.iloc[te].to_numpy()))
    return pd.DataFrame(rows), pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_crypto.yaml")
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    model_name = cfg["model"]["name"]
    fee = float(cfg["costs"]["taker_fee_bp"])

    res, meta = run(cfg)
    pd.set_option("display.width", 160)
    print("--- Couverture ---")
    print(meta.to_string(index=False))

    print("\n--- R2_OOS moyen par symbole x estimateur (pooled folds) ---")
    piv = res.pivot_table(index="symbol", columns="estimator", values="r2_oos", aggfunc="mean")
    print(piv.round(6).to_string())

    print("\n--- Agrégat (pooled) ---")
    print(res.groupby("estimator")[["r2_oos", "dir_acc", "gross_bp"]].mean().round(5).to_string())
    res.to_csv(os.path.join(args.out, "crypto_phase0.csv"), index=False)

    g = res[res.estimator == model_name]["gross_bp"].mean()
    print(f"\nCheck couts : edge brut moyen modele = {g:.3f} bp/barre vs frais taker "
          f"{fee} bp -> {'POSITIF net' if g > fee else 'NEGATIF net (mange par les frais)'}")

    if MPL:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for sym in cfg["data"]["symbols"]:
            d = res[(res.symbol == sym) & (res.estimator == model_name)]
            ax.plot(d["fold"], d["r2_oos"], marker="o", label=sym)
        ax.axhline(0, color="grey", ls="--", lw=1, label="random walk (=0)")
        ax.set_xlabel("fold walk-forward (≈ période, 2024)")
        ax.set_ylabel("R²_OOS (modèle)")
        ax.set_title("Crypto 1 min — R²_OOS par période (robustesse temporelle)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.out, "crypto_phase0_r2.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"Figure : {p}")


if __name__ == "__main__":
    main()
