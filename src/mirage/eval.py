"""Boucle d'évaluation Phase 0 : walk-forward purgé, MULTI-TICKERS, Go/No-Go.

Chaque ticker est évalué dans SON propre temps (walk-forward intraday). On agrège
ensuite : un edge crédible doit battre les baselines de façon COHÉRENTE d'un ticker
à l'autre (cross-sectional), pas sur un seul coup de chance.

Usage :
    python -m mirage.eval --config configs/phase0.yaml
"""
from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from .data.bars import to_clock_bars
from .data.lobster import load_sample
from .features import build_features
from .metrics import directional_accuracy, mae, r2_oos, rmse
from .models import make_estimators
from .splits import walk_forward_splits


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def instrument_paths(cfg: dict, inst: dict) -> tuple[str, str]:
    d = cfg["data"]
    tag = f"{inst['ticker']}_{d['date']}_{inst['start_ms']}_{inst['end_ms']}"
    lvl = inst["file_level"]
    base = d["raw_dir"]
    return (
        os.path.join(base, f"{tag}_message_{lvl}.csv"),
        os.path.join(base, f"{tag}_orderbook_{lvl}.csv"),
    )


def prepare_instrument(cfg: dict, inst: dict):
    msg_path, book_path = instrument_paths(cfg, inst)
    for p in (msg_path, book_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Données absentes : {p}\n"
                "→ déposer le sample LOBSTER dans data/raw/ (voir data/README.md)"
            )
    d = cfg["data"]
    book = load_sample(msg_path, book_path, d["levels"],
                       d.get("session_start_s"), d.get("session_end_s"))
    bars = to_clock_bars(book, cfg["bars"]["freq"], d["date"])
    X, y, persist = build_features(
        bars,
        levels=cfg["features"]["levels"],
        lags=tuple(cfg["features"]["lags"]),
        use_ofi=cfg["features"]["ofi"],
        horizon=int(cfg["target"]["horizon"]),
    )
    return X, y, persist


def standardize(train: pd.DataFrame, test: pd.DataFrame):
    """Scaler fit sur TRAIN seulement (anti-fuite I3)."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0).replace(0.0, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def _score(ticker, fold, name, y, pred) -> dict:
    return dict(
        ticker=ticker, fold=fold, estimator=name,
        r2_oos=r2_oos(y, pred), rmse=rmse(y, pred),
        mae=mae(y, pred), dir_acc=directional_accuracy(y, pred),
    )


def run_phase0(cfg: dict):
    sp = cfg["split"]
    horizon = int(cfg["target"]["horizon"])
    # ANTI-FUITE : une cible qui regarde h barres en avant exige un embargo >= h,
    # sinon le dernier label du train empiète sur le test.
    embargo = max(int(sp["embargo"]), horizon)

    rows, meta = [], []
    for inst in cfg["data"]["instruments"]:
        ticker = inst["ticker"]
        X, y, persist = prepare_instrument(cfg, inst)
        n = len(X)
        folds = list(walk_forward_splits(
            n, sp["n_folds"], embargo, sp["min_train_frac"], sp["scheme"]
        ))
        meta.append(dict(ticker=ticker, n_bars=n, n_folds=len(folds),
                         horizon=horizon, embargo=embargo))
        if not folds:
            print(f"[!] {ticker}: trop peu de barres ({n}) → ignoré")
            continue

        for fi, (tr, te) in enumerate(folds):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()

            if sp.get("standardize", True):
                Xtr_s, Xte_s = standardize(Xtr, Xte)
            else:
                Xtr_s, Xte_s = Xtr, Xte
            Xtr_a, Xte_a = Xtr_s.to_numpy(), Xte_s.to_numpy()

            est, _ = make_estimators(cfg["model"], cfg["baselines"])
            for name, model in est.items():
                model.fit(Xtr_a, ytr)
                rows.append(_score(ticker, fi, name, yte, model.predict(Xte_a)))

            if "persistence" in cfg["baselines"]:
                rows.append(_score(ticker, fi, "persistence", yte,
                                   persist.iloc[te].to_numpy()))

    return pd.DataFrame(rows), pd.DataFrame(meta)


def go_no_go(res: pd.DataFrame, model_name: str) -> str:
    """GO si le modèle bat zero (R2_OOS>0) ET linear EN MOYENNE, et sur la
    MAJORITÉ des tickers. Sinon NO-GO (résultat valide)."""
    per_tk = res.pivot_table(index="ticker", columns="estimator",
                             values="r2_oos", aggfunc="mean")
    pooled = res.groupby("estimator")["r2_oos"].mean()
    m_mean = pooled.get(model_name, float("nan"))
    lin_mean = pooled.get("linear", float("-inf"))

    beats_zero_tk = (per_tk[model_name] > 0)
    beats_lin_tk = (per_tk[model_name] > per_tk.get("linear", 0))
    n_tk = len(per_tk)
    maj_zero = int(beats_zero_tk.sum())
    maj_lin = int(beats_lin_tk.sum())

    beats_zero = m_mean > 0
    beats_linear = m_mean > lin_mean
    majority = (maj_zero > n_tk / 2) and (maj_lin > n_tk / 2)
    verdict = "GO" if (beats_zero and beats_linear and majority) else "NO-GO"

    return (
        f"\n=== Go/No-Go ({model_name}) ===\n"
        f"  R2_OOS moyen (pooled)  modèle  : {m_mean:+.6f}  (bat zero ? {beats_zero})\n"
        f"  R2_OOS moyen (pooled)  linéaire: {lin_mean:+.6f}  (modèle > linéaire ? {beats_linear})\n"
        f"  modèle bat zero    sur {maj_zero}/{n_tk} tickers\n"
        f"  modèle bat linéaire sur {maj_lin}/{n_tk} tickers\n"
        f"  --> VERDICT : {verdict}\n"
        f"  (NO-GO = résultat valide et documenté, pas un échec — I3)\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res, meta = run_phase0(cfg)
    model_name = cfg["model"]["name"]

    pd.set_option("display.width", 140)
    print("\n--- Couverture données ---")
    print(meta.to_string(index=False))

    print("\n--- R2_OOS moyen par ticker x estimateur ---")
    piv = res.pivot_table(index="ticker", columns="estimator",
                          values="r2_oos", aggfunc="mean")
    print(piv.round(6).to_string())

    print("\n--- Agrégat (pooled, moyenne sur tous folds/tickers) ---")
    print(res.groupby("estimator")[["r2_oos", "rmse", "mae", "dir_acc"]]
          .mean().round(6).to_string())

    print(go_no_go(res, model_name))

    if cfg.get("wandb", {}).get("enabled", False):
        import wandb
        wandb.init(project=cfg["wandb"]["project"], config=cfg)
        for _, row in res.iterrows():
            wandb.log({f"{row.ticker}/{row.estimator}/r2_oos": row.r2_oos})
        wandb.finish()


if __name__ == "__main__":
    main()
