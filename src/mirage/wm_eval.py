"""Éval Phase 1a : world model d'état passif, walk-forward purgé.

Deux mesures :
  1) 1-step, R²_OOS PAR DIMENSION d'état vs persistence (le WM bat-il « rien ne
     change » sur chaque composante de l'état ?) ;
  2) ROLLOUT autorégressif : R²_OOS du rendement CUMULÉ vs random walk (= 0), en
     fonction de l'horizon -> jusqu'où le world model prédit avant de retomber.

    python -m mirage.wm_eval --config configs/phase1.yaml --out experiments
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .data.bars import to_clock_bars
from .data.lobster import load_sample
from .eval import instrument_paths, load_config
from .splits import walk_forward_splits
from .state import RET_IDX, STATE_COLS, build_state
from .wm import LinearWM, MeanWM, MLPWM, PersistenceWM, make_model, make_supervised, rollout

warnings.filterwarnings("ignore", category=ConvergenceWarning)
matplotlib_ok = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib_ok = False


def _prep(cfg, inst):
    d = cfg["data"]
    msg, book = instrument_paths(cfg, inst)
    b = load_sample(msg, book, d["levels"], d.get("session_start_s"), d.get("session_end_s"))
    bars = to_clock_bars(b, cfg["bars"]["freq"], d["date"])
    S, _ = build_state(bars, levels=d["levels"])
    return make_supervised(S.values, int(cfg["state"]["lookback"]))


def _r2_cols(Y, pred, base):
    """R²_OOS par colonne vs `base` (persistence)."""
    sse_m = np.sum((Y - pred) ** 2, axis=0)
    sse_b = np.sum((Y - base) ** 2, axis=0)
    return 1.0 - sse_m / np.where(sse_b == 0, np.nan, sse_b)


def run(cfg):
    lookback = int(cfg["state"]["lookback"])
    sp = cfg["split"]
    embargo = max(int(sp["embargo"]), lookback)  # anti-fuite : fenêtres qui se chevauchent
    horizons = cfg["rollout"]["horizons"]
    maxH = max(horizons)
    max_starts = int(cfg["rollout"]["max_starts"])
    model_name = cfg["model"]["name"]

    rows1, rowsR = [], []
    for inst in cfg["data"]["instruments"]:
        tk = inst["ticker"]
        X, Y, pos, d = _prep(cfg, inst)
        for fi, (tr, te) in enumerate(walk_forward_splits(
                len(X), sp["n_folds"], embargo, sp["min_train_frac"], sp["scheme"])):
            Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]

            models = {
                "persistence": PersistenceWM().fit(Xtr, Ytr),
                "mean": MeanWM().fit(Xtr, Ytr),
                "linear": LinearWM().fit(Xtr, Ytr),
            }
            if model_name not in models:
                models[model_name] = make_model(model_name, cfg["model"]).fit(Xtr, Ytr)

            # --- 1-step, par dimension, vs baseline naïve appropriée ---
            # ret = un CHANGEMENT -> baseline = 0 (random walk), comme Phase 0.
            # dims de forme (spread, imbalance...) = des NIVEAUX -> baseline = no-change.
            base = Xte[:, -d:].copy()
            base[:, RET_IDX] = 0.0
            for name in ("mean", "linear", model_name):
                r2 = _r2_cols(Yte, models[name].predict(Xte), base)
                for c, dim in enumerate(STATE_COLS):
                    rows1.append(dict(ticker=tk, fold=fi, model=name, dim=dim, r2_oos=r2[c]))

            # --- rollout, rendement cumulé, vs random walk ---
            a, b = int(te[0]), int(te[-1]) + 1
            starts = np.arange(a, b - maxH)
            if len(starts) == 0:
                continue
            if len(starts) > max_starts:
                starts = starts[np.linspace(0, len(starts) - 1, max_starts).astype(int)]
            windows = X[starts].reshape(-1, lookback, d)
            hidx = starts[:, None] + np.arange(maxH)[None, :]
            actual_cum = np.cumsum(Y[hidx, RET_IDX], axis=1)  # (m, maxH)

            for name in ("persistence", "linear", model_name):
                preds = rollout(models[name], windows, maxH)
                pred_cum = np.cumsum(preds[:, :, RET_IDX], axis=1)
                for h in horizons:
                    ac, pc = actual_cum[:, h - 1], pred_cum[:, h - 1]
                    sse_b = np.sum(ac ** 2)
                    r2 = np.nan if sse_b == 0 else 1.0 - np.sum((ac - pc) ** 2) / sse_b
                    rowsR.append(dict(ticker=tk, fold=fi, model=name, horizon=h, r2_oos_cumret=r2))

    return pd.DataFrame(rows1), pd.DataFrame(rowsR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    model_name = cfg["model"]["name"]

    res1, resR = run(cfg)

    pd.set_option("display.width", 160)
    print("=== 1-step : R²_OOS par dimension (pooled) ===")
    print("    (baseline : random-walk=0 pour 'ret' ; no-change pour les niveaux)")
    piv = res1.pivot_table(index="dim", columns="model", values="r2_oos", aggfunc="mean")
    piv = piv.reindex(STATE_COLS)
    print(piv.round(5).to_string())
    res1.to_csv(os.path.join(args.out, "phase1_1step.csv"), index=False)

    print("\n=== Rollout : R²_OOS du rendement cumulé vs random walk (pooled) ===")
    pivR = resR.pivot_table(index="horizon", columns="model", values="r2_oos_cumret", aggfunc="mean")
    print(pivR.round(5).to_string())
    resR.to_csv(os.path.join(args.out, "phase1_rollout.csv"), index=False)

    if matplotlib_ok:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in pivR.columns:
            ax.plot(pivR.index, pivR[m], marker="o", label=m)
        ax.axhline(0, color="grey", ls="--", lw=1, label="random walk (=0)")
        # zoom sur la zone utile : persistence (catastrophique) écraserait l'échelle
        learned = [m for m in pivR.columns if m != "persistence"]
        lo = float(pivR[learned].min().min()) * 1.4 - 0.02
        ax.set_ylim(lo, 0.05)
        if "persistence" in pivR.columns:
            worst = float(pivR["persistence"].min())
            ax.text(0.03, 0.06, f"persistence : hors échelle (→ ~{worst:.0f} à 30 s)",
                    transform=ax.transAxes, fontsize=8, color="green")
        ax.set_xlabel("horizon de rollout (secondes)")
        ax.set_ylabel("R²_OOS rendement cumulé")
        ax.set_title("World model d'état — rollout vs random walk")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(args.out, "phase1_rollout.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"\nFigure : {path}")

    # synthèse honnête
    wm_1s, lin_1s = piv[model_name], piv["linear"]
    beats = [dim for dim in STATE_COLS if wm_1s.get(dim, float("nan")) > 0]
    beats_lin = [dim for dim in STATE_COLS if lin_1s.get(dim, float("nan")) > 0]
    print(f"\n1-step : le WM ({model_name}) bat la baseline sur : {beats or 'aucune dimension'}")
    print(f"1-step : le linéaire bat la baseline sur : {beats_lin or 'aucune dimension'}")
    roll_wm = pivR[model_name]
    pos_h = [h for h in roll_wm.index if roll_wm[h] > 0]
    print(f"Rollout : R²_OOS cumret > 0 jusqu'à l'horizon {max(pos_h) if pos_h else 0}s "
          f"(au-delà, le world model ne bat plus le random walk).")


if __name__ == "__main__":
    main()
