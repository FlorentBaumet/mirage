"""Microstructure crypto (carnet Bybit L2) : Phase 1a (état) + Phase 1b (impact).

Rejoue, sur le carnet crypto reconstruit (barres 1 s cachées en .pkl), exactement la
même analyse que sur LOBSTER — sans toucher au code d'analyse :
  - Phase 1a : world model d'état (R²_OOS 1-step par dim + rollout du rendement cumulé) ;
  - Phase 1b : courbes de coût d'exécution (slippage/impact vs taille).

    python scripts/crypto_lob.py --out experiments
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mirage.impact import cost_curve  # noqa: E402
from mirage.splits import walk_forward_splits  # noqa: E402
from mirage.state import RET_IDX, STATE_COLS, build_state  # noqa: E402
from mirage.wm import (LinearWM, MeanWM, MLPWM, PersistenceWM,  # noqa: E402
                       make_supervised, rollout)
from mirage.wm_eval import _r2_cols  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DIR = os.path.join("data", "raw", "crypto_lob")
LOOKBACK, NFOLDS, MINTRAIN = 16, 5, 0.4
HORIZONS, MAXH, MAXSTARTS = [1, 2, 3, 5, 10, 20, 30], 30, 2000
KS = [0.25, 0.5, 1, 2, 4, 8]


def load_cached():
    out = {}
    for pf in sorted(glob.glob(os.path.join(DIR, "*_1s_book.pkl"))):
        sym = re.match(r"\d{4}-\d{2}-\d{2}_([A-Z]+)_1s_book", os.path.basename(pf)).group(1)
        out.setdefault(sym, pf)
    return out


def phase1a(sym, bars):
    S, _ = build_state(bars)
    X, Y, pos, d = make_supervised(S.values, LOOKBACK)
    embargo = max(10, LOOKBACK)
    rows1, rowsR = [], []
    for tr, te in walk_forward_splits(len(X), NFOLDS, embargo, MINTRAIN, "expanding"):
        Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
        models = {"persistence": PersistenceWM().fit(Xtr, Ytr),
                  "mean": MeanWM().fit(Xtr, Ytr),
                  "linear": LinearWM().fit(Xtr, Ytr),
                  "mlp": MLPWM().fit(Xtr, Ytr)}
        base = Xte[:, -d:].copy()
        base[:, RET_IDX] = 0.0
        for name in ("mean", "linear", "mlp"):
            r2 = _r2_cols(Yte, models[name].predict(Xte), base)
            for c, dim in enumerate(STATE_COLS):
                rows1.append(dict(symbol=sym, model=name, dim=dim, r2=r2[c]))
        a, b = int(te[0]), int(te[-1]) + 1
        starts = np.arange(a, b - MAXH)
        if len(starts) > MAXSTARTS:
            starts = starts[np.linspace(0, len(starts) - 1, MAXSTARTS).astype(int)]
        if len(starts) == 0:
            continue
        win = X[starts].reshape(-1, LOOKBACK, d)
        hidx = starts[:, None] + np.arange(MAXH)[None, :]
        ac = np.cumsum(Y[hidx, RET_IDX], axis=1)
        for name in ("persistence", "linear", "mlp"):
            pc = np.cumsum(rollout(models[name], win, MAXH)[:, :, RET_IDX], axis=1)
            for h in HORIZONS:
                sse_b = np.sum(ac[:, h - 1] ** 2)
                r2 = np.nan if sse_b == 0 else 1 - np.sum((ac[:, h - 1] - pc[:, h - 1]) ** 2) / sse_b
                rowsR.append(dict(symbol=sym, model=name, horizon=h, r2=r2))
    return rows1, rowsR


SPREAD_IDX = STATE_COLS.index("spread_rel")


def economic_check(sym, bars, fees_bp=(0.0, 2.0, 5.5)):
    """LE verdict edge-vs-mirage : un R²_OOS positif survit-il aux frais ?

    Stratégie naïve : position = signe de la prédiction (linéaire OOS) du return
    next-step. Coût à chaque changement de position = demi-spread (réel, mesuré) +
    frais taker. On reporte gross vs net par barre, pour plusieurs niveaux de frais.
    """
    S, _ = build_state(bars)
    X, Y, pos, d = make_supervised(S.values, LOOKBACK)
    embargo = max(10, LOOKBACK)
    P, A, SP = [], [], []
    for tr, te in walk_forward_splits(len(X), NFOLDS, embargo, MINTRAIN, "expanding"):
        m = LinearWM().fit(X[tr], Y[tr])
        P.append(m.predict(X[te])[:, RET_IDX])
        A.append(Y[te][:, RET_IDX])
        SP.append(S.values[pos[te] - 1, SPREAD_IDX])    # spread_rel à la barre de décision
    pred, y, spread = np.concatenate(P), np.concatenate(A), np.concatenate(SP)
    posn = np.sign(pred)
    gross = posn * y
    dpos = np.abs(np.diff(posn, prepend=0))             # 0 ou 2 à chaque flip
    half = spread / 2.0
    rows = []
    for fee in fees_bp:
        net = gross - dpos * (half + fee * 1e-4)
        rows.append(dict(symbol=sym, fee_bp=fee,
                         gross_bp=round(float(gross.mean()) * 1e4, 4),
                         turnover=round(float(dpos.mean()) / 2, 3),
                         net_bp=round(float(net.mean()) * 1e4, 4),
                         net_cumul_pct=round(float(net.sum()) * 100, 3)))
    return pd.DataFrame(rows), dict(gross=gross, dpos=dpos, half=half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = load_cached()
    if not cache:
        raise SystemExit("Aucun .pkl dans data/raw/crypto_lob — lance d'abord "
                         "scripts/build_bybit_bars.py")
    print("Symboles :", list(cache))

    all1, allR, costs = [], [], {}
    for sym, pf in cache.items():
        bars = pd.read_pickle(pf)
        r1, rR = phase1a(sym, bars)
        all1 += r1
        allR += rR
        costs[sym] = cost_curve(bars, KS, side=1, levels=10)

    res1, resR = pd.DataFrame(all1), pd.DataFrame(allR)

    print("\n=== Phase 1a — R²_OOS 1-step par dimension (baseline: 0 pour ret, no-change sinon) ===")
    print(res1.pivot_table(index="dim", columns="model", values="r2", aggfunc="mean")
          .reindex(STATE_COLS).round(5).to_string())

    print("\n=== Phase 1a — rollout : R²_OOS rendement cumulé vs random walk ===")
    pivR = resR.pivot_table(index="horizon", columns="model", values="r2", aggfunc="mean")
    print(pivR.round(5).to_string())

    print("\n=== Phase 1b — coût d'exécution (achat), slippage en bp vs taille ===")
    for sym, c in costs.items():
        print(f"\n[{sym}]")
        print(c.to_string(index=False))

    # figures
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sym, c in costs.items():
        ax.plot(c["k_x_L1"], c["slippage_bp"], marker="o", label=sym)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("taille (k × meilleur niveau)")
    ax.set_ylabel("slippage moyen (bp)")
    ax.set_title("Crypto (Bybit L2) — coût d'exécution vs taille")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_cost.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in pivR.columns:
        ax.plot(pivR.index, pivR[m], marker="o", label=m)
    ax.axhline(0, color="grey", ls="--", lw=1)
    learned = [m for m in pivR.columns if m != "persistence"]
    ax.set_ylim(float(pivR[learned].min().min()) * 1.4 - 0.02,
                max(0.05, float(pivR[learned].max().max()) * 1.2))
    ax.set_xlabel("horizon de rollout (s)")
    ax.set_ylabel("R²_OOS rendement cumulé")
    ax.set_title("Crypto (Bybit L2) — world model d'état, rollout vs random walk")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_rollout.png"), dpi=130)
    plt.close(fig)

    print("\n=== VERDICT économique — l'edge survit-il aux frais ? (stratégie signe, net de coûts) ===")
    econ_rows, detail = [], {}
    for sym, pf in cache.items():
        df, det = economic_check(sym, pd.read_pickle(pf))
        econ_rows.append(df)
        detail[sym] = det
    econ = pd.concat(econ_rows, ignore_index=True)
    print(econ.to_string(index=False))
    econ.to_csv(os.path.join(args.out, "crypto_lob_economic.csv"), index=False)
    verdict = "EDGE (net>0 à frais réalistes)" if (econ[econ.fee_bp >= 2.0]["net_bp"] > 0).any() \
        else "MIRAGE (net<=0 dès des frais réalistes)"
    print(f"\n-> {verdict}")

    # LA figure : gross (mirage) vs net de frais — l'écart EST le mirage
    sym0 = next(iter(detail))
    det = detail[sym0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fee, lbl in [(0.0, "sans frais (gross)"), (2.0, "net frais 2 bp"),
                     (5.5, "net frais 5.5 bp (taker Bybit)")]:
        net = det["gross"] - det["dpos"] * (det["half"] + fee * 1e-4)
        ax.plot(np.cumsum(net) * 100, label=lbl)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("barres de test (1 s)")
    ax.set_ylabel("PnL cumulé (%)")
    ax.set_title(f"Crypto {sym0} — un edge réel qui est un mirage net de frais")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_mirage.png"), dpi=130)
    plt.close(fig)

    res1.to_csv(os.path.join(args.out, "crypto_lob_1step.csv"), index=False)
    resR.to_csv(os.path.join(args.out, "crypto_lob_rollout.csv"), index=False)
    print(f"\nFigures + CSV dans {args.out}/")


if __name__ == "__main__":
    main()
