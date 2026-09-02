"""Microstructure crypto (carnet Bybit L2) : Phase 1a (état) + Phase 1b (impact).

MULTI-JOURS : chaque symbole agrège plusieurs journées. Rigueur temporelle :
  - l'état et les fenêtres sont construits PAR JOUR -> aucune fenêtre à cheval sur
    deux jours, aucun rendement calculé par-dessus une nuit ;
  - les jours sont ensuite concaténés dans l'ordre -> le walk-forward purgé devient
    INTER-JOURS (train sur les jours passés, test sur les jours futurs) ;
  - le rollout ne démarre que s'il tient dans la même journée ;
  - le backtest repart à plat chaque jour (pas de faux retournement à la frontière).

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

from mirage.backtest import intraday_starts, position_changes  # noqa: E402
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
SPREAD_IDX = STATE_COLS.index("spread_rel")


def load_cached() -> dict[str, list[str]]:
    """{symbole: [chemins .pkl triés par date]}"""
    out: dict[str, list[str]] = {}
    for pf in sorted(glob.glob(os.path.join(DIR, "*_1s_book.pkl"))):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)_1s_book", os.path.basename(pf))
        out.setdefault(m.group(2), []).append(pf)
    return {k: sorted(v) for k, v in out.items()}


def build_symbol(pkls: list[str]):
    """Construit les échantillons multi-jours d'un symbole (par jour, puis concaténés)."""
    Xs, Ys, days, spr, costs = [], [], [], [], []
    d = None
    for di, pf in enumerate(pkls):
        bars = pd.read_pickle(pf)
        costs.append(cost_curve(bars, KS, side=1, levels=10))
        S, _ = build_state(bars)
        X, Y, pos, d = make_supervised(S.values, LOOKBACK)   # fenêtres internes au jour
        Xs.append(X)
        Ys.append(Y)
        days.append(np.full(len(X), di))
        spr.append(S.values[pos - 1, SPREAD_IDX])            # spread à la barre de décision
        del bars, S
    cost = pd.concat(costs).groupby("k_x_L1", as_index=False).mean(numeric_only=True)
    return (np.vstack(Xs), np.vstack(Ys), np.concatenate(days),
            np.concatenate(spr), d, cost)


def phase1a(sym, X, Y, days, d):
    rows1, rowsR = [], []
    embargo = max(10, LOOKBACK)
    for fi, (tr, te) in enumerate(walk_forward_splits(len(X), NFOLDS, embargo,
                                                      MINTRAIN, "expanding")):
        Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
        models = {"persistence": PersistenceWM().fit(Xtr, Ytr),
                  "mean": MeanWM().fit(Xtr, Ytr),
                  "linear": LinearWM().fit(Xtr, Ytr),
                  "mlp": MLPWM().fit(Xtr, Ytr)}
        base = Xte[:, -d:].copy()
        base[:, RET_IDX] = 0.0                     # ret : baseline = random walk
        for name in ("mean", "linear", "mlp"):
            r2 = _r2_cols(Yte, models[name].predict(Xte), base)
            for c, dim in enumerate(STATE_COLS):
                rows1.append(dict(symbol=sym, fold=fi, model=name, dim=dim, r2=r2[c]))

        a, b = int(te[0]), int(te[-1]) + 1
        starts = intraday_starts(np.arange(a, b - MAXH), days, MAXH)   # rollout intra-jour
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
                rowsR.append(dict(symbol=sym, fold=fi, model=name, horizon=h, r2=r2))
    return rows1, rowsR


def economic_check(sym, X, Y, days, spread, fees_bp=(0.0, 2.0, 5.5)):
    """LE verdict edge-vs-mirage : le R²_OOS positif survit-il aux frais ?"""
    embargo = max(10, LOOKBACK)
    P, A, SP, D = [], [], [], []
    for tr, te in walk_forward_splits(len(X), NFOLDS, embargo, MINTRAIN, "expanding"):
        m = LinearWM().fit(X[tr], Y[tr])
        P.append(m.predict(X[te])[:, RET_IDX])
        A.append(Y[te][:, RET_IDX])
        SP.append(spread[te])
        D.append(days[te])
    pred, y = np.concatenate(P), np.concatenate(A)
    spr, dd = np.concatenate(SP), np.concatenate(D)

    posn = np.sign(pred)
    gross = posn * y
    dpos = position_changes(posn, dd)          # remise à plat à chaque nouveau jour
    half = spr / 2.0

    rows = []
    for fee in fees_bp:
        net = gross - dpos * (half + fee * 1e-4)
        rows.append(dict(symbol=sym, fee_bp=fee,
                         gross_bp=round(float(gross.mean()) * 1e4, 4),
                         turnover=round(float(dpos.mean()) / 2, 3),
                         net_bp=round(float(net.mean()) * 1e4, 4),
                         net_cumul_pct=round(float(net.sum()) * 100, 2)))
    return pd.DataFrame(rows), dict(gross=gross, dpos=dpos, half=half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = load_cached()
    if not cache:
        raise SystemExit("Aucun .pkl dans data/raw/crypto_lob — lance d'abord "
                         "scripts/fetch_bybit_batch.py")
    print("=== Couverture ===")
    for sym, pk in cache.items():
        dates = [os.path.basename(p)[:10] for p in pk]
        print(f"  {sym} : {len(pk)} jours  ({dates[0]} .. {dates[-1]})")

    all1, allR, econ_rows, detail, costs = [], [], [], {}, {}
    for sym, pkls in cache.items():
        X, Y, days, spread, d, cost = build_symbol(pkls)
        print(f"  [{sym}] {len(X)} échantillons sur {days.max() + 1} jours", flush=True)
        costs[sym] = cost
        r1, rR = phase1a(sym, X, Y, days, d)
        all1 += r1
        allR += rR
        e, det = economic_check(sym, X, Y, days, spread)
        econ_rows.append(e)
        detail[sym] = det
        del X, Y

    res1, resR = pd.DataFrame(all1), pd.DataFrame(allR)
    pd.set_option("display.width", 160)

    print("\n=== Phase 1a — R²_OOS 1-step par dimension (baseline: 0 pour ret, no-change sinon) ===")
    print(res1.pivot_table(index="dim", columns="model", values="r2", aggfunc="mean")
          .reindex(STATE_COLS).round(5).to_string())
    print("\n--- 'ret' par symbole ---")
    print(res1[res1.dim == "ret"].pivot_table(index="symbol", columns="model",
                                              values="r2", aggfunc="mean").round(5).to_string())

    print("\n=== Phase 1a — rollout : R²_OOS rendement cumulé vs random walk ===")
    pivR = resR.pivot_table(index="horizon", columns="model", values="r2", aggfunc="mean")
    print(pivR.round(5).to_string())

    print("\n=== Phase 1b — coût d'exécution (achat), moyenné sur les jours ===")
    for sym, c in costs.items():
        print(f"\n[{sym}]")
        print(c.round(4).to_string(index=False))

    print("\n=== VERDICT économique — l'edge survit-il aux frais ? ===")
    econ = pd.concat(econ_rows, ignore_index=True)
    print(econ.to_string(index=False))
    econ.to_csv(os.path.join(args.out, "crypto_lob_economic.csv"), index=False)
    verdict = "EDGE (net>0 à frais réalistes)" if (econ[econ.fee_bp >= 2.0]["net_bp"] > 0).any() \
        else "MIRAGE (net<=0 dès des frais réalistes)"
    print(f"\n-> {verdict}")

    res1.to_csv(os.path.join(args.out, "crypto_lob_1step.csv"), index=False)
    resR.to_csv(os.path.join(args.out, "crypto_lob_rollout.csv"), index=False)

    # --- figures ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sym, c in costs.items():
        ax.plot(c["k_x_L1"], c["slippage_bp"], marker="o", label=sym)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("taille (k × meilleur niveau)")
    ax.set_ylabel("slippage moyen (bp)")
    ax.set_title("Crypto (Bybit L2) — coût d'exécution vs taille")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_cost.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in pivR.columns:
        ax.plot(pivR.index, pivR[m], marker="o", label=m)
    ax.axhline(0, color="grey", ls="--", lw=1)
    learned = [m for m in pivR.columns if m != "persistence"]
    ax.set_ylim(float(pivR[learned].min().min()) * 1.4 - 0.02,
                max(0.05, float(pivR[learned].max().max()) * 1.2))
    ax.set_xlabel("horizon de rollout (s)"); ax.set_ylabel("R²_OOS rendement cumulé")
    ax.set_title("Crypto (Bybit L2) — world model d'état, rollout vs random walk")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_rollout.png"), dpi=130); plt.close(fig)

    # stabilité temporelle du signal : R²_OOS(ret) du linéaire par fold
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = res1[(res1.dim == "ret") & (res1.model == "linear")]
    for sym in sorted(sub.symbol.unique()):
        s = sub[sub.symbol == sym].groupby("fold")["r2"].mean()
        ax.plot(s.index, s.values, marker="o", label=sym)
    ax.axhline(0, color="grey", ls="--", lw=1, label="random walk (=0)")
    ax.set_xlabel("fold walk-forward (≈ temps, inter-jours)")
    ax.set_ylabel("R²_OOS du rendement (linéaire)")
    ax.set_title("Stabilité du signal carnet dans le temps")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_stability.png"), dpi=130); plt.close(fig)

    sym0 = next(iter(detail))
    det = detail[sym0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fee, lbl in [(0.0, "sans frais (gross)"), (2.0, "net frais 2 bp"),
                     (5.5, "net frais 5.5 bp (taker Bybit)")]:
        net = det["gross"] - det["dpos"] * (det["half"] + fee * 1e-4)
        ax.plot(np.cumsum(net) * 100, label=lbl)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("barres de test (1 s, multi-jours)"); ax.set_ylabel("PnL cumulé (%)")
    ax.set_title(f"Crypto {sym0} — un edge réel qui est un mirage net de frais")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_mirage.png"), dpi=130); plt.close(fig)

    print(f"\nFigures + CSV dans {args.out}/")


if __name__ == "__main__":
    main()
