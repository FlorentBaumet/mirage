"""Phase 1b — caractérisation de l'impact + démo du world model action-conditionné.

1) Courbes de COÛT (mesurables, vrai carnet) : slippage & impact immédiat vs taille
   d'ordre, par ticker. Relie le « mirage » de Phase 0 : à quel point trader coûte.
2) Démo ACTION-CONDITIONNÉE : trajectoire de prix passive (WM Phase 1a) vs
   « et si j'achète maintenant » (overlay d'impact). Montre le world model qui
   répond à une action.

    python scripts/impact_curves.py --out experiments
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mirage.data.bars import to_clock_bars  # noqa: E402
from mirage.data.lobster import load_sample  # noqa: E402
from mirage.eval import instrument_paths, load_config  # noqa: E402
from mirage.impact import cost_curve, execution_cost, impact_decay_path  # noqa: E402
from mirage.state import RET_IDX, build_state  # noqa: E402
from mirage.wm import LinearWM, make_supervised, rollout  # noqa: E402

KS = [0.25, 0.5, 1, 2, 4, 8]


def _bars(cfg, inst):
    d = cfg["data"]
    msg, book = instrument_paths(cfg, inst)
    b = load_sample(msg, book, d["levels"], d.get("session_start_s"), d.get("session_end_s"))
    return to_clock_bars(b, cfg["bars"]["freq"], d["date"])


def action_demo(bars, horizon=30, k=2.0, lookback=16, perm=0.3, decay=0.7, seed=0):
    """Trajectoire de mid (bp) : passive (WM) vs avec un achat de taille k×L1 à t0."""
    S, _ = build_state(bars)
    X, Y, pos, d = make_supervised(S.values, lookback)
    cut = int(len(X) * 0.6)
    wm = LinearWM().fit(X[:cut], Y[:cut])

    rng = np.random.default_rng(seed)
    starts = rng.choice(np.arange(cut, len(X) - horizon), size=min(1000, len(X) - cut - horizon),
                        replace=False)
    preds = rollout(wm, X[starts].reshape(-1, lookback, d), horizon)
    passive_bp = np.cumsum(preds[:, :, RET_IDX], axis=1).mean(0) * 1e4  # (horizon,)

    _, impact, _ = execution_cost(bars, k * float(np.nanmean(
        bars["ask_size_1"])), side=1)
    immediate = float(np.nanmean(impact))                # impact immédiat relatif
    impact_bp = impact_decay_path(immediate, horizon, perm, decay) * 1e4
    return passive_bp, passive_bp + impact_bp, immediate * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--demo_ticker", default="MSFT")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    # --- 1) courbes de coût par ticker ---
    print("=== Coût d'exécution (achat) — slippage en bp vs taille (k × L1) ===")
    curves = {}
    for inst in cfg["data"]["instruments"]:
        bars = _bars(cfg, inst)
        c = cost_curve(bars, KS, side=1, levels=cfg["data"]["levels"])
        curves[inst["ticker"]] = c
        print(f"\n[{inst['ticker']}]")
        print(c.to_string(index=False))
        c.to_csv(os.path.join(args.out, f"phase1b_cost_{inst['ticker']}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tk, c in curves.items():
        ax.plot(c["k_x_L1"], c["slippage_bp"], marker="o", label=tk)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("taille d'ordre  (k × taille du meilleur niveau)")
    ax.set_ylabel("slippage moyen (bp)")
    ax.set_title("Coût d'exécution vs taille — manger le carnet")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(args.out, "phase1b_cost_curves.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # --- 2) démo action-conditionnée ---
    inst = next(i for i in cfg["data"]["instruments"] if i["ticker"] == args.demo_ticker)
    passive_bp, action_bp, imm_bp = action_demo(_bars(cfg, inst))
    h = np.arange(1, len(passive_bp) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(h, passive_bp, marker="o", label="passif (world model seul)")
    ax.plot(h, action_bp, marker="o", label=f"+ achat à t0 ({args.demo_ticker})")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("horizon (secondes)")
    ax.set_ylabel("déviation du mid (bp)")
    ax.set_title("World model action-conditionné — « et si j'achète maintenant ? »")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(args.out, "phase1b_action_demo.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    print(f"\nDemo action ({args.demo_ticker}) : impact immediat ~ {imm_bp:.2f} bp, "
          f"puis decroissance vers la part permanente.")
    print(f"Figures : {p1} | {p2}")


if __name__ == "__main__":
    main()
