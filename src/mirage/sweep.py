"""Horizon sweep PRÉ-ENREGISTRÉ — caractérise 'ce qui est prédictible vs pas'.

Balaie le grid figé `target.horizons` (en barres = secondes), sur tous les tickers,
pour tous les estimateurs. On reporte TOUT (aucun cherry-pick) : tables + figures.

Discipline (I3) :
  - le grid est figé dans configs/phase0.yaml AVANT de regarder les résultats ;
  - embargo = max(embargo, horizon) (géré dans eval.run_phase0) -> pas de fuite ;
  - cibles à horizon h CHEVAUCHANTES => autocorrélation : le R²_OOS reste un
    estimateur valide, mais la SIGNIFICATIVITÉ exigerait un block bootstrap
    (cf. option 'valider le signal'). On le note, on ne le cache pas.

Usage :
    python -m mirage.sweep --config configs/phase0.yaml --out experiments
"""
from __future__ import annotations

import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .eval import load_config, run_phase0  # noqa: E402


def run_sweep(cfg: dict) -> pd.DataFrame:
    horizons = cfg["target"]["horizons"]
    frames = []
    for h in horizons:
        cfg_h = copy.deepcopy(cfg)
        cfg_h["target"]["horizon"] = int(h)
        res, _ = run_phase0(cfg_h)
        res["horizon"] = int(h)
        frames.append(res)
        pooled = res.groupby("estimator")["r2_oos"].mean()
        print(f"horizon={h:>3}s  |  " +
              "  ".join(f"{k}={v:+.4f}" for k, v in pooled.items()))
    return pd.concat(frames, ignore_index=True)


def _plot(piv: pd.DataFrame, ylabel: str, ref: float, ref_label: str, path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for est in piv.columns:
        if est == "zero":
            continue
        ax.plot(piv.index, piv[est], marker="o", label=est)
    ax.axhline(ref, color="grey", ls="--", lw=1, label=ref_label)
    ax.set_xlabel("horizon (secondes)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs horizon (pooled sur les tickers)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    print("=== Horizon sweep (grid pré-enregistré :", cfg["target"]["horizons"], ") ===")
    allres = run_sweep(cfg)
    allres.to_csv(os.path.join(args.out, "horizon_sweep_raw.csv"), index=False)

    r2 = allres.pivot_table(index="horizon", columns="estimator",
                            values="r2_oos", aggfunc="mean")
    da = allres.pivot_table(index="horizon", columns="estimator",
                            values="dir_acc", aggfunc="mean")
    r2.to_csv(os.path.join(args.out, "horizon_sweep_r2oos.csv"))

    print("\n--- R2_OOS pooled (horizon x estimateur) ---")
    print(r2.round(6).to_string())
    print("\n--- dir_acc pooled (horizon x estimateur) ---")
    print(da.round(4).to_string())

    _plot(r2, "R2_OOS", 0.0, "zero (=0)",
          os.path.join(args.out, "horizon_sweep_r2oos.png"))
    _plot(da, "directional accuracy", 0.5, "hasard (0.5)",
          os.path.join(args.out, "horizon_sweep_diracc.png"))

    # report honnête : meilleur (horizon, estimateur) hors baselines triviales
    model_rows = allres[~allres.estimator.isin(["zero", "persistence"])]
    g = model_rows.groupby(["estimator", "horizon"])["r2_oos"].mean()
    best = g.idxmax()
    print(f"\nMeilleur R2_OOS pooled (hors baselines triviales) : "
          f"{best[0]} @ {best[1]}s -> {g.max():+.6f}")
    print("(reporté pour transparence, PAS sélectionné a posteriori comme 'le' résultat)")
    print(f"\nFigures + CSV écrits dans : {args.out}/")


if __name__ == "__main__":
    main()
