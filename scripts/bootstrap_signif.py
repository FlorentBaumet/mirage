"""Significativité du signal directionnel @1s : block bootstrap + analyse coûts.

Pour chaque ticker (horizon 1) :
  1. reconstruit les prédictions OOS du modèle linéaire (ridge) via le MÊME
     walk-forward purgé que l'éval (folds concaténés dans l'ordre temporel) ;
  2. block bootstrap CIRCULAIRE (blocs contigus -> respecte l'autocorrélation des
     cibles chevauchantes) -> IC 95% de dir_acc et de R²_OOS ;
  3. analyse économique : edge directionnel brut/barre vs demi-spread -> net de coûts.

Question : le signe @1s est-il (a) statistiquement > hasard, et (b) exploitable net
de coûts ? Attendu : (a) OUI sur les large-tick, (b) NON => mirage confirmé.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mirage.baselines import LinearBaseline
from mirage.eval import load_config, prepare_instrument
from mirage.metrics import directional_accuracy, r2_oos
from mirage.splits import walk_forward_splits


def collect_oos(cfg, inst):
    """Prédictions OOS du ridge + cibles + rel_spread BRUT, concaténées sur les folds."""
    X, y, _ = prepare_instrument(cfg, inst)
    sp = cfg["split"]
    horizon = int(cfg["target"]["horizon"])
    embargo = max(int(sp["embargo"]), horizon)
    n = len(X)
    yt, pl, rs = [], [], []
    for tr, te in walk_forward_splits(n, sp["n_folds"], embargo,
                                      sp["min_train_frac"], sp["scheme"]):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        mu = Xtr.mean()
        sd = Xtr.std().replace(0.0, 1.0)
        m = LinearBaseline().fit(((Xtr - mu) / sd).to_numpy(), y.iloc[tr].to_numpy())
        pl.append(m.predict(((Xte - mu) / sd).to_numpy()))
        yt.append(y.iloc[te].to_numpy())
        rs.append(Xte["rel_spread"].to_numpy())  # rel_spread BRUT = (ask1-bid1)/mid
    return np.concatenate(yt), np.concatenate(pl), np.concatenate(rs)


def block_boot(y, p, stat, n_boot=2000, block=60, seed=0):
    """Bootstrap par blocs circulaires de longueur `block`."""
    rng = np.random.default_rng(seed)
    n = len(y)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        out[i] = stat(y[idx], p[idx])
    return out


def main(config="configs/phase0.yaml", n_boot=2000, block=60):
    cfg = load_config(config)
    rows = []
    for inst in cfg["data"]["instruments"]:
        tk = inst["ticker"]
        y, p, rs = collect_oos(cfg, inst)

        da, r2 = directional_accuracy(y, p), r2_oos(y, p)
        da_b = block_boot(y, p, directional_accuracy, n_boot, block, seed=0)
        r2_b = block_boot(y, p, r2_oos, n_boot, block, seed=1)
        da_lo, da_hi = np.percentile(da_b, [2.5, 97.5])
        r2_lo, r2_hi = np.percentile(r2_b, [2.5, 97.5])
        p_chance = float(np.mean(da_b <= 0.5))  # p-value bootstrap unilatérale vs 0.5

        # --- économie : strategie 'trade le signe', coûts = spread payé par flip ---
        sign = np.sign(p)
        gross = float(np.mean(sign * y))                      # edge brut / barre
        half_spread = float(np.mean(0.5 * rs))                # demi-spread relatif moyen
        flips = float(np.mean(np.abs(np.diff(sign)) / 2.0))   # taux de retournement
        cost = flips * float(np.mean(rs))                     # ~1 spread payé par flip
        net = gross - cost

        rows.append({
            "ticker": tk, "n_oos": len(y),
            "dir_acc": round(da, 4), "da_CI95": f"[{da_lo:.3f},{da_hi:.3f}]",
            "p_vs_chance": round(p_chance, 4),
            "r2_oos": round(r2, 5), "r2_CI95": f"[{r2_lo:.4f},{r2_hi:.4f}]",
            "gross_bp": round(gross * 1e4, 3),
            "halfspread_bp": round(half_spread * 1e4, 2),
            "net_bp": round(net * 1e4, 3),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv("experiments/bootstrap_signif.csv", index=False)
    print("\nLecture :")
    print("  da_CI95 exclut 0.5  => signe significativement > hasard.")
    print("  net_bp < 0          => inexploitable net de coûts (mirage confirmé).")
    print("  (1 bp = 1e-4 de rendement)")


if __name__ == "__main__":
    main()
