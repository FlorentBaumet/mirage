"""Regime de cout du carnet Bybit L2 : spread par symbole et par annee.

Le pre-enregistrement (configs/phase1_crypto_prereg.yaml, `rapport.doit_contenir`) exige
le spread moyen par symbole et la comparaison 2023 / 2024 / 2025. C'est la piece qui dit
si le mirage change de nature selon l'epoque : tue par le spread (annees de faible
liquidite) ou par les seuls frais (annees liquides).

DEUX mesures, volontairement separees :

  - Fenetre COMPLETE (les 44 journees, depuis les .pkl) : c'est le descripteur de regime.
    Il porte sur toute la periode telechargee, y compris les journees que le walk-forward
    garde en train.
  - Journees de TEST (depuis les .npz) : c'est ce que l'evaluation a reellement vu. Avec
    min_train_frac = 0.4 en schema expanding, le train initial avale les 17 premieres
    journees (les 16 de 2023 plus le 2024-01-14) : le test ne commence qu'en cours de
    journee le 2024-02-06 et court jusqu'au 2025-08-06, soit 27 journees.

Sortie ASCII (console cp1252).

    python scripts/crypto/crypto_lob.py --out experiments
    python scripts/crypto/spread_regime.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

from mirage.state import STATE_COLS, build_state

PREREG = os.path.join("configs", "phase1_crypto_prereg.yaml")
DIR = os.path.join("data", "raw", "crypto_lob")
OOS_DIR = "experiments"
SPREAD_IDX = STATE_COLS.index("spread_rel")


def spread_par_jour_pleine_fenetre(symboles: list[str]) -> pd.DataFrame:
    """Spread mesure sur TOUTES les journees en cache, depuis les barres."""
    rows = []
    for pf in sorted(glob.glob(os.path.join(DIR, "*_1s_book.pkl"))):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)_1s_book", os.path.basename(pf))
        date, sym = m.group(1), m.group(2)
        if sym not in symboles:
            continue
        # read_pickle est sur ici : le .pkl est ecrit par notre propre
        # scripts/fetch_bybit_batch.py depuis l'archive Bybit, jamais par un tiers.
        S, _ = build_state(pd.read_pickle(pf))
        spr = S.values[:, SPREAD_IDX] * 1e4          # spread relatif en points de base
        rows.append(dict(symbol=sym, date=date, annee=date[:4],
                         n=len(spr), spread_bp=float(spr.mean())))
    return pd.DataFrame(rows)


def main() -> None:
    with open(PREREG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dates = list(cfg["donnees"]["dates"])
    symbols = list(cfg["donnees"]["symboles"])

    # --- A. regime sur la fenetre complete ---
    plein = spread_par_jour_pleine_fenetre(symbols)
    piv = plein.pivot_table(index="symbol", columns="annee", values="spread_bp",
                            aggfunc="mean").reindex(symbols)
    nj = plein.pivot_table(index="symbol", columns="annee", values="date",
                           aggfunc="count").reindex(symbols)
    piv.insert(0, "toutes", plein.groupby("symbol")["spread_bp"].mean().reindex(symbols))

    pd.set_option("display.width", 200)
    print("=== A. Spread mesure sur les BARRES, fenetre complete (44 journees par symbole) ===")
    print("    en points de base ; entre parentheses : nombre de journees")
    aff = pd.DataFrame(index=piv.index)
    aff["toutes"] = [f"{piv.loc[s, 'toutes']:.4f}" for s in piv.index]
    for c in [c for c in piv.columns if c != "toutes"]:
        aff[c] = [f"{piv.loc[s, c]:.4f} ({int(nj.loc[s, c])})"
                  if np.isfinite(piv.loc[s, c]) else "-" for s in piv.index]
    print(aff.to_string())

    print("\n--- Bascule de regime 2023 -> 2025 (fenetre complete) ---")
    for sym in symbols:
        a, b = piv.loc[sym, "2023"], piv.loc[sym, "2025"]
        if np.isfinite(a) and np.isfinite(b):
            print(f"  {sym:9s} {a:8.4f} bp -> {b:8.4f} bp  (x{b / a:.3f})")

    # --- B. ce que l'evaluation a reellement vu ---
    rows, per_day = [], []
    for sym in symbols:
        pf = os.path.join(OOS_DIR, f"crypto_lob_oos_{sym}.npz")
        if not os.path.exists(pf):
            continue
        with np.load(pf) as z:
            half = z["half"].astype(float)
            day = z["day"].astype(int)
        spread_bp = half * 2.0 * 1e4
        n_days = int(day.max()) + 1
        if n_days > len(dates):
            raise SystemExit(f"{sym} : {n_days} journees > {len(dates)} dates du prereg")
        year = np.array([dates[d][:4] for d in range(n_days)])
        row = {"symbol": sym, "n_oos": len(half),
               "spread_bp_oos": float(spread_bp.mean())}
        for y in sorted(set(year.tolist())):
            sel = year[day] == y
            row[f"jours_test_{y}"] = int(np.unique(day[sel]).size)
        rows.append(row)
        for d in np.unique(day):
            sel = day == d
            per_day.append(dict(symbol=sym, jour=dates[d], spread_bp=float(spread_bp[sel].mean())))

    oos = pd.DataFrame(rows)
    print("\n=== B. Journees de TEST du walk-forward (ce que l'evaluation a vu) ===")
    print(oos.to_string(index=False))

    jours = sorted({d for sym in symbols
                    for d in np.unique(np.load(
                        os.path.join(OOS_DIR, f"crypto_lob_oos_{sym}.npz"))["day"])})
    premier, dernier = dates[int(jours[0])], dates[int(jours[-1])]
    print(f"\n  Fenetre reellement testee : {premier} -> {dernier}, {len(jours)} journees.")
    print("  Les 17 premieres journees (les 16 de 2023 + le 2024-01-14) restent en TRAIN :")
    print(f"  2023 n'apparait jamais en test, et le test demarre en cours de journee le {premier}.")

    plein.to_csv(os.path.join(OOS_DIR, "crypto_lob_spread_pleine_fenetre.csv"), index=False)
    oos.to_csv(os.path.join(OOS_DIR, "crypto_lob_spread_regime.csv"), index=False)
    pd.DataFrame(per_day).to_csv(
        os.path.join(OOS_DIR, "crypto_lob_spread_par_jour.csv"), index=False)
    print(f"\n-> {OOS_DIR}/crypto_lob_spread_regime.csv (OOS)")
    print(f"-> {OOS_DIR}/crypto_lob_spread_pleine_fenetre.csv (barres)")
    print(f"-> {OOS_DIR}/crypto_lob_spread_par_jour.csv (OOS, par journee)")


if __name__ == "__main__":
    main()
