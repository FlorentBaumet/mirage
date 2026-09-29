"""Significativité du signal carnet crypto — block bootstrap PAR JOUR.

Le jeu publié (6 jours) ne portait aucun intervalle de confiance : R²_OOS ≈ +0.067 et
« net négatif dès 2 bp » étaient des moyennes ponctuelles. Ici on rééchantillonne les
JOURNÉES (les blocs sont les journées) pour obtenir des IC95 % qui respectent la
dépendance intra-jour et la persistance inter-jours des régimes.

Le protocole — B, graines, unité de bloc, niveaux de frais, règles de décision — est
FIGÉ dans `configs/phase1_crypto_prereg.yaml`, commité avant toute exécution.

Ce script réutilise EXACTEMENT les prédictions OOS et les coûts produits par
`crypto_lob.py` (fichiers npz) : aucune divergence possible entre le verdict
économique et son intervalle de confiance.

Chaque statistique se réduit à des sommes par jour, donc le bootstrap est exact et
instantané malgré ~4 M d'observations :

    R²_OOS      = 1 - Σ(y-ŷ)² / Σy²
    net_bp(fee) = moyenne de gross - dpos·(demi-spread + fee)

    python scripts/crypto/crypto_lob.py --out experiments
    python scripts/crypto/bootstrap_signif.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

PREREG = os.path.join("configs", "phase1_crypto_prereg.yaml")
OOS_DIR = "experiments"
OUT_CSV = os.path.join(OOS_DIR, "crypto_lob_bootstrap.csv")


def load_prereg(path: str = PREREG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def boot_mult(rng, n_pool: int, n_boot: int, block_days: int) -> np.ndarray:
    """Multiplicités par jour (n_boot x n_pool) pour chaque réplication bootstrap.

    block_days == 1 : tirage i.i.d. de journées avec remise.
    block_days  > 1 : blocs circulaires de journées contiguës (un régime de marché dure
                      plusieurs jours, donc les journées ne sont pas échangeables).
    """
    if block_days <= 1:
        picks = rng.integers(0, n_pool, size=(n_boot, n_pool))
    else:
        n_blocks = -(-n_pool // block_days)
        starts = rng.integers(0, n_pool, size=(n_boot, n_blocks))
        offs = np.arange(block_days)[None, None, :]
        picks = ((starts[:, :, None] + offs) % n_pool).reshape(n_boot, -1)[:, :n_pool]
    mult = np.zeros((n_boot, n_pool), dtype=np.int32)
    for b in range(n_boot):
        mult[b] = np.bincount(picks[b], minlength=n_pool)
    return mult


def symbol_report(sym: str, npz: dict, cfg: dict) -> dict:
    y = npz["y"].astype(float)
    pred = npz["pred"].astype(float)
    gross = npz["gross"].astype(float)
    dpos = npz["dpos"].astype(float)
    half = npz["half"].astype(float)
    day = npz["day"].astype(int)

    boot_cfg = cfg["evaluation"]["bootstrap"]
    n_boot = int(boot_cfg["n_boot"])
    seed = int(boot_cfg["seed"])
    blk = int(boot_cfg["variante_robustesse"]["longueur_bloc_jours"])
    fees = [float(f) for f in cfg["evaluation"]["frais_bp"]]

    n_days = int(day.max()) + 1
    counts = np.bincount(day, minlength=n_days).astype(float)
    pool = np.where(counts > 0)[0]                 # journées réellement présentes en test
    c = counts[pool]

    # --- sommes par jour : suffisantes pour toutes les statistiques d'intérêt ---
    sse_m = np.bincount(day, weights=(y - pred) ** 2, minlength=n_days)[pool]
    sse_b = np.bincount(day, weights=y**2, minlength=n_days)[pool]
    gross_s = np.bincount(day, weights=gross, minlength=n_days)[pool]
    net_s = {f: np.bincount(day, weights=gross - dpos * (half + f * 1e-4),
                            minlength=n_days)[pool] for f in fees}

    # --- estimateurs ponctuels (identiques à ceux de crypto_lob.py) ---
    out = {"symbol": sym, "n_oos": len(y), "n_days": len(pool),
           "r2_oos": 1.0 - sse_m.sum() / sse_b.sum(),
           "gross_bp": gross_s.sum() / c.sum() * 1e4}

    # --- le R²_OOS moyen est-il porté par quelques journées ? ---
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_day = 1.0 - sse_m / sse_b
    r2_day = r2_day[np.isfinite(r2_day)]
    out |= {"r2_day_frac_pos": float(np.mean(r2_day > 0)),
            "r2_day_med": float(np.median(r2_day)),
            "r2_day_q25": float(np.percentile(r2_day, 25)),
            "r2_day_q75": float(np.percentile(r2_day, 75))}

    # --- bootstrap : on ne rééchantillonne que des sommes par jour ---
    rng = np.random.default_rng(seed)

    def one_pass(block_days: int, suffix: str) -> dict:
        m = boot_mult(rng, len(pool), n_boot, block_days)
        res = {}
        r2_b = 1.0 - (m @ sse_m) / (m @ sse_b)
        res[f"r2_lo{suffix}"] = float(np.percentile(r2_b, 2.5))
        res[f"r2_hi{suffix}"] = float(np.percentile(r2_b, 97.5))
        if not suffix:
            res["r2_p_le0"] = float(np.mean(r2_b <= 0))
            g_b = (m @ gross_s) / (m @ c) * 1e4
            res["gross_lo"] = float(np.percentile(g_b, 2.5))
            res["gross_hi"] = float(np.percentile(g_b, 97.5))
        for f in fees:
            tag = f"f{f:g}"
            n_b = (m @ net_s[f]) / (m @ c) * 1e4
            res[f"net_{tag}{suffix}"] = (float(net_s[f].sum() / c.sum() * 1e4)
                                        if not suffix else float(np.mean(n_b)))
            res[f"net_lo_{tag}{suffix}"] = float(np.percentile(n_b, 2.5))
            res[f"net_hi_{tag}{suffix}"] = float(np.percentile(n_b, 97.5))
            if not suffix:
                res[f"net_exploitable_{tag}"] = bool(np.percentile(n_b, 2.5) > 0)
        return res

    out |= one_pass(1, "")
    out |= one_pass(blk, f"_blk{blk}")
    return out


def main() -> None:
    cfg = load_prereg()
    symbols = cfg["donnees"]["symboles"]
    fees = [float(f) for f in cfg["evaluation"]["frais_bp"]]
    boot_cfg = cfg["evaluation"]["bootstrap"]
    blk = int(boot_cfg["variante_robustesse"]["longueur_bloc_jours"])

    files = {}
    for pf in sorted(glob.glob(os.path.join(OOS_DIR, "crypto_lob_oos_*.npz"))):
        m = re.match(r"crypto_lob_oos_([A-Z]+)\.npz", os.path.basename(pf))
        if m and m.group(1) in symbols:
            files[m.group(1)] = pf
    if not files:
        raise SystemExit(f"Aucun crypto_lob_oos_*.npz dans {OOS_DIR}/ — lance d'abord "
                         "scripts/crypto/crypto_lob.py")

    rows = []
    for sym in symbols:
        if sym in files:
            with np.load(files[sym]) as z:
                rows.append(symbol_report(sym, {k: z[k] for k in z.files}, cfg))
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    print(f"=== Bootstrap par jour — B={boot_cfg['n_boot']}, unité = la journée ; "
          f"variante robustesse = blocs contigus de {blk} jours ===")

    print("\n--- Signal : R²_OOS(rendement) ---")
    print(df[["symbol", "n_days", "n_oos", "r2_oos", "r2_lo", "r2_hi", "r2_p_le0",
              "r2_day_med", "r2_day_q25", "r2_day_q75", "r2_day_frac_pos"]]
          .round(4).to_string(index=False))

    print("\n--- Verdict économique : net par barre (bp), IC95 par jour ---")
    econ = df[["symbol", "gross_bp", "gross_lo", "gross_hi"]].copy()
    for f in fees:
        tag = f"f{f:g}"
        econ[f"net_{f:g}"] = df[f"net_{tag}"]
        econ[f"IC95_{f:g}"] = [f"[{a:.3f},{b:.3f}]" for a, b in
                               zip(df[f"net_lo_{tag}"], df[f"net_hi_{tag}"], strict=True)]
    print(econ.round(4).to_string(index=False))

    print(f"\n--- Robustesse : blocs contigus de {blk} jours ---")
    rob = df[["symbol", "r2_oos"]].copy()
    rob["r2_IC95"] = [f"[{a:.4f},{b:.4f}]" for a, b in
                      zip(df[f"r2_lo_blk{blk}"], df[f"r2_hi_blk{blk}"], strict=True)]
    for f in fees:
        tag = f"net_{f:g}"
        rob[tag] = [f"[{a:.3f},{b:.3f}]" for a, b in
                    zip(df[f"net_lo_f{f:g}_blk{blk}"], df[f"net_hi_f{f:g}_blk{blk}"],
                        strict=True)]
    print(rob.round(4).to_string(index=False))

    # --- verdicts, appliqués mécaniquement depuis les règles pré-enregistrées ---
    n_sig = int((df["r2_lo"] > 0).sum())
    print("\n=== Verdicts (règles pré-enregistrées, configs/phase1_crypto_prereg.yaml) ===")
    v_sig = "SIGNAL RÉEL" if n_sig >= 4 else "SIGNAL NON ÉTABLI"
    print(f"  signal      : IC95 de R²_OOS > 0 sur {n_sig}/{len(df)} symboles "
          f"-> {v_sig} (seuil pré-enregistré : >= 4/5)")

    hi2 = df["net_hi_f2"]
    if bool((hi2 < 0).all()):
        print("  mirage      : IC95 de net à 2 bp entièrement < 0 sur TOUS les symboles "
              "-> MIRAGE CONFIRMÉ")
    else:
        print(f"  mirage      : IC95 de net à 2 bp non entièrement < 0 sur "
              f"{int((~(hi2 < 0)).sum())}/{len(df)} symbole(s) -> NON confirmé au sens "
              f"strict ; net moyen à 2 bp = {df['net_f2'].round(3).tolist()}")
    explo = int(df["net_exploitable_f2"].sum())
    print(f"  exploitable : borne basse > 0 à 2 bp sur {explo}/{len(df)} symbole(s) "
          f"-> {'EXPLOITABLE' if explo else 'AUCUN SYMBOLE EXPLOITABLE'}")
    print(f"  à 0 frais   : net moyen {df['net_f0'].round(3).tolist()} bp ; "
          f"borne basse > 0 sur {int((df['net_lo_f0'] > 0).sum())}/{len(df)} "
          f"(borne optimiste : frais nuls irréalistes, ne vaut pas edge)")

    df.to_csv(OUT_CSV, index=False)
    print(f"\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
