"""Phase 2b, etape B : calibration du bras myope (diagnostic, LECTURE SEULE).

Le prereg (`configs/phase2b_crypto_prereg.yaml`, bloc `regles_B`) demande trois chiffres,
sans aucun re-entrainement :

  1. la pente beta de y sur pred, IC95 par jour : la prevision est-elle exploitable
     lineairement, et de combien ;
  2. E[y | decile de pred] contre E[pred | decile] : la prevision est-elle monotone, ou
     portee par ses extremes (c'est la que le planificateur a cout agit) ;
  3. alpha / n_train par pli : chiffre a citer. L'hypothese "contraction Ridge" qui servait
     d'explication dans la Phase 2 tombe si alpha / n_train est de l'ordre de 1e-6.

Aucune de ces mesures ne depend du planificateur ni d'un bras : elles ne peuvent donc pas
servir a en choisir un. Le fichier ne lit que `experiments/` (artefact publie) et
`experiments_2b/p2b_exploit.csv` s'il existe ; il n'ecrit que `experiments_2b/calib_*.csv`.

  python scripts/crypto/calib_diag.py
  python scripts/crypto/calib_diag.py --symbols BTCUSDT DOGEUSDT --no-folds
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from agent_plan import BLK, N_BOOT, SEED
from arm_eval import EMBARGO, LOOKBACK, MINTRAIN, NFOLDS
from bootstrap_signif import boot_mult
from crypto_lob import _merge_csv, load_cached

from mirage.splits import walk_forward_splits
from mirage.wm import LinearWM

OUT = "experiments_2b"
REF = "experiments"
BP = 1e4                       # les rendements sont des fractions : x 1e4 -> points de base


# --------------------------------------------------------------------------------------
# Pente et IC, par jour
# --------------------------------------------------------------------------------------

def slope(y: np.ndarray, p: np.ndarray) -> float:
    """Pente OLS de y sur p (ordonnee a l'origine incluse, donc sur les donnees centrees)."""
    p = p - p.mean()
    den = float(p @ p)
    if den <= 0.0:
        return float("nan")
    return float(p @ (y - y.mean()) / den)


def _day_sums(y, p, dd, n_days):
    """Sommes par jour. Suffisent a toute moyenne ou pente ponderee par le tirage."""
    n = np.bincount(dd, minlength=n_days).astype(float)
    return (n,
            np.bincount(dd, weights=y, minlength=n_days),
            np.bincount(dd, weights=p, minlength=n_days),
            np.bincount(dd, weights=y * p, minlength=n_days),
            np.bincount(dd, weights=p * p, minlength=n_days))


def _pooled_slope(sums, mult):
    """Pente sur le pool re-tire, a partir des seules sommes par jour.

    Le cout est en (n_boot x n_jours) et non en (n_boot x 2,3 M) : on re-tire les JOURNEES
    (c'est la convention du projet, cf. `bootstrap_signif.boot_mult`), pas les lignes.
    """
    n, sy, sp, syp, spp = sums
    N = mult @ n
    Sy, Sp = mult @ sy, mult @ sp
    cov = (mult @ syp) - Sy * Sp / N
    var = (mult @ spp) - Sp * Sp / N
    tol = np.finfo(float).tiny
    return np.where(var > tol, cov / var, np.nan)


def boot_slope(y, p, dd, n_days, n_boot=N_BOOT, blk=BLK, seed=SEED):
    """(pente, lo, hi) : point sur le pool des jours presents, IC95 par tirage de jours."""
    k = np.bincount(dd, minlength=n_days) > 0
    if int(k.sum()) < 3:
        return slope(y, p), float("nan"), float("nan")
    sums = tuple(a[k] for a in _day_sums(y, p, dd, n_days))
    m = boot_mult(np.random.default_rng(seed), int(k.sum()), n_boot, blk)
    b = _pooled_slope(sums, m)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return slope(y, p), float("nan"), float("nan")
    return slope(y, p), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------------------
# Deciles
# --------------------------------------------------------------------------------------

def deciles(y: np.ndarray, p: np.ndarray, n: int = 10) -> list[dict]:
    """E[pred] contre E[y] par decile de pred, et la pente interne a chaque decile."""
    q = np.quantile(p, np.linspace(0.0, 1.0, n + 1))
    q[0] -= 1e-15                                  # pour inclure le minimum dans le decile 1
    idx = np.clip(np.digitize(p, q[1:-1]), 0, n - 1)
    rows = []
    for d in range(n):
        k = idx == d
        rows.append({
            "decile": d + 1,
            "n": int(k.sum()),
            "e_pred_bp": float(p[k].mean() * BP) if k.any() else np.nan,
            "e_y_bp": float(y[k].mean() * BP) if k.any() else np.nan,
            "pente": slope(y[k], p[k]) if int(k.sum()) > 2 else np.nan,
        })
    return rows


def top_abs_slope(y, p, dd, n_days, frac=0.1, **kw):
    """Pente dans le decile superieur de |pred| : c'est la que la prevision est forte.

    L'IC est un tirage de jours sur le SOUS-ECHANTILLON (les jours a forte volatilite le
    portent presque seuls : c'est precisement ce que l'IC doit montrer).
    """
    k = np.abs(p) >= np.quantile(np.abs(p), 1.0 - frac)
    return boot_slope(y[k], p[k], dd[k], n_days, **kw)


# --------------------------------------------------------------------------------------
# alpha / n_train par pli
# --------------------------------------------------------------------------------------

def _taille(idx) -> int:
    a = np.asarray(idx)
    return int(np.count_nonzero(a)) if a.dtype == bool else int(a.size)


def n_train_par_pli(sym: str, n_oos: int, pkls: list[str]) -> list[dict]:
    """Reconstruit les plis et le n_train de chacun, a partir du NOMBRE DE BARRES par jour.

    `len(S_jour) = len(bars_jour) - 1` : la premiere barre n'a pas de rendement. Verifie sur
    les 44 jours publies, ou `somme(len(bars)) - n_jours x (LOOKBACK + 1)` redonne exactement
    le total d'echantillons imprime par le harnais. La somme des tailles de test doit, elle,
    redonner le nombre d'echantillons OOS stockes dans l'artefact publie : c'est le controle
    qui dit que la reconstruction est la bonne, et il est bloquant.
    """
    # Pickle local, produit par nos propres scripts dans data/raw/crypto_lob : meme source
    # de confiance que `arm_eval.build_wide`, et rien d'autre n'est lu du contenu.
    n_barres = sum(len(pd.read_pickle(pf)) for pf in pkls)
    n = n_barres - len(pkls) * (LOOKBACK + 1)
    alpha = float(LinearWM().alpha)
    rows, total = [], 0
    for i, (tr, te) in enumerate(walk_forward_splits(n, NFOLDS, EMBARGO, MINTRAIN,
                                                     "expanding")):
        n_tr, n_te = _taille(tr), _taille(te)
        total += n_te
        rows.append({"symbol": sym, "pli": i, "n": int(n), "n_train": n_tr, "n_test": n_te,
                     "alpha": alpha, "alpha_sur_n_train": alpha / n_tr})
    if total != n_oos:
        raise SystemExit(
            f"ECHEC [{sym}] : les plis reconstruits portent {total} echantillons de test, "
            f"l'artefact publie en contient {n_oos}. La reconstruction (n = {n}) n'est pas "
            f"celle du run publie : alpha / n_train serait faux.")
    return rows


# --------------------------------------------------------------------------------------

def symbol_rows(sym: str, ref: str, out: str, n_dec: int, folds: bool, pkls) -> tuple:
    z = np.load(os.path.join(ref, f"crypto_lob_oos_{sym}.npz"))
    y = np.asarray(z["y"], float)
    p = np.asarray(z["pred"], float)
    dd = np.asarray(z["day"]).astype(int)
    n_days = int(dd.max()) + 1

    b, lo, hi = boot_slope(y, p, dd, n_days)
    b1, lo1, hi1 = boot_slope(y, p, dd, n_days, blk=1)
    s_top, t_lo, t_hi = top_abs_slope(y, p, dd, n_days)

    print(f"  [{sym}] n = {len(y)} sur {int(np.bincount(dd, minlength=n_days).astype(bool).sum())} "
          f"jours")
    print(f"    pente y~pred : {b:.4f} [{lo:.4f}, {hi:.4f}] (blocs de {BLK} jours) ; "
          f"blocs de 1 : {b1:.4f} [{lo1:.4f}, {hi1:.4f}]")
    print(f"    pente |pred| top 10% : {s_top:.4f} [{t_lo:.4f}, {t_hi:.4f}]")
    print(f"    E[y] = {y.mean() * BP:+.4f} bp ; E[pred] = {p.mean() * BP:+.4f} bp")

    sl = {"symbol": sym, "n": len(y), "n_jours": n_days,
          "pente": b, "pente_lo": lo, "pente_hi": hi,
          "pente_blk1": b1, "pente_blk1_lo": lo1, "pente_blk1_hi": hi1,
          "pente_top10": s_top, "pente_top10_lo": t_lo, "pente_top10_hi": t_hi,
          "e_y_bp": float(y.mean() * BP), "e_pred_bp": float(p.mean() * BP),
          "r2_oos": float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    dc = [{"symbol": sym, **r} for r in deciles(y, p, n_dec)]
    fo = n_train_par_pli(sym, len(y), pkls) if folds else []
    if fo:
        f = fo[0]
        print(f"    n = {f['n']} ; n_train {fo[0]['n_train']} -> {fo[-1]['n_train']} ; "
              f"alpha/n_train {fo[0]['alpha_sur_n_train']:.3e} -> "
              f"{fo[-1]['alpha_sur_n_train']:.3e}")
    return sl, dc, fo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF, help="dossier de l'artefact publie (lecture seule)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--no-folds", dest="folds", action="store_false",
                    help="saute alpha / n_train (evite de relire les .pkl)")
    args = ap.parse_args()

    cache = load_cached()
    syms = args.symbols or sorted(cache)
    for s in syms:
        if s not in cache:
            raise SystemExit(f"Aucun .pkl pour : {s}.")
    os.makedirs(args.out, exist_ok=True)

    sl_rows, dc_rows, fo_rows = [], [], []
    for sym in syms:
        s, d, f = symbol_rows(sym, args.ref, args.out, args.deciles, args.folds, cache[sym])
        sl_rows.append(s)
        dc_rows += d
        fo_rows += f

    # Fusion par symbole, comme partout ailleurs : un run partiel ne doit pas amputer un
    # CSV deja complet.
    _merge_csv(os.path.join(args.out, "calib_slope.csv"), pd.DataFrame(sl_rows))
    _merge_csv(os.path.join(args.out, "calib_deciles.csv"), pd.DataFrame(dc_rows))
    if fo_rows:
        _merge_csv(os.path.join(args.out, "calib_folds.csv"), pd.DataFrame(fo_rows))

    # La prediction pre-enregistree de B porte sur l'ecart d'exploitation de agent_causal@2,
    # calcule par le harnais. On ne le recalcule pas ici : on l'affiche tel qu'il a ete ecrit.
    ep = os.path.join(args.out, "p2b_exploit.csv")
    if os.path.exists(ep):
        df = pd.read_csv(ep)
        sel = df[df.bras.astype(str).str.startswith("agent_causal@2")]
        print("\nEcart d'exploitation (p2b_exploit.csv), agent_causal@2 :")
        print(sel.to_string(index=False) if len(sel) else "  (aucune ligne)")
    else:
        print(f"\n({ep} absent : lancer d'abord le harnais 2b.)")

    print(f"\nSorties : {args.out}/calib_slope.csv, calib_deciles.csv"
          + (", calib_folds.csv" if fo_rows else ""))


if __name__ == "__main__":
    main()
