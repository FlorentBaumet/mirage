"""Evaluation multi-bras / multi-modeles en UN SEUL passage par symbole.

Questions pre-enregistrees (configs/phase1d_crypto_prereg.yaml, phase1e_crypto_prereg.yaml) :
  - a etat IDENTIQUE, le MLP extrait-il de l'OFI une information que le lineaire manque ?
  - une description de la PROFONDEUR hors meilleur niveau (imb_deep, slope_asym) aide-t-elle
    a prevoir le rendement ?

POURQUOI UNE MATRICE LARGE, DECOUPEE PAR BRAS
---------------------------------------------
Les trois bras (base, ofi, deep) ne different que par leurs colonnes. Construire UNE matrice
d'etat large (base 5 + deep 2 + ofi) puis la DECOUPER garantit que les trois bras portent
EXACTEMENT les memes lignes, les memes folds et les memes cibles : la comparaison est
appariee par construction. Construire les bras separement rouvrirait la porte a un decalage
silencieux des echantillons.

Le bras `base` + modele `linear` doit reproduire A L'IDENTIQUE les predictions publiees dans
`experiments/crypto_lob_oos_{SYM}.npz` (cle `pred`). C'est le controle d'integrite, active
par `--check`.

    python scripts/crypto/arm_eval.py --symbols DOGEUSDT --check
    python scripts/crypto/arm_eval.py
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from crypto_lob import _merge_csv, load_cached, ofi_path_for
from sklearn.exceptions import ConvergenceWarning

from mirage.backtest import position_changes
from mirage.metrics import r2_per_dim
from mirage.splits import walk_forward_splits
from mirage.state import DEEP_COLS, OFI_COL, RET_IDX, STATE_COLS, build_state
from mirage.wm import MLPWM, LinearWM, make_supervised

warnings.filterwarnings("ignore", category=ConvergenceWarning)

LOOKBACK, NFOLDS, MINTRAIN = 16, 5, 0.4
EMBARGO = max(10, LOOKBACK)
FEES_BP = (0.0, 2.0, 5.5)
REF_DIR = "experiments"  # bras de base publie (controle d'integrite)
SPREAD_IDX = STATE_COLS.index("spread_rel")

# Ordre des colonnes de la matrice large : build_state(..., deep=True, ofi=...) produit
# STATE_COLS + DEEP_COLS + [OFI_COL]. On derive les indices de cet ordre, jamais en dur.
WIDE_COLS = STATE_COLS + DEEP_COLS + [OFI_COL]
_IDX = {c: i for i, c in enumerate(WIDE_COLS)}
ARMS = {
    "base": [_IDX[c] for c in STATE_COLS],
    "ofi": [_IDX[c] for c in STATE_COLS + [OFI_COL]],
    "deep": [_IDX[c] for c in STATE_COLS + DEEP_COLS],
}
ARM_DIMS = {
    "base": list(STATE_COLS),
    "ofi": list(STATE_COLS) + [OFI_COL],
    "deep": list(STATE_COLS) + list(DEEP_COLS),
}
MODELS = ("linear", "mlp")


def build_wide(pkls: list[str]) -> list[np.ndarray]:
    """UN passage sur les donnees : etat large PAR JOUR (aucune fenetre a cheval).

    Renvoie une liste de tableaux (n_jour, len(WIDE_COLS)) : l'etat large brut de chaque
    journee, dans l'ordre. La supervision par bras se fait ensuite jour par jour, comme
    dans crypto_lob.build_symbol, pour que les fenetres ne traversent jamais la nuit.
    """
    S_days = []
    for pf in pkls:
        # pickle local, produit par nos propres scripts (data/raw/crypto_lob) : source de
        # confiance, comme dans crypto_lob.build_symbol.
        bars = pd.read_pickle(pf)
        op = ofi_path_for(pf)
        if not os.path.exists(op):
            raise SystemExit(
                f"OFI absent : {op}\nCouverture incomplete : lance d'abord "
                f"scripts/crypto/fetch_ofi.py (un jour manquant avantagerait silencieusement "
                f"un bras d'un echantillon different).")
        ofi = pd.read_pickle(op)
        S, _ = build_state(bars, ofi=ofi, deep=True)
        if list(S.columns) != WIDE_COLS:  # garde-fou : l'ordre des colonnes porte le sens
            raise SystemExit(f"Ordre de colonnes inattendu : {list(S.columns)} != {WIDE_COLS}")
        S_days.append(S.values)
        del bars, S
    return S_days


def _days_and_spread(S_days: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Identifiants de journee et spread a la barre de decision (identiques pour tout bras)."""
    days, spr = [], []
    for di, Sd in enumerate(S_days):
        pos = np.arange(LOOKBACK, len(Sd))          # positions testables dans le jour
        days.append(np.full(len(pos), di))
        spr.append(Sd[pos - 1, SPREAD_IDX])         # spread a la barre de decision
    return np.concatenate(days), np.concatenate(spr)


def eval_symbol(sym: str, S_days: list[np.ndarray], days, spread,
                arms=ARMS, models=MODELS):
    """Evalue les bras x modeles demandes sur les MEMES folds. Renvoie (rows_1step, arrays).

    Chaque bras est supervise JOUR PAR JOUR a partir des colonnes de la matrice large :
    les trois bras portent donc exactement les memes lignes, dans le meme ordre.
    """
    rows1, arrays = [], {}
    y_ref = None
    for arm, idx in arms.items():
        idx = np.asarray(idx)
        Xs, Ys = [], []
        for Sd in S_days:
            X, Y, _, _ = make_supervised(Sd[:, idx], LOOKBACK)
            Xs.append(X)
            Ys.append(Y)
        Xa, Ya = np.vstack(Xs), np.vstack(Ys)
        del Xs, Ys
        # APPARIEMENT, verifie la OU il a un sens. Chaque bras vient d'etre supervise a
        # partir de SES colonnes : comparer les cibles ici teste donc reellement
        # l'echantillon. Le faire a la relecture du .npz ne testerait rien, puisque ce
        # fichier ne stocke qu'UNE cible (celle du bras de reference) - c'est exactement
        # ce qui rendait l'ancienne garde tautologique.
        if y_ref is None:
            y_ref = Ya[:, RET_IDX]
        elif not np.array_equal(y_ref, Ya[:, RET_IDX]):
            raise SystemExit(f"[{sym}] le bras '{arm}' ne porte pas la meme cible que le "
                             f"premier bras construit : echantillons non appariables.")
        dims = ARM_DIMS[arm]
        for mname in models:
            P, A, D, SP = [], [], [], []
            for fi, (tr, te) in enumerate(walk_forward_splits(len(Xa), NFOLDS, EMBARGO,
                                                              MINTRAIN, "expanding")):
                model = LinearWM() if mname == "linear" else MLPWM()
                model.fit(Xa[tr], Ya[tr])
                pred_te = model.predict(Xa[te])
                # meme baseline que phase1a : dernier etat de la fenetre, ret remis a 0.
                base = Xa[te][:, -len(idx):].copy()
                base[:, RET_IDX] = 0.0
                r2 = r2_per_dim(Ya[te], pred_te, base)
                for c, dim in enumerate(dims):
                    rows1.append(dict(symbol=sym, fold=fi, arm=arm, model=mname,
                                      dim=dim, r2=r2[c]))
                P.append(pred_te[:, RET_IDX])
                A.append(Ya[te][:, RET_IDX])
                D.append(days[te])
                SP.append(spread[te])
            pred, y = np.concatenate(P), np.concatenate(A)
            dd, sp = np.concatenate(D), np.concatenate(SP)
            posn = np.sign(pred)                       # position = signe de la prediction
            arrays[(arm, mname)] = dict(
                pred=pred, y=y, day=dd,
                gross=posn * y,
                dpos=position_changes(posn, dd),       # remise a plat a chaque jour
                half=sp / 2.0)
        del Xa, Ya
    return rows1, arrays


def _per_day(npz: dict, arrays: dict) -> None:
    """Reduit chaque (bras, modele) a des SOMMES PAR JOUR, suffisantes au bootstrap.

    Chaque entree est reduite depuis SA PROPRE cible et SES PROPRES journees
    (`arrays[(arm, mname)]["y"]` / `["day"]`), jamais depuis celles d'un autre bras.
    C'est ce qui donne a `sse_b_{tag}` la valeur d'une EMPREINTE : si un bras portait un
    echantillon decale, sa somme de y^2 differerait et `paired_arms.assert_apparies`
    echouerait. Reduire tous les bras avec la cible de reference - ce que faisait la
    version precedente - rendait la garde tautologique, puisqu'elle comparait deux fois
    le meme calcul.
    """
    ref = arrays[("base", "linear")]
    y_ref, day_ref = ref["y"], ref["day"]
    n_days = int(day_ref.max()) + 1
    counts = np.bincount(day_ref, minlength=n_days).astype(float)
    pool = np.where(counts > 0)[0]
    npz.update({"n_days": len(pool), "n_oos": len(y_ref), "day_pool": pool,
                "count": counts[pool], "y": y_ref, "day": day_ref})
    for (arm, mname), d in arrays.items():
        tag = f"{arm}_{mname}"
        y, day = d["y"], d["day"]
        c = np.bincount(day, minlength=n_days)
        if not np.array_equal(np.where(c > 0)[0], pool):
            raise SystemExit(f"[_per_day] le bras {tag} ne porte pas les memes journees "
                             f"que base_linear : non appariable.")
        npz[f"sse_b_{tag}"] = np.bincount(day, weights=y**2, minlength=n_days)[pool]
        npz[f"sse_m_{tag}"] = np.bincount(day, weights=(y - d["pred"]) ** 2,
                                          minlength=n_days)[pool]
        npz[f"gross_{tag}"] = np.bincount(day, weights=d["gross"], minlength=n_days)[pool]
        npz[f"dpos_{tag}"] = np.bincount(day, weights=d["dpos"], minlength=n_days)[pool]
        npz[f"dposhalf_{tag}"] = np.bincount(day, weights=d["dpos"] * d["half"],
                                             minlength=n_days)[pool]


def check_base_linear(sym: str, arrays: dict) -> bool:
    """CONTROLE D'INTEGRITE : base+linear doit reproduire le bras publie a l'identique."""
    pf = os.path.join(REF_DIR, f"crypto_lob_oos_{sym}.npz")
    if not os.path.exists(pf):
        raise SystemExit(f"controle impossible : {pf} introuvable.")
    with np.load(pf) as z:
        ref = {k: z[k] for k in z.files}
    d = arrays[("base", "linear")]
    ok = True
    print(f"\n=== Controle d'integrite [{sym}] : bras base + lineaire vs {pf} ===")
    for k, mine in (("y", d["y"]), ("day", d["day"]), ("pred", d["pred"])):
        r = ref[k].astype(float)
        if len(r) != len(mine):
            print(f"  {k:5s} : TAILLES DIFFERENTES {len(r)} (publie) vs {len(mine)} (calcule)")
            ok = False
            continue
        diff = float(np.max(np.abs(r - mine))) if len(r) else 0.0
        eq = np.array_equal(r, np.asarray(mine, float))
        print(f"  {k:5s} : identique (np.array_equal) = {eq} ; max|diff| = {diff:.3e}")
        ok &= eq
    y, p = d["y"], d["pred"]
    r2_mine = 1.0 - np.sum((y - p) ** 2) / np.sum(y**2)
    y, p = ref["y"].astype(float), ref["pred"].astype(float)
    r2_ref = 1.0 - np.sum((y - p) ** 2) / np.sum(y**2)
    print(f"  R2_OOS(ret) publie = {r2_ref:.8f} ; calcule = {r2_mine:.8f} ; "
          f"ecart = {abs(r2_ref - r2_mine):.3e}")
    if not ok:
        raise SystemExit("ECHEC du controle d'integrite : le bras base+lineaire ne "
                         "reproduit pas les predictions publiees. C'EST UN BUG.")
    print("  -> OK : le bras base+lineaire est identique au bras publie.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments_arms",
                    help="dossier des sorties (npz + CSV).")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="sous-ensemble de symboles (defaut : tous ceux qui ont des .pkl).")
    ap.add_argument("--check", action="store_true",
                    help="verifie que base+linear reproduit experiments/crypto_lob_oos_<SYM>.npz "
                         "(force le recalcul du symbole meme si le npz existe deja).")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = load_cached()
    if not cache:
        raise SystemExit("Aucun .pkl dans data/raw/crypto_lob.")
    if args.symbols:
        inconnus = [s for s in args.symbols if s not in cache]
        if inconnus:
            raise SystemExit(f"Aucun .pkl pour : {', '.join(inconnus)}. "
                             f"Disponibles : {', '.join(cache)}")
        cache = {s: cache[s] for s in args.symbols}

    all_rows, checked = [], 0
    for sym, pkls in cache.items():
        out_npz = os.path.join(args.out, f"arms_oos_{sym}.npz")
        if os.path.exists(out_npz) and not args.check:
            print(f"  [{sym}] deja fait ({out_npz}) - saute.")
            continue
        print(f"  [{sym}] construction de l'etat large ({len(pkls)} jours)...", flush=True)
        S_days = build_wide(pkls)
        days, spread = _days_and_spread(S_days)
        if args.check:
            # Controle d'integrite seul : on ne calcule QUE le bras publie base+lineaire
            # (le controle ne porte que sur lui), sans ecrire de npz - un run complet de
            # 3 bras x 2 modeles serait bien plus long pour la meme verification.
            print(f"  [{sym}] controle base+lineaire...", flush=True)
            _, arrays = eval_symbol(sym, S_days, days, spread,
                                    arms={"base": ARMS["base"]}, models=("linear",))
            del S_days
            check_base_linear(sym, arrays)
            checked += 1
            continue
        print(f"  [{sym}] {len(days)} echantillons sur {days.max() + 1} jours, "
              f"3 bras x 2 modeles...", flush=True)
        rows1, arrays = eval_symbol(sym, S_days, days, spread)
        del S_days
        npz: dict = {}
        _per_day(npz, arrays)
        np.savez_compressed(out_npz, **npz)
        all_rows += rows1
        print(f"  [{sym}] -> {out_npz}", flush=True)

    if all_rows:
        _merge_csv(os.path.join(args.out, "arms_1step.csv"), pd.DataFrame(all_rows))
        print(f"\nCSV 1-step : {os.path.join(args.out, 'arms_1step.csv')}")
    elif not args.check:
        print("\nAucun symbole traite : rien a ecrire.")
    if args.check:
        print(f"\nControles d'integrite passes : {checked} symbole(s).")


if __name__ == "__main__":
    main()
