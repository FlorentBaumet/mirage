"""Phase 2 : un AGENT qui planifie dans le world model appris.

Question pre-enregistree (configs/phase2_crypto_prereg.yaml) : un agent qui planifie a
horizon dans le modele trouve-t-il un edge REEL, ou exploite-t-il les ERREURS du modele ?

LA MESURE CENTRALE
------------------
L'agent choisit ses positions en n'optimisant que la recompense IMAGINEE, celle que lui
promet le modele. On rejoue ensuite CES MEMES positions sur les rendements VRAIS, avec
les memes couts. L'ecart net_imagine - net_reel est la model exploitation, quantifiee.

    agent : positions -> { net imagine (rendements predits), net reel (rendements vrais) }
    myope : p = signe(rendement predit)  -> le bras publie des phases 1/1c/1d/1e

L'AGENT EST UN PLANIFICATEUR EXACT, PAS UNE POLITIQUE APPRISE
-------------------------------------------------------------
Choix de methode, pas de facilite : tout echec doit etre imputable au MONDE appris, jamais
a la variance d'un optimiseur. `mirage.plan.plan_positions` donne l'optimal sous le modele,
exactement (verifie par enumeration exhaustive dans tests/test_plan.py). S'il ne gagne pas,
c'est que le modele est faux - point.

LE COUT ENTRE DANS LE PLAN
--------------------------
Le planificateur paie le demi-spread + les frais a chaque changement de position, donc il
connait le regime de frais. Les positions de l'agent sont donc recalculees PAR REGIME DE
FRAIS : un agent qui planifie a 5,5 bp ne trade pas comme un agent qui planifie a 0 bp.

REUTILISATION
-------------
build_wide / _days_and_spread viennent d'arm_eval.py : memes donnees, memes folds, memes
cout, meme appariement que les phases 1d/1e. Le bras `myope` doit reproduire a l'identique
le bras publie.

    python scripts/crypto/agent_plan.py --check --symbols DOGEUSDT
    python scripts/crypto/agent_plan.py
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from arm_eval import ARMS, EMBARGO, LOOKBACK, MINTRAIN, NFOLDS, _days_and_spread, build_wide
from bootstrap_signif import boot_mult
from crypto_lob import _merge_csv, load_cached
from sklearn.exceptions import ConvergenceWarning

from mirage.backtest import position_changes
from mirage.plan import plan_positions
from mirage.splits import walk_forward_splits
from mirage.state import RET_IDX
from mirage.wm import MLPWM, LinearWM, make_supervised

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HORIZON = 10                      # horizon primaire FIGE (pre-enregistrement)
HORIZONS_ROBUSTESSE = (1, 5, 20)  # diagnostics, ne servent JAMAIS a choisir
FEES_BP = (0.0, 2.0, 5.5)
FEE_PRIMAIRE = 2.0
N_BOOT, SEED, BLK = 2000, 0, 3
REF_DIR = "experiments"

BRAS_FIXES = ("myope", "plat", "oracle")


def tag(base: str, fee: float) -> str:
    """Nom d'un bras planifie, qui depend du regime de frais dans lequel il a planifie."""
    return f"{base}@{fee:g}"


def _run_bounds(days: np.ndarray) -> np.ndarray:
    """Indices de debut de chaque bloc contigu de meme journee.

    Meme decoupage que `mirage.backtest.position_changes` : une frontiere de journee
    remet la position a plat, et un fold qui couperait une journee en deux ne cree PAS
    de frontiere (les deux morceaux portent le meme identifiant de jour).
    """
    if len(days) == 0:
        return np.zeros(0, dtype=int)
    brk = np.flatnonzero(days[1:] != days[:-1]) + 1
    return np.concatenate([[0], brk, [len(days)]])


def plan_by_run(r: np.ndarray, c: np.ndarray, days: np.ndarray, horizon: int) -> np.ndarray:
    """Planifie par blocs de journee. Un plan ne traverse jamais une frontiere de jour,
    et chaque bloc repart a plat : c'est la convention de cout du projet."""
    pos = np.empty(len(r))
    b = _run_bounds(days)
    for k in range(len(b) - 1):
        s, e = b[k], b[k + 1]
        pos[s:e] = plan_positions(r[s:e], c[s:e], horizon)
    return pos


def _reduce(dst: dict, name: str, gross_img, gross_real, pos, half, days, n_days):
    dpos = position_changes(pos, days)
    dst[f"gi_{name}"] = np.bincount(days, weights=gross_img, minlength=n_days)
    dst[f"gr_{name}"] = np.bincount(days, weights=gross_real, minlength=n_days)
    dst[f"dp_{name}"] = np.bincount(days, weights=dpos, minlength=n_days)
    dst[f"dph_{name}"] = np.bincount(days, weights=dpos * half, minlength=n_days)


def eval_symbol(S_days, days, spread, model_name: str = "linear") -> dict:
    """Entraine le monde une fois, puis fait planifier l'agent dans chaque regime de frais."""
    idx = np.asarray(ARMS["base"])
    Xs, Ys = [], []
    for Sd in S_days:
        X, Y, _, _ = make_supervised(Sd[:, idx], LOOKBACK)
        Xs.append(X)
        Ys.append(Y)
    Xa, Ya = np.vstack(Xs), np.vstack(Ys)
    del Xs, Ys

    splits = list(walk_forward_splits(len(Xa), NFOLDS, EMBARGO, MINTRAIN, "expanding"))
    used = np.zeros(len(Xa), dtype=bool)
    for _tr, te in splits:
        used[te] = True
    keep = np.flatnonzero(used)

    rhat = np.empty(len(keep))
    rtrue = np.empty(len(keep))
    for tr, te in splits:
        model = LinearWM() if model_name == "linear" else MLPWM()
        model.fit(Xa[tr], Ya[tr])
        sel = np.searchsorted(keep, te)      # les index de test sont contigus et croissants
        rhat[sel] = model.predict(Xa[te])[:, RET_IDX]
        rtrue[sel] = Ya[te][:, RET_IDX]
    del Xa, Ya

    dd = days[keep]
    sp = spread[keep]
    half = sp / 2.0
    n_days = int(dd.max()) + 1
    npz: dict = {"y": rtrue, "pred": rhat, "day": dd, "n_oos": len(rtrue),
                 "count": np.bincount(dd, minlength=n_days).astype(float)}

    # --- references sans planification ------------------------------------------------
    _reduce(npz, "myope", np.sign(rhat) * rhat, np.sign(rhat) * rtrue,
            np.sign(rhat), half, dd, n_days)
    z = np.zeros_like(rhat)
    _reduce(npz, "plat", z, z, z, half, dd, n_days)
    _reduce(npz, "oracle", np.sign(rtrue) * rhat, np.sign(rtrue) * rtrue,
            np.sign(rtrue), half, dd, n_days)

    # --- agent et clairvoyant, UN PLAN PAR REGIME DE FRAIS -----------------------------
    for fee in FEES_BP:
        c = half + fee * 1e-4
        p_ag = plan_by_run(rhat, c, dd, HORIZON)
        _reduce(npz, tag("agent", fee), p_ag * rhat, p_ag * rtrue, p_ag, half, dd, n_days)
        p_cl = plan_by_run(rtrue, c, dd, HORIZON)
        _reduce(npz, tag("clairv", fee), p_cl * rtrue, p_cl * rtrue, p_cl, half, dd, n_days)

    # --- robustesse d'horizon, au regime primaire seulement ----------------------------
    c0 = half + FEE_PRIMAIRE * 1e-4
    for H in HORIZONS_ROBUSTESSE:
        p = plan_by_run(rhat, c0, dd, H)
        _reduce(npz, tag(f"agent_H{H}", FEE_PRIMAIRE), p * rhat, p * rtrue, p, half,
                dd, n_days)
    return npz


def _net_day(npz: dict, name: str, fee: float, imaginee: bool) -> np.ndarray:
    g = npz[f"gi_{name}"] if imaginee else npz[f"gr_{name}"]
    return g - npz[f"dph_{name}"] - fee * 1e-4 * npz[f"dp_{name}"]


def _boot_mean(m, nd, c):
    return (m @ nd) / (m @ c) * 1e4


def _pool(npz: dict, nd: np.ndarray):
    """Restreint le tirage aux journees REELLEMENT presentes en test.

    C'est la convention de `bootstrap_signif.py`, celle des IC publies. `paired_arms.py`
    ne filtre pas et dilue donc ses tirages dans les journees vides ; ce n'est pas faux
    (l'appariement tient dans les deux cas) mais c'est inutilement large. Ici on filtre :
    une journee sans echantillon de test n'est pas une journee a tirer.
    """
    c = npz["count"].astype(float)
    k = c > 0
    return nd[k], c[k]


def arm_stats(npz: dict, sym: str, name: str, fee: float, imaginee: bool) -> dict:
    """Net_bp d'un bras (moyenne + IC95 par jour). `imaginee` choisit le flux de rendement."""
    nd, c = _pool(npz, _net_day(npz, name, fee, imaginee))
    nb = _boot_mean(boot_mult(np.random.default_rng(SEED), len(c), N_BOOT, 1), nd, c)
    return {"symbol": sym, "bras": name, "fee_bp": fee, "nature": "imag" if imaginee
            else "reel", "net_bp": float(nd.sum() / c.sum() * 1e4),
            "net_lo": float(np.percentile(nb, 2.5)), "net_hi": float(np.percentile(nb, 97.5)),
            "turnover": float(npz[f"dp_{name}"].sum() / c.sum())}


def compare(npz: dict, sym: str, gain: str, ref: str, fee: float, imaginee: bool) -> dict:
    """Compare APPARIEE (gain - ref) : MEMES journees tirees pour les deux bras."""
    k = npz["count"] > 0
    c = npz["count"].astype(float)[k]
    ng = _net_day(npz, gain, fee, imaginee)[k]
    nr = _net_day(npz, ref, fee, imaginee)[k]
    m = boot_mult(np.random.default_rng(SEED), len(c), N_BOOT, 1)
    mb = boot_mult(np.random.default_rng(SEED), len(c), N_BOOT, BLK)
    d = _boot_mean(m, ng, c) - _boot_mean(m, nr, c)
    db = _boot_mean(mb, ng, c) - _boot_mean(mb, nr, c)
    return {"symbol": sym, "comparaison": f"{gain} - {ref}", "fee_bp": fee,
            "nature": "imag" if imaginee else "reel",
            "delta_bp": float((ng.sum() - nr.sum()) / c.sum() * 1e4),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "lo_blk": float(np.percentile(db, 2.5)), "hi_blk": float(np.percentile(db, 97.5))}


def ecart_exploitation(npz: dict, sym: str, name: str, fee: float) -> dict:
    """LA MESURE DU SUJET : net imagine - net reel, memes positions, memes couts.

    Par journee l'ecart vaut exactement (rendement imagine - rendement reel) du meme bras :
    les couts sont identiques dans les deux flux, ils s'annulent. L'IC95 est donc celui
    d'une moyenne par jour, construit avec le meme tirage que les autres comparaisons.
    """
    d, c = _pool(npz, _net_day(npz, name, fee, True) - _net_day(npz, name, fee, False))
    m = boot_mult(np.random.default_rng(SEED), len(c), N_BOOT, 1)
    mb = boot_mult(np.random.default_rng(SEED), len(c), N_BOOT, BLK)
    db = _boot_mean(m, d, c)
    dbb = _boot_mean(mb, d, c)
    return {"symbol": sym, "bras": name, "fee_bp": fee,
            "ecart_bp": float(d.sum() / c.sum() * 1e4),
            "lo": float(np.percentile(db, 2.5)), "hi": float(np.percentile(db, 97.5)),
            "lo_blk": float(np.percentile(dbb, 2.5)),
            "hi_blk": float(np.percentile(dbb, 97.5))}


def check_myope(sym: str, npz: dict) -> None:
    """CONTROLE : le bras myope doit reproduire les predictions publiees a l'identique."""
    pf = os.path.join(REF_DIR, f"crypto_lob_oos_{sym}.npz")
    if not os.path.exists(pf):
        raise SystemExit(f"controle impossible : {pf} introuvable.")
    with np.load(pf) as z:
        ref = {k: z[k] for k in z.files}
    print(f"\n=== Controle [{sym}] : bras myope vs {pf} ===")
    for k, mine in (("y", npz["y"]), ("day", npz["day"]), ("pred", npz["pred"])):
        r = ref[k].astype(float)
        if len(r) != len(mine):
            raise SystemExit(f"ECHEC : {k} n'a pas la meme taille ({len(r)} vs {len(mine)}).")
        d = float(np.max(np.abs(r - np.asarray(mine, float)))) if len(r) else 0.0
        print(f"  {k:5s} : identique = {d == 0.0} ; max|diff| = {d:.3e}")
        if d != 0.0:
            raise SystemExit(f"ECHEC : {k} differe du bras publie. C'EST UN BUG.")
    r2 = 1.0 - float(np.sum((ref["y"] - ref["pred"]) ** 2)) / float(np.sum(ref["y"] ** 2))
    print(f"  R2_OOS(ret) = {r2:.8f}")
    print("  -> OK : le bras myope est identique au bras publie.")


def verdicts(net: pd.DataFrame, cmp_: pd.DataFrame, explo: pd.DataFrame,
             syms: list[str]) -> None:
    """Applique MECANIQUEMENT les regles du pre-enregistrement. Aucune interpretation.

    C'est le seul endroit qui decide. Les seuils sont ceux de
    configs/phase2_crypto_prereg.yaml et ne sont pas ajustables apres coup.
    """
    p = FEE_PRIMAIRE
    n = net[(net.fee_bp == p)].set_index(["symbol", "bras", "nature"])
    c2 = cmp_[(cmp_.fee_bp == p) & (cmp_.nature == "reel")].set_index(
        ["symbol", "comparaison"])
    e = explo[explo.fee_bp == p].set_index("symbol")
    ag, my, cl = tag("agent", p), "myope", tag("clairv", p)
    print("\n=== VERDICTS (regles figees du pre-enregistrement) ===")

    expl = [s for s in syms
            if n.loc[(s, ag, "imag"), "net_bp"] > 0 and n.loc[(s, ag, "reel"), "net_bp"] <= 0]
    print(f"exploitation_du_modele     : {len(expl)}/5  {expl}")

    battus = [s for s in syms if c2.loc[(s, f"{ag} - {my}"), "lo"] > 0]
    print(f"agent_bat_le_myope (>=4/5) : {len(battus)}/5  {battus}  -> "
          f"{'OUI' if len(battus) >= 4 else 'NON'}")

    edge = [s for s in syms if n.loc[(s, ag, "reel"), "net_lo"] > 0]
    print(f"edge_reel (>=4/5)          : {len(edge)}/5  {edge}  -> "
          f"{'OUI' if len(edge) >= 4 else 'NON'}")

    print("\n  symbole    net imagine   net reel   ecart      IC95 ecart      "
          "net reel myope   net reel clairv  net reel oracle")
    for s in syms:
        ec = e.loc[s]
        print(f"  {s:10s} {n.loc[(s, ag, 'imag'), 'net_bp']:+10.4f} "
              f"{n.loc[(s, ag, 'reel'), 'net_bp']:+10.4f} "
              f"{ec['ecart_bp']:+9.4f}  [{ec['lo']:+.4f}, {ec['hi']:+.4f}] "
              f"{n.loc[(s, my, 'reel'), 'net_bp']:+13.4f} "
              f"{n.loc[(s, cl, 'reel'), 'net_bp']:+15.4f} "
              f"{n.loc[(s, 'oracle', 'reel'), 'net_bp']:+15.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments_agent")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--model", default="linear", choices=("linear", "mlp"))
    ap.add_argument("--check", action="store_true",
                    help="verifie que le bras myope reproduit le bras publie, puis sort.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = load_cached()
    if not cache:
        raise SystemExit("Aucun .pkl dans data/raw/crypto_lob.")
    if args.symbols:
        inconnus = [s for s in args.symbols if s not in cache]
        if inconnus:
            raise SystemExit(f"Aucun .pkl pour : {', '.join(inconnus)}.")
        cache = {s: cache[s] for s in args.symbols}

    net_rows, cmp_rows, exp_rows = [], [], []
    for sym, pkls in cache.items():
        t0 = time.time()
        print(f"  [{sym}] etat large ({len(pkls)} jours)...", flush=True)
        S_days = build_wide(pkls)
        days, spread = _days_and_spread(S_days)
        print(f"  [{sym}] {len(days)} echantillons, {days.max() + 1} jours ; "
              f"entrainement + planification (H={HORIZON} x {len(FEES_BP)} regimes + "
              f"robustesse)...", flush=True)
        npz = eval_symbol(S_days, days, spread, model_name=args.model)
        del S_days
        if args.check:
            check_myope(sym, npz)
            continue
        # On ne sauve que les SOMMES PAR JOURNEE : suffisantes pour toute statistique
        # d'interet, et `count` en fait partie (sans lui le fichier est inanalysable).
        # Les tableaux par echantillon (3,8 M lignes) restent en memoire seulement.
        drop = {"y", "pred", "day"}
        np.savez_compressed(os.path.join(args.out, f"agent_oos_{sym}.npz"),
                            **{k: v for k, v in npz.items() if k not in drop})

        # On ECRIT PAR SYMBOLE (fusion, comme crypto_lob.py) : un run interrompu ne perd
        # rien, et le relancer reprend ou il s'est arrete. Sans cela, un arret en cours
        # de route jetterait plusieurs heures de calcul.
        n_s, c_s, e_s = [], [], []
        for fee in FEES_BP:
            for name in BRAS_FIXES + (tag("agent", fee), tag("clairv", fee)):
                n_s.append(arm_stats(npz, sym, name, fee, imaginee=False))
            for name in ("myope", tag("agent", fee)):
                n_s.append(arm_stats(npz, sym, name, fee, imaginee=True))
            e_s.append(ecart_exploitation(npz, sym, tag("agent", fee), fee))
        for H in HORIZONS_ROBUSTESSE:
            name = tag(f"agent_H{H}", FEE_PRIMAIRE)
            n_s.append(arm_stats(npz, sym, name, FEE_PRIMAIRE, imaginee=False))
            n_s.append(arm_stats(npz, sym, name, FEE_PRIMAIRE, imaginee=True))

        ag, my = tag("agent", FEE_PRIMAIRE), "myope"
        c_s.append(compare(npz, sym, ag, my, FEE_PRIMAIRE, imaginee=False))
        c_s.append(compare(npz, sym, ag, my, 0.0, imaginee=False))
        c_s.append(compare(npz, sym, ag, tag("clairv", FEE_PRIMAIRE), FEE_PRIMAIRE,
                           imaginee=False))
        c_s.append(compare(npz, sym, tag("clairv", FEE_PRIMAIRE), "oracle",
                           FEE_PRIMAIRE, imaginee=False))
        _merge_csv(os.path.join(args.out, "agent_net.csv"), pd.DataFrame(n_s))
        _merge_csv(os.path.join(args.out, "agent_compare.csv"), pd.DataFrame(c_s))
        _merge_csv(os.path.join(args.out, "agent_exploit.csv"), pd.DataFrame(e_s))
        net_rows += n_s
        cmp_rows += c_s
        exp_rows += e_s
        print(f"  [{sym}] fait en {time.time() - t0:.0f}s ; CSV fusionnes.", flush=True)

    if args.check:
        print("\nControle passe.")
        return

    df_net = pd.DataFrame(net_rows)
    df_cmp = pd.DataFrame(cmp_rows)
    df_exp = pd.DataFrame(exp_rows)

    # --- CONTROLE ANTI-BUG : le DP maximise le net imagine, donc il ne peut pas perdre
    # contre le myope sur ce meme critere. Une violation n'est pas un resultat.
    # Le controle est relu DEPUIS LES CSV : un run reparti par sous-ensemble de symboles
    # doit verifier le corpus entier, pas seulement ce qu'il vient de calculer.
    ag = tag("agent", FEE_PRIMAIRE)
    im = df_net[(df_net.fee_bp == FEE_PRIMAIRE) & (df_net.nature == "imag")].set_index(
        ["symbol", "bras"])["net_bp"]
    tous = sorted(df_net.symbol.unique())
    ok = [s for s in tous if im.get((s, ag), -1e9) >= im.get((s, "myope"), -1e9)]
    print(f"\nCONTROLE anti-bug : net imagine agent >= net imagine myope sur "
          f"{len(ok)}/{len(tous)}")
    for s in tous:
        print(f"  {s:10s} agent imagine {im.get((s, ag), float('nan')):+.4f} bp   "
              f"myope imagine {im.get((s, 'myope'), float('nan')):+.4f} bp")
    if len(ok) < len(tous):
        raise SystemExit("ECHEC DU CONTROLE : le planificateur n'est pas optimal. C'EST UN BUG.")

    verdicts(df_net, df_cmp, df_exp, tous)
    print(f"\nSorties : {args.out}")


if __name__ == "__main__":
    main()
