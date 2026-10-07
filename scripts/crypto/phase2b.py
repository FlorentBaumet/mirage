"""Phase 2b : re-mesure CAUSALE de l'agent de la Phase 2.

Motif. Le planificateur publie (`agent_plan.plan_by_run` sur la serie rhat) consomme a la
barre t les predictions rhat[t+1 .. t+H-1], qui sont faites PLUS TARD que t. Or
`make_supervised` aligne X[k] = etats k..k+L-1 et Y[k] = etat k+L : rhat[t+1] est donc
calculee sur une fenetre qui CONTIENT Y[t], c'est-a-dire le rendement que la position de t
encaisse. Le verdict `edge_reel` de la Phase 2 est suspendu depuis le 2026-10-08
(voir l'erratum en tete de reports/PHASE2_AGENT.md).

Ce harnais refait la meme evaluation avec un planificateur honnete : a la barre t, l'agent
ne dispose que des previsions FAITES A t, soit le rollout autoregressif du world model du
fold depuis la fenetre de t et elle seule. Regles, seuils et bras :
configs/phase2b_crypto_prereg.yaml, fige et commite le 2026-10-08 AVANT tout calcul.

    python scripts/crypto/phase2b.py --symbols DOGEUSDT --check
    python scripts/crypto/phase2b.py --out experiments_2b
    python scripts/crypto/phase2b.py --diag --perm 39 --shift 400

DEUX CONTROLES D'INTEGRITE BLOQUANTS (`--check`) : le bras myope doit reproduire
experiments/crypto_lob_oos_<SYM>.npz AU BIT PRES, et `agent_fuite` doit redonner le net
publie de la Phase 2 a 1e-9. Si l'un echoue, ce n'est pas un resultat, c'est un bug : le
harnais s'arrete et n'ecrit rien. Aucun de ces deux controles ne se negocie.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from agent_diag import diag_arrays
from agent_plan import (
    BRAS_FIXES,
    FEE_PRIMAIRE,
    FEES_BP,
    HORIZON,
    HORIZONS_ROBUSTESSE,
    N_BOOT,
    SEED,
    _reduce,
    arm_stats,
    check_myope,
    compare,
    ecart_exploitation,
    plan_by_run,
    tag,
)
from arm_eval import ARMS, EMBARGO, LOOKBACK, MINTRAIN, NFOLDS, _days_and_spread, build_wide
from crypto_lob import _merge_csv, load_cached

from mirage.plan import mask_day_end, plan_positions_causal
from mirage.splits import walk_forward_splits
from mirage.state import RET_IDX
from mirage.wm import MLPWM, LinearWM, make_supervised, rollout

OUT = "experiments_2b"
REF_AGENT = "experiments_agent"     # sorties de la Phase 2 : le controle d'integrite
H_MAX = 20                          # plus grand H de robustesse du prereg : un seul rollout
CHUNK = 250_000                     # lignes de rollout par bloc (memoire)
K_PERM, K_SHIFT = 39, 400


def eval_symbol_2b(sym: str, S_days, days, spread, model_name: str = "linear"):
    """Le monde de la Phase 2, plus le rollout causal dans la foulee des memes folds.

    Renvoie (npz, ext). `ext` porte ce que les diagnostics rejouent : les chemins R
    (masques, H_MAX colonnes), rhat, rtrue, half et dd.
    """
    idx = np.asarray(ARMS["base"])
    D = len(idx)
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

    n_oos = len(keep)
    rhat = np.empty(n_oos)
    rtrue = np.empty(n_oos)
    R = np.empty((n_oos, H_MAX))
    for tr, te in splits:
        model = LinearWM() if model_name == "linear" else MLPWM()
        model.fit(Xa[tr], Ya[tr])
        sel = np.searchsorted(keep, te)      # les index de test sont contigus et croissants
        rhat[sel] = model.predict(Xa[te])[:, RET_IDX]
        rtrue[sel] = Ya[te][:, RET_IDX]
        # Le rollout part des MEMES fenetres que la prevision a un pas : X[k] = etats
        # k..k+L-1, donc a la barre t il ne voit que la fenetre de t. Une fenetre a cheval
        # sur deux journees est impossible, elle ne l'etait deja pas pour l'entrainement.
        base = int(sel[0])
        te0 = int(te[0])
        for a in range(0, len(te), CHUNK):
            e = min(a + CHUNK, len(te))
            win = Xa[te0 + a:te0 + e].reshape(-1, LOOKBACK, D)
            R[base + a:base + e] = rollout(model, win, H_MAX)[:, :, RET_IDX]
    del Xa, Ya

    dd = days[keep]
    half = spread[keep] / 2.0
    n_days = int(dd.max()) + 1
    npz: dict = {"y": rtrue, "pred": rhat, "half": half, "day": dd, "n_oos": n_oos,
                 "count": np.bincount(dd, minlength=n_days).astype(float),
                 "symbole": sym}

    # --- references sans planification ------------------------------------------------
    _reduce(npz, "myope", np.sign(rhat) * rhat, np.sign(rhat) * rtrue,
            np.sign(rhat), half, dd, n_days)
    z = np.zeros_like(rhat)
    _reduce(npz, "plat", z, z, z, half, dd, n_days)
    _reduce(npz, "oracle", np.sign(rtrue) * rhat, np.sign(rtrue) * rtrue,
            np.sign(rtrue), half, dd, n_days)

    Rm = mask_day_end(R, dd)

    # --- les trois planificateurs, UN PLAN PAR REGIME DE FRAIS -------------------------
    for fee in FEES_BP:
        c = half + fee * 1e-4
        # `agent_fuite` : le bras PUBLIE de la Phase 2, recalcule ici. Le planificateur
        # relit rhat[t+1..t+H-1], donc l'ecart a `agent_causal` EST la mesure de la fuite.
        p_f = plan_by_run(rhat, c, dd, HORIZON)
        _reduce(npz, tag("agent_fuite", fee), p_f * rhat, p_f * rtrue, p_f, half, dd, n_days)
        # `agent_causal` : le bras PRIMAIRE. R ne contient que des previsions faites a t.
        Cm = mask_day_end(np.repeat(c[:, None], H_MAX, axis=1), dd)
        p_c = plan_positions_causal(Rm[:, :HORIZON], Cm[:, :HORIZON], dd)
        _reduce(npz, tag("agent_causal", fee), p_c * rhat, p_c * rtrue, p_c, half, dd, n_days)
        # borne a information parfaite, inchangee par la correction
        p_cl = plan_by_run(rtrue, c, dd, HORIZON)
        _reduce(npz, tag("clairv", fee), p_cl * rtrue, p_cl * rtrue, p_cl, half, dd, n_days)

    # --- robustesse d'horizon et agent plat, au regime primaire seulement ---------------
    Cm2 = mask_day_end(np.repeat((half + FEE_PRIMAIRE * 1e-4)[:, None], H_MAX, axis=1), dd)
    for h in HORIZONS_ROBUSTESSE:
        p = plan_positions_causal(Rm[:, :h], Cm2[:, :h], dd)
        _reduce(npz, tag(f"agent_causal_H{h}", FEE_PRIMAIRE), p * rhat, p * rtrue, p,
                half, dd, n_days)
    # `agent_plat` : la barre t seule, puis plus rien. Isole ce que le rollout ajoute.
    Rp = mask_day_end(np.zeros((n_oos, HORIZON)), dd)
    Rp[:, 0] = rhat
    p_p = plan_positions_causal(Rp, Cm2[:, :HORIZON], dd)
    _reduce(npz, tag("agent_plat", FEE_PRIMAIRE), p_p * rhat, p_p * rtrue, p_p, half, dd,
            n_days)

    ext = {"R": Rm[:, :HORIZON], "C": Cm2[:, :HORIZON], "rhat": rhat, "rtrue": rtrue,
           "half": half, "dd": dd}
    return npz, ext


def causal_planner(C: np.ndarray):
    """`planner(X, c, dd, horizon)` du bras causal, pour `agent_diag.diag_arrays`.

    `X` est la matrice des chemins (n, H), permutee par le diagnostic. Le second argument
    est la serie de couts scalaire du harnais publie : le planificateur causal n'en veut
    pas (son cout est la MATRICE C, persistance du demi-spread courant), il est donc ignore.
    """
    def planner(X, c, dd, horizon):
        return plan_positions_causal(X[:, :horizon], C[:, :horizon], dd)
    return planner


def check_fuite(sym: str, npz: dict) -> None:
    """CONTROLE : `agent_fuite` doit redonner le net publie de la Phase 2 a 1e-9.

    `agent_fuite@f` est exactement le planificateur publie, applique au meme echantillon.
    S'il ne reproduit pas experiments_agent/agent_net.csv, alors ce harnais ne tourne pas
    sur le meme echantillon que la Phase 2 et AUCUNE de ses comparaisons n'aurait de sens.
    """
    cf = os.path.join(REF_AGENT, "agent_net.csv")
    if not os.path.exists(cf):
        raise SystemExit(f"controle impossible : {cf} introuvable.")
    df = pd.read_csv(cf)
    print(f"\n=== Controle [{sym}] : agent_fuite vs Phase 2 publiee ({cf}) ===", flush=True)
    for fee in FEES_BP:
        for nature in ("reel", "imag"):
            q = df[(df.symbol == sym) & (df.bras == tag("agent", fee))
                   & (df.fee_bp == fee) & (df.nature == nature)]
            if q.empty:
                continue
            ref = float(q.net_bp.iloc[0])
            mine = arm_stats(npz, sym, tag("agent_fuite", fee), fee,
                             nature == "imag")["net_bp"]
            d = abs(ref - mine)
            print(f"  agent@{fee:g} {nature:4s} : publie {ref:+.6f} ; ici {mine:+.6f} ; "
                  f"ecart {d:.2e}", flush=True)
            if d > 1e-9:
                raise SystemExit("ECHEC : agent_fuite ne redonne pas le net publie de la "
                                 "Phase 2. C'EST UN BUG.")
    print("  -> OK : le harnais 2b tourne bien sur l'echantillon de la Phase 2.", flush=True)


def check_rollout(ext: dict) -> None:
    """CONTROLE : la colonne 0 du rollout doit etre la prevision a un pas, au bit pres.

    Elle est calculee par le meme `model.predict` mais dans un autre chemin de code (le
    rollout reinjecte la prediction). Une divergence signalerait un decalage de fenetres.
    """
    d = float(np.max(np.abs(ext["R"][:, 0] - ext["rhat"]))) if len(ext["rhat"]) else 0.0
    print(f"\n=== Controle : R[:, 0] vs rhat ===\n  max|diff| = {d:.3e}", flush=True)
    if d != 0.0:
        raise SystemExit("ECHEC : R[:, 0] n'est pas la prevision a un pas. C'EST UN BUG.")


def verdicts_2b(net: pd.DataFrame, cmp_: pd.DataFrame, syms: list[str],
                conf: pd.DataFrame | None = None) -> dict:
    """Applique MECANIQUEMENT les regles de configs/phase2b_crypto_prereg.yaml.

    Aucune interpretation, aucun seuil ajustable : ce qui n'atteint pas le seuil est NON.
    Le seul cas ou ce harnais s'arrete est `planificateur_coherent`, que le prereg qualifie
    de bug (un DP exact ne peut pas perdre sur son propre critere).
    """
    p = FEE_PRIMAIRE
    n = net[net.fee_bp == p].set_index(["symbol", "bras", "nature"])
    c = cmp_[(cmp_.fee_bp == p) & (cmp_.nature == "reel")].set_index(
        ["symbol", "comparaison"])
    ag, fu, my = tag("agent_causal", p), tag("agent_fuite", p), "myope"

    coh = [s for s in syms
           if n.loc[(s, ag, "imag"), "net_bp"] >= n.loc[(s, my, "imag"), "net_bp"]]
    edge = [s for s in syms if n.loc[(s, ag, "reel"), "net_lo"] > 0]
    bat = [s for s in syms if c.loc[(s, f"{ag} - {my}"), "lo"] > 0]
    fuite = [s for s in syms if c.loc[(s, f"{fu} - {ag}"), "lo"] > 0]
    expl = [s for s in syms
            if n.loc[(s, ag, "imag"), "net_bp"] > 0 and n.loc[(s, ag, "reel"), "net_bp"] <= 0]

    v = {
        "planificateur_coherent": {"symboles": coh, "n": len(coh), "seuil": "5/5",
                                   "ok": len(coh) == len(syms)},
        "edge_reel_causal": {"symboles": edge, "n": len(edge), "seuil": ">= 4/5",
                             "ok": len(edge) >= 4, "primaire": True},
        "agent_causal_bat_le_myope": {"symboles": bat, "n": len(bat), "seuil": ">= 4/5",
                                      "ok": len(bat) >= 4},
        "fuite_significative": {"symboles": fuite, "n": len(fuite), "seuil": ">= 4/5",
                                "ok": len(fuite) >= 4},
        "exploitation_du_modele": {"symboles": expl, "n": len(expl)},
    }
    if conf is not None and len(conf):
        cf = conf.set_index("symbol")
        dedans = [s for s in syms if s in cf.index]
        bruit = [s for s in dedans if cf.loc[s, "obs_bp"] > cf.loc[s, "bruit_max"]]
        derive = [s for s in dedans
                  if cf.loc[s, "contrib_derive_lo"] <= 0 <= cf.loc[s, "contrib_derive_hi"]
                  and cf.loc[s, "contrib_timing_lo"] > 0]
        v["bruit_perd"] = {"symboles": bruit, "n": len(bruit), "seuil": ">= 4/5",
                           "ok": len(bruit) >= 4, "k_perm": K_PERM}
        v["derive_nulle"] = {"symboles": derive, "n": len(derive), "seuil": ">= 4/5",
                             "ok": len(derive) >= 4}

    print("\n=== VERDICTS 2b (regles figees de configs/phase2b_crypto_prereg.yaml) ===")
    for k, d in v.items():
        if k == "exploitation_du_modele":
            print(f"  {k:26s} : {d['n']}/{len(syms)}  {d['symboles']}")
        else:
            print(f"  {k:26s} : {d['n']}/{len(syms)}  seuil {d['seuil']}  -> "
                  f"{'OUI' if d['ok'] else 'NON'}  {d['symboles']}")

    print("\n  symbole    causal imag   causal reel  IC95 reel                "
          "fuite reel   myope reel   fuite-causal  causal-myope")
    for s in syms:
        print(f"  {s:10s} {n.loc[(s, ag, 'imag'), 'net_bp']:+10.4f} "
              f"{n.loc[(s, ag, 'reel'), 'net_bp']:+12.4f} "
              f"[{n.loc[(s, ag, 'reel'), 'net_lo']:+.4f}, "
              f"{n.loc[(s, ag, 'reel'), 'net_hi']:+.4f}]  "
              f"{n.loc[(s, fu, 'reel'), 'net_bp']:+11.4f}  "
              f"{n.loc[(s, my, 'reel'), 'net_bp']:+11.4f}  "
              f"{c.loc[(s, f'{fu} - {ag}'), 'delta_bp']:+12.4f}  "
              f"{c.loc[(s, f'{ag} - {my}'), 'delta_bp']:+11.4f}")

    if not v["planificateur_coherent"]["ok"]:
        raise SystemExit("ECHEC DU CONTROLE : net imagine agent_causal < net imagine myope. "
                         "Le planificateur n'est pas optimal. C'EST UN BUG.")
    return v


def write_results(out: str, v: dict, net: pd.DataFrame, cmp_: pd.DataFrame,
                  exp: pd.DataFrame, conf: pd.DataFrame | None, syms: list[str]) -> None:
    res = {
        "meta": {"nom": "phase2b_crypto_causal",
                 "prereg": "configs/phase2b_crypto_prereg.yaml",
                 "harnais": "scripts/crypto/phase2b.py",
                 "frais_primaire": FEE_PRIMAIRE, "horizon": HORIZON,
                 "horizons_robustesse": list(HORIZONS_ROBUSTESSE),
                 "n_boot": N_BOOT, "seed": SEED, "blocs_jours": 3,
                 "symboles": syms,
                 "echantillon": "44 journees publiees (configs/phase1c_crypto_prereg.yaml)",
                 "genere_le": "2026-10-08"},
        "verdicts": v,
        "net": json.loads(net.to_json(orient="records")),
        "compare": json.loads(cmp_.to_json(orient="records")),
        "exploit": json.loads(exp.to_json(orient="records")),
    }
    if conf is not None and len(conf):
        res["diag_confondants"] = json.loads(conf.to_json(orient="records"))
    p = os.path.join(out, "phase2b_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--model", default="linear", choices=("linear", "mlp"))
    ap.add_argument("--dates", nargs="*", default=None,
                    help="journees a utiliser ; defaut = les 44 publiees")
    ap.add_argument("--check", action="store_true",
                    help="controles d'integrite seuls, puis sortie sans rien ecrire.")
    ap.add_argument("--diag", action="store_true", help="ajoute les diagnostics des 2b regles.")
    ap.add_argument("--perm", type=int, default=K_PERM)
    ap.add_argument("--shift", type=int, default=K_SHIFT)
    args = ap.parse_args()

    cache = load_cached(args.dates)
    if not cache:
        raise SystemExit("Aucun .pkl dans data/raw/crypto_lob.")
    if args.symbols:
        inconnus = [s for s in args.symbols if s not in cache]
        if inconnus:
            raise SystemExit(f"Aucun .pkl pour : {', '.join(inconnus)}.")
        cache = {s: cache[s] for s in args.symbols}
    if not args.check:
        os.makedirs(args.out, exist_ok=True)

    net_rows, cmp_rows, exp_rows, conf_rows = [], [], [], []
    for sym, pkls in cache.items():
        t0 = time.time()
        print(f"  [{sym}] etat large ({len(pkls)} jours)...", flush=True)
        S_days = build_wide(pkls)
        days, spread = _days_and_spread(S_days)
        print(f"  [{sym}] {len(days)} echantillons, {days.max() + 1} jours ; monde + "
              f"rollout (H={H_MAX}) + planification...", flush=True)
        npz, ext = eval_symbol_2b(sym, S_days, days, spread, model_name=args.model)
        del S_days

        if args.check:
            check_myope(sym, npz)
            check_fuite(sym, npz)
            check_rollout(ext)
            continue

        # Un fichier PAR SYMBOLE fait foi : un run interrompu ne perd rien, et le relancer
        # reprend ou il s'est arrete. Les CSV sont FUSIONNES par symbole comme partout
        # ailleurs dans le projet : un run partiel n'ampute pas les symboles deja faits.
        np.savez_compressed(os.path.join(args.out, f"p2b_oos_{sym}.npz"), **npz)

        n_s, c_s, e_s = [], [], []
        for fee in FEES_BP:
            for name in BRAS_FIXES + (tag("agent_fuite", fee), tag("agent_causal", fee),
                                      tag("clairv", fee)):
                n_s.append(arm_stats(npz, sym, name, fee, imaginee=False))
            for name in ("myope", tag("agent_fuite", fee), tag("agent_causal", fee)):
                n_s.append(arm_stats(npz, sym, name, fee, imaginee=True))
            e_s.append(ecart_exploitation(npz, sym, tag("agent_causal", fee), fee))
            e_s.append(ecart_exploitation(npz, sym, tag("agent_fuite", fee), fee))
        for h in HORIZONS_ROBUSTESSE:
            name = tag(f"agent_causal_H{h}", FEE_PRIMAIRE)
            n_s.append(arm_stats(npz, sym, name, FEE_PRIMAIRE, imaginee=False))
            n_s.append(arm_stats(npz, sym, name, FEE_PRIMAIRE, imaginee=True))
        n_s.append(arm_stats(npz, sym, tag("agent_plat", FEE_PRIMAIRE), FEE_PRIMAIRE,
                             imaginee=False))
        n_s.append(arm_stats(npz, sym, tag("agent_plat", FEE_PRIMAIRE), FEE_PRIMAIRE,
                             imaginee=True))

        ag, fu, my = tag("agent_causal", FEE_PRIMAIRE), tag("agent_fuite", FEE_PRIMAIRE), "myope"
        cl = tag("clairv", FEE_PRIMAIRE)
        # LA mesure de la fuite : les deux bras ne different QUE par l'information dont le
        # planificateur dispose a t, et sont apparies par journee.
        c_s.append(compare(npz, sym, fu, ag, FEE_PRIMAIRE, imaginee=False))
        c_s.append(compare(npz, sym, ag, my, FEE_PRIMAIRE, imaginee=False))
        c_s.append(compare(npz, sym, fu, my, FEE_PRIMAIRE, imaginee=False))
        c_s.append(compare(npz, sym, ag, cl, FEE_PRIMAIRE, imaginee=False))
        c_s.append(compare(npz, sym, ag, my, 0.0, imaginee=False))
        c_s.append(compare(npz, sym, ag, tag("agent_causal_H1", FEE_PRIMAIRE), FEE_PRIMAIRE,
                           imaginee=False))

        _merge_csv(os.path.join(args.out, "p2b_net.csv"), pd.DataFrame(n_s))
        _merge_csv(os.path.join(args.out, "p2b_compare.csv"), pd.DataFrame(c_s))
        _merge_csv(os.path.join(args.out, "p2b_exploit.csv"), pd.DataFrame(e_s))
        net_rows += n_s
        cmp_rows += c_s
        exp_rows += e_s

        if args.diag:
            planner = causal_planner(ext["C"])
            o, v_my, conf = diag_arrays(
                sym, ext["rhat"], ext["rtrue"], ext["half"], ext["dd"], planner,
                args.perm, args.shift, x=ext["R"],
                net_csv=os.path.join(args.out, "p2b_net.csv"))
            for nm, rr in (("ordre", o), ("confondants", conf), ("vs_myope", v_my)):
                df = pd.DataFrame(rr)
                df.to_csv(os.path.join(args.out, f"p2b_diag_{nm}_{sym}.csv"), index=False)
                _merge_csv(os.path.join(args.out, f"p2b_diag_{nm}.csv"), df)
            conf_rows += conf

        print(f"  [{sym}] fait en {time.time() - t0:.0f}s ; CSV fusionnes.", flush=True)

    if args.check:
        print("\nControles passes.")
        return

    # Les verdicts sont relus DEPUIS LES CSV : un run reparti par sous-ensemble de symboles
    # doit juger le corpus entier, pas seulement ce qu'il vient de calculer.
    df_net = pd.read_csv(os.path.join(args.out, "p2b_net.csv"))
    df_cmp = pd.read_csv(os.path.join(args.out, "p2b_compare.csv"))
    df_exp = pd.read_csv(os.path.join(args.out, "p2b_exploit.csv"))
    conf = None
    cp = os.path.join(args.out, "p2b_diag_confondants.csv")
    if os.path.exists(cp):
        conf = pd.read_csv(cp)
    tous = sorted(df_net.symbol.unique())
    v = verdicts_2b(df_net, df_cmp, tous, conf)
    write_results(args.out, v, df_net, df_cmp, df_exp, conf, tous)
    print(f"\nSorties : {args.out}/p2b_net.csv, p2b_compare.csv, p2b_exploit.csv, "
          f"phase2b_results.json")


if __name__ == "__main__":
    main()
