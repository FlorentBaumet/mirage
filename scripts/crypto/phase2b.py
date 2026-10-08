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
import yaml
from agent_diag import diag_arrays
from agent_plan import (
    BLK,
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
from arm_eval import (
    ARMS,
    EMBARGO,
    LOOKBACK,
    MINTRAIN,
    NFOLDS,
    SPREAD_IDX,
    _days_and_spread,
    build_wide,
)
from bootstrap_signif import boot_mult
from crypto_lob import DATES_PUBLIEES_1C, _merge_csv, date_of, load_cached

from mirage.plan import mask_day_end, plan_positions_causal
from mirage.splits import walk_forward_splits
from mirage.state import RET_IDX
from mirage.wm import MLPWM, LinearWM, cumulative_target, make_supervised, rollout
from mirage.wm_eval import rollout_r2

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
    # `len(X_jour) = len(S_jour) - LOOKBACK` : on connait donc la taille totale AVANT de
    # construire quoi que ce soit, et on remplit deux tableaux prealloues jour par jour.
    # Un `np.vstack` des X de chaque jour doublerait le pic (la liste des jours, puis la
    # copie) : ~4,9 Go de trop sur l'union des 88 journees. L'egalite ci-dessus est la meme
    # que celle dont `calib_diag.n_train_par_pli` tire les tailles de pli, donc elle est
    # verifiee ici plutot que supposee.
    ns = [len(Sd) - LOOKBACK for Sd in S_days]
    Xa = np.empty((sum(ns), LOOKBACK * D))
    Ya = np.empty((sum(ns), D))
    o = 0
    for Sd, n in zip(S_days, ns, strict=True):
        X, Y, _, _ = make_supervised(Sd[:, idx], LOOKBACK)
        if len(X) != n:
            raise SystemExit(f"ECHEC [{sym}] : make_supervised rend {len(X)} fenetres, "
                             f"len(S) - LOOKBACK en annonce {n}. La reconstruction des "
                             f"tailles de pli serait fausse.")
        Xa[o:o + n] = X
        Ya[o:o + n] = Y
        o += n
        del X, Y
    del S_days

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


# --------------------------------------------------------------------------------------
# Etape 2b-3 : A2, cible cumulee (EXPLORATOIRE). Echantillon DIFFERENT de 2b-2 : les
# chiffres A2 ne se comparent jamais a ceux de 2b-2 (prereg, `regles_A2`).
#
# yH[k] = somme des H rendements a partir de k, DANS la journee : la cible chevauche H pas
# en avant, donc l'embargo doit valoir LOOKBACK + H et pas LOOKBACK + 10. Les H-1 derniers
# echantillons de chaque jour sont exclus (jamais de somme par-dessus la nuit).
# --------------------------------------------------------------------------------------

EMBARGO_A2 = LOOKBACK + HORIZON     # 26 : la cible chevauche HORIZON pas en avant
if EMBARGO_A2 != 26:
    # Le prereg est fige (`regles_A2.embargo: 26`). Changer l'horizon sans changer le
    # prereg ferait tomber l'embargo sous LOOKBACK + H : le harnais doit s'arreter plutot
    # que de tourner avec une regle qui n'est plus celle qui a ete annoncee.
    raise SystemExit(f"regles_A2 fige l'embargo a 26 ; LOOKBACK + HORIZON = {EMBARGO_A2}. "
                     f"Le prereg doit etre modifie AVANT le run.")


def _r2_ci(y, p, dd, n_days, n_boot=N_BOOT, blk=BLK, seed=SEED):
    """(R², lo, hi) : R² = 1 - SSres/SSbase, dont les deux sommes sont tirees par jour."""
    ssb = np.bincount(dd, weights=y * y, minlength=n_days)
    ssr = np.bincount(dd, weights=(y - p) ** 2, minlength=n_days)
    k = np.bincount(dd, minlength=n_days) > 0
    m = boot_mult(np.random.default_rng(seed), int(k.sum()), n_boot, blk)
    B, R = m @ ssb[k], m @ ssr[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(B > 0, 1.0 - R / B, np.nan)
    r2 = r2[np.isfinite(r2)]
    point = float(1.0 - ssr.sum() / ssb.sum())
    if r2.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))


def eval_symbol_a2(sym: str, S_days, model_name: str = "linear"):
    """Le monde de la Phase 2 sur la cible cumulee `yH`, plus les deux bras A2.

    Deux modeles sont ajustes sur les MEMES plis : le modele a un pas (etat complet), qu'on
    deroule et qui porte `agent_causal_A2`, et le modele de la cible cumulee (cible
    scalaire), qui porte `agent_direct`. Les deux bras sont donc apparies par construction.
    """
    idx = np.asarray(ARMS["base"])
    D = len(idx)
    Xs, Ys, dds, hfs, cums = [], [], [], [], []
    for di, Sd in enumerate(S_days):
        X, Y, _, _ = make_supervised(Sd[:, idx], LOOKBACK)
        yH, k = cumulative_target(Y, HORIZON)
        if len(k) == 0:                       # journee plus courte que l'horizon
            continue
        Xs.append(X[k])
        Ys.append(Y[k])
        # Verite cumulee sur les pas 1..H, lue dans la MEME journee : cum[k+g] - cum[k].
        cum = np.concatenate([[0.0], np.cumsum(Y[:, RET_IDX])])
        g = np.arange(1, HORIZON + 1)[None, :]
        cums.append(cum[k[:, None] + g] - cum[k[:, None]])
        dds.append(np.full(len(k), di))
        hfs.append(Sd[k + LOOKBACK - 1, SPREAD_IDX] / 2.0)   # spread a la barre de decision
    if not Xs:
        raise SystemExit(f"[{sym}] aucune journee plus longue que H = {HORIZON}.")
    X2, Y2 = np.vstack(Xs), np.vstack(Ys)
    act, dd2, half2 = np.vstack(cums), np.concatenate(dds), np.concatenate(hfs)
    yH = act[:, -1]
    del Xs, Ys, cums
    n_days = int(dd2.max()) + 1

    splits = list(walk_forward_splits(len(X2), NFOLDS, EMBARGO_A2, MINTRAIN, "expanding"))
    used = np.zeros(len(X2), dtype=bool)
    for _tr, te in splits:
        used[te] = True
    keep = np.flatnonzero(used)
    n2 = len(keep)

    pred1 = np.empty(n2)
    y1 = np.empty(n2)
    yHp = np.empty(n2)
    R = np.empty((n2, HORIZON))
    for tr, te in splits:
        m1 = LinearWM() if model_name == "linear" else MLPWM()
        m1.fit(X2[tr], Y2[tr])
        mc = LinearWM() if model_name == "linear" else MLPWM()
        mc.fit(X2[tr], yH[tr, None])
        sel = np.searchsorted(keep, te)
        pred1[sel] = m1.predict(X2[te])[:, RET_IDX]
        y1[sel] = Y2[te][:, RET_IDX]
        # `mc` est ajuste sur une cible a UNE colonne, et sklearn >= 1.9 APLATIT la prediction
        # d'un modele mono-sortie ((m,) et non (m, 1)). On ravel au lieu d'indexer [:, 0] :
        # ca marche pour les deux formes, et ca ne depend pas de la version de sklearn.
        yHp[sel] = np.ravel(mc.predict(X2[te]))
        base, te0 = int(sel[0]), int(te[0])
        for a in range(0, len(te), CHUNK):
            e = min(a + CHUNK, len(te))
            win = X2[te0 + a:te0 + e].reshape(-1, LOOKBACK, D)
            R[base + a:base + e] = rollout(m1, win, HORIZON)[:, :, RET_IDX]
    del X2, Y2, yH

    dk, hk = dd2[keep], half2[keep]
    # `yHp` est DEJA compacte : il est rempli aux positions `sel` dans une taille `n2`, comme
    # `pred1` et `y1`. Le re-indexer par `keep` le decalait une seconde fois et sortait des
    # bornes (le premier `keep` vaut ~0,4*len(X) alors que `n2` en vaut ~0,6). Les tableaux
    # restes a la taille de `len(X2)` sont ceux d'AVANT compaction : `act`, `dd2`, `half2`.
    actk = act[keep]
    yHk, yHk_p = actk[:, -1], yHp
    npz: dict = {"y": y1, "pred": pred1, "half": hk, "day": dk, "n_oos": n2,
                 "count": np.bincount(dk, minlength=n_days).astype(float), "symbole": sym}
    _reduce(npz, "myope", np.sign(pred1) * pred1, np.sign(pred1) * y1,
            np.sign(pred1), hk, dk, n_days)

    Rm = mask_day_end(R, dk)
    for fee in FEES_BP:
        c = hk + fee * 1e-4
        Cm = mask_day_end(np.repeat(c[:, None], HORIZON, axis=1), dk)
        p_c = plan_positions_causal(Rm, Cm, dk)
        _reduce(npz, tag("agent_causal_A2", fee), p_c * pred1, p_c * y1, p_c, hk, dk, n_days)
        # `agent_direct` : la cible cumulee, divisee par H, repete sur les H pas du jour.
        Rd = np.repeat((yHk_p / HORIZON)[:, None], HORIZON, axis=1)
        mask_day_end(Rd, dk)
        p_d = plan_positions_causal(Rd, Cm, dk)
        _reduce(npz, tag("agent_direct", fee), p_d * pred1, p_d * y1, p_d, hk, dk, n_days)

    # --- R² : la cible cumulee, et le rollout du modele a un pas comme reference ---------
    pred_cum = np.cumsum(Rm, axis=1)
    horizons = (1, 5, HORIZON)
    ref = rollout_r2(actk, pred_cum, horizons)          # reference deja publiee (wm_eval)
    rows = []
    for h in horizons:
        r2, lo, hi = _r2_ci(actk[:, h - 1], pred_cum[:, h - 1], dk, n_days)
        # Les deux chemins de code doivent tomber sur le meme point : sinon l'IC ci-dessus
        # n'est pas l'IC du chiffre cite.
        if abs(r2 - ref[h]) > 1e-12:
            raise SystemExit(f"[{sym}] R² du rollout inconsistant a h={h} : {r2} vs {ref[h]}")
        rows.append({"symbol": sym, "type": "rollout_1pas", "h": h, "r2": r2, "lo": lo,
                     "hi": hi})
    r2, lo, hi = _r2_ci(yHk, yHk_p, dk, n_days)
    rows.append({"symbol": sym, "type": "direct_yH", "h": HORIZON, "r2": r2, "lo": lo,
                 "hi": hi})
    print(f"  [{sym}] A2 : R²(yH) {r2:+.5f} [{lo:+.5f}, {hi:+.5f}] ; rollout h={HORIZON} "
          f"{ref[HORIZON]:+.5f}", flush=True)
    return npz, rows


def verdicts_a2(df_net: pd.DataFrame, df_cmp: pd.DataFrame, df_r2: pd.DataFrame) -> dict:
    """Les deux regles A2 du prereg. A2 est EXPLORATOIRE : ces verdicts ne touchent jamais
    au verdict principal de 2b-2."""
    syms = sorted(df_r2.symbol.unique())
    sig = {s: bool(df_r2[(df_r2.symbol == s) & (df_r2.type == "direct_yH")].lo.iloc[0] > 0)
           for s in syms}
    d = df_cmp[(df_cmp.gain.astype(str).str.startswith("agent_direct@2"))
               & (df_cmp.ref.astype(str).str.startswith("agent_causal_A2@2"))
               & (df_cmp.nature == "reel")]
    bat = {s: bool(d[d.symbol == s].lo.iloc[0] > 0) for s in syms if (d.symbol == s).any()}
    out = {
        "echantillon": "A2 (cible cumulee, embargo 26) : NON comparable aux chiffres de 2b-2",
        "signal_horizon": {"regle": "borne basse IC95 de R²_OOS(yH) > 0", "seuil": ">= 4/5",
                           "n": f"{sum(sig.values())}/{len(sig)}", "detail": sig,
                           "ok": sum(sig.values()) * 5 >= 4 * len(sig)},
        "direct_bat_rollout": {"regle": "borne basse IC95 appariée de Δnet(agent_direct@2 - "
                               "agent_causal_A2@2) > 0", "seuil": ">= 4/5",
                               "n": f"{sum(bat.values())}/{len(bat)}", "detail": bat,
                               "ok": sum(bat.values()) * 5 >= 4 * len(bat)},
    }
    return out


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
        # `p2b_diag_confondants.csv` porte DEUX lignes par symbole : celle du plan (derive,
        # timing, achat-conservation) et celle des permutations (`obs_bp`, `bruit_*`,
        # `k_perm`), aux colonnes disjointes et donc trouees de NaN. `groupby.first()` prend
        # la premiere valeur non nulle de chaque colonne, ce qui recolle les deux.
        cf = conf.groupby("symbol").first()
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


def write_a2_results(out: str, v: dict, net: pd.DataFrame, cmp_: pd.DataFrame,
                     r2: pd.DataFrame) -> None:
    res = {
        "meta": {"nom": "phase2b_crypto_causal_A2",
                 "prereg": "configs/phase2b_crypto_prereg.yaml (bloc regles_A2)",
                 "harnais": "scripts/crypto/phase2b.py --a2",
                 "frais_primaire": FEE_PRIMAIRE, "horizon": HORIZON,
                 "embargo": EMBARGO_A2, "n_boot": N_BOOT, "seed": SEED,
                 "symboles": sorted(r2.symbol.unique()),
                 "echantillon": "OOS prive des H-1 dernieres barres par jour : "
                                "NON comparable aux chiffres de 2b-2",
                 "genere_le": "2026-10-08"},
        "verdicts": v,
        "net": json.loads(net.to_json(orient="records")),
        "compare": json.loads(cmp_.to_json(orient="records")),
        "r2": json.loads(r2.to_json(orient="records")),
    }
    p = os.path.join(out, "phase2b_a2_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
        f.write("\n")


# --------------------------------------------------------------------------------------
# Etape 2b-5 : extension a l'union des 88 journees, et masques par sous-ensemble.
#
# L'union est la liste TRIEE des 44 journees publiees et des 44 journees neuves. Les dates
# neuves sont les points milieux des publiees (prereg, `extension_88_jours`), donc le tri
# chronologique alterne exactement publiee / neuve : les deux sous-ensembles font 44 jours
# chacun, et c'est la seule propriete dont depend la lecture par masque.
# --------------------------------------------------------------------------------------

def dates_neuves_du_prereg(p: str) -> list[str]:
    """Les 44 dates neuves, lues dans le prereg : jamais en dur ici."""
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        neuves = [str(d) for d in cfg["extension_88_jours"]["dates_neuves"]]
    except (KeyError, TypeError) as e:
        raise SystemExit(f"{p} : illisible dans extension_88_jours.dates_neuves "
                         f"({e}).") from None
    if len(neuves) != 44:
        raise SystemExit(f"{p} : {len(neuves)} dates neuves, 44 attendues.")
    if set(neuves) & set(DATES_PUBLIEES_1C):
        raise SystemExit("Une date neuve figure aussi parmi les publiees : l'extension n'est "
                         "plus disjointe, les masques n'auraient plus de sens.")
    return neuves


def dates_du_dateset(nom: str, prereg: str) -> tuple[list[str], list[str]]:
    """(dates ordonnees, neuves) du jeu demande. `phase1` = les 44 publiees."""
    if nom == "phase1":
        return list(DATES_PUBLIEES_1C), []
    if nom == "union":
        neuves = dates_neuves_du_prereg(prereg)
        return sorted(set(DATES_PUBLIEES_1C) | set(neuves)), neuves
    raise SystemExit(f"--dateset inconnu : {nom} (phase1 | union).")


def applique_masque(npz: dict, day_dates: list[str], neuves: list[str], masque: str) -> None:
    """Restreint `count` aux journees du sous-ensemble. Mute `npz`.

    Toutes les statistiques du harnais (`arm_stats`, `compare`, `ecart_exploitation`) se
    ponderent par `count` : mettre a zero les journees hors masque les restreint donc au
    sous-ensemble SANS toucher aux tableaux par barre, qui sont identiques d'un masque a
    l'autre. Un masque vide est un ECHEC BLOQUANT : il rendrait des statistiques sur zero
    journee, et un zero presente comme un resultat est exactement ce que le prereg interdit.
    """
    if masque == "union":
        keep = np.ones(len(day_dates), dtype=bool)
    elif masque in ("neuf", "ancien"):
        dedans = np.isin(day_dates, list(neuves))
        keep = dedans if masque == "neuf" else ~dedans
    else:
        raise SystemExit(f"--masque inconnu : {masque} (neuf | union | ancien).")
    c = np.asarray(npz["count"], float)
    if len(keep) != len(c):
        raise SystemExit(f"masque {masque} : {len(keep)} journees pour {len(c)} compteurs. "
                         f"Les identifiants de journee ne sont pas ceux du cache.")
    if not keep.any():
        raise SystemExit(f"masque {masque} vide : aucune journee de ce sous-ensemble.")
    out = c.copy()
    out[~keep] = 0.0
    if not (out > 0).any():
        raise SystemExit(f"masque {masque} vide : aucune journee TESTEE dans ce "
                         f"sous-ensemble (les plis n'y passent pas).")
    npz["count"] = out


def chiffre_bras(npz: dict, sym: str, out: str, suffix: str = "") -> None:
    """Toutes les statistiques d'un symbole, ecrites dans `p2b_*{suffix}.csv`.

    Un fichier PAR SYMBOLE fait foi : un run interrompu ne perd rien, et le relancer reprend
    ou il s'est arrete. Les CSV sont FUSIONNES par symbole comme partout ailleurs dans le
    projet : un run partiel n'ampute pas les symboles deja faits.
    """
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

    _merge_csv(os.path.join(out, f"p2b_net{suffix}.csv"), pd.DataFrame(n_s))
    _merge_csv(os.path.join(out, f"p2b_compare{suffix}.csv"), pd.DataFrame(c_s))
    _merge_csv(os.path.join(out, f"p2b_exploit{suffix}.csv"), pd.DataFrame(e_s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="defaut : experiments_2b (phase1) | experiments_2b_88 (union). "
                         "Les deux jeux ne partagent JAMAIS un dossier : leurs chiffres ne "
                         "sont pas comparables.")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--model", default="linear", choices=("linear", "mlp"))
    ap.add_argument("--dates", nargs="*", default=None,
                    help="journees explicites ; prioritaire sur --dateset")
    ap.add_argument("--prereg", default="configs/phase2b_crypto_prereg.yaml",
                    help="source des 44 dates neuves (--dateset union)")
    ap.add_argument("--dateset", default="phase1", choices=("phase1", "union"))
    ap.add_argument("--masques", nargs="*", default=None,
                    help="sous-ensembles a chiffrer SEULEMENT pour --dateset union "
                         "(neuf | union | ancien). Le premier est le primaire.")
    ap.add_argument("--check", action="store_true",
                    help="controles d'integrite seuls, puis sortie sans rien ecrire.")
    ap.add_argument("--diag", action="store_true", help="ajoute les diagnostics des 2b regles.")
    ap.add_argument("--a2", action="store_true",
                    help="etape 2b-3 : cible cumulee, echantillon A2 (EXPLORATOIRE). Ecrit "
                         "a2_*.csv et phase2b_a2_results.json, jamais p2b_*.csv.")
    ap.add_argument("--rapport-seul", dest="rapport_seul", action="store_true",
                    help="reapplique les verdicts aux CSV deja ecrits, sans rien recalculer. "
                         "Les verdicts sont une fonction pure des CSV : un plantage dans la "
                         "queue du run ne doit pas couter les 10 a 35 minutes de calcul.")
    ap.add_argument("--perm", type=int, default=K_PERM)
    ap.add_argument("--shift", type=int, default=K_SHIFT)
    args = ap.parse_args()
    if args.a2 and args.check:
        raise SystemExit("--a2 et --check ensemble n'ont pas de sens : les controles "
                         "d'integrite portent sur l'echantillon de la Phase 2.")
    if args.masques and args.dateset != "union":
        raise SystemExit("--masques ne sert qu'a --dateset union : sur les 44 journees "
                         "publiees il n'y a pas de journee neuve.")
    masques: list[str | None] = ["union"] if args.dateset == "union" else [None]
    if args.masques:
        masques = list(args.masques)
    args.out = args.out or ("experiments_2b" if args.dateset == "phase1"
                            else "experiments_2b_88")

    if args.rapport_seul and (args.a2 or args.check):
        raise SystemExit("--rapport-seul ne se combine ni avec --a2 ni avec --check : ceux-ci "
                         "calculent des chiffres, lui n'en relit que.")
    cache: dict = {}
    if not args.rapport_seul:
        dates, neuves = dates_du_dateset(args.dateset, args.prereg)
        cache = load_cached(args.dates or dates)
        if not cache:
            raise SystemExit("Aucun .pkl dans data/raw/crypto_lob.")
        if args.symbols:
            inconnus = [s for s in args.symbols if s not in cache]
            if inconnus:
                raise SystemExit(f"Aucun .pkl pour : {', '.join(inconnus)}.")
            cache = {s: cache[s] for s in args.symbols}
        if not args.check:
            os.makedirs(args.out, exist_ok=True)

    a2_net, a2_cmp, a2_r2 = [], [], []
    for sym, pkls in cache.items():
        t0 = time.time()
        print(f"  [{sym}] etat large ({len(pkls)} jours)...", flush=True)
        S_days = build_wide(pkls)
        days, spread = _days_and_spread(S_days)
        # Les identifiants de journee suivent l'ordre du cache, date par date : c'est ce qui
        # permet de rattacher chaque journee a son sous-ensemble.
        day_dates = [date_of(pf) for pf in pkls]

        if args.a2:
            print(f"  [{sym}] A2 : cible cumulee H={HORIZON}, embargo {EMBARGO_A2}, "
                  f"echantillon distinct de 2b-2...", flush=True)
            npz2, rows = eval_symbol_a2(sym, S_days, model_name=args.model)
            del S_days
            np.savez_compressed(os.path.join(args.out, f"a2_oos_{sym}.npz"), **npz2)
            for fee in FEES_BP:
                for name in ("myope", tag("agent_causal_A2", fee),
                             tag("agent_direct", fee)):
                    a2_net.append(arm_stats(npz2, sym, name, fee, imaginee=False))
                    a2_net.append(arm_stats(npz2, sym, name, fee, imaginee=True))
                for nat in (False, True):
                    a2_cmp.append(compare(npz2, sym, tag("agent_direct", fee),
                                          tag("agent_causal_A2", fee), fee, nat))
            a2_r2 += rows
            _merge_csv(os.path.join(args.out, "a2_net.csv"), pd.DataFrame(a2_net))
            _merge_csv(os.path.join(args.out, "a2_compare.csv"), pd.DataFrame(a2_cmp))
            _merge_csv(os.path.join(args.out, "a2_r2.csv"), pd.DataFrame(a2_r2))
            print(f"  [{sym}] A2 fait en {time.time() - t0:.0f}s ; CSV fusionnes.", flush=True)
            continue

        print(f"  [{sym}] {len(days)} echantillons, {days.max() + 1} jours ; monde + "
              f"rollout (H={H_MAX}) + planification...", flush=True)
        npz, ext = eval_symbol_2b(sym, S_days, days, spread, model_name=args.model)
        del S_days

        if args.check:
            check_myope(sym, npz)
            check_fuite(sym, npz)
            check_rollout(ext)
            continue

        np.savez_compressed(os.path.join(args.out, f"p2b_oos_{sym}.npz"), **npz)

        # `union` chiffre le corpus entier, les autres le restreignent : les CSV sont
        # suffixes par masque, et JAMAIS fusionnes entre eux.
        c_plein = np.asarray(npz["count"], float).copy()
        for m in masques:
            if m not in (None, "union"):
                applique_masque(npz, day_dates, neuves, m)
            chiffre_bras(npz, sym, args.out, "" if m == "union" else f"_{m}" if m else "")
            npz["count"] = c_plein.copy()

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

        print(f"  [{sym}] fait en {time.time() - t0:.0f}s ; CSV fusionnes.", flush=True)

    if args.check:
        print("\nControles passes.")
        return

    if args.a2:
        d_net = pd.read_csv(os.path.join(args.out, "a2_net.csv"))
        d_cmp = pd.read_csv(os.path.join(args.out, "a2_compare.csv"))
        d_r2 = pd.read_csv(os.path.join(args.out, "a2_r2.csv"))
        v = verdicts_a2(d_net, d_cmp, d_r2)
        write_a2_results(args.out, v, d_net, d_cmp, d_r2)
        print("\nA2 (exploratoire) :")
        for k in ("signal_horizon", "direct_bat_rollout"):
            print(f"  {k:20s} {v[k]['n']:>5s} (seuil {v[k]['seuil']}) -> "
                  f"{'OK' if v[k]['ok'] else 'non'}")
        print(f"\nSorties : {args.out}/a2_net.csv, a2_compare.csv, a2_r2.csv, "
              f"phase2b_a2_results.json")
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
