"""Phase 2 : DIAGNOSTICS des trois confondants du resultat positif.

Le harnais `agent_plan.py` applique les regles figees et sort un verdict. Ce script
n'applique AUCUNE regle : il instrumente le resultat pour determiner ce que le rapport a
le droit d'affirmer. Trois questions, posees dans le prompt de reprise :

  1. COUT OU SIGNAL ?  Le planificateur trade rarement, donc paie moins de demi-spread
     que le myope. Une partie du gain peut etre purement mecanique.
     Instrument : rejouer le MEME planificateur sur une prediction BRUITEE (rhat permute
     dans le temps, a l'interieur de chaque journee) avec les MEMES couts, K fois. Si un
     agent nourri de bruit gagne aussi, le gain n'est pas du signal.
     Decomposition directe : net(agent) - net(myope) = [d_gross] - [d_cout].

  2. CONFONDANT DE DERIVE DIRECTIONNELLE.  Si le DP tient longtemps des positions longues,
     le net positif peut n'etre qu'une exposition a la derive de l'actif sur la fenetre.
     Instruments : decomposition EXACTE  somme p_t*r_t = pbar*somme(r_t) + somme((p_t-pbar)*r_t)
     (exposition a la derive, puis timing), bras achat-et-conservation, et rejeu des MEMES
     positions sur des rendements decales circulairement dans la journee -- un decalage
     circulaire preserve exactement la derive du jour et ne detruit que l'alignement.

  3. ORDRE DE GRANDEUR.  Turnover, nombre de changements de position, fraction de temps
     en position, part de l'extractible captee.

D'OU VIENNENT LES DONNEES
-------------------------
`experiments/crypto_lob_oos_<SYM>.npz` est le bras PUBLIE des phases 1/1c/1d/1e. Il
contient exactement ce qu'il faut : `pred` (= rhat), `y` (= r_true), `half` (demi-spread a
la barre) et `day`. `agent_plan.py --check` a verifie que ce bras est reproduit au bit
pres par la chaine de la phase 2 : les diagnostics portent donc sur l'artefact publie
lui-meme, pas sur une reimplementation.

    python scripts/crypto/agent_diag.py --symbols BTCUSDT
    python scripts/crypto/agent_diag.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from agent_plan import (
    BLK,
    FEE_PRIMAIRE,
    HORIZON,
    N_BOOT,
    REF_DIR,
    SEED,
    _boot_mean,
    _run_bounds,
    plan_by_run,
)
from bootstrap_signif import boot_mult
from crypto_lob import _merge_csv, load_cached

from mirage.backtest import position_changes

K_PERM = 39          # permutations de rhat (bruit) : resolution 1/(K+1) = 0.025
K_SHIFT = 400        # decalages circulaires de rendement (positions fixes)
OUT = "experiments_agent"


# ---------------------------------------------------------------------------------------
# Briques de comptabilite : memes conventions que agent_plan.py, au symbole par symbole.
# ---------------------------------------------------------------------------------------

def load_published(sym: str) -> dict:
    """Le bras publie : pred = rhat, y = r_true, half = demi-spread, day = journee."""
    pf = os.path.join(REF_DIR, f"crypto_lob_oos_{sym}.npz")
    if not os.path.exists(pf):
        raise SystemExit(f"introuvable : {pf}")
    with np.load(pf) as z:
        d = {k: z[k] for k in z.files}
    for k in ("pred", "y", "half", "day"):
        if k not in d:
            raise SystemExit(f"{pf} ne contient pas '{k}' : impossible d'instrumenter.")
    return d


def arm_block(pos: np.ndarray, rhat, rtrue, half, dd, n_days: int) -> dict:
    """Sommes par journee d'un bras, au format de `agent_plan._reduce`."""
    dpos = position_changes(pos, dd)
    return {"gi": np.bincount(dd, weights=pos * rhat, minlength=n_days),
            "gr": np.bincount(dd, weights=pos * rtrue, minlength=n_days),
            "dp": np.bincount(dd, weights=dpos, minlength=n_days),
            "dph": np.bincount(dd, weights=dpos * half, minlength=n_days)}


def net_day(blk: dict, fee: float, imaginee: bool) -> np.ndarray:
    g = blk["gi"] if imaginee else blk["gr"]
    return g - blk["dph"] - fee * 1e-4 * blk["dp"]


def pooled(series: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Restreint aux journees reellement en test (count > 0), comme agent_plan._pool."""
    k = count > 0
    return series[k], count[k]


def bp(series: np.ndarray, count: np.ndarray) -> dict:
    """Moyenne par barre en bp + IC95 par journee."""
    s, c = pooled(series, count)
    m = boot_mult(np.random.default_rng(SEED), len(c), N_BOOT, 1)
    nb = _boot_mean(m, s, c)
    return {"bp": float(s.sum() / c.sum() * 1e4),
            "lo": float(np.percentile(nb, 2.5)), "hi": float(np.percentile(nb, 97.5))}


def bp_paired(a: np.ndarray, b: np.ndarray, count: np.ndarray) -> dict:
    """(a - b) apparie : MEMES journees tirees pour les deux flux (convention du projet)."""
    sa, c = pooled(a, count)
    sb, _ = pooled(b, count)
    m = boot_mult(np.random.default_rng(SEED), len(c), N_BOOT, BLK)
    d = _boot_mean(m, sa, c) - _boot_mean(m, sb, c)
    return {"bp": float((sa.sum() - sb.sum()) / c.sum() * 1e4),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def permute_within_days(rng, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Permutation temporelle de `v` a l'interieur de chaque journee.

    On ne permute PAS globalement : la structure par journee (et donc la serie de couts)
    reste intacte, seul l'appariement entre la prediction et le rendement est detruit.
    """
    out = np.empty_like(v)
    for k in range(len(b) - 1):
        s, e = b[k], b[k + 1]
        out[s:e] = v[s:e][rng.permutation(e - s)]
    return out


def shift_within_days(rng, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Decalage circulaire de `v` dans chaque journee.

    Un decalage circulaire est une permutation de la journee : la SOMME du jour est
    preservee exactement, donc la derive directionnelle aussi. Seul l'alignement
    intra-journee est detruit. C'est l'instrument propre pour le confondant n°2.
    """
    out = np.empty_like(v)
    for k in range(len(b) - 1):
        s, e = b[k], b[k + 1]
        n = e - s
        out[s:e] = np.roll(v[s:e], int(rng.integers(1, n))) if n > 1 else v[s:e]
    return out


# ---------------------------------------------------------------------------------------

def control_myope(sym: str, blk_my: dict, count: np.ndarray, fee: float) -> None:
    """CONTROLE : mon bras myope doit valoir celui du harnais principal (CSV du run)."""
    cf = os.path.join(OUT, "agent_net.csv")
    if not os.path.exists(cf):
        print("  [controle] agent_net.csv absent : controle saute.", flush=True)
        return
    df = pd.read_csv(cf)
    q = df[(df.symbol == sym) & (df.bras == "myope") & (df.fee_bp == fee)
           & (df.nature == "reel")]
    if q.empty:
        print("  [controle] pas de ligne myope : controle saute.", flush=True)
        return
    ref = float(q.net_bp.iloc[0])
    mine = bp(net_day(blk_my, fee, imaginee=False), count)["bp"]
    d = abs(ref - mine)
    print(f"  [controle] myope reel @{fee:g}bp : harnais {ref:+.6f} ; ici {mine:+.6f} ; "
          f"ecart {d:.2e}")
    if d > 1e-9:
        raise SystemExit("ECHEC : le diagnostic ne voit pas le meme bras myope que le "
                         "harnais principal. C'EST UN BUG.")


def diag_symbol(sym: str, k_perm: int, k_shift: int) -> tuple[list, list, list]:
    z = load_published(sym)
    rhat = np.asarray(z["pred"], float)
    rtrue = np.asarray(z["y"], float)
    half = np.asarray(z["half"], float)
    dd = np.asarray(z["day"]).astype(int)
    n_days = int(dd.max()) + 1
    count = np.bincount(dd, minlength=n_days).astype(float)
    b = _run_bounds(dd)
    c2 = half + FEE_PRIMAIRE * 1e-4

    t0 = time.time()
    blocks: dict = {}
    blocks["myope"] = arm_block(np.sign(rhat), rhat, rtrue, half, dd, n_days)
    blocks["plat"] = arm_block(np.zeros_like(rhat), rhat, rtrue, half, dd, n_days)
    blocks["bh"] = arm_block(np.ones_like(rhat), rhat, rtrue, half, dd, n_days)

    p_ag = plan_by_run(rhat, c2, dd, HORIZON)
    blocks["agent"] = arm_block(p_ag, rhat, rtrue, half, dd, n_days)
    t_plan = time.time() - t0
    control_myope(sym, blocks["myope"], count, FEE_PRIMAIRE)

    gross_ag = p_ag * rtrue
    net_ag = net_day(blocks["agent"], FEE_PRIMAIRE, imaginee=False)
    net_my = net_day(blocks["myope"], FEE_PRIMAIRE, imaginee=False)
    net_bh = net_day(blocks["bh"], FEE_PRIMAIRE, imaginee=False)

    # --- 3. ORDRE DE GRANDEUR -----------------------------------------------------------
    n_pool = count[count > 0].sum()
    frac = {k: float(np.mean(p_ag == k)) for k in (-1.0, 0.0, 1.0)}
    dp_ag = blocks["agent"]["dp"].sum()
    dp_my = blocks["myope"]["dp"].sum()
    dp_bh = blocks["bh"]["dp"].sum()
    ordre = [dict(symbol=sym, bras="agent", turnover=dp_ag / n_pool,
                  changements_par_jour=dp_ag / (count > 0).sum(),
                  frac_long=frac[1.0], frac_plat=frac[0.0], frac_short=frac[-1.0],
                  position_moyenne=float(p_ag.mean()),
                  cout_bp=float(blocks["agent"]["dph"].sum() / n_pool * 1e4
                                + FEE_PRIMAIRE * 1e-4 * dp_ag / n_pool * 1e4)),
             dict(symbol=sym, bras="myope", turnover=dp_my / n_pool,
                  changements_par_jour=dp_my / (count > 0).sum(),
                  frac_long=float(np.mean(np.sign(rhat) == 1)),
                  frac_plat=0.0, frac_short=float(np.mean(np.sign(rhat) == -1)),
                  position_moyenne=float(np.sign(rhat).mean()),
                  cout_bp=float(blocks["myope"]["dph"].sum() / n_pool * 1e4
                                + FEE_PRIMAIRE * 1e-4 * dp_my / n_pool * 1e4)),
             dict(symbol=sym, bras="bh", turnover=dp_bh / n_pool,
                  changements_par_jour=dp_bh / (count > 0).sum(),
                  frac_long=1.0, frac_plat=0.0, frac_short=0.0, position_moyenne=1.0,
                  cout_bp=float(blocks["bh"]["dph"].sum() / n_pool * 1e4
                                + FEE_PRIMAIRE * 1e-4 * dp_bh / n_pool * 1e4))]

    # --- 1a. DECOMPOSITION net(agent) - net(myope) = d_gross + gain_cout ------------------
    # net = brut - cout, donc net_agent - net_myope = d_brut + (cout_myope - cout_agent).
    # Le second terme est la part PUREMENT MECANIQUE : ce que l'agent economise en ne
    # reversant plus sa position a chaque barre.
    d_gross = float((gross_ag - np.sign(rhat) * rtrue).sum() / n_pool * 1e4)
    gain_cout = float((blocks["myope"]["dph"].sum() - blocks["agent"]["dph"].sum()) / n_pool
                      * 1e4 + FEE_PRIMAIRE * 1e-4 * (dp_my - dp_ag) / n_pool * 1e4)
    paired = bp_paired(net_ag, net_my, count)
    if abs((d_gross + gain_cout) - paired["bp"]) > 1e-6:
        raise SystemExit("ECHEC : la decomposition brut/cout ne somme pas au net. BUG.")
    dec_myope = dict(symbol=sym, d_net=paired["bp"], d_net_lo=paired["lo"],
                     d_net_hi=paired["hi"], d_gross=d_gross, gain_cout=gain_cout)

    # --- 2. DERIVE : decomposition exacte + achat-conservation --------------------------
    pbar = float(p_ag.mean())
    drift = float(rtrue.sum() * 1e4)                     # derive totale, en bp
    a_drift = np.bincount(dd, weights=pbar * rtrue, minlength=n_days)
    b_timing = np.bincount(dd, weights=(p_ag - pbar) * rtrue, minlength=n_days)
    if not np.allclose(a_drift + b_timing, blocks["agent"]["gr"], atol=0, rtol=1e-12):
        raise SystemExit("ECHEC : derive + timing ne redonne pas le brut. BUG.")
    st_a = bp(a_drift, count)
    st_b = bp(b_timing, count)
    st_bh = bp(net_bh, count)
    derive = dict(symbol=sym, position_moyenne=pbar,
                  derive_totale_bp=drift,
                  contrib_derive_bp=st_a["bp"], contrib_derive_lo=st_a["lo"],
                  contrib_derive_hi=st_a["hi"],
                  contrib_timing_bp=st_b["bp"], contrib_timing_lo=st_b["lo"],
                  contrib_timing_hi=st_b["hi"],
                  bh_bp=st_bh["bp"], bh_lo=st_bh["lo"], bh_hi=st_bh["hi"])

    # Rejeu des MEMES positions sur des rendements decales circulairement (derive preservee)
    rng = np.random.default_rng(SEED + 1)
    shift_bp = np.empty(k_shift)
    for i in range(k_shift):
        rs = shift_within_days(rng, rtrue, b)
        shift_bp[i] = (p_ag * rs).sum() / n_pool * 1e4
    obs_gross = float(np.sum(gross_ag) / n_pool * 1e4)
    derive.update(d_gross_observe=obs_gross,
                  d_gross_decale_moy=float(shift_bp.mean()),
                  d_gross_decale_lo=float(np.percentile(shift_bp, 2.5)),
                  d_gross_decale_hi=float(np.percentile(shift_bp, 97.5)),
                  d_gross_decale_max=float(shift_bp.max()),
                  d_gross_decale_pct=float((shift_bp < obs_gross).mean()))

    # --- 1b. BRUIT : meme planificateur nourri de rhat permute --------------------------
    rng = np.random.default_rng(SEED)
    null_bp = np.empty(k_perm)
    null_img = np.empty(k_perm)
    null_to = np.empty(k_perm)
    for i in range(k_perm):
        rp = permute_within_days(rng, rhat, b)
        pp = plan_by_run(rp, c2, dd, HORIZON)
        blk = arm_block(pp, rhat, rtrue, half, dd, n_days)
        null_bp[i] = net_day(blk, FEE_PRIMAIRE, imaginee=False).sum() / n_pool * 1e4
        null_img[i] = net_day(blk, FEE_PRIMAIRE, imaginee=True).sum() / n_pool * 1e4
        null_to[i] = blk["dp"].sum() / n_pool

    obs = float(net_ag.sum() / n_pool * 1e4)
    st_obs = bp(net_ag, count)
    bruit = dict(symbol=sym, obs_bp=obs, obs_lo=st_obs["lo"], obs_hi=st_obs["hi"],
                 bruit_moy=float(null_bp.mean()), bruit_med=float(np.median(null_bp)),
                 bruit_lo=float(np.percentile(null_bp, 2.5)),
                 bruit_hi=float(np.percentile(null_bp, 97.5)),
                 bruit_max=float(null_bp.max()), bruit_min=float(null_bp.min()),
                 p_conservateur=float((1 + np.sum(null_bp >= obs)) / (k_perm + 1)),
                 k_perm=k_perm,
                 agent_imag_bp=float(net_day(blocks["agent"], FEE_PRIMAIRE, True).sum()
                                     / n_pool * 1e4),
                 bruit_imag_moy=float(null_img.mean()),
                 turnover_agent=dp_ag / n_pool, turnover_bruit=float(null_to.mean()))

    print(f"    PLANIFIE en {t_plan:.0f}s pour {1 + k_perm} plans.", flush=True)
    print(f"    obs {obs:+.4f} bp [{st_obs['lo']:+.4f}, {st_obs['hi']:+.4f}] ; "
          f"bruit {null_bp.mean():+.4f} bp "
          f"[{np.percentile(null_bp, 2.5):+.4f}, {np.percentile(null_bp, 97.5):+.4f}] "
          f"max {null_bp.max():+.4f} ; p <= {bruit['p_conservateur']:.3f}", flush=True)
    print(f"    brut {st_a['bp'] + st_b['bp']:+.4f} = derive {st_a['bp']:+.4f} "
          f"[{st_a['lo']:+.4f}, {st_a['hi']:+.4f}] + timing {st_b['bp']:+.4f} "
          f"[{st_b['lo']:+.4f}, {st_b['hi']:+.4f}] ; achat-conservation "
          f"{st_bh['bp']:+.4f} ; brut decale {shift_bp.mean():+.4f}", flush=True)
    print(f"    turnover agent {dp_ag / n_pool:.4f} vs bruit {null_to.mean():.4f} "
          f"(myope {dp_my / n_pool:.4f}) ; long/flat/short "
          f"{frac[1.0]:.2f}/{frac[0.0]:.2f}/{frac[-1.0]:.2f}", flush=True)
    return ordre, [dec_myope], [derive, bruit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--perm", type=int, default=K_PERM)
    ap.add_argument("--shift", type=int, default=K_SHIFT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = load_cached()
    if not cache:
        raise SystemExit("Aucun .pkl dans data/raw/crypto_lob.")
    syms = args.symbols or [s for s in cache]
    inconnus = [s for s in syms if s not in cache and
                not os.path.exists(os.path.join(REF_DIR, f"crypto_lob_oos_{s}.npz"))]
    if inconnus:
        raise SystemExit(f"Aucun artefact publie pour : {', '.join(inconnus)}.")

    for sym in syms:
        t0 = time.time()
        print(f"  [{sym}] chargement du bras publie...", flush=True)
        o, c, d = diag_symbol(sym, args.perm, args.shift)
        # Un fichier PAR SYMBOLE fait foi : il permet de lancer les symboles en processus
        # separes sans course sur les CSV fusionnes (lecture-modification-ecriture).
        for nm, rr in (("ordre", o), ("confondants", d), ("vs_myope", c)):
            df = pd.DataFrame(rr)
            df.to_csv(os.path.join(args.out, f"diag_{nm}_{sym}.csv"), index=False)
            _merge_csv(os.path.join(args.out, f"diag_{nm}.csv"), df)
        print(f"  [{sym}] fait en {time.time() - t0:.0f}s ; CSV ecrits.", flush=True)
    print(f"\nSorties : {args.out}/diag_ordre.csv, diag_confondants.csv, diag_vs_myope.csv")


if __name__ == "__main__":
    main()
