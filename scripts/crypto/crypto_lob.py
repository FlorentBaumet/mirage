"""Microstructure crypto (carnet Bybit L2) : Phase 1a (état) + Phase 1b (impact).

MULTI-JOURS : chaque symbole agrège plusieurs journées. Rigueur temporelle :
  - l'état et les fenêtres sont construits PAR JOUR -> aucune fenêtre à cheval sur
    deux jours, aucun rendement calculé par-dessus une nuit ;
  - les jours sont ensuite concaténés dans l'ordre -> le walk-forward purgé devient
    INTER-JOURS (train sur les jours passés, test sur les jours futurs) ;
  - le rollout ne démarre que s'il tient dans la même journée ;
  - le backtest repart à plat chaque jour (pas de faux retournement à la frontière).

    python scripts/crypto_lob.py --out experiments
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mirage.backtest import intraday_starts, position_changes  # noqa: E402
from mirage.impact import cost_curve  # noqa: E402
from mirage.metrics import r2_per_dim  # noqa: E402
from mirage.splits import walk_forward_splits  # noqa: E402
from mirage.state import OFI_COL, RET_IDX, STATE_COLS, build_state  # noqa: E402
from mirage.wm import LinearWM, make_supervised, make_wm_suite, rollout  # noqa: E402
from mirage.wm_eval import rollout_r2, subsample_starts  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DIR = os.path.join("data", "raw", "crypto_lob")
OFI_DIR = os.path.join("data", "raw", "crypto_ofi")
LOOKBACK, NFOLDS, MINTRAIN = 16, 5, 0.4
HORIZONS, MAXH, MAXSTARTS = [1, 2, 3, 5, 10, 20, 30], 30, 2000
KS = [0.25, 0.5, 1, 2, 4, 8]
SPREAD_IDX = STATE_COLS.index("spread_rel")


def ofi_path_for(pf: str) -> str:
    """Chemin du .pkl OFI evenementiel correspondant a une journee de barres."""
    return os.path.join(OFI_DIR, os.path.basename(pf).replace("_1s_book.pkl", "_ofi.pkl"))


# Les 44 journees de l'echantillon publie (configs/phase1c_crypto_prereg.yaml, inchangees
# depuis la phase 1). Elles sont NOMMEES ici parce que `data/raw/crypto_lob` en contient
# d'autres depuis l'extension 2b : sans liste explicite, deposer 44 journees de plus
# changerait en silence l'echantillon de TOUS les scripts existants (phases 1 a 2), et
# leurs controles d'integrite tomberaient pour une raison qui n'a rien a voir avec le code.
# tests/test_phase2b.py verifie que cette liste est bien celle du pre-enregistrement.
DATES_PUBLIEES_1C = (
    "2023-01-20", "2023-02-11", "2023-03-06", "2023-03-28", "2023-04-20", "2023-05-12",
    "2023-06-04", "2023-06-26", "2023-07-19", "2023-08-10", "2023-09-02", "2023-09-24",
    "2023-10-17", "2023-11-08", "2023-11-30", "2023-12-23", "2024-01-14", "2024-02-06",
    "2024-02-28", "2024-03-22", "2024-04-13", "2024-05-06", "2024-05-28", "2024-06-20",
    "2024-07-12", "2024-08-03", "2024-08-26", "2024-09-17", "2024-10-10", "2024-11-01",
    "2024-11-24", "2024-12-16", "2025-01-08", "2025-01-30", "2025-02-22", "2025-03-16",
    "2025-04-08", "2025-04-30", "2025-05-08", "2025-05-22", "2025-06-02", "2025-06-18",
    "2025-07-09", "2025-08-06",
)


def date_of(pf: str) -> str:
    """Journee d'un cache, lue dans son nom de fichier."""
    return os.path.basename(pf)[:10]


def load_cached(dates=None) -> dict[str, list[str]]:
    """{symbole: [chemins .pkl triés par date]}, restreint aux journees demandees.

    `dates` : iterable de "AAAA-MM-JJ" ; None -> les 44 journees publiees (DATES_PUBLIEES_1C).
    Une journee demandee et absente du disque est une ERREUR : un echantillon incomplet
    passerait sinon pour un echantillon complet, et les chiffres ne seraient plus ceux de
    l'echantillon annonce.
    """
    wanted = set(DATES_PUBLIEES_1C if dates is None else dates)
    out: dict[str, list[str]] = {}
    vues: set[str] = set()
    for pf in sorted(glob.glob(os.path.join(DIR, "*_1s_book.pkl"))):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)_1s_book", os.path.basename(pf))
        if m.group(1) in wanted:
            out.setdefault(m.group(2), []).append(pf)
            vues.add(m.group(1))
    manquantes = wanted - vues
    if manquantes:
        raise SystemExit(f"Journees demandees absentes de {DIR} : "
                         f"{', '.join(sorted(manquantes))}.")
    return {k: sorted(v) for k, v in out.items()}


def _merge_csv(path: str, new: pd.DataFrame) -> None:
    """Ecrit `new` en CONSERVANT les lignes des symboles non relances.

    Un run partiel (--symbols) ne doit jamais amputer les CSV : sans cette fusion,
    relancer un symbole seul effacerait les resultats deja acquis pour les autres.
    """
    if os.path.exists(path):
        old = pd.read_csv(path)
        if "symbol" in old.columns:
            old = old[~old["symbol"].isin(new["symbol"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def build_symbol(pkls: list[str], with_ofi: bool = False):
    """Construit les échantillons multi-jours d'un symbole (par jour, puis concaténés).

    `with_ofi` ajoute la dimension OFI en DERNIÈRE position. Le reste est inchangé, y
    compris le nombre de lignes : le comparatif base/enrichi doit porter sur exactement
    les mêmes échantillons, sinon les deux bras ne sont pas appariables.
    """
    Xs, Ys, days, spr, costs = [], [], [], [], []
    d = None
    for di, pf in enumerate(pkls):
        bars = pd.read_pickle(pf)
        ofi = None
        if with_ofi:
            op = ofi_path_for(pf)
            if not os.path.exists(op):
                # La couverture exigée est bloquante : un jour manquant avantagerait
                # silencieusement les deux bras d'un échantillon différent.
                raise SystemExit(
                    f"OFI absent : {op}\nCouverture incomplete : lance d'abord "
                    f"scripts/crypto/fetch_ofi.py (le pre-enregistrement interdit "
                    f"d'abandonner silencieusement un couple symbole/date).")
            ofi = pd.read_pickle(op)
        costs.append(cost_curve(bars, KS, side=1, levels=10))
        S, _ = build_state(bars, ofi=ofi)
        X, Y, pos, d = make_supervised(S.values, LOOKBACK)   # fenêtres internes au jour
        Xs.append(X)
        Ys.append(Y)
        days.append(np.full(len(X), di))
        spr.append(S.values[pos - 1, SPREAD_IDX])            # spread à la barre de décision
        del bars, S
    cost = pd.concat(costs).groupby("k_x_L1", as_index=False).mean(numeric_only=True)
    return (np.vstack(Xs), np.vstack(Ys), np.concatenate(days),
            np.concatenate(spr), d, cost)


def phase1a(sym, X, Y, days, d, dims=STATE_COLS):
    rows1, rowsR = [], []
    embargo = max(10, LOOKBACK)
    for fi, (tr, te) in enumerate(walk_forward_splits(len(X), NFOLDS, embargo,
                                                      MINTRAIN, "expanding")):
        Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
        models = make_wm_suite(Xtr, Ytr, "mlp")
        base = Xte[:, -d:].copy()
        base[:, RET_IDX] = 0.0                     # ret : baseline = random walk
        for name in ("mean", "linear", "mlp"):
            r2 = r2_per_dim(Yte, models[name].predict(Xte), base)
            for c, dim in enumerate(dims):
                rows1.append(dict(symbol=sym, fold=fi, model=name, dim=dim, r2=r2[c]))

        a, b = int(te[0]), int(te[-1]) + 1
        # rollout intra-jour uniquement (aucune fenêtre à cheval sur la nuit)
        starts = subsample_starts(intraday_starts(np.arange(a, b - MAXH), days, MAXH), MAXSTARTS)
        if len(starts) == 0:
            continue
        win = X[starts].reshape(-1, LOOKBACK, d)
        hidx = starts[:, None] + np.arange(MAXH)[None, :]
        ac = np.cumsum(Y[hidx, RET_IDX], axis=1)
        for name in ("persistence", "linear", "mlp"):
            pc = np.cumsum(rollout(models[name], win, MAXH)[:, :, RET_IDX], axis=1)
            for h, r2 in rollout_r2(ac, pc, HORIZONS).items():
                rowsR.append(dict(symbol=sym, fold=fi, model=name, horizon=h, r2=r2))
    return rows1, rowsR


def economic_check(sym, X, Y, days, spread, fees_bp=(0.0, 2.0, 5.5)):
    """LE verdict edge-vs-mirage : le R²_OOS positif survit-il aux frais ?"""
    embargo = max(10, LOOKBACK)
    P, A, SP, D = [], [], [], []
    for tr, te in walk_forward_splits(len(X), NFOLDS, embargo, MINTRAIN, "expanding"):
        m = LinearWM().fit(X[tr], Y[tr])
        P.append(m.predict(X[te])[:, RET_IDX])
        A.append(Y[te][:, RET_IDX])
        SP.append(spread[te])
        D.append(days[te])
    pred, y = np.concatenate(P), np.concatenate(A)
    spr, dd = np.concatenate(SP), np.concatenate(D)

    posn = np.sign(pred)
    gross = posn * y
    dpos = position_changes(posn, dd)          # remise à plat à chaque nouveau jour
    half = spr / 2.0

    rows = []
    for fee in fees_bp:
        net = gross - dpos * (half + fee * 1e-4)
        rows.append(dict(symbol=sym, fee_bp=fee,
                         gross_bp=round(float(gross.mean()) * 1e4, 4),
                         turnover=round(float(dpos.mean()) / 2, 3),
                         net_bp=round(float(net.mean()) * 1e4, 4),
                         net_cumul_pct=round(float(net.sum()) * 100, 2)))
    # y / pred / day sont conservés pour que le bootstrap par jour (bootstrap_signif.py)
    # réutilise EXACTEMENT ces prédictions OOS au lieu de refaire le walk-forward à sa façon.
    return pd.DataFrame(rows), dict(gross=gross, dpos=dpos, half=half,
                                    y=y, pred=pred, day=dd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--state", choices=("base", "ofi"), default="base",
                    help="'ofi' ajoute l'order flow imbalance evenementiel en 6e dim. "
                         "Ecrire dans un --out distinct : les .npz du bras de base sont "
                         "ceux publies, et l'appariement les relit.")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="sous-ensemble de symboles a traiter (defaut : tous ceux qui ont "
                         "des .pkl). Les CSV et le JSON sont fusionnes par symbole, donc un "
                         "run partiel ne detruit rien ; les figures ne sont regenerees que "
                         "sur un run complet.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dims = list(STATE_COLS) + ([OFI_COL] if args.state == "ofi" else [])

    if args.state == "ofi" and os.path.abspath(args.out) == os.path.abspath("experiments"):
        # Les .npz de experiments/ sont ceux publies : l'appariement les relit comme bras
        # de base. Les ecraser par le bras enrichi detruirait la comparaison.
        raise SystemExit("--state ofi doit ecrire ailleurs que dans experiments/ "
                         "(ex. --out experiments_ofi) : experiments/ est le bras de base.")

    cache = load_cached()
    if not cache:
        raise SystemExit("Aucun .pkl dans data/raw/crypto_lob - lance d'abord "
                         "scripts/fetch_bybit_batch.py")
    tous = list(cache)
    if args.symbols:
        # Un symbole a la fois : pic memoire bas, et chaque run partiel est acquis
        # (le script ne reprend pas ou il s'est arrete, il repart de zero).
        inconnus = [s for s in args.symbols if s not in cache]
        if inconnus:
            raise SystemExit(f"Aucun .pkl pour : {', '.join(inconnus)}. "
                             f"Disponibles : {', '.join(tous)}")
        cache = {s: cache[s] for s in args.symbols}
    complet = set(cache) == set(tous)

    print("=== Couverture ===")
    for sym, pk in cache.items():
        dates = [os.path.basename(p)[:10] for p in pk]
        print(f"  {sym} : {len(pk)} jours  ({dates[0]} .. {dates[-1]})")

    all1, allR, econ_rows, detail, costs = [], [], [], {}, {}
    n_ech: dict[str, int] = {}
    for sym, pkls in cache.items():
        X, Y, days, spread, d, cost = build_symbol(pkls, with_ofi=(args.state == "ofi"))
        n_ech[sym] = len(X)
        print(f"  [{sym}] {len(X)} échantillons sur {days.max() + 1} jours", flush=True)
        costs[sym] = cost
        r1, rR = phase1a(sym, X, Y, days, d, dims)
        all1 += r1
        allR += rR
        e, det = economic_check(sym, X, Y, days, spread)
        econ_rows.append(e)
        detail[sym] = det
        np.savez_compressed(os.path.join(args.out, f"crypto_lob_oos_{sym}.npz"),
                            y=det["y"], pred=det["pred"], gross=det["gross"],
                            dpos=det["dpos"], half=det["half"], day=det["day"])
        del X, Y

    res1, resR = pd.DataFrame(all1), pd.DataFrame(allR)
    pd.set_option("display.width", 160)

    print("\n=== Phase 1a - R²_OOS 1-step par dimension "
          "(baseline: 0 pour ret, no-change sinon) ===")
    print(res1.pivot_table(index="dim", columns="model", values="r2", aggfunc="mean")
          .reindex(dims).round(5).to_string())
    print("\n--- 'ret' par symbole ---")
    print(res1[res1.dim == "ret"].pivot_table(index="symbol", columns="model",
                                              values="r2", aggfunc="mean").round(5).to_string())

    print("\n=== Phase 1a - rollout : R²_OOS rendement cumulé vs random walk ===")
    pivR = resR.pivot_table(index="horizon", columns="model", values="r2", aggfunc="mean")
    print(pivR.round(5).to_string())

    print("\n=== Phase 1b - coût d'exécution (achat), moyenné sur les jours ===")
    for sym, c in costs.items():
        print(f"\n[{sym}]")
        print(c.round(4).to_string(index=False))

    print("\n=== VERDICT économique - l'edge survit-il aux frais ? ===")
    econ = pd.concat(econ_rows, ignore_index=True)
    print(econ.to_string(index=False))
    _merge_csv(os.path.join(args.out, "crypto_lob_economic.csv"), econ)
    verdict = "EDGE (net>0 à frais réalistes)" if (econ[econ.fee_bp >= 2.0]["net_bp"] > 0).any() \
        else "MIRAGE (net<=0 dès des frais réalistes)"
    print(f"\n-> {verdict}")

    _merge_csv(os.path.join(args.out, "crypto_lob_1step.csv"), res1)
    _merge_csv(os.path.join(args.out, "crypto_lob_rollout.csv"), resR)
    # Trace du nombre d'echantillons par symbole : le bootstrap apparie verifie que les
    # deux bras ont exactement le meme echantillon avant de comparer quoi que ce soit.
    ns_path = os.path.join(args.out, "crypto_lob_nsample.json")
    prev: dict[str, int] = {}
    if os.path.exists(ns_path):
        with open(ns_path, encoding="utf-8") as fh:
            prev = json.load(fh).get("nsample", {})
    prev.update(n_ech)
    with open(ns_path, "w", encoding="utf-8") as fh:
        json.dump({"state": args.state, "nsample": prev}, fh, indent=2, sort_keys=True)

    if not complet:
        # Les figures (cout, rollout, stabilite, mirage) porteraient sur les seuls
        # symboles relances : une version incomplete remplacerait une version complete.
        print("\nRun partiel : figures non regenerees. Relance sans --symbols pour les "
              "produire sur l'ensemble des symboles.")
        print(f"\nPredictions OOS par symbole : {args.out}/crypto_lob_oos_<SYM>.npz")
        return

    # --- figures ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sym, c in costs.items():
        ax.plot(c["k_x_L1"], c["slippage_bp"], marker="o", label=sym)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("taille (k × meilleur niveau)")
    ax.set_ylabel("slippage moyen (bp)")
    ax.set_title("Crypto (Bybit L2) - coût d'exécution vs taille")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_cost.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in pivR.columns:
        ax.plot(pivR.index, pivR[m], marker="o", label=m)
    ax.axhline(0, color="grey", ls="--", lw=1)
    learned = [m for m in pivR.columns if m != "persistence"]
    ax.set_ylim(float(pivR[learned].min().min()) * 1.4 - 0.02,
                max(0.05, float(pivR[learned].max().max()) * 1.2))
    ax.set_xlabel("horizon de rollout (s)")
    ax.set_ylabel("R²_OOS rendement cumulé")
    ax.set_title("Crypto (Bybit L2) - world model d'état, rollout vs random walk")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_rollout.png"), dpi=130)
    plt.close(fig)

    # stabilité temporelle du signal : R²_OOS(ret) du linéaire par fold
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = res1[(res1.dim == "ret") & (res1.model == "linear")]
    for sym in sorted(sub.symbol.unique()):
        s = sub[sub.symbol == sym].groupby("fold")["r2"].mean()
        ax.plot(s.index, s.values, marker="o", label=sym)
    ax.axhline(0, color="grey", ls="--", lw=1, label="random walk (=0)")
    ax.set_xlabel("fold walk-forward (≈ temps, inter-jours)")
    ax.set_ylabel("R²_OOS du rendement (linéaire)")
    ax.set_title("Stabilité du signal carnet dans le temps")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_stability.png"), dpi=130)
    plt.close(fig)

    sym0 = next(iter(detail))
    det = detail[sym0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fee, lbl in [(0.0, "sans frais (gross)"), (2.0, "net frais 2 bp"),
                     (5.5, "net frais 5.5 bp (taker Bybit)")]:
        net = det["gross"] - det["dpos"] * (det["half"] + fee * 1e-4)
        ax.plot(np.cumsum(net) * 100, label=lbl)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("barres de test (1 s, multi-jours)")
    ax.set_ylabel("PnL cumulé (%)")
    ax.set_title(f"Crypto {sym0} - un edge réel qui est un mirage net de frais")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crypto_lob_mirage.png"), dpi=130)
    plt.close(fig)

    print(f"\nPrédictions OOS par symbole : {args.out}/crypto_lob_oos_<SYM>.npz")
    print(f"\nFigures + CSV dans {args.out}/")


if __name__ == "__main__":
    main()
