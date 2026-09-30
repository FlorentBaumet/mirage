"""Apport de l'OFI evenementiel : test APPARIE base vs enrichi.

Question pre-enregistree (configs/phase1c_crypto_prereg.yaml, regle `apport`) :
l'ajout de l'OFI ameliore-t-il la prevision du rendement ?

POURQUOI APPARIE, ET PAS DEUX IC SEPARES
----------------------------------------
Les deux bras partagent tout sauf une colonne : memes barres, memes jours, memes folds
walk-forward (le nombre d'echantillons est identique), memes cibles. Comparer deux
intervalles de confiance calcules sur des reechantillonnages INDEPENDANTS reviendrait a
traiter deux modeles presque identiques comme deux experiences distinctes : l'IC de la
difference serait faussement large, et un apport reel pourrait passer pour nul.

Ici, chaque replication tire UNE fois les journees et les applique aux DEUX bras. La
statistique est la difference, et l'IC est celui de la loi de la difference sur
echantillons apparies.

Le reechantillonnage est celui de bootstrap_signif.py (meme fonction), garantissant que
les deux verdicts ne divergent pas par la seule mecanique du tirage.

    python scripts/crypto/crypto_lob.py --out experiments
    python scripts/crypto/crypto_lob.py --out experiments_ofi --state ofi
    python scripts/crypto/paired_ofi.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from bootstrap_signif import boot_mult, load_prereg  # meme tirage pour les deux scripts

BASE_DIR = "experiments"
OFI_DIR = "experiments_ofi"


def load_arm(directory: str, sym: str) -> dict:
    pf = os.path.join(directory, f"crypto_lob_oos_{sym}.npz")
    if not os.path.exists(pf):
        raise SystemExit(f"npz manquant : {pf}\nLance d'abord crypto_lob.py pour ce bras.")
    with np.load(pf) as z:
        return {k: z[k] for k in z.files}


def assert_apparies(sym: str, b: dict, e: dict) -> None:
    """Les deux bras doivent porter EXACTEMENT le meme echantillon.

    Sans cette verification, une colonne ajoutee qui decalerait la fenetre (ou ferait
    disparaitre les premieres lignes) produirait une comparaison silencieusement fausse.
    """
    for k in ("y", "day"):
        if len(b[k]) != len(e[k]):
            raise SystemExit(f"[{sym}] {k} : {len(b[k])} (base) vs {len(e[k])} "
                             f"(enrichi) - echantillons non appariables.")
        if not np.array_equal(b[k], e[k]):
            raise SystemExit(f"[{sym}] {k} : valeurs differentes entre les deux bras "
                             f"- les echantillons ne sont pas appariables.")


def symbol_paired(sym: str, cfg: dict) -> dict:
    b, e = load_arm(BASE_DIR, sym), load_arm(OFI_DIR, sym)
    assert_apparies(sym, b, e)

    y = b["y"].astype(float)
    day = b["day"].astype(int)
    boot_cfg = cfg["evaluation"]["bootstrap"]
    n_boot, seed = int(boot_cfg["n_boot"]), int(boot_cfg["seed"])
    blk = int(boot_cfg["variante_robustesse"]["longueur_bloc_jours"])
    fees = [float(f) for f in cfg["evaluation"]["frais_bp"]]

    n_days = int(day.max()) + 1
    counts = np.bincount(day, minlength=n_days).astype(float)
    pool = np.where(counts > 0)[0]
    c = counts[pool]

    sse_b = np.bincount(day, weights=y**2, minlength=n_days)[pool]
    sse_m = {arm: np.bincount(day, weights=(y - d["pred"].astype(float)) ** 2,
                              minlength=n_days)[pool] for arm, d in (("base", b), ("ofi", e))}
    net_s = {arm: {f: np.bincount(day, weights=d["gross"].astype(float)
                                  - d["dpos"].astype(float)
                                  * (d["half"].astype(float) + f * 1e-4),
                                  minlength=n_days)[pool] for f in fees}
             for arm, d in (("base", b), ("ofi", e))}

    def r2(sse):  # estimateur ponctuel, identique a bootstrap_signif.py
        return 1.0 - sse.sum() / sse_b.sum()

    out = {"symbol": sym, "n_oos": len(y), "n_days": len(pool),
           "r2_base": r2(sse_m["base"]), "r2_ofi": r2(sse_m["ofi"])}
    out["delta_r2"] = out["r2_ofi"] - out["r2_base"]

    rng = np.random.default_rng(seed)

    def one_pass(block_days: int, suffix: str) -> dict:
        m = boot_mult(rng, len(pool), n_boot, block_days)
        d_r2 = ((1.0 - (m @ sse_m["ofi"]) / (m @ sse_b))
                - (1.0 - (m @ sse_m["base"]) / (m @ sse_b)))
        res = {f"dlo{suffix}": float(np.percentile(d_r2, 2.5)),
               f"dhi{suffix}": float(np.percentile(d_r2, 97.5))}
        if not suffix:
            res["d_p_le0"] = float(np.mean(d_r2 <= 0))
        for f in fees:
            tag = f"f{f:g}"
            d_net = ((m @ net_s["ofi"][f]) - (m @ net_s["base"][f])) / (m @ c) * 1e4
            res[f"dnet_{tag}{suffix}"] = float(np.mean(d_net))
            res[f"dnet_lo_{tag}{suffix}"] = float(np.percentile(d_net, 2.5))
            res[f"dnet_hi_{tag}{suffix}"] = float(np.percentile(d_net, 97.5))
        return res

    out |= one_pass(1, "")
    out |= one_pass(blk, f"_blk{blk}")
    return out


def one_step_table() -> None:
    """R²_OOS 1-step par dimension, base vs enrichi (lecture des CSV de chaque bras)."""
    paths = {a: os.path.join(d, "crypto_lob_1step.csv") for a, d in
             (("base", BASE_DIR), ("ofi", OFI_DIR))}
    if not all(os.path.exists(p) for p in paths.values()):
        print("\n(CSV 1-step d'un bras absent : tableau par dimension non produit)")
        return
    piv = {}
    for arm, p in paths.items():
        df = pd.read_csv(p)
        piv[arm] = df.pivot_table(index="dim", columns="model", values="r2", aggfunc="mean")
    dims = [d for d in piv["ofi"].index if d in piv["base"].index]
    print("\n--- R2_OOS 1-step par dimension : lineaire, base vs enrichi ---")
    tbl = pd.DataFrame({
        "r2_base": piv["base"].loc[dims, "linear"],
        "r2_ofi": piv["ofi"].loc[dims, "linear"],
    })
    tbl["delta"] = tbl["r2_ofi"] - tbl["r2_base"]
    print(tbl.round(5).to_string())


def main() -> None:
    cfg = load_prereg()
    symbols = cfg["donnees"]["symboles"]
    boot_cfg = cfg["evaluation"]["bootstrap"]
    blk = int(boot_cfg["variante_robustesse"]["longueur_bloc_jours"])

    rows = [symbol_paired(s, cfg) for s in symbols]
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    print(f"=== Apport de l'OFI : bootstrap APPARIE par jour, B={boot_cfg['n_boot']}, "
          f"memes journees tirees pour les deux bras ===")

    print("\n--- delta R2_OOS(rendement) = enrichi - base ---")
    show = df[["symbol", "n_days", "r2_base", "r2_ofi", "delta_r2", "dlo", "dhi", "d_p_le0"]]
    print(show.round(5).to_string(index=False))

    print(f"\n--- Robustesse : blocs contigus de {blk} jours ---")
    rb = df[["symbol", "delta_r2"]].copy()
    rb["delta_IC95_blk"] = [f"[{a:.5f},{b:.5f}]" for a, b in
                            zip(df[f"dlo_blk{blk}"], df[f"dhi_blk{blk}"], strict=True)]
    print(rb.round(5).to_string(index=False))

    print("\n--- delta net_bp = enrichi - base (moyenne, IC95 apparie) ---")
    dn = df[["symbol"]].copy()
    for f in [float(x) for x in cfg["evaluation"]["frais_bp"]]:
        tag = f"f{f:g}"
        dn[f"dnet_{f:g}"] = df[f"dnet_{tag}"]
        dn[f"IC95_{f:g}"] = [f"[{a:.3f},{b:.3f}]" for a, b in
                             zip(df[f"dnet_lo_{tag}"], df[f"dnet_hi_{tag}"], strict=True)]
    print(dn.round(4).to_string(index=False))

    one_step_table()

    # --- application mecanique de la regle `apport` ---
    n_apporte = int((df["dlo"] > 0).sum())
    print("\n=== Verdicts (regles pre-enregistrees, configs/phase1c_crypto_prereg.yaml) ===")
    verdict = ("OFI APPORTE UNE INFORMATION REELLE" if n_apporte >= 4
               else "APPORT NON ETABLI")
    print(f"  apport : borne basse de l'IC95 apparie de delta_r2_oos > 0 sur "
          f"{n_apporte}/{len(df)} symboles -> {verdict} (seuil pre-enregistre : >= 4/5)")
    print(f"  ecart-type de la difference : IC95 le plus etroit = "
          f"{float((df['dhi'] - df['dlo']).min()):.5f}, le plus large = "
          f"{float((df['dhi'] - df['dlo']).max()):.5f}")

    out = os.path.join(OFI_DIR, "crypto_ofi_paired.csv")
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
