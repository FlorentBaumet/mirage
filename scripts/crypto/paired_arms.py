"""Analyse appariee multi-bras : delta R2 et delta net, avec IC95 par jour.

Applique mecaniquement les regles des deux pre-enregistrements :
  - configs/phase1d_crypto_prereg.yaml : OFI + classe de modele (lineaire vs MLP).
  - configs/phase1e_crypto_prereg.yaml : profondeur hors meilleur niveau (deep).

Le dispositif de bootstrap est celui de paired_ofi.py / bootstrap_signif.py : B=2000,
graine=0, unite = la journee, variante robustesse = blocs contigus de 3 jours. A chaque
replication, UNE seule serie de journees tirees avec remise sert aux DEUX bras ; la
statistique est leur DIFFERENCE (appariement). Chaque comparaison repart d'un generateur
de graine fixe, donc toutes les comparaisons d'un symbole partagent le meme tirage.

    python scripts/crypto/arm_eval.py
    python scripts/crypto/paired_arms.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml
from bootstrap_signif import boot_mult

ARMS_DIR = "experiments_arms"
PHASE1D = os.path.join("configs", "phase1d_crypto_prereg.yaml")
PHASE1E = os.path.join("configs", "phase1e_crypto_prereg.yaml")


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_arm(sym: str) -> dict:
    pf = os.path.join(ARMS_DIR, f"arms_oos_{sym}.npz")
    if not os.path.exists(pf):
        raise SystemExit(f"npz manquant : {pf}\nLance d'abord arm_eval.py pour ce symbole.")
    with np.load(pf) as z:
        return {k: z[k] for k in z.files}


def assert_apparies(sym: str, npz: dict, tag_gain: str, tag_ref: str) -> None:
    """Les deux bras doivent porter EXACTEMENT le meme echantillon.

    Les sommes de y^2 par jour doivent coincider : c'est l'empreinte de l'echantillon
    teste. Un decalage (colonne qui ferait disparaitre ou glisser des lignes) produirait
    une comparaison silencieusement fausse -> on echoue bruyamment.
    """
    for k in ("y", "day"):
        if len(npz[k]) != int(npz["n_oos"]):
            raise SystemExit(f"[{sym}] {k} : {len(npz[k])} vs n_oos={int(npz['n_oos'])}.")
    if not np.array_equal(npz[f"sse_b_{tag_gain}"], npz[f"sse_b_{tag_ref}"]):
        raise SystemExit(f"[{sym}] {tag_gain} et {tag_ref} n'ont pas les memes journees "
                         f"d'echantillons (sommes de y^2 differentes) : non appariables.")


def net_day(npz: dict, arm: str, model: str, fee: float) -> np.ndarray:
    """Net par jour (somme des net par echantillon) a un niveau de frais donne."""
    return (npz[f"gross_{arm}_{model}"] - npz[f"dposhalf_{arm}_{model}"]
            - fee * 1e-4 * npz[f"dpos_{arm}_{model}"])


def paired_compare(sym: str, npz: dict, gain: str, model_gain: str,
                   ref: str, model_ref: str, fees, n_boot: int, seed: int,
                   blk: int) -> dict:
    """delta R2 et delta net apparies (gain - ref), IC95 par jour."""
    tg, tr = f"{gain}_{model_gain}", f"{ref}_{model_ref}"
    assert_apparies(sym, npz, tg, tr)
    sb = npz[f"sse_b_{tg}"]                      # identiques (verifie ci-dessus)
    sm_g, sm_r = npz[f"sse_m_{tg}"], npz[f"sse_m_{tr}"]
    c = npz["count"].astype(float)

    def r2(sse):
        return 1.0 - sse.sum() / sb.sum()

    out = {"symbol": sym, "comparison": f"{tg} - {tr}", "n_days": int(npz["n_days"]),
           "n_oos": int(npz["n_oos"]), "r2_gain": r2(sm_g), "r2_ref": r2(sm_r)}
    out["delta_r2"] = out["r2_gain"] - out["r2_ref"]

    rng = np.random.default_rng(seed)

    def one_pass(block: int, suffix: str) -> dict:
        m = boot_mult(rng, len(c), n_boot, block)
        d_r2 = ((1.0 - (m @ sm_g) / (m @ sb)) - (1.0 - (m @ sm_r) / (m @ sb)))
        res = {f"dlo{suffix}": float(np.percentile(d_r2, 2.5)),
               f"dhi{suffix}": float(np.percentile(d_r2, 97.5))}
        if not suffix:
            res["d_p_le0"] = float(np.mean(d_r2 <= 0))
        for f in fees:
            t = f"f{f:g}"
            dn = ((m @ net_day(npz, gain, model_gain, f)) / (m @ c)
                  - (m @ net_day(npz, ref, model_ref, f)) / (m @ c)) * 1e4
            res[f"dnet_{t}{suffix}"] = float(np.mean(dn))
            res[f"dnet_lo_{t}{suffix}"] = float(np.percentile(dn, 2.5))
            res[f"dnet_hi_{t}{suffix}"] = float(np.percentile(dn, 97.5))
        return res

    out |= one_pass(1, "")
    out |= one_pass(blk, f"_blk{blk}")
    return out


def arm_net(sym: str, npz: dict, arm: str, model: str, fee: float,
            n_boot: int, seed: int) -> dict:
    """Net_bp d'UN bras (moyenne + IC95), pour les regles mirage/bascule.

    Bootstrap PRIMAIRE pre-enregistre (unite = jour, tirage i.i.d. avec remise) : c'est
    celui que la regle `mirage` emploie. La variante blocs de 3 jours est une robustesse.
    """
    c = npz["count"].astype(float)
    nd = net_day(npz, arm, model, fee)
    rng = np.random.default_rng(seed)
    m = boot_mult(rng, len(c), n_boot, 1)
    nb = (m @ nd) / (m @ c) * 1e4
    return {"symbol": sym, "arm_model": f"{arm}_{model}", "fee_bp": fee,
            "net_bp": float(nd.sum() / c.sum() * 1e4),
            "net_lo": float(np.percentile(nb, 2.5)),
            "net_hi": float(np.percentile(nb, 97.5))}


def main() -> None:
    global ARMS_DIR
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ARMS_DIR)
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    ARMS_DIR = args.out

    cfg_d, cfg_e = load_cfg(PHASE1D), load_cfg(PHASE1E)
    # Les deux pre-enregistrements partagent le meme dispositif de bootstrap : on VERIFIE
    # les champs qui le gouvernent (et non l'objet entier, qui porte des cles propres a
    # chaque phase) plutot que de le supposer.
    def boot_sig(cfg):
        b = cfg["evaluation"]["bootstrap"]
        return (b["n_boot"], b["seed"], b["unite"],
                b["variante_robustesse"]["longueur_bloc_jours"])

    if (boot_sig(cfg_d) != boot_sig(cfg_e)
            or cfg_d["evaluation"]["frais_bp"] != cfg_e["evaluation"]["frais_bp"]):
        raise SystemExit("phase1d et phase1e divergent sur le bootstrap ou les frais : "
                         "les deux phases ne sont plus comparables.")
    boot_cfg = cfg_d["evaluation"]["bootstrap"]
    n_boot, seed = int(boot_cfg["n_boot"]), int(boot_cfg["seed"])
    blk = int(boot_cfg["variante_robustesse"]["longueur_bloc_jours"])
    fees = [float(f) for f in cfg_d["evaluation"]["frais_bp"]]
    symbols = cfg_d["donnees"]["symboles"]
    if args.symbols:
        symbols = [s for s in symbols if s in args.symbols]

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    rows = []
    arm_rows = []
    for sym in symbols:
        npz = load_arm(sym)
        for g, mg, r, mr in (("ofi", "mlp", "base", "mlp"),
                             ("ofi", "linear", "base", "linear"),
                             ("deep", "mlp", "base", "mlp"),
                             ("deep", "linear", "base", "linear"),
                             ("base", "mlp", "base", "linear")):
            rows.append(paired_compare(sym, npz, g, mg, r, mr, fees, n_boot, seed, blk))
        for arm in ("base", "ofi", "deep"):
            for model in ("linear", "mlp"):
                arm_rows.append(arm_net(sym, npz, arm, model, 2.0, n_boot, seed))

    df = pd.DataFrame(rows)
    arm_df = pd.DataFrame(arm_rows)
    pd.set_option("display.width", 200)

    log(f"=== Analyse appariee multi-bras - B={n_boot}, graine={seed}, unite = jour, "
        f"robustesse = blocs de {blk} jours ===")
    log("\n--- delta R2_OOS(ret) = gain - ref (IC95 apparie par jour) ---")
    show = df[["symbol", "comparison", "n_days", "r2_ref", "r2_gain", "delta_r2",
               "dlo", "dhi"]].copy()
    show["IC95"] = [f"[{a:.5f},{b:.5f}]" for a, b in zip(df["dlo"], df["dhi"], strict=True)]
    log(show.round(5).to_string(index=False))

    log(f"\n--- Robustesse : blocs contigus de {blk} jours (delta R2) ---")
    rb = df[["symbol", "comparison", "delta_r2"]].copy()
    rb["IC95_blk"] = [f"[{a:.5f},{b:.5f}]" for a, b in
                      zip(df[f"dlo_blk{blk}"], df[f"dhi_blk{blk}"], strict=True)]
    log(rb.round(5).to_string(index=False))

    log("\n--- delta net_bp = gain - ref (moyenne bootstrap, IC95 apparie) ---")
    dn = df[["symbol", "comparison"]].copy()
    for f in fees:
        t = f"f{f:g}"
        dn[f"dnet_{f:g}"] = df[f"dnet_{t}"]
        dn[f"IC95_{f:g}"] = [f"[{a:.3f},{b:.3f}]" for a, b in
                             zip(df[f"dnet_lo_{t}"], df[f"dnet_hi_{t}"], strict=True)]
    log(dn.round(4).to_string(index=False))

    log("\n--- net_bp a 2 bp par bras et modele (IC95 par jour) ---")
    arm_df["IC95"] = [f"[{a:.3f},{b:.3f}]" for a, b in
                      zip(arm_df["net_lo"], arm_df["net_hi"], strict=True)]
    log(arm_df.round(4).to_string(index=False))

    # --- regles appliquees mecaniquement -------------------------------------
    def ic_low_pos(comp: str) -> int:
        sub = df[df["comparison"] == comp]
        return int((sub["dlo"] > 0).sum())

    def verdict_apport(n: int) -> str:
        if n >= 4:
            return f"APPORTE UNE INFORMATION REELLE ({n}/{len(symbols)} >= 4/5)"
        if n >= 1:
            return f"APPORT PARTIEL ({n}/{len(symbols)}, non promu en decouverte)"
        return "APPORT NON ETABLI (plancher renforce)"

    def mirage_verdict(arm: str, model: str) -> str:
        sub = arm_df[(arm_df["arm_model"] == f"{arm}_{model}") & (arm_df["fee_bp"] == 2.0)]
        n_bad = int((~(sub["net_hi"] < 0)).sum())
        return ("MIRAGE CONFIRME (net a 2 bp entierement < 0 partout)" if n_bad == 0
                else f"MIRAGE NON confirme au sens strict ({n_bad}/{len(sub)} symbole(s))")

    def bascule_verdict(arm: str, model: str) -> str:
        sub = arm_df[(arm_df["arm_model"] == f"{arm}_{model}") & (arm_df["fee_bp"] == 2.0)]
        n_pos = int((sub["net_lo"] > 0).sum())
        return (f"BASCULE ({n_pos} symbole(s), borne basse > 0 a 2 bp)" if n_pos
                else "AUCUNE BASCULE (aucune borne basse > 0 a 2 bp)")

    log("\n" + "=" * 78)
    log("VERDICTS - configs/phase1d_crypto_prereg.yaml (OFI / classe de modele)")
    log("=" * 78)
    n_ofl = ic_low_pos("ofi_mlp - base_mlp")
    log(f"  apport_non_lineaire      : OFI vs base, MLP, IC95 borne basse > 0 sur "
        f"{n_ofl}/{len(symbols)} -> {verdict_apport(n_ofl)}")
    n_olf = ic_low_pos("ofi_linear - base_linear")
    log(f"  controle lineaire        : OFI vs base, lineaire, borne basse > 0 sur "
        f"{n_olf}/{len(symbols)} -> {verdict_apport(n_olf)}")
    n_mod = ic_low_pos("base_mlp - base_linear")
    log(f"  modele_gagne_t-il        : MLP vs lineaire a etat de base, borne basse > 0 sur "
        f"{n_mod}/{len(symbols)} -> " +
        ("LE MLP BAT LE LINEAIRE" if n_mod >= 4 else "DIFFERENCE DE CLASSE NON ETABLIE"))
    log(f"  mirage (bras ofi, MLP)   : {mirage_verdict('ofi', 'mlp')}")
    log(f"  bascule (bras ofi, MLP)  : {bascule_verdict('ofi', 'mlp')}")

    log("\n" + "=" * 78)
    log("VERDICTS - configs/phase1e_crypto_prereg.yaml (profondeur hors meilleur niveau)")
    log("=" * 78)
    n_dl = ic_low_pos("deep_linear - base_linear")
    log(f"  apport_profondeur        : deep vs base, LINEAIRE (primaire), borne basse > 0 sur "
        f"{n_dl}/{len(symbols)} -> {verdict_apport(n_dl)}")
    n_dm = ic_low_pos("deep_mlp - base_mlp")
    log(f"  apport_profondeur (MLP)  : deep vs base, MLP (secondaire), borne basse > 0 sur "
        f"{n_dm}/{len(symbols)} -> {verdict_apport(n_dm)}")
    log(f"  mirage (bras deep, lin.) : {mirage_verdict('deep', 'linear')}")
    log(f"  bascule (bras deep, lin.): {bascule_verdict('deep', 'linear')}")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "paired_arms.csv")
    df.to_csv(csv_path, index=False)
    arm_csv = os.path.join(args.out, "paired_arms_net_by_arm.csv")
    arm_df.to_csv(arm_csv, index=False)
    log_path = os.path.join(args.out, "paired_arms_log.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n-> {csv_path}\n-> {log_path}")


if __name__ == "__main__":
    main()
