"""Batch OFI evenementiel Bybit : telecharge -> extrait l'OFI -> supprime le zip.

Ne conserve qu'une Series par (symbole, date) : ~86 400 flottants, quelques centaines
de Ko, contre 28 Mo pour le .pkl de barres et ~280 Mo pour le zip. Les archives ne sont
donc pas conservees, et le pic disque reste ~300 Mo.

Les dates ne sont PAS un argument de ce script : elles sont lues dans le
pre-enregistrement `configs/phase1c_crypto_prereg.yaml`. C'est deliberé - une date
choisie en ligne de commande apres avoir vu un resultat serait du data snooping, et le
fait de ne pas pouvoir la passer est ce qui rend ce point verifiable.

Idempotent : saute tout (symbole, date) dont le .pkl OFI existe deja.

    python scripts/crypto/fetch_ofi.py
    python scripts/crypto/fetch_ofi.py --symbols BTCUSDT      # sous-ensemble (reprise)
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request

import yaml

from mirage.data.bybit_ofi import load_ofi

BASE = "https://quote-saver.bycsi.com/orderbook/linear"
PREREG = os.path.join("configs", "phase1c_crypto_prereg.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="sous-ensemble ; par defaut tous ceux du pre-enregistrement")
    ap.add_argument("--out", default=os.path.join("data", "raw", "crypto_ofi"))
    args = ap.parse_args()

    with open(PREREG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = args.symbols or list(cfg["donnees"]["symboles"])
    dates = list(cfg["donnees"]["dates"])
    os.makedirs(args.out, exist_ok=True)
    print(f"Pre-enregistrement : {len(symbols)} symboles x {len(dates)} dates "
          f"= {len(symbols) * len(dates)} couples", flush=True)

    todo = [(s, d) for s in symbols for d in dates]
    done = skipped = failed = 0
    for i, (sym, date) in enumerate(todo, 1):
        pkl = os.path.join(args.out, f"{date}_{sym}_ofi.pkl")
        if os.path.exists(pkl):
            skipped += 1
            continue

        stem = f"{date}_{sym}_ob500.data.zip"
        zpath = os.path.join(args.out, stem)
        part = zpath + ".part"
        t0 = time.time()
        try:
            # Telechargement en .part puis renommage : un arret brutal laisse un .part
            # qui ne sera jamais pris pour un zip complet
            urllib.request.urlretrieve(f"{BASE}/{sym}/{stem}", part)
            os.replace(part, zpath)
        except urllib.error.HTTPError as e:
            print(f"[{i}/{len(todo)}] ABSENT ({e.code}) : {date} {sym}", flush=True)
            failed += 1
            for p in (part, zpath):
                if os.path.exists(p):
                    os.remove(p)
            continue

        mb = os.path.getsize(zpath) / 1e6
        ofi = load_ofi(zpath)
        ofi.to_pickle(pkl)
        os.remove(zpath)

        abs_ = ofi.abs()
        print(f"[{i}/{len(todo)}] OK {date} {sym} : {len(ofi)} secondes actives | "
              f"zip {mb:.0f} Mo | |OFI| moy {abs_.mean():.3f} | "
              f"max {abs_.max():.1f} | {time.time() - t0:.0f}s", flush=True)
        done += 1

    print(f"\n=== Batch termine : {done} extraits, {skipped} deja en cache, "
          f"{failed} absents ===")


if __name__ == "__main__":
    main()
