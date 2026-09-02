"""Batch carnet Bybit : télécharge -> reconstruit les barres 1 s -> supprime le zip.

Traite chaque (symbole, date) séquentiellement pour que le pic disque reste ~300 Mo
au lieu de plusieurs Go : on ne conserve que le cache .pkl (~28 Mo/jour-symbole).
Idempotent : saute tout (symbole, date) dont le .pkl existe déjà.

    python scripts/fetch_bybit_batch.py --symbols BTCUSDT ETHUSDT SOLUSDT \
        --dates 2025-05-08 2025-05-22 2025-06-02 2025-06-18 2025-07-09 2025-08-06
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request

from mirage.data.bybit_lob import load_book_bars

BASE = "https://quote-saver.bycsi.com/orderbook/linear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join("data", "raw", "crypto_lob"))
    ap.add_argument("--keep-zip", action="store_true",
                    help="conserver le zip brut (par défaut il est supprimé après reconstruction)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    todo = [(s, d) for s in args.symbols for d in args.dates]
    done = skipped = failed = 0
    for i, (sym, date) in enumerate(todo, 1):
        pkl = os.path.join(args.out, f"{date}_{sym}_1s_book.pkl")
        if os.path.exists(pkl):
            print(f"[{i}/{len(todo)}] skip (cache présent) : {date} {sym}", flush=True)
            skipped += 1
            continue

        stem = f"{date}_{sym}_ob500.data.zip"
        zpath = os.path.join(args.out, stem)
        t0 = time.time()
        if not os.path.exists(zpath):
            try:
                urllib.request.urlretrieve(f"{BASE}/{sym}/{stem}", zpath)
            except urllib.error.HTTPError as e:
                print(f"[{i}/{len(todo)}] ABSENT ({e.code}) : {date} {sym}", flush=True)
                failed += 1
                continue
        mb = os.path.getsize(zpath) / 1e6

        bars = load_book_bars(zpath, levels=10, freq_s=1)
        bars.to_pickle(pkl)
        if not args.keep_zip:
            os.remove(zpath)                      # le .pkl suffit ; le zip est re-téléchargeable

        mid = (bars["ask_price_1"] + bars["bid_price_1"]) / 2
        spr = (bars["ask_price_1"] - bars["bid_price_1"]) / mid * 1e4
        print(f"[{i}/{len(todo)}] OK {date} {sym} : {len(bars)} barres 1s | "
              f"zip {mb:.0f} Mo | mid~{mid.mean():.2f} | spread~{spr.mean():.3f} bp | "
              f"{time.time() - t0:.0f}s", flush=True)
        done += 1

    print(f"\n=== Batch terminé : {done} construits, {skipped} déjà en cache, {failed} absents ===")


if __name__ == "__main__":
    main()
