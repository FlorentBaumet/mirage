"""Télécharge le carnet L2 Bybit depuis quote-saver.bycsi.com (gratuit, sans clé).

URL : https://quote-saver.bycsi.com/orderbook/linear/{SYMBOL}/{YYYY-MM-DD}_{SYMBOL}_ob500.data.zip
⚠️ ~250-300 Mo par jour-symbole. Données disponibles à partir de mai 2025.

    python scripts/download_bybit_lob.py --symbols BTCUSDT ETHUSDT --dates 2025-06-02
"""
from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request

BASE = "https://quote-saver.bycsi.com/orderbook/linear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--dates", nargs="+", default=["2025-06-02"])
    ap.add_argument("--out", default=os.path.join("data", "raw", "crypto_lob"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for sym in args.symbols:
        for date in args.dates:
            stem = f"{date}_{sym}_ob500.data.zip"
            out = os.path.join(args.out, stem)
            if os.path.exists(out):
                print("déjà présent :", stem)
                continue
            url = f"{BASE}/{sym}/{stem}"
            try:
                urllib.request.urlretrieve(url, out)
            except urllib.error.HTTPError as e:
                print(f"absent ({e.code}) : {stem}")
                continue
            print(f"ok {round(os.path.getsize(out)/1e6, 1)} Mo : {stem}")


if __name__ == "__main__":
    main()
