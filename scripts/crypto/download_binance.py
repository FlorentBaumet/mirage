"""Télécharge des klines historiques depuis Binance Vision (gratuit, sans clé).

Dumps mensuels zip/CSV : data.binance.vision/data/spot/monthly/klines/SYM/INT/...
Idempotent (saute les fichiers déjà présents). Stdlib only.

    python scripts/download_binance.py --symbols BTCUSDT ETHUSDT --start 2024-01 --end 2024-12
"""
from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
import zipfile

BASE = "https://data.binance.vision/data/spot/monthly/klines"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def months(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--out", default=os.path.join("data", "raw", "crypto"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    got, skipped, missing = 0, 0, 0
    for sym in args.symbols:
        for mo in months(args.start, args.end):
            stem = f"{sym}-{args.interval}-{mo}"
            csv = os.path.join(args.out, stem + ".csv")
            if os.path.exists(csv):
                skipped += 1
                continue
            url = f"{BASE}/{sym}/{args.interval}/{stem}.zip"
            zpath = os.path.join(args.out, stem + ".zip")
            try:
                urllib.request.urlretrieve(url, zpath)
            except urllib.error.HTTPError as e:
                print(f"  absent ({e.code}) : {stem}")
                missing += 1
                continue
            with zipfile.ZipFile(zpath) as z:
                z.extractall(args.out)
            os.remove(zpath)
            got += 1
            print(f"  ok : {stem}")

    print(f"\nTerminé : {got} téléchargés, {skipped} déjà présents, {missing} absents.")


if __name__ == "__main__":
    main()
