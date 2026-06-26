"""Reconstruit les barres 1 s du carnet Bybit (zip -> .pkl en cache).

Parse les gros dumps .data.zip UNE fois et sauve des barres 1 s top-10 (format
LOBSTER) en pickle, pour que l'analyse soit rapide ensuite.

    python scripts/build_bybit_bars.py
"""
from __future__ import annotations

import glob
import os
import re

from mirage.data.bybit_lob import load_book_bars

DIR = os.path.join("data", "raw", "crypto_lob")


def main():
    for zf in sorted(glob.glob(os.path.join(DIR, "*_ob500.data.zip"))):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)_ob500", os.path.basename(zf))
        date, sym = m.group(1), m.group(2)
        out = os.path.join(DIR, f"{date}_{sym}_1s_book.pkl")
        if os.path.exists(out):
            print("skip (déjà fait) :", out)
            continue
        bars = load_book_bars(zf, levels=10, freq_s=1)
        bars.to_pickle(out)
        mid = (bars["ask_price_1"] + bars["bid_price_1"]) / 2
        spr = bars["ask_price_1"] - bars["bid_price_1"]
        print(f"{sym} {date}: {len(bars)} barres 1s | "
              f"{bars.index[0]}..{bars.index[-1]} | mid~{mid.mean():.1f} | "
              f"spread~{spr.mean():.4f} ({(spr/mid*1e4).mean():.2f} bp)")


if __name__ == "__main__":
    main()
