"""Chargeur de klines Binance (dumps data.binance.vision).

Klines (12 colonnes) : open_time, open, high, low, close, volume, close_time,
quote_volume, trades, taker_buy_base, taker_buy_quote, ignore.
Prix/volumes en flottant ; temps en ms. taker_buy_base = volume initié à l'achat
(agresseur) -> proxy d'order-flow gratuit.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
NUM = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
       "taker_buy_base", "taker_buy_quote"]


def load_klines_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=KLINE_COLS)
    if str(df.iloc[0, 0]).strip().lower() == "open_time":   # certains dumps ont un en-tête
        df = df.iloc[1:].reset_index(drop=True)
    df["open_time"] = df["open_time"].astype("int64")
    df[NUM] = df[NUM].astype(float)
    return df


def load_symbol(symbol: str, interval: str, raw_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, f"{symbol}-{interval}-*.csv")))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier {symbol}-{interval}-*.csv dans {raw_dir}\n"
            "→ python scripts/download_binance.py --symbols " + symbol)
    df = pd.concat([load_klines_file(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    return df
