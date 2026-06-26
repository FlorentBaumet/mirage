"""Reconstruction du carnet à partir des dumps Bybit (quote-saver.bycsi.com).

Format (orderbook v5, fichiers .data.zip = JSON par ligne) :
  {"topic":"orderbook.500.BTCUSDT","type":"snapshot|delta","ts":<ms>,
   "data":{"s":"BTCUSDT","b":[["price","qty"],...],"a":[...],"u":..,"seq":..},"cts":..}
  - snapshot : b/a = carnet complet ; delta : b/a = changements (qty "0" = suppression).

On rejoue le flux (snapshot puis deltas), et on émet l'état du carnet à la fin de
chaque seconde, en gardant les `levels` meilleurs niveaux -> mêmes colonnes que LOBSTER
(ask_price_i/ask_size_i/bid_price_i/bid_size_i), donc compatible avec state.py / impact.py.

Lecture en STREAMING depuis le zip (le décompressé fait ~1-2 Go).
"""
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd


def _iter_lines(data_zip: str):
    with zipfile.ZipFile(data_zip) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        name = names[0]
        with z.open(name) as f:
            for raw in io.TextIOWrapper(f, encoding="utf-8"):
                raw = raw.strip()
                if raw:
                    yield raw


def _levels(o: dict):
    """Extrait (type, ts_ms, bids, asks) en tolérant quelques variantes de schéma."""
    d = o.get("data", o)
    typ = o.get("type") or d.get("type") or "delta"
    ts = int(o.get("ts") or d.get("ts"))
    b = d.get("b", d.get("bids", []))
    a = d.get("a", d.get("asks", []))
    return typ, ts, b, a


def load_book_bars(data_zip: str, levels: int = 10, freq_s: int = 1) -> pd.DataFrame:
    bids: dict[str, float] = {}
    asks: dict[str, float] = {}
    rows, secs = [], []
    prev_bucket = None

    def emit(bucket: int):
        a_top = sorted(asks.items(), key=lambda kv: float(kv[0]))[:levels]
        b_top = sorted(bids.items(), key=lambda kv: -float(kv[0]))[:levels]
        if len(a_top) < levels or len(b_top) < levels:
            return
        row = {}
        for i, (p, q) in enumerate(a_top, 1):
            row[f"ask_price_{i}"] = float(p)
            row[f"ask_size_{i}"] = q
        for i, (p, q) in enumerate(b_top, 1):
            row[f"bid_price_{i}"] = float(p)
            row[f"bid_size_{i}"] = q
        rows.append(row)
        secs.append(bucket * freq_s)

    for line in _iter_lines(data_zip):
        o = json.loads(line)
        typ, ts, b, a = _levels(o)
        bucket = ts // (1000 * freq_s)
        if prev_bucket is not None and bucket > prev_bucket:
            emit(prev_bucket)               # état du carnet à la fin du bucket précédent
        if typ == "snapshot":
            bids.clear()
            asks.clear()
        for p, q in b:
            q = float(q)
            bids.pop(p, None) if q == 0 else bids.__setitem__(p, q)
        for p, q in a:
            q = float(q)
            asks.pop(p, None) if q == 0 else asks.__setitem__(p, q)
        prev_bucket = bucket
    if prev_bucket is not None:
        emit(prev_bucket)

    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(secs, unit="s")
    return df
