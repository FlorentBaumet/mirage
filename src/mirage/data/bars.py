"""Agrégation des events LOBSTER en barres clock-time (1 s par défaut).

Causalité : pour chaque barre on prend le DERNIER snapshot du carnet dans la
seconde (état connu à la clôture de barre), puis forward-fill des secondes vides
(dernier état connu = info passée, donc causale).
"""
from __future__ import annotations

import pandas as pd


def to_clock_bars(book: pd.DataFrame, freq: str = "1s", date: str = "2012-06-21") -> pd.DataFrame:
    df = book.copy()
    ts = pd.to_datetime(date) + pd.to_timedelta(df["time"], unit="s")
    df = df.drop(columns=["time"]).set_index(ts)
    bars = df.resample(freq).last().ffill().dropna()
    return bars


def add_mid(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["mid"] = (bars["ask_price_1"] + bars["bid_price_1"]) / 2.0
    return bars
