"""Caractérise le régime tick (large-tick vs small-tick) par ticker.

Sert à substantier l'explication de l'hétérogénéité de la prédictibilité
directionnelle (large-tick => mid très prédictible à court terme).
"""
from __future__ import annotations

import pandas as pd

from mirage.data.bars import to_clock_bars
from mirage.data.lobster import load_sample
from mirage.eval import instrument_paths, load_config

TICK = 0.01


def main(config="configs/phase0.yaml"):
    cfg = load_config(config)
    d = cfg["data"]
    rows = []
    for inst in d["instruments"]:
        msg, book = instrument_paths(cfg, inst)
        b = load_sample(msg, book, d["levels"],
                        d.get("session_start_s"), d.get("session_end_s"))
        bars = to_clock_bars(b, cfg["bars"]["freq"], d["date"])
        mid = (bars["ask_price_1"] + bars["bid_price_1"]) / 2.0
        spread = bars["ask_price_1"] - bars["bid_price_1"]
        ticks = spread / TICK
        rows.append({
            "ticker": inst["ticker"],
            "mid_usd": round(float(mid.mean()), 2),
            "spread_ticks": round(float(ticks.mean()), 2),
            "pct_spread_1tick": round(float((ticks.round() <= 1).mean()) * 100, 1),
            "regime": "large-tick" if ticks.mean() < 2 else "small-tick",
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
