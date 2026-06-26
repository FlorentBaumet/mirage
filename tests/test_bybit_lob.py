"""Sanity check du reconstructeur de carnet Bybit (snapshot + delta -> barres 1 s)."""
from __future__ import annotations

import json
import zipfile

from mirage.data.bybit_lob import load_book_bars


def _write_zip(path, lines):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.data", "\n".join(json.dumps(o) for o in lines))


def test_snapshot_delta_and_emit(tmp_path):
    asks = [["100.1", "5"], ["100.2", "3"], ["100.3", "2"], ["100.4", "1"], ["100.5", "1"]]
    bids = [["100.0", "4"], ["99.9", "2"], ["99.8", "1"], ["99.7", "1"], ["99.6", "1"]]
    lines = [
        {"type": "snapshot", "ts": 1000, "data": {"s": "X", "a": asks, "b": bids}},
        {"type": "delta", "ts": 1500, "data": {"s": "X", "a": [["100.1", "0"]], "b": []}},  # vide le best ask
        {"type": "delta", "ts": 2000, "data": {"s": "X", "a": [], "b": []}},                # change de seconde -> émet sec 1
    ]
    zf = tmp_path / "2025-01-01_X_ob500.data.zip"
    _write_zip(zf, lines)

    bars = load_book_bars(str(zf), levels=3, freq_s=1)
    # à la fin de la seconde 1, le best ask (100.1) a été retiré -> top-3 = 100.2, 100.3, 100.4
    assert bars.iloc[0]["ask_price_1"] == 100.2
    assert bars.iloc[0]["ask_price_2"] == 100.3
    assert bars.iloc[0]["bid_price_1"] == 100.0       # meilleur bid inchangé
    assert bars.iloc[0]["ask_size_1"] == 3.0          # taille du nouveau best ask
