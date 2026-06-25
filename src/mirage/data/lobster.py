"""Parsing des fichiers LOBSTER (message + orderbook).

Format LOBSTER :
  - message  (6 col) : time, event_type, order_id, size, price, direction
  - orderbook(4*N col): ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...
  - prix = dollars x 10000 (entiers) ; time = secondes après minuit.
  - les deux fichiers sont alignés 1:1 (ligne à ligne).
"""
from __future__ import annotations

import pandas as pd

MESSAGE_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]
PRICE_SCALE = 10000  # LOBSTER : prix en dollars x 10000


def orderbook_columns(levels: int) -> list[str]:
    cols: list[str] = []
    for i in range(1, levels + 1):
        cols += [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
    return cols


def load_orderbook(path: str, levels: int = 10) -> pd.DataFrame:
    """Lit les `levels` premiers niveaux du carnet.

    Robuste aux fichiers plus profonds que demandé : un sample L50 a 200 colonnes,
    on ne parse que les 4*levels premières (= meilleurs niveaux). usecols évite de
    charger les colonnes inutiles en mémoire.
    """
    cols = orderbook_columns(levels)
    df = pd.read_csv(path, header=None, usecols=range(4 * levels))
    df.columns = cols
    price_cols = [c for c in cols if "price" in c]
    df[price_cols] = df[price_cols] / PRICE_SCALE
    return df


def load_sample(
    message_file: str,
    orderbook_file: str,
    levels: int = 10,
    session_start_s: float | None = None,
    session_end_s: float | None = None,
) -> pd.DataFrame:
    """Charge message + orderbook, filtre la session, renvoie le carnet horodaté.

    Le filtrage session est appliqué AUX DEUX fichiers via le même masque pour
    préserver l'alignement 1:1.
    """
    msg = pd.read_csv(message_file, header=None, names=MESSAGE_COLS)
    book = load_orderbook(orderbook_file, levels)
    if len(msg) != len(book):
        raise ValueError(
            f"message ({len(msg)}) et orderbook ({len(book)}) non alignés."
        )

    mask = pd.Series(True, index=msg.index)
    if session_start_s is not None:
        mask &= msg["time"] >= session_start_s
    if session_end_s is not None:
        mask &= msg["time"] <= session_end_s

    book = book[mask.values].reset_index(drop=True)
    book.insert(0, "time", msg.loc[mask, "time"].to_numpy())
    return book
