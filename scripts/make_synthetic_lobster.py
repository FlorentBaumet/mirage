"""Génère un échantillon SYNTHÉTIQUE au format LOBSTER (message + orderbook L10).

But : tester le pipeline et les tests anti-fuite SANS le vrai sample.
⚠️ Données bidon : ne sert PAS à conclure quoi que ce soit (I3).
Le mid suit une marche aléatoire en ticks ; le carnet est construit autour.
"""
from __future__ import annotations

import os

import numpy as np

LEVELS = 10
TICK = 0.01
PRICE_SCALE = 10000
START_S, END_S = 34200, 57600  # 09:30 -> 16:00
OUT_DIR = os.path.join("data", "raw")
TICKER, DATE = "AAPL", "2012-06-21"


def main(n_events: int = 12000, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    # horodatages croissants sur la session
    dt = rng.exponential(scale=(END_S - START_S) / n_events, size=n_events)
    times = START_S + np.cumsum(dt)
    times = times[times <= END_S]
    n = len(times)

    # mid en ticks = marche aléatoire bornée
    steps = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
    mid_ticks = 5000 + np.cumsum(steps)  # ~ 50.00 $
    mid_ticks = np.clip(mid_ticks, 100, None)

    msg_lines, book_lines = [], []
    for i in range(n):
        m = int(mid_ticks[i])
        half = rng.integers(1, 3)  # demi-spread en ticks
        ask1, bid1 = m + half, m - half

        # message (valeurs plausibles, non utilisées au-delà de l'alignement)
        etype = int(rng.choice([1, 2, 3, 4]))
        size = int(rng.integers(1, 500))
        direction = int(rng.choice([-1, 1]))
        price = (ask1 if direction == -1 else bid1) * (PRICE_SCALE // 100)
        msg_lines.append(f"{times[i]:.9f},{etype},{i + 1},{size},{price},{direction}")

        # orderbook L10
        row = []
        for lvl in range(LEVELS):
            ap = (ask1 + lvl) * (PRICE_SCALE // 100)
            bp = (bid1 - lvl) * (PRICE_SCALE // 100)
            asz = int(rng.integers(1, 1000))
            bsz = int(rng.integers(1, 1000))
            row += [ap, asz, bp, bsz]
        book_lines.append(",".join(str(v) for v in row))

    tag = f"{TICKER}_{DATE}_{START_S}000_{END_S}000"
    msg_path = os.path.join(OUT_DIR, f"{tag}_message_{LEVELS}.csv")
    book_path = os.path.join(OUT_DIR, f"{tag}_orderbook_{LEVELS}.csv")
    with open(msg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(msg_lines) + "\n")
    with open(book_path, "w", encoding="utf-8") as f:
        f.write("\n".join(book_lines) + "\n")

    print(f"Écrit {n} events synthétiques :\n  {msg_path}\n  {book_path}")


if __name__ == "__main__":
    main()
