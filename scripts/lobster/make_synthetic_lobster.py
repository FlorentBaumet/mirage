"""Génère un échantillon SYNTHÉTIQUE au format LOBSTER, pour faire tourner le
pipeline sans télécharger les vraies données.

Les fichiers produits portent exactement les noms, sessions et profondeurs que
`configs/phase0.yaml` attend (mêmes 5 instruments que le sample réel). Le mid suit
une marche aléatoire en cents et le carnet est reconstruit autour : **aucun
résultat obtenu dessus n'a de signification** — c'est un banc d'essai du code,
pas une mesure de marché.

    python scripts/lobster/make_synthetic_lobster.py
    python -m mirage.eval --config configs/phase0.yaml --raw-dir data/raw/synthetic
"""
from __future__ import annotations

import os

import numpy as np

PRICE_SCALE = 10000  # LOBSTER : prix en dollars x 10000, soit cents x 100
DATE = "2012-06-21"
FULL_DAY = (34200, 57600)   # 09:30 -> 16:00
ONE_HOUR = (34200, 37800)   # 09:30 -> 10:30 (le sample AAPL L50 réel)

# (ticker, profondeur du fichier, session, prix initial $) — aligné sur phase0.yaml
INSTRUMENTS = [
    ("AMZN", 10, FULL_DAY, 222.0),
    ("GOOG", 10, FULL_DAY, 570.0),
    ("INTC", 10, FULL_DAY, 27.0),
    ("MSFT", 10, FULL_DAY, 30.0),
    ("AAPL", 50, ONE_HOUR, 586.0),
]
OUT_DIR = os.path.join("data", "raw", "synthetic")
N_EVENTS = 12000


def write_instrument(ticker, levels, session, price0, n_events, rng, out_dir) -> int:
    """Écrit les CSV message + orderbook d'un instrument. Renvoie le nombre d'events."""
    start_s, end_s = session
    dt = rng.exponential(scale=(end_s - start_s) / n_events, size=n_events)
    times = start_s + np.cumsum(dt)
    times = times[times <= end_s]
    n = len(times)

    # mid en cents, marche aléatoire bornée (le pas de prix LOBSTER est le cent)
    mid_cents = int(round(price0 * 100)) + np.cumsum(
        rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
    )
    mid_cents = np.clip(mid_cents, 100, None)

    msg_lines, book_lines = [], []
    for i in range(n):
        m = int(mid_cents[i])
        half = int(rng.integers(1, 3))  # demi-spread en cents
        ask1, bid1 = m + half, m - half

        # message : valeurs plausibles ; seul l'horodatage sert vraiment au pipeline
        etype = int(rng.choice([1, 2, 3, 4]))
        size = int(rng.integers(1, 500))
        direction = int(rng.choice([-1, 1]))
        price = (ask1 if direction == -1 else bid1) * (PRICE_SCALE // 100)
        msg_lines.append(f"{times[i]:.9f},{etype},{i + 1},{size},{price},{direction}")

        row = []
        for lvl in range(levels):
            row += [
                (ask1 + lvl) * (PRICE_SCALE // 100), int(rng.integers(1, 1000)),
                (bid1 - lvl) * (PRICE_SCALE // 100), int(rng.integers(1, 1000)),
            ]
        book_lines.append(",".join(str(v) for v in row))

    tag = f"{ticker}_{DATE}_{start_s}000_{end_s}000"
    for kind, lines in (("message", msg_lines), ("orderbook", book_lines)):
        path = os.path.join(out_dir, f"{tag}_{kind}_{levels}.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return n


def main(seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Échantillon synthétique dans {OUT_DIR}/ — données bidon, ne rien en conclure.")
    for ticker, levels, session, price0 in INSTRUMENTS:
        n = write_instrument(ticker, levels, session, price0, N_EVENTS, rng, OUT_DIR)
        print(f"  {ticker}: {n} events, {levels} niveaux")


if __name__ == "__main__":
    main()
