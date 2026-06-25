"""Run unique du world model GRU (torch) sur données réelles, horizon 1.

Vérifie que le modèle de séquence non-linéaire ne change PAS le verdict NO-GO.
Réutilise le même protocole figé (configs/phase0.yaml), n'override que le modèle.

    python scripts/run_gru.py
"""
from __future__ import annotations

import pandas as pd

from mirage.eval import go_no_go, load_config, run_phase0


def main():
    cfg = load_config("configs/phase0.yaml")
    cfg["model"] = {"name": "gru", "seq_len": 16, "hidden_gru": 32,
                    "epochs": 8, "seed": 0}
    res, meta = run_phase0(cfg)

    pd.set_option("display.width", 160)
    print("--- Couverture ---")
    print(meta.to_string(index=False))
    print("\n--- R2_OOS moyen par ticker x estimateur (GRU) ---")
    print(res.pivot_table(index="ticker", columns="estimator",
                          values="r2_oos", aggfunc="mean").round(5).to_string())
    print("\n--- dir_acc moyen par ticker x estimateur (GRU) ---")
    print(res.pivot_table(index="ticker", columns="estimator",
                          values="dir_acc", aggfunc="mean").round(4).to_string())
    print(go_no_go(res, "gru"))


if __name__ == "__main__":
    main()
