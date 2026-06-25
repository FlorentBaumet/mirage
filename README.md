# 🌐 MIRAGE — world model de marché + éval honnête

Pièce de portfolio. **Spec complète** : [`PROJET_MARKET_WORLD_MODEL.md`](PROJET_MARKET_WORLD_MODEL.md).
Le cœur du projet = un **world model de marché** action-conditionné **évalué honnêtement**
(anti-lookahead, walk-forward, out-of-sample). L'agent/edge est un bonus (Stage 2).

## Invariants (jamais violés)
- **I1** World model d'abord ; agent/edge en bonus.
- **I2** Zéro argent réel, zéro trade réel (sim / historique uniquement).
- **I3** L'évaluation honnête est la colonne vertébrale.
- **I4** Petit modèle, données basse dimension, low-compute.

## État : Phase 0 (MVP) — ✅ terminée

**Question** : un world model **simple** (séquence MLP/GRU) prédit-il le **rendement
next-step** (barres 1 s, cible = mid-price log-return) **mieux** que des baselines
naïves (zero-forecast / persistence / linéaire), **en out-of-sample honnête** ?

**Résultat → 📄 [`reports/PHASE0.md`](reports/PHASE0.md)** : **NO-GO** assumé (aucun
modèle ne bat le zero-forecast en R²_OOS, à aucun horizon de 1 à 60 s). Mais une
prédictibilité **directionnelle** réelle existe à 1 s, **concentrée sur les actions
large-tick** (INTC/MSFT ~80 %), **sous l'échelle du spread** → un *edge statistique
qui est un mirage net de coûts*. La thèse du projet, démontrée dès la Phase 0.

Décisions kickoff figées :
- Données : **LOBSTER** (samples gratuits 2012-06-21 : AMZN/GOOG/INTC/MSFT jour
  complet + AAPL 1 h) → puis **ABIDES** (Stage 2).
- Cible : **mid-price log-return**, horizon 1 barre, barres **1 s**.
- Modèle : **séquence simple** (MLP par défaut ; GRU optionnel via extra `torch`).
- Action-conditioning : **Phase 1** (la Phase 0 prédit le marché « passif »).
- Protocole d'éval **figé** dans [`configs/phase0.yaml`](configs/phase0.yaml) **avant** tout entraînement.

## Setup (Windows, venv dédié — jamais le Python global)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# (optionnel, pour le GRU)  .\.venv\Scripts\python.exe -m pip install -e ".[torch]"
```

## Lancer la Phase 0
```powershell
# 1) données : déposer le sample LOBSTER dans data/raw/  (voir data/README.md)
#    OU générer un échantillon synthétique pour tester le pipeline :
.\.venv\Scripts\python.exe scripts\make_synthetic_lobster.py

# 2) tests anti-fuite (la colonne vertébrale)
.\.venv\Scripts\python.exe -m pytest -q

# 3) éval walk-forward + Go/No-Go
.\.venv\Scripts\python.exe -m mirage.eval --config configs\phase0.yaml
```

## Structure
```
src/mirage/
  data/lobster.py   parse message + orderbook LOBSTER
  data/bars.py      event -> barres clock-time, mid-price
  features.py       features CAUSALES (imbalance, OFI, returns laggés)
  splits.py         walk-forward purgé + embargo   <- cœur I3
  baselines.py      zero-forecast / linéaire
  models/seq.py     MLP (défaut) + GRU (optionnel)
  metrics.py        R²_OOS, RMSE, hit-rate
  eval.py           boucle walk-forward + agrégation + Go/No-Go
tests/test_no_lookahead.py   tests anti-lookahead
configs/phase0.yaml          protocole FIGÉ
```
