# MIRAGE - world model de marché et évaluation honnête

Un **world model de marché** (carnet d'ordres et prix) évalué par une procédure
**anti-lookahead et pré-enregistrée**, entièrement out-of-sample. Pas d'agent, pas de
trade réel : la question posée est de savoir si l'on peut *mesurer* un edge - et le
projet existe pour montrer comment on le **réfute**.

Le résultat porteur : sur le carnet crypto réel, le world model trouve un signal de prix
**authentique et statistiquement significatif** (R²_OOS 0.052 / 0.050 / 0.037 / 0.021 / 0.016
sur BTC / ETH / SOL / XRP / DOGE, intervalle de confiance à 95 % qui exclut zéro sur **5
symboles sur 5**). Une stratégie naïve dessus perd de l'argent **dès 2 bp de frais** sur les
cinq, et sur XRP elle est déjà morte **à zéro frais**. Le signal est réel ; l'edge est un
mirage.

![Un edge réel qui est un mirage net de frais](reports/figures/crypto_lob_mirage.png)

## Le résultat central - Phase 1 (carnet Bybit L2)

5 symboles × 44 journées (2023-01-20 → 2025-08-06), 19 M barres 1 s, dont 27 journées et
11,4 M points réellement testés. Position = signe de la prédiction linéaire out-of-sample ;
coût à chaque changement de position = demi-spread **réellement mesuré** + frais taker.
IC95 par bootstrap de journées (B = 2000).

| Symbole | R²_OOS [IC95] | brut/barre | turnover | net à 0 bp | net à 2 bp [IC95] | net à 5,5 bp |
|---|---|---|---|---|---|---|
| BTCUSDT | 0.052 [0.035, 0.079] | +0.164 bp | 0.287 | +0.160 bp | **−0.987 bp** [−1.06, −0.92] | −2.992 bp |
| ETHUSDT | 0.050 [0.036, 0.068] | +0.231 bp | 0.367 | +0.216 bp | **−1.254 bp** [−1.31, −1.20] | −3.825 bp |
| SOLUSDT | 0.037 [0.021, 0.064] | +0.271 bp | 0.387 | +0.036 bp | **−1.514 bp** [−1.59, −1.44] | −4.225 bp |
| XRPUSDT | 0.021 [0.008, 0.041] | +0.222 bp | 0.358 | **−0.133 bp** | **−1.566 bp** [−1.70, −1.45] | −4.074 bp |
| DOGEUSDT | 0.016 [0.002, 0.047] | +0.271 bp | 0.384 | +0.029 bp | **−1.505 bp** [−1.61, −1.40] | −4.190 bp |

Le net à 2 bp est négatif sur les cinq **avec un IC95 entièrement sous zéro** : la perte n'est
pas une moyenne qui pourrait basculer. Et XRP est négatif **sans le moindre frais** - son
demi-spread × son turnover consomme plus que son gain brut, exactement le mécanisme qui tuait
SOL sur le jeu précédent. Le mirage ne repose donc sur aucune hypothèse de frais : il est
structurel. Détail, limites et chiffres complets :
[`reports/PHASE1_CRYPTO.md`](reports/PHASE1_CRYPTO.md).

## La thèse, démontrée sur cinq jeux de données

| Phase | Données | Question | Verdict |
|---|---|---|---|
| [**0**](reports/PHASE0.md) - LOBSTER | 5 actions, 1 journée 2012, barres 1 s | Le rendement du mid est-il prédictible ? | **NO-GO** : R²_OOS ≤ 0 de 1 à 60 s. Le *signe* est prédictible sur les large-tick (82–83 % à 1 s) mais pour un gain 20–50× plus petit que le demi-spread. |
| [**0 (crypto)**](reports/PHASE0_CRYPTO.md) | 4 cryptos, 1 an de klines 1 min | La question tient-elle sur des mois ? | Random walk (R²_OOS ≤ 0 sur 4 coins × 6 périodes, dir ≈ 50 %). Le signal sous-seconde a disparu à 1 min. |
| [**1a**](reports/PHASE1.md) | LOBSTER | Prédire le *vecteur d'état* du carnet plutôt qu'un scalaire | Le prix reste un random walk ; la forme du carnet (spread, imbalances, micro-price) est légèrement prévisible linéairement (R²_OOS 0.02–0.09). |
| [**1b**](reports/PHASE1B.md) | LOBSTER | Action-conditionner le world model (impact de ses propres ordres) | Le plus petit ordre coûte déjà **1.7–3 bp** contre un edge prédictible **≤ 0.1 bp**. Mirage confirmé côté exécution. |
| [**1 (crypto LOB)**](reports/PHASE1_CRYPTO.md) | 5 symboles × 44 journées, carnet Bybit L2, 19 M barres 1 s | Sait-on distinguer un vrai edge d'un mirage quand le signal existe ? | **Le cas d'école** : signal réel et significatif (R²_OOS 0.016–0.052, IC95 > 0 sur 5/5), backtest brut flatteur, net de coûts négatif partout dès 2 bp, et mort sans frais sur XRP. |

## Méthodologie - ce qui rend l'évaluation crédible

- **Walk-forward purgé** : train sur le passé, test sur le futur, jamais l'inverse.
  Embargo ≥ horizon et ≥ lookback, pour qu'une cible regardant `h` barres en avant ne
  puisse pas empiéter sur le train.
- **Scaler ajusté sur le train seul**, puis appliqué au test - aucune statistique du futur.
- **Baselines obligatoires** à chaque étape (`zero`/random walk, `persistence`, `linear`) :
  un modèle qui ne les bat pas n'apporte rien, et on le dit.
- **Métrique primaire pré-enregistrée** (R²_OOS = 1 − SSE(modèle)/SSE(baseline)) et grille
  d'horizons figée dans les `configs/*.yaml` **avant** tout entraînement. Tout est reporté,
  aucun cherry-pick.
- **Tests anti-fuite automatisés** ([`tests/test_no_lookahead.py`](tests/test_no_lookahead.py)) :
  alignement cible = futur, embargo respecté, scaler train-only, absence de signal fantôme
  sur données aléatoires.
- **Significativité et coûts** : block bootstrap circulaire pour les intervalles de
  confiance, et un verdict économique chiffré à côté du verdict statistique. Sur le carnet
  crypto, les blocs sont des **journées** entières, pas des observations : 2,3 M de barres
  d'une seconde ne font pas 2,3 M d'observations indépendantes.
- **Pré-enregistrement** : sur les phases où un chiffre publié peut être révisé, le protocole
  (dates, modèle, grille de frais, unité de bootstrap, graine) **et les règles de décision**
  sont commités avant d'exécuter quoi que ce soit - cf.
  [`configs/phase1_crypto_prereg.yaml`](configs/phase1_crypto_prereg.yaml).

« Significatif » n'est pas « rentable ». L'écart entre les deux est le sujet du projet.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# (optionnel, pour le GRU)  .\.venv\Scripts\python.exe -m pip install -e ".[torch]"
```

## Reproduire

**Sans télécharger la moindre donnée** - le pipeline complet tourne sur un échantillon
synthétique, et l'évaluateur y rend NO-GO (comportement attendu sur des données sans
signal) :

```powershell
.\.venv\Scripts\python.exe -m pytest -q                                  # tests anti-fuite
.\.venv\Scripts\python.exe scripts\lobster\make_synthetic_lobster.py     # échantillon bidon
.\.venv\Scripts\python.exe -m mirage.eval --config configs\phase0.yaml --raw-dir data\raw\synthetic
```

**Sur les vraies données** (voir [`data/README.md`](data/README.md)) :

```powershell
# Phase 0 (LOBSTER) - éval + horizon sweep + régime tick + significativité + GRU
.\.venv\Scripts\python.exe -m mirage.eval --config configs\phase0.yaml
.\.venv\Scripts\python.exe -m mirage.sweep --config configs\phase0.yaml
.\.venv\Scripts\python.exe scripts\lobster\tick_regime.py
.\.venv\Scripts\python.exe scripts\lobster\bootstrap_signif.py
.\.venv\Scripts\python.exe scripts\lobster\run_gru.py              # extra [torch]

# Phase 1a (world model d'état) et 1b (impact / action-conditioning)
.\.venv\Scripts\python.exe -m mirage.wm_eval --config configs\phase1.yaml
.\.venv\Scripts\python.exe scripts\lobster\impact_curves.py

# Phase 0 crypto (klines Binance, 1 an)
.\.venv\Scripts\python.exe scripts\crypto\download_binance.py --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --start 2024-01 --end 2024-12
.\.venv\Scripts\python.exe -m mirage.crypto_eval --config configs\phase0_crypto.yaml

# Phase 1 crypto (carnet Bybit L2 ; ~28 Mo de .pkl par jour-symbole, 44 jours x 5 symboles)
# La liste des dates fait partie du pre-enregistrement : configs\phase1_crypto_prereg.yaml
.\.venv\Scripts\python.exe scripts\crypto\fetch_bybit_batch.py `
    --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT --dates <les 44 dates du prereg>
.\.venv\Scripts\python.exe scripts\crypto\crypto_lob.py --out experiments
.\.venv\Scripts\python.exe scripts\crypto\bootstrap_signif.py    # IC95 par jour + verdicts
.\.venv\Scripts\python.exe scripts\crypto\spread_regime.py       # regime de cout par annee
```

## Structure

```
src/mirage/
  data/lobster.py       parsing des fichiers LOBSTER (message + orderbook)
  data/bars.py          agrégation en barres clock-time (le dernier état de la seconde)
  data/crypto.py        chargeur klines Binance
  data/bybit_lob.py     reconstructeur carnet Bybit L2 (snapshot + deltas -> barres 1 s)
  features.py           features causales LOBSTER (imbalance, OFI, rendements laggés)
  crypto_features.py    features causales OHLCV
  splits.py             walk-forward purgé + embargo
  baselines.py          zero-forecast / persistence / linéaire
  models/seq.py         MLP (défaut) + GRU (optionnel, extra torch)
  metrics.py            R²_OOS, RMSE, MAE, directional accuracy
  eval.py               boucle walk-forward Phase 0 + agrégation + Go/No-Go
  sweep.py              horizon sweep pré-enregistré + figures
  crypto_eval.py        Phase 0 crypto (walk-forward inter-périodes)
  state.py              vecteur d'état compact (Phase 1)
  wm.py                 world model d'état + rollout autorégressif
  wm_eval.py            Phase 1a : 1-step par dimension + rollout
  impact.py             overlay d'impact mécaniste (Phase 1b)
  backtest.py           règles de frontière de journée (multi-jours), testées
scripts/
  lobster/              bootstrap_signif, impact_curves, run_gru, tick_regime,
                        make_synthetic_lobster
  crypto/               download_binance, fetch_bybit_batch, crypto_lob
configs/                protocoles FIGÉS (phase0, phase0_crypto, phase1)
reports/                les 5 write-ups + figures
tests/                  tests anti-fuite, frontières de journée, impact
```

## Limites

- **Phase 0** : une seule journée (2012), 5 tickers → l'out-of-sample est intraday, sans
  prétention à généraliser dans le temps.
- **Phase 1 crypto** : 44 journées échelonnées (2023-01-20 → 2025-08-06, soit un peigne sur
  les 947 jours de l'archive Bybit, pas une série continue), 5 symboles, top-10 niveaux. Les
  17 premières journées restent en train : **2023 n'est jamais testé**, et l'out-of-sample ne
  couvre que 27 journées (2024-02-06 → 2025-08-06).
- La stratégie économique testée est **taker** et naïve (signe, sans sizing ni filtre).
  L'angle maker/post-only est **explicitement non revendiqué** : il hérite d'un risque de
  non-exécution non modélisé ici. Le trancher demande un simulateur à impact natif.
- Le développement post-trade de l'impact (Phase 1b) est **modélisé**, pas mesuré : il n'est
  pas testable sans contrefactuel.
- Zéro argent réel, zéro ordre réel : tout tourne sur données historiques.
