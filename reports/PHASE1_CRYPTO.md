# Phase 1 (crypto) - microstructure du carnet Bybit L2

> **TL;DR.** Sur le carnet crypto (3 symboles × 6 jours, mai→août 2025, 1,55 M barres
> 1 s), le world model d'état trouve une prédictibilité réelle du prochain mouvement de
> prix : R²_OOS ≈ +0.067 (BTC 0.072 / ETH 0.061 / SOL 0.066), positive sur 5 folds
> walk-forward inter-jours sur 5, et sur 3 symboles sur 3. Le signal n'est pas pour autant
> un edge : une stratégie naïve « signe de la prédiction » tourne ~tous les 5 s, et dès
> 2 bp de frais elle perd lourdement sur les 3 symboles. SOL est négatif à zéro frais :
> son spread (0.6 bp, ~60× celui de BTC) absorbe 100 % du gain brut. Le mirage est donc
> démontré deux fois, par les frais et par le spread seul.

---

## 1. Question

Ailleurs dans le projet, l'absence d'edge venait d'abord de l'absence de signal (prix ≈
random walk). Ici le signal existe : c'est le cas de test décisif - un signal de
microstructure réel survit-il aux coûts d'exécution, ou n'est-il qu'un mirage statistique ?
La réponse nette de coûts est non, et elle est chiffrée ci-dessous.

## 2. Données

Carnet L2 **Bybit** (gratuit, scriptable - `quote-saver.bycsi.com`, dumps `ob500`,
500 niveaux, snapshot + deltas ~10 ms), reconstruit en **barres 1 s top-10** au format
LOBSTER.

| | |
|---|---|
| Symboles | BTCUSDT, ETHUSDT, SOLUSDT |
| Dates | 2025-05-08, 05-22, 06-02, 06-18, 07-09, 08-06 |
| Volume | ~86 400 barres/jour → **518 k échantillons/symbole**, **1,55 M au total** |
| Régimes | BTC 100 k→114 k ; ETH 1 983→3 629 $ (gros rally) ; SOL 147→178 |

Pipeline : `scripts/crypto/fetch_bybit_batch.py` → `mirage/data/bybit_lob.py` →
`scripts/crypto/crypto_lob.py`.

## 3. Protocole

**Construction multi-jours.** L'état et les fenêtres sont construits par journée puis
concaténés : aucune fenêtre à cheval sur deux jours, aucun rendement calculé par-dessus la
nuit, aucun faux retournement de position à minuit. Le walk-forward purgé devient
inter-jours - train sur les jours passés, test sur les jours futurs. Ces règles sont isolées
et testées dans [`mirage/backtest.py`](../src/mirage/backtest.py), avec des tests dédiés aux
frontières de jours.

**Modèle de coût.** Position = signe de la prédiction linéaire OOS. Coût à chaque changement
de position = demi-spread réel mesuré + frais taker. Jeu de frais testé : {0, 2, 5.5} bp.

## 4. Résultats

### 4.1 - Une prédictibilité réelle et robuste

R²_OOS 1-step par dimension (baseline : 0 pour `ret`, no-change pour les niveaux) :

| Dim | linear | mlp | mean |
|---|---|---|---|
| `ret` (prix) | **+0.067** | +0.070 | ~0 |
| `spread_rel` | 0.481 | 0.493 | −0.150 |
| `imb1` | 0.233 | 0.259 | −0.340 |
| `depth_imb` | 0.184 | 0.203 | −1.248 |
| `micro_dev` | 0.383 | 0.391 | 0.241 |

Par symbole (`ret`, linéaire) : BTC **+0.072**, ETH **+0.061**, SOL **+0.066**.

![stabilité](figures/crypto_lob_stability.png)

Le signal est positif sur les 5 folds × 3 symboles (0.02–0.10), sans jamais toucher zéro :
c'est une prédictibilité réelle et persistante, pas un artefact d'une journée.

Rollout du rendement cumulé : positif jusqu'à 30 s pour le linéaire (0.071 → 0.012) ; le MLP
repasse négatif vers 10–20 s (il overfit et compose l'erreur).

![rollout](figures/crypto_lob_rollout.png)

Mécanisme : la micro-price et l'imbalance des files prédisent le prochain micro-mouvement du
mid - effet documenté, fort sur les perps crypto. Le signal ne vient pas du prix passé : la
persistence est franchement anti-prédictive (−0.81).

### 4.2 - Le coût dépend du symbole

Slippage du plus petit ordre testé (0.25 × meilleur niveau), moyenné sur les 6 jours :

| Symbole | slippage min | spread moyen | fill-rate à 8×L1 |
|---|---|---|---|
| BTCUSDT | **0.010 bp** | ~0.009 bp | 0.2 % |
| ETHUSDT | **0.046 bp** | ~0.04 bp | 0.7 % |
| SOLUSDT | **0.400 bp** | **~0.6 bp** | 100 % |

![courbes de coût](figures/crypto_lob_cost.png)

BTC/ETH ont des spreads minuscules mais des carnets fins au-delà des 10 premiers niveaux :
le fill-rate s'effondre avec la taille. SOL a un carnet profond mais un spread ~60× plus
large.

### 4.3 - Le verdict économique

| Symbole | frais | gross/barre | turnover | **net/barre** |
|---|---|---|---|---|
| BTCUSDT | 0 bp | +0.107 bp | 0.216 | **+0.105 bp** |
| BTCUSDT | 2 bp | +0.107 bp | 0.216 | **−0.760 bp** |
| BTCUSDT | 5.5 bp | +0.107 bp | 0.216 | **−2.275 bp** |
| ETHUSDT | 0 bp | +0.213 bp | 0.349 | **+0.200 bp** |
| ETHUSDT | 2 bp | +0.213 bp | 0.349 | **−1.196 bp** |
| ETHUSDT | 5.5 bp | +0.213 bp | 0.349 | **−3.640 bp** |
| **SOLUSDT** | **0 bp** | +0.228 bp | 0.352 | **−0.0004 bp** |
| SOLUSDT | 2 bp | +0.228 bp | 0.352 | **−1.409 bp** |
| SOLUSDT | 5.5 bp | +0.228 bp | 0.352 | **−3.874 bp** |

![gross vs net = le mirage](figures/crypto_lob_mirage.png)

Le cas SOL est le plus net. SOL porte le gross le plus élevé des trois (+0.228 bp/barre,
signal aussi fort qu'ETH) et, à zéro frais, son net est nul (−0.0004 bp) : son demi-spread
(~0.3 bp) × son turnover (0.35) consomme exactement le gain brut. Autrement dit :

> Même sans aucun frais, ce signal n'est pas exploitable dès que le spread est réaliste. Le
> mirage ne tient pas à un choix d'hypothèse de frais - il est structurel.

Note de lecture : les « PnL cumulés » (ex. +327 % / −7 073 %) sont la somme arithmétique des
rendements par barre sur ~310 k barres de test, pas un rendement de compte composé - au-delà
de −100 % le compte serait liquidé bien avant. Le chiffre à retenir est le net par barre ; le
cumul n'illustre que l'ampleur.

## 5. Interprétation

Le signal est réel et robuste : positif sur 3 symboles × 5 folds inter-jours, mécanisme
documenté, décroissance de rollout lisse - ni bruit, ni fuite (cf. §5.1).

Il n'est pas un edge : le coût de l'exécuter (spread + frais, × un turnover élevé) dépasse le
gain sur les 3 symboles. Significatif ≠ rentable. Un backtest brut naïf aurait crié victoire ;
l'éval honnête, pré-enregistrée et nette de coûts, dit non.

Élargir les données a renforcé les deux moitiés du résultat : le signal est plus crédible
(6 jours, 3 coins) et le verdict mirage plus solide (SOL le tue sans frais).

### 5.1 - Pourquoi c'est réel, pas une fuite

- Mécanisme documenté (micro-price / imbalance → prochain mid), pas une corrélation fortuite.
- Décroissance lisse du rollout : signature d'un signal éphémère réel ; une fuite gonflerait
  tous les horizons.
- Causalité vérifiée : les features à `t` (carnet à la clôture de la seconde `t`) prédisent
  le mouvement `t→t+1`. Tests anti-fuite du repo et tests dédiés aux frontières de jours.
- Le MLP overfit en rollout là où le linéaire tient, ce qui est cohérent avec un signal réel,
  faible et essentiellement linéaire.

### 5.2 - L'angle ouvert : maker / post-only

La stratégie testée est taker. En maker / post-only (frais nul voire rebate), l'arithmétique
change, mais le risque de non-exécution n'est pas modélisé ici et n'est donc pas revendiqué.
C'est le terrain où les backtests optimistes prospèrent : supposer des fills parfaits et
gratuits. Le trancher demande un simulateur à impact natif (ABIDES, Stage 2) ou des données
d'exécution réelles. Sur SOL, même un frais nul ne suffit pas : le spread seul tue le signal.

## 6. Limites

- **6 jours** sur mai→août 2025 (fenêtre réelle de l'archive Bybit), **3 symboles**.
- Stratégie signe naïve : pas de sizing, pas de filtre de confiance, pas de maker.
- Frais = hypothèses (2 / 5.5 bp) ; pas de modèle de file d'attente ni de latence.
- Top-10 niveaux (les carnets BTC/ETH ont beaucoup de profondeur au-delà).

## 7. Reproduire

```powershell
.\.venv\Scripts\python.exe scripts\crypto\fetch_bybit_batch.py --symbols BTCUSDT ETHUSDT SOLUSDT `
    --dates 2025-05-08 2025-05-22 2025-06-02 2025-06-18 2025-07-09 2025-08-06
.\.venv\Scripts\python.exe scripts\crypto\crypto_lob.py
.\.venv\Scripts\python.exe -m pytest -q     # règles multi-jours et frontières de jours
```
