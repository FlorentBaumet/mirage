# Phase 1 (crypto) - microstructure du carnet Bybit L2

> **TL;DR.** Sur le carnet crypto (5 symboles × 44 journées échelonnées de janvier 2023 à
> août 2025, 19 M barres 1 s), le world model d'état trouve une prédictibilité réelle du
> prochain mouvement de prix : **R²_OOS poolé 0.052 (BTC), 0.050 (ETH), 0.037 (SOL), 0.021
> (XRP), 0.016 (DOGE)**, et l'intervalle de confiance à 95 % par bootstrap de journées
> **exclut 0 sur 5 symboles sur 5**. Le signal n'est pas un edge : une stratégie naïve
> « signe de la prédiction » tourne ~tous les 3 s, et **dès 2 bp de frais le net est négatif
> sur les 5 symboles, IC95 entièrement sous zéro**. Le mirage est structurel, pas une affaire
> d'hypothèse de frais : XRP est négatif **même à zéro frais**, son spread absorbant 100 % du
> gain brut (SOL l'était déjà sur le jeu précédent).
>
> **Ce rapport remplace le précédent** (3 symboles × 6 jours). Le passage de 6 à 44 journées et
> de 3 à 5 symboles **abaisse** le R²_OOS de ~25-30 % (il était de 0.072/0.061/0.066) mais lui
> ajoute ce qui manquait : un intervalle de confiance, une distribution par jour, et une
> couverture de régimes. Le résultat n'est pas détruit, il est consolidé.

---

## 1. Question

Ailleurs dans le projet, l'absence d'edge venait d'abord de l'absence de signal (prix ≈
random walk). Ici le signal existe : c'est le cas de test décisif - un signal de
microstructure réel survit-il aux coûts d'exécution, ou n'est-il qu'un mirage statistique ?

La réponse nette de coûts est non, et elle est chiffrée ci-dessous. Mais la première version de
ce résultat ne reposait que sur **6 journées et 3 symboles** : trop peu pour savoir si
R²_OOS ≈ +0.07 était un régime ou une coïncidence, et sans aucun intervalle de confiance. La
sonde du 2026-09-29 a montré que l'archive Bybit remonte en réalité à janvier 2023, pas à mai
2025 comme on le croyait : le jeu a donc été élargi, et **le protocole de décision figé avant
de regarder les nouveaux chiffres** (§3.4).

## 2. Données

Carnet L2 **Bybit** (gratuit, scriptable - `quote-saver.bycsi.com`, dumps `ob500`,
500 niveaux, snapshot + deltas ~10 ms), reconstruit en **barres 1 s top-10** au format
LOBSTER.

| | |
|---|---|
| Symboles | BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT |
| Journées | **44**, échelonnées du 2023-01-20 au 2025-08-06 |
| Volume construit | 3,80 M échantillons/symbole → **19,0 M au total** |
| **Fenêtre testée (OOS)** | **27 journées, 2024-02-06 → 2025-08-06**, 2,28 M points/symbole → **11,4 M au total** |

Deux précisions qui comptent pour lire la suite :

- **L'échantillonnage est en peigne, pas continu.** 44 journées sur les ~947 que contient
  l'archive (2023-01-18 → 2025-08-20), soit environ une journée tous les 23 jours. C'est un
  échantillon de régimes, pas une série temporelle continue : chaque journée est analysée
  pour elle-même, et aucune position ne traverse une nuit. Un backtest continu sur 2,6 ans
  n'est pas ce qui est fait ici.
- **2023 n'apparaît jamais en test.** Avec `min_train_frac = 0.4` en schéma *expanding*, le
  train initial consomme les 17 premières journées (les 16 de 2023 plus le 2024-01-14). Le
  test ne commence qu'en cours de journée le 2024-02-06. Le modèle est donc **entraîné** sur
  des journées 2023 qu'il ne sera jamais évalué dessus - détail repris en §6.

Pipeline : `scripts/crypto/fetch_bybit_batch.py` → `mirage/data/bybit_lob.py` →
`scripts/crypto/crypto_lob.py` → `scripts/crypto/bootstrap_signif.py`.

## 3. Protocole

### 3.1 Construction multi-jours

L'état et les fenêtres sont construits par journée puis concaténés : aucune fenêtre à cheval
sur deux jours, aucun rendement calculé par-dessus la nuit, aucun faux retournement de
position à minuit. Le walk-forward purgé devient inter-jours - train sur les jours passés,
test sur les jours futurs. Ces règles sont isolées et testées dans
[`mirage/backtest.py`](../src/mirage/backtest.py), avec des tests dédiés aux frontières de
jours.

### 3.2 Modèle et métrique

Modèle linéaire (`LinearWM`), lookback 16 s, cible = rendement du mid à +1 s. Walk-forward
purgé à 5 folds, embargo 16 (= max(lookback, horizon)), schéma *expanding*. Baseline = 0 pour
`ret` (random walk), no-change pour les niveaux. Métrique principale **pré-enregistrée** :
R²_OOS.

### 3.3 Modèle de coût

Position = signe de la prédiction linéaire OOS. Coût à chaque changement de position =
**demi-spread réellement mesuré à la barre de décision** + frais taker. Jeu de frais testé :
{0, 2, 5,5} bp. Le cas 0 bp est une borne optimiste (frais nuls irréalistes), jamais un edge.

### 3.4 Pré-enregistrement

Le fichier [`configs/phase1_crypto_prereg.yaml`](../configs/phase1_crypto_prereg.yaml) fixe,
**avant toute exécution sur le jeu élargi** : la liste des 44 dates, le modèle (inchangé par
rapport au jeu publié, pour rester comparable), la grille de frais, le dispositif de
bootstrap (unités, B, graine) et **les règles de décision**. Il a été commité et poussé avant
que le premier chiffre du jeu élargi soit calculé. Rien n'a été retouché après.

Les quatre règles, appliquées mécaniquement par le script :

| Règle | Énoncé | Verdict |
|---|---|---|
| `signal` | IC95 de R²_OOS(ret) exclut 0 pour ≥ 4 des 5 symboles | **SIGNAL RÉEL** (5/5) |
| `mirage` | IC95 de net à 2 bp entièrement < 0 pour chaque symbole | **MIRAGE CONFIRMÉ** |
| `exploitable` | borne basse de l'IC95 du net à 2 bp > 0 | **AUCUN SYMBOL EXPLOITABLE** (0/5) |
| `zero_frais` | 0 bp = borne optimiste, jamais un edge | rapporté, ne vaut pas verdict |

### 3.5 Bootstrap par journées

Le jeu précédent ne portait **aucun intervalle de confiance** : R²_OOS et net étaient des
moyennes ponctuelles. Les 2,28 M de points OOS par symbole ne sont pas 2,28 M d'observations
indépendantes - ce sont 27 journées de 86 400 barres fortement autocorrélées. L'unité de
rééchantillonnage est donc **la journée**, pas l'observation : block bootstrap circulaire,
B = 2000, graine 0, rééchantillonnage de journées entières avec remise. Une variante
**pré-enregistrée** en blocs contigus de 3 journées teste la sensibilité à l'autocorrélation
inter-jours (les régimes durent plusieurs jours, donc les journées ne sont pas échangeables).

Chaque statistique se réduit à des sommes par journée, donc le bootstrap est exact et
instantané malgré ~11 M d'observations. Il réutilise **les prédictions OOS exportées** par
`crypto_lob.py` (fichiers `.npz`), et non un second walk-forward : le verdict économique et
son intervalle ne peuvent pas diverger.

## 4. Résultats

### 4.1 - Une prédictibilité réelle

R²_OOS 1-step par dimension (moyenne sur 5 folds × 5 symboles ; baseline : 0 pour `ret`,
no-change pour les niveaux) :

| Dim | linear | mlp | mean |
|---|---|---|---|
| `ret` (prix) | **+0.0530** | +0.0543 | ~0 |
| `spread_rel` | 0.4180 | 0.3540 | −48.45 |
| `imb1` | 0.2657 | 0.3047 | +0.0427 |
| `depth_imb` | 0.1938 | 0.2186 | −1.0707 |
| `micro_dev` | 0.2907 | 0.3666 | +0.2192 |

Par symbole (`ret`) :

| Symbole | linear | mlp |
|---|---|---|
| BTCUSDT | **+0.0706** | +0.0777 |
| ETHUSDT | **+0.0585** | +0.0580 |
| SOLUSDT | **+0.0552** | +0.0540 |
| DOGEUSDT | **+0.0435** | +0.0365 |
| XRPUSDT | **+0.0372** | +0.0450 |

**Par fold** (R²_OOS `ret`, modèle linéaire) - la stabilité temporelle, fold par fold :

| Symbole | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | positifs |
|---|---|---|---|---|---|---|
| BTCUSDT | 0.0251 | 0.0842 | 0.0771 | 0.0825 | 0.0839 | 5/5 |
| ETHUSDT | 0.0239 | 0.0838 | 0.0714 | 0.0568 | 0.0566 | 5/5 |
| SOLUSDT | 0.0146 | 0.0744 | 0.0600 | 0.0582 | 0.0686 | 5/5 |
| DOGEUSDT | **−0.0084** | 0.0848 | 0.0278 | 0.0578 | 0.0556 | 4/5 |
| XRPUSDT | 0.0215 | 0.0675 | **−0.0139** | 0.0445 | 0.0666 | 4/5 |

**23 folds sur 25 sont positifs.** Le fold 0 est systématiquement le plus faible : c'est le
premier test, celui dont le train est le plus court (17 journées) et le plus éloigné en
régime. Les deux seuls folds négatifs (DOGE fold 0, XRP fold 2) sont des cas isolés, pas une
dérive.

### 4.2 - Le signal est-il significatif, jour par jour ?

Bootstrap par journées, B = 2000, graine 0 :

| Symbole | R²_OOS poolé | IC95 | p(R²≤0) | médiane/jour | IQR/jour | % jours > 0 |
|---|---|---|---|---|---|---|
| BTCUSDT | **0.0522** | [0.0349, 0.0793] | 0.0000 | 0.0804 | [0.0632, 0.0953] | **100 %** |
| ETHUSDT | **0.0496** | [0.0360, 0.0677] | 0.0000 | 0.0684 | [0.0538, 0.0834] | **100 %** |
| SOLUSDT | **0.0370** | [0.0211, 0.0636] | 0.0000 | 0.0656 | [0.0503, 0.0731] | **100 %** |
| XRPUSDT | **0.0212** | [0.0084, 0.0405] | 0.0010 | 0.0510 | [0.0314, 0.0715] | 85,2 % |
| DOGEUSDT | **0.0158** | [0.0018, 0.0471] | 0.0115 | 0.0478 | [0.0339, 0.0735] | 88,9 % |

Le signal est **positif sur 100 % des journées de test** pour BTC/ETH/SOL, et sur 85-89 % pour
XRP/DOGE : il n'est porté ni par quelques journées exceptionnelles, ni par un seul symbole.

**Robustesse, blocs contigus de 3 journées** (variante pré-enregistrée) :

| Symbole | IC95 R²_OOS |
|---|---|
| BTCUSDT | [0.0327, 0.0841] |
| ETHUSDT | [0.0340, 0.0700] |
| SOLUSDT | [0.0202, 0.0681] |
| XRPUSDT | [0.0042, 0.0442] |
| **DOGEUSDT** | **[−0.0005, 0.0569]** |

Sous la variante la plus conservatrice, **4 symboles sur 5 excluent 0** ; DOGE l'encadre. Le
seuil pré-enregistré (≥ 4/5) tient, mais de justesse pour DOGE, et c'est écrit ici plutôt que
masqué.

**Deux R²_OOS différents, et pourquoi.** Le R²_OOS « poolé » de ce tableau (0.0522 pour BTC)
n'est pas la moyenne des R² par fold du §4.1 (0.0706). Le premier agrège toutes les
prédictions OOS avant de calculer `1 − SSE/SSE_baseline` ; le second moyenne cinq ratios
calculés sur des folds de variances différentes. Les folds les plus volatils pèsent plus
lourd dans le pool et ont un R² plus faible. La métrique pré-enregistrée est le **poolé** ;
le découpage par fold sert de diagnostic de stabilité, pas de mesure de référence.

### 4.3 - Rollout : le signal décroît, il ne disparaît pas

R²_OOS du rendement cumulé par horizon (linéaire / MLP / persistence) :

| Horizon (s) | 1 | 2 | 3 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|---|
| linear | **0.0583** | 0.0514 | 0.0428 | 0.0284 | 0.0150 | 0.0089 | 0.0057 |
| mlp | 0.0558 | 0.0418 | 0.0354 | 0.0168 | **−0.0067** | −0.0235 | −0.0382 |
| persistence | −0.854 | −1.651 | −2.537 | −4.094 | −8.236 | −16.571 | −23.436 |

![rollout](figures/crypto_lob_rollout.png)

Décroissance lisse et monotone du linéaire jusqu'à 30 s : c'est la signature d'un signal
éphémère réel. Le MLP repasse négatif vers 10 s (il overfit et compose son erreur), là où le
linéaire tient. La persistence est franchement anti-prédictive : **le signal ne vient pas du
prix passé**, mais de l'état du carnet.

### 4.4 - Le coût dépend du symbole

Slippage du plus petit ordre testé (0,25 × meilleur niveau) et fill-rate au plus gros
(8 × L1), moyennés sur les journées :

| Symbole | slippage min | spread moyen | fill-rate à 8×L1 |
|---|---|---|---|
| BTCUSDT | **0.048 bp** | 0.027 bp | 0,3 % |
| ETHUSDT | 0.098 bp | 0.046 bp | 0,9 % |
| SOLUSDT | 0.422 bp | 0.608 bp | 80,8 % |
| DOGEUSDT | 0.662 bp | 0.932 bp | 95,9 % |
| XRPUSDT | **0.877 bp** | **1.458 bp** | 98,6 % |

![courbes de coût](figures/crypto_lob_cost.png)

BTC/ETH ont des spreads minuscules mais des carnets fins au-delà des 10 premiers niveaux : le
fill-rate s'effondre avec la taille. SOL/DOGE/XRP ont des carnets profonds mais des spreads
50 à 300 fois plus larges. **Il n'y a pas un marché crypto, il y a deux régimes de liquidité
opposés.**

### 4.5 - Le régime de coût change avec l'époque

Spread mesuré sur **toutes** les barres téléchargées (les 44 journées, pas seulement celles du
test), en points de base :

| Symbole | toutes | 2023 | 2024 | 2025 | bascule 2023→2025 |
|---|---|---|---|---|---|
| BTCUSDT | 0.0268 | 0.0480 (16 j) | 0.0179 (16 j) | 0.0105 (12 j) | **÷ 4,6** |
| ETHUSDT | 0.0458 | 0.0571 (16 j) | 0.0366 (16 j) | 0.0432 (12 j) | ÷ 1,3 |
| SOLUSDT | 0.6083 | 0.6587 (16 j) | 0.5332 (16 j) | 0.6413 (12 j) | ≈ stable |
| XRPUSDT | 1.4579 | 1.9916 (16 j) | 1.6950 (16 j) | 0.4303 (12 j) | **÷ 4,6** |
| DOGEUSDT | 0.9316 | 1.3691 (16 j) | 0.8130 (16 j) | 0.5063 (12 j) | ÷ 2,7 |

Sur les journées réellement testées, le spread effectivement facturé est plus étroit encore
(BTC 0.0142 bp, ETH 0.0393, SOL 0.6074, XRP 1.1120, DOGE 0.6468).

**Lecture.** La nature du mirage change selon l'époque. En 2023, BTC et XRP se négociaient avec
un spread 4,6 fois plus large qu'en 2025 : un signal identique y aurait coûté bien plus cher.
Comme l'évaluation ne porte que sur 2024-2025 (§2), elle se place dans le régime **le plus
favorable** de la période disponible - et le verdict reste négatif. Autrement dit, le mirage
est robuste au choix de régime : il n'est pas un artefact d'une période de spreads serrés.

### 4.6 - Le verdict économique, avec intervalles

Net par barre, en points de base ; IC95 par bootstrap de journées :

| Symbole | frais | gross/barre | turnover | **net/barre** | IC95 net |
|---|---|---|---|---|---|
| BTCUSDT | 0 bp | +0.164 | 0.287 | **+0.160** | [+0.141, +0.178] |
| BTCUSDT | 2 bp | +0.164 | 0.287 | **−0.987** | [−1.058, −0.916] |
| BTCUSDT | 5,5 bp | +0.164 | 0.287 | **−2.992** | [−3.218, −2.768] |
| ETHUSDT | 0 bp | +0.231 | 0.367 | **+0.216** | [+0.203, +0.231] |
| ETHUSDT | 2 bp | +0.231 | 0.367 | **−1.254** | [−1.313, −1.198] |
| ETHUSDT | 5,5 bp | +0.231 | 0.367 | **−3.825** | [−4.012, −3.653] |
| SOLUSDT | 0 bp | +0.271 | 0.387 | **+0.036** | [+0.002, +0.069] |
| SOLUSDT | 2 bp | +0.271 | 0.387 | **−1.514** | [−1.588, −1.443] |
| SOLUSDT | 5,5 bp | +0.271 | 0.387 | **−4.225** | [−4.413, −4.040] |
| **XRPUSDT** | **0 bp** | +0.222 | 0.358 | **−0.133** | **[−0.203, −0.063]** |
| XRPUSDT | 2 bp | +0.222 | 0.358 | **−1.566** | [−1.701, −1.447] |
| XRPUSDT | 5,5 bp | +0.222 | 0.358 | **−4.074** | [−4.451, −3.736] |
| DOGEUSDT | 0 bp | +0.271 | 0.384 | **+0.029** | [−0.006, +0.064] |
| DOGEUSDT | 2 bp | +0.271 | 0.384 | **−1.505** | [−1.611, −1.404] |
| DOGEUSDT | 5,5 bp | +0.271 | 0.384 | **−4.190** | [−4.459, −3.921] |

![gross vs net = le mirage](figures/crypto_lob_mirage.png)

*(figure : BTCUSDT, net cumulé à 0 / 2 / 5,5 bp)*

Le verdict est sans ambiguïté et ne dépend d'aucun choix de lecture :

- **À 2 bp**, le net est négatif sur les cinq symboles et l'IC95 est **entièrement sous zéro**
  pour chacun : la perte n'est pas une moyenne qui pourrait basculer. Règle `mirage` :
  **CONFIRMÉ**.
- **À 0 bp**, trois symboles sur cinq ont une borne basse strictement positive (BTC, ETH, SOL)
  et deux encadrent zéro (XRP est négatif, DOGE est à cheval sur 0). C'est la borne
  optimiste - frais nuls, irréalistes - et elle ne vaut pas edge.
- **XRP est le cas le plus net** : son net est négatif **même sans aucun frais**. Son
  demi-spread (0.556 bp sur les journées testées) × son turnover (0.358 × 2) consomme plus que
  son gain brut. C'est le même mécanisme qui tuait SOL sur le jeu précédent, appliqué à un
  second symbole.

> Même sans aucun frais, ce signal n'est pas exploitable dès que le spread est réaliste. Le
> mirage ne tient pas à un choix d'hypothèse de frais - il est structurel.

Note de lecture : les « PnL cumulés » (`crypto_lob_economic.csv`, ex. +3 639 % / −22 497 %)
sont la **somme arithmétique** des rendements par barre sur ~2,3 M barres de test, pas un
rendement de compte composé - au-delà de −100 % le compte serait liquidé bien avant. Le
chiffre à retenir est le net par barre ; le cumul n'illustre que l'ampleur.

### 4.7 - Stabilité du signal dans le temps

![stabilité](figures/crypto_lob_stability.png)

Les cinq symboles suivent la même forme : un premier fold plus faible, puis un plateau autour
de 0.05-0.08. Aucun fold ne touche zéro pour BTC/ETH/SOL. C'est un signal de microstructure
persistant, pas un artefact d'une journée.

## 5. Interprétation

Le signal est réel et robuste : positif sur 5 symboles × 5 folds inter-jours (23/25), IC95 qui
exclut 0 sur 5/5, positif sur 100 % des journées de test pour BTC/ETH/SOL, mécanisme documenté,
décroissance de rollout lisse. Ni bruit, ni fuite (cf. §5.1).

Il n'est pas un edge : le coût de l'exécuter (spread + frais, × un turnover élevé) dépasse le
gain sur les cinq symboles, et pour XRP il le dépasse déjà sans frais. **Significatif ≠
rentable.** Un backtest brut naïf aurait crié victoire ; l'éval honnête, pré-enregistrée et
nette de coûts, dit non.

Élargir les données a **renforcé la crédibilité des deux moitiés du résultat** tout en
**abaissant la valeur** du signal : 0.072 → 0.052 sur BTC. C'est exactement ce qu'un jeu plus
large doit faire - un chiffre plus petit et plus sûr vaut mieux qu'un chiffre plus flatteur et
non testé. Le verdict mirage, lui, n'a fait que gagner : il est passé de « SOL est tué par le
spread » à « XRP aussi », avec des intervalles à l'appui.

### 5.1 - Pourquoi c'est réel, pas une fuite

- Mécanisme documenté (micro-price / imbalance → prochain mid), pas une corrélation fortuite.
- Décroissance lisse du rollout : signature d'un signal éphémère réel ; une fuite gonflerait
  tous les horizons.
- Causalité vérifiée : les features à `t` (carnet à la clôture de la seconde `t`) prédisent
  le mouvement `t→t+1`. Tests anti-fuite du repo et tests dédiés aux frontières de jours.
- Le signal se retrouve **indépendamment sur cinq actifs** aux liquidités très différentes
  (spread de 0.027 à 1.458 bp), ce qu'une fuite propre à un symbole ne produirait pas.
- Le MLP overfit en rollout là où le linéaire tient, ce qui est cohérent avec un signal réel,
  faible et essentiellement linéaire.

### 5.2 - L'angle ouvert : maker / post-only

La stratégie testée est taker. En maker / post-only (frais nul voire rebate), l'arithmétique
change, mais le risque de non-exécution n'est pas modélisé ici et n'est donc pas revendiqué.
C'est le terrain où les backtests optimistes prospèrent : supposer des fills parfaits et
gratuits. Le trancher demande un simulateur à impact natif (ABIDES, Stage 2) ou des données
d'exécution réelles.

Le nouveau résultat borne ce qu'on peut en attendre : sur XRP, même un frais **nul** ne suffit
pas - le spread seul tue le signal. Le maker ne peut donc sauver au mieux que les symboles dont
le spread est déjà minuscule (BTC/ETH), précisément ceux où le fill-rate s'effondre avec la
taille.

## 6. Limites

- **2023 n'est jamais testé.** Les 17 premières journées (16 de 2023 + le 2024-01-14) tombent
  dans le train initial. L'OOS ne couvre que 2024-02-06 → 2025-08-06. Le modèle est entraîné
  sur un régime de spreads bien plus larges (§4.5) qu'il n'est évalué - décalage de
  distribution non mesuré ici.
- **27 journées effectives, pas 2,3 M d'observations.** Le bootstrap traite correctement la
  dépendance intra-jour, mais la taille d'échantillon réelle pour l'inférence est de 27
  journées. C'est ce qui rend l'IC de DOGE si large.
- **DOGE encadre 0 sous la variante à blocs de 3 journées.** Le seuil pré-enregistré tient
  (4/5), mais le signal de DOGE est le plus fragile des cinq.
- **Échantillonnage en peigne** (44 journées sur 947) : couverture de régimes, pas série
  continue. Aucun résultat sur la persistance du signal sur plusieurs jours consécutifs.
- Stratégie signe naïve : pas de sizing, pas de filtre de confiance, pas de maker.
- Frais = hypothèses (2 / 5,5 bp) ; pas de modèle de file d'attente, pas de latence, pas de
  coût de financement (perps).
- Top-10 niveaux (les carnets BTC/ETH ont beaucoup de profondeur au-delà).
- Le R²_OOS est une métrique d'erreur quadratique : un prédicteur directionnellement
  informatif mais bruité y score mal. Les valeurs ici sont des planchers de ce qu'un modèle
  mieux calibré pourrait extraire.

## 7. Reproduire

```powershell
# 1. Télécharger et reconstruire le carnet (44 journées x 5 symboles, ~28 Go de .pkl)
.\.venv\Scripts\python.exe scripts\crypto\fetch_bybit_batch.py `
    --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT `
    --dates 2023-01-20 2023-02-11 ... 2025-08-06     # liste complète dans le pré-enregistrement

# 2. Évaluation (walk-forward purgé, coûts, export des prédictions OOS)
.\.venv\Scripts\python.exe scripts\crypto\crypto_lob.py --out experiments

# 3. Intervalle de confiance par bootstrap de journées + verdicts pré-enregistrés
.\.venv\Scripts\python.exe scripts\crypto\bootstrap_signif.py

# 4. Régime de coût (spread par symbole et par année)
.\.venv\Scripts\python.exe scripts\crypto\spread_regime.py

# 5. Règles multi-jours et frontières de jours
.\.venv\Scripts\python.exe -m pytest -q
```

La liste exacte des 44 dates est dans
[`configs/phase1_crypto_prereg.yaml`](../configs/phase1_crypto_prereg.yaml) - elle fait partie
du pré-enregistrement, et non du script de téléchargement, précisément pour qu'aucune date ne
puisse être choisie après avoir vu les résultats.

Sorties : `experiments/crypto_lob_run.txt` (log complet), `crypto_lob_bootstrap.csv`,
`crypto_lob_economic.csv`, `crypto_lob_1step.csv`, `crypto_lob_rollout.csv`,
`crypto_lob_spread_regime.csv`, et les `.npz` de prédictions OOS.
