# Phase 1e - la profondeur du carnet au-delà du meilleur niveau

**Verdict : APPORT PARTIEL, non promu en découverte - `2/5` symboles (modèle linéaire,
bras primaire).** Deux dimensions qui décrivent le carnet au-delà du meilleur niveau
améliorent significativement la prévision sur BTC et ETH, et **dégradent
significativement** sur XRP et DOGE. Le MLP ne confirme rien (`0/5`). Ce n'est donc pas une
découverte, et ce n'est pas non plus un zéro : c'est un signal réel sur une partie du
panel, avec un mécanisme identifié, et une réserve pré-enregistrée qui a été **à moitié
réfutée par les données**.

Protocole figé et commité avant toute exécution : `configs/phase1e_crypto_prereg.yaml`.

## 1. La question

La limite n°1 du rapport de phase 1c était que l'état ne regarde que le **meilleur niveau**
du carnet (`imb1`, `depth_imb`, `micro_dev` portent tous sur le best bid/ask ou le total).
Toute l'information en attente aux niveaux 2 à 10 était ignorée. Cette phase ajoute deux
dimensions qui la captent, et demande si elles apportent quelque chose au-delà de l'état
existant.

Le flux brut niveau-par-niveau n'est pas archivé (seules les barres top-10 le sont, cf.
`reserve_honnete` du pré-enregistrement) : ces deux dimensions sont donc **reconstruites
depuis les agrégats** déjà disponibles, pas depuis les niveaux individuels. Elles sont une
approximation de l'information de profondeur, pas l'information de profondeur elle-même.

## 2. Définitions exactes

Conformes au pré-enregistrement, sans ajustement a posteriori :

```
imb_deep   = (Σ_{i=2..10} bid_size_i − Σ_{i=2..10} ask_size_i)
           / (Σ_{i=2..10} bid_size_i + Σ_{i=2..10} ask_size_i + eps)

slope_asym = bid_share − ask_share
             bid_share = bid_size_1 / (Σ_{i=1..10} bid_size_i + eps)
             ask_share = ask_size_1 / (Σ_{i=1..10} ask_size_i + eps)
```

`imb_deep` est le déséquilibre des **niveaux 2 à 10**, c'est-à-dire hors du meilleur
niveau. `slope_asym` mesure si le premier niveau pèse plus lourd que la moyenne sur un côté
que sur l'autre - une asymétrie de « pente » du carnet.

Ces deux dimensions ont été ajoutées d'un bloc (`imb_deep`, `slope_asym`), comme le
pré-enregistrement le fixe : le bras `deep` teste la paire, pas chaque dimension isolément.

## 3. Corrélation des dimensions ajoutées avec l'état existant

Exigé par `doit_contenir`. Pearson, sur les 44 journées concaténées par symbole
(≈ 3,8 M lignes/symbole, aucune ligne non finie). Colonnes : `imb_deep` / `slope_asym`.

| Symbole | ret | spread_rel | imb1 | depth_imb | micro_dev | ofi |
|---|---|---|---|---|---|---|
| BTCUSDT | 0.192 / 0.084 | −0.009 / 0.006 | 0.544 / 0.703 | **0.733** / 0.503 | 0.254 / 0.343 | 0.257 / 0.247 |
| ETHUSDT | 0.188 / 0.070 | −0.001 / −0.002 | 0.497 / 0.754 | **0.791** / 0.469 | 0.327 / 0.505 | 0.277 / 0.228 |
| SOLUSDT | 0.099 / 0.087 | −0.009 / 0.012 | 0.311 / 0.685 | **0.952** / 0.231 | 0.190 / 0.445 | 0.111 / 0.217 |
| XRPUSDT | 0.101 / 0.104 | 0.046 / −0.007 | 0.223 / **0.834** | **0.977** / 0.195 | 0.098 / **0.752** | 0.056 / 0.270 |
| DOGEUSDT | 0.040 / 0.075 | −0.038 / 0.007 | 0.244 / 0.715 | **0.982** / 0.107 | 0.177 / **0.662** | 0.073 / 0.247 |

Corrélation entre les deux dimensions ajoutées : BTC −0.085, ETH −0.087, SOL −0.051,
XRP −0.014, DOGE −0.070 - **mutuellement quasi orthogonales**.

### La réserve pré-enregistrée, confrontée aux données

Le pré-enregistrement avait écrit, **avant** toute mesure, une réserve en deux points.
Voici ce que les données en font :

| Réserve pré-enregistrée | Confrontation |
|---|---|
| « `imb_deep` attendu quasi-redondant avec (`imb1`, `depth_imb`) » | **CONFIRMÉ** - mais la redondance est **avec `depth_imb` seul**, pas avec `imb1` : r = 0.73/0.79 sur BTC/ETH et **0.95/0.98/0.98** sur SOL/XRP/DOGE. |
| « `slope_asym` est le seul axe orthogonal à l'état existant » | **RÉFUTÉ.** `slope_asym` est fortement corrélé à **`imb1`** (0.685–0.834), et sur XRP/DOGE aussi à `micro_dev` (0.752/0.662). Il ne porte pas l'axe orthogonal annoncé. |

C'est le second point qui compte : la prévision centrale de la phase 1e - que l'axe neuf
serait la pente du carnet - est **démentie par les données**. Le seul axe véritablement
orthogonal observé est celui qui sépare les deux dimensions ajoutées l'une de l'autre, pas
celui qui les sépare de l'état existant.

### Variance brute - le mécanisme du résultat

| Symbole | var(`imb_deep`) | var(`slope_asym`) |
|---|---|---|
| BTCUSDT | 0.357 | 0.141 |
| ETHUSDT | 0.289 | 0.150 |
| SOLUSDT | 0.062 | 0.013 |
| XRPUSDT | 0.028 | 0.004 |
| DOGEUSDT | 0.045 | 0.005 |

Les deux dimensions sont des ratios bornés dans [−1, 1]. Sur les alts, elles sont **presque
dégénérées** : variance 0.004–0.062 contre 0.14–0.36 sur BTC/ETH. Une dimension quasi
constante ne peut rien apporter, et offre en revanche un axe d'ajustement parasitaire au
modèle. C'est exactement le motif des résultats ci-dessous : les gains significatifs sont
là où la variance est réelle (BTC, ETH), les pertes là où elle est quasi nulle (XRP, DOGE).

## 4. Contrôle d'intégrité

Le bras `base`+`linear` de cette phase reproduit `experiments/crypto_lob_oos_DOGEUSDT.npz`
avec `pred` **bit à bit identique** (`max|diff| = 0.000e+00`), `R2_OOS(ret) = 0.01584620`
exact. Les ajouts de dimensions ne touchent pas le bras de référence.

## 5. ΔR²_OOS(ret) apparié

Bootstrap apparié, B = 2000, graine 0, unité = la journée (27 journées OOS), même tirage
pour les deux bras de chaque comparaison (condition d'appariement, cf. phase 1d).

**Bras primaire - modèle linéaire** (`deep − base`), celui sur lequel porte la règle :

| Symbole | ΔR² | IC95 | |
|---|---|---|---|
| BTCUSDT | **+0.00100** | [0.00056, 0.00165] | significatif |
| ETHUSDT | **+0.00039** | [0.00022, 0.00062] | significatif |
| SOLUSDT | +0.00001 | [−0.00017, 0.00033] | - |
| XRPUSDT | **−0.01342** | [−0.03277, −0.00307] | significativement négatif |
| DOGEUSDT | **−0.00462** | [−0.00647, −0.00281] | significativement négatif |

**→ 2/5. APPORT PARTIEL.**

**Bras secondaire - MLP** (`deep − base`) :

| Symbole | ΔR² | IC95 | |
|---|---|---|---|
| BTCUSDT | **−0.00619** | [−0.01083, −0.00027] | significativement négatif |
| ETHUSDT | −0.01208 | [−0.03081, 0.00178] | - |
| SOLUSDT | **−0.00332** | [−0.00508, −0.00032] | significativement négatif |
| XRPUSDT | −0.00031 | [−0.00969, 0.01368] | - |
| DOGEUSDT | +0.00620 | [−0.00387, 0.01178] | - |

**→ 0/5. APPORT NON ÉTABLI.**

**Le signe s'inverse entre les deux classes de modèles** - c'est le fait le plus important de
cette section. Sur BTC, les dimensions de profondeur **aident** le linéaire (+0.001) et
**nuisent** au MLP (−0.006). Un résultat qui dépend de la classe du modèle dans ce sens
n'est pas un apport robuste : c'est un signal fragile, sensible au modèle.

Robustesse (blocs contigus de 3 jours) : les intervalles s'élargissent, les conclusions ne
changent pas. `deep_linear − base_linear` reste significativement positif sur BTC
[0.00061, 0.00167] et ETH [0.00024, 0.00057], significativement négatif sur XRP
[−0.03358, −0.00211] et DOGE [−0.00691, −0.00183], nul sur SOL. Le 2/5 tient sous
l'échantillonnage par blocs.

## 6. Effet économique - Δnet_bp entre bras

Différence de net par barre entre le bras `deep` et le bras `base`, au même modèle.
Trois niveaux de frais, tiretés `0 bp | 2 bp | 5.5 bp` (les trois valeurs pré-enregistrées).

| Symbole | `deep_linear − base_linear` | `deep_mlp − base_mlp` |
|---|---|---|
| BTCUSDT | −0.0021 / −0.0432 / −0.1152 | −0.0021 / −0.0346 / −0.0916 |
| ETHUSDT | −0.0008 / −0.0204 / −0.0548 | +0.0014 / −0.0509 / −0.1424 |
| SOLUSDT | +0.0017 / +0.0084 / +0.0201 | −0.0017 / −0.0143 / −0.0365 |
| XRPUSDT | −0.0018 / −0.0055 / −0.0119 | **+0.0164 / +0.0667 / +0.1549** |
| DOGEUSDT | −0.0038 / −0.0237 / −0.0587 | +0.0030 / +0.0251 / +0.0636 |

Deux lectures, toutes deux contre-intuitives au vu du ΔR² :

- **SOL, `deep_linear`** : ΔR² nul (+0.00001) mais Δnet **positif et significatif aux trois
  niveaux de frais** ([0.003, 0.035] à 5.5 bp). Le modèle linéaire enrichi prédit la
  *direction* un peu mieux sans mieux expliquer la *variance* - deux critères distincts, et
  seul le premier a un intérêt économique.
- **XRP, `deep_mlp`** : ΔR² nul (−0.00031) mais Δnet **fortement positif et significatif**
  (+0.0164 / +0.0667 / +0.1549). Le gain **croît avec les frais**, ce qui signifie que le
  bras enrichi **traite beaucoup moins** que le bras de base : l'avantage est un
  *évitement de coût*, pas une meilleure prévision du prix.

Ces deux effets sont réels mais **ils ne sauvent pas le bras** : en valeur absolue, le net
du bras `deep` reste négatif partout (section suivante). Un Δnet positif entre deux bras
qui perdent tous les deux de l'argent ne constitue pas une découverte économique.

## 7. Net absolu à 2 bp - le mirage ne se dissout pas

| Symbole | deep linear | IC95 | deep MLP | IC95 |
|---|---|---|---|---|
| BTCUSDT | −1.0297 | [−1.096, −0.963] | −0.9308 | [−1.006, −0.859] |
| ETHUSDT | −1.2739 | [−1.331, −1.221] | −1.1994 | [−1.259, −1.143] |
| SOLUSDT | −1.5050 | [−1.579, −1.436] | −1.4744 | [−1.538, −1.410] |
| XRPUSDT | −1.5711 | [−1.698, −1.461] | −1.3910 | [−1.485, −1.305] |
| DOGEUSDT | −1.5287 | [−1.645, −1.420] | −1.3648 | [−1.442, −1.289] |

*(bp par barre)*

**L'IC95 de chaque bras est entièrement sous zéro, sur les cinq symboles, aux deux
modèles.** Enrichir l'état de la profondeur du carnet ne fait pas franchir zéro.

## 8. Verdicts - appliqués mécaniquement

| Règle (pré-enregistrée) | Résultat | Verdict |
|---|---|---|
| `apport_profondeur` (bras primaire, linéaire) | **2/5** | **APPORT PARTIEL - non promu en découverte** |
| `apport_profondeur` (bras secondaire, MLP) | **0/5** | APPORT NON ÉTABLI |
| `mirage` (bras deep, linéaire) | net < 0 partout à 2 bp | **MIRAGE CONFIRMÉ** |
| `bascule` (bras deep, linéaire) | aucune borne basse > 0 à 2 bp | **AUCUNE BASCULE** |

« Non promu en découverte » est une règle pré-enregistrée, pas une prudence ajoutée après
coup : le pré-enregistrement ne reconnaît une découverte qu'à partir de 4/5. Un signal qui
apparaît sur 2 symboles sur 5, qui **change de signe** entre classes de modèles, et dont la
dimension est quasi dégénérée sur les 3 symboles où elle nuit, n'est pas une découverte.

## 9. Limites

- **Les dimensions sont reconstruites depuis des agrégats**, pas depuis les niveaux
  individuels : l'archive Bybit ne conserve pas le carnet niveau par niveau. `imb_deep` et
  `slope_asym` approximent l'information de profondeur, elles ne la constituent pas. Un
  résultat négatif ici ne condamne pas le carnet profond en général.
- **Les deux dimensions sont testées comme un bloc**, conformément au pré-enregistrement.
  Ce test ne dit pas laquelle des deux porte le signal sur BTC/ETH ni la dégradation sur
  XRP/DOGE.
- **Le signal dépend de la classe de modèle** (signe opposé sur BTC entre linéaire et MLP),
  ce qui est en soi un argument contre sa robustesse.
- **La variance des dimensions est très inégale** entre symboles (0.004 à 0.36) : le test
  n'a pas la même puissance sur BTC/ETH que sur SOL/XRP/DOGE, et une partie du résultat est
  probablement un artefact de cette disparité.
- **27 journées OOS** (2024-02-06 → 2025-08-06), 5 symboles, échantillonnage en peigne. IC
  larges ; couverture de régimes bornée par construction.
- **Cible `ret` uniquement.** La profondeur du carnet pourrait informer le spread ou la
  liquidité future sans informer le rendement. Non testé.

## 10. Reproduire

```powershell
# Les 5 symboles, un seul passage ; le bras base+linear sert de controle d'integrite
.\.venv\Scripts\python.exe scripts\crypto\arm_eval.py --out experiments_arms
# Bootstrap apparie : delta R2, delta net, IC95, verdicts mecaniques des deux preregistrements
.\.venv\Scripts\python.exe scripts\crypto\paired_arms.py
# Correlations des dimensions ajoutees (section 3) : voir experiments_arms/deep_corr.csv
```

Sorties : `experiments_arms/arms_1step.csv`, `paired_arms.csv`,
`paired_arms_net_by_arm.csv`, `paired_arms_log.txt`, `deep_corr.csv`.
Tous les chiffres de ce rapport proviennent de ces fichiers, sans sélection.
