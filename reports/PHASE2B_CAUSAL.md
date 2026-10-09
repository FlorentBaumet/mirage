# Phase 2b (crypto) - le planificateur causal, ou comment le seul résultat positif du projet s'est dissous

> **TL;DR.** Le résultat positif de la Phase 2 était un **artefact de fuite**. Le
> planificateur publié lisait, à la barre `t`, les prédictions `rhat[t+1..t+H-1]`, faites
> **plus tard** que `t` : `rhat[t+1]` est calculée sur une fenêtre qui contient le rendement
> que la position `p_t` encaisse. Re-mesuré avec un planificateur **causal**, pré-enregistré
> avant tout calcul, sur les **mêmes 44 journées, les mêmes folds et les mêmes frais** :
>
> - `edge_reel_causal` : **NON, 0/5**. Le net réel de l'agent honnête à 2 bp vaut de
>   **-0.0056 à +0.0003** bp/barre, IC95 jamais au-dessus de 0 (entièrement négatif sur
>   DOGE, SOL et XRP ; contenant 0 sur BTC et ETH).
> - `fuite_significative` : **OUI, 5/5**. L'écart apparié par journée entre le bras publié et
>   l'agent causal vaut **+0.021 à +0.096** bp, IC95 > 0 partout. La fuite portait le net
>   publié, et son ampleur est du même ordre que lui.
> - Ce qui tombait avec elle : le **contrôle du bruit**, où l'agent publié gagnait 5/5, ne
>   tient plus que sur **2/5** ; l'**écart d'exploitation** négatif et significatif de la
>   Phase 2 **disparaît** (désormais IC95 contenant 0 ou > 0 sur 5/5).
> - Ce qui survit : l'agent honnête bat toujours le myope sur **5/5**, de +0.99 à +1.56
>   bp/barre, mais **entièrement en coût évité**. Son brut directionnel est *inférieur* à
>   celui du myope sur 5/5 : il ne lit pas mieux le marché, il s'abstient de le payer
>   (turnover de 1e-4 à 3e-3 contre 0.56 à 0.77 pour le myope).
>
> Le mirage tient donc au niveau de l'agent aussi : le seul résultat positif du projet est
> celui qui n'a pas survécu à son propre contrôle d'honnêteté. C'est, pour ce projet, le
> résultat le plus utile.

---

## 1. Question

La Phase 2 avait produit le premier résultat **positif** du projet : un planificateur exact
qui maximise la récompense imaginée par le world model appris, en payant le spread dans son
plan, dégageait un net *réel* positif à 2 bp sur 5/5 symboles. Le projet, dont la colonne
vertébrale est l'évaluation honnête, devait donc attaquer son propre résultat plus durement
que les autres.

Un contrôle de lecture a montré que ce résultat était porté par une **fuite d'information** :
le planificateur consommait à `t` des prévisions faites après `t` (§3.1). La question de
cette phase est simple : **que reste-t-il du résultat quand on interdit cette fuite ?**

Elle est posée de façon **pré-enregistrée** : le planificateur causal, les métriques, les
seuils et la liste des journées d'extension sont figés dans
`configs/phase2b_crypto_prereg.yaml` **avant** le premier calcul, et ce fichier impose la
lecture à en faire. Sa règle `echecs_et_lectures` prévoyait trois cas ; les données en ont
sélectionné un, et il était écrit d'avance :

> `fuite_significative` OUI et `edge_reel_causal` NON : le résultat publié était un artefact
> de fuite. C'est la conclusion à écrire en tête du rapport.

## 2. Données

Exactement le jeu de la Phase 1c et de la Phase 2, inchangé : carnet L2 **Bybit**, 5 symboles
(BTC, ETH, SOL, XRP, DOGE en USDT perpétuel linéaire), **44 journées** échelonnées du
2023-01-18 au 2025-08-20, reconstruites en **barres 1 s top-10**. Les dates, le format et le
protocole de reconstruction sont ceux de `configs/phase1c_crypto_prereg.yaml` ; le harnais
les recharge via `load_cached`, qui **filtre sur ces 44 dates** et lève si l'une manque. Les
44 journées supplémentaires de l'étape 2b-5 vivent dans le même dossier mais sont hors de
cet échantillon tant que le filtre n'est pas levé explicitement.

Après découpage walk-forward (5 folds, entraînement expansif, `min_train_frac = 0.4`) et
purge, l'échantillon hors échantillon compte **2 280 200 à 2 280 495 barres par symbole**,
soit ≈ 11,4 M de barres au total, réparties sur 27 journées de test.

## 3. Protocole

### 3.1 La fuite, exactement

`make_supervised(S, lookback)` aligne `X[k] = états k..k+L-1` et `Y[k] = état k+L`. Une
prédiction `rhat[j]` est donc calculée sur une fenêtre qui **contient** `Y[j-1]`, le rendement
de la barre `j-1`.

Le planificateur publié (`plan_by_run`) prend la série `rhat` et, à chaque barre `t`, choisit
la position qui maximise la récompense imaginée sur les `H` pas suivants, en consommant
`rhat[t+1..t+H-1]`. Or `rhat[t+1]` est calculée sur une fenêtre qui contient `Y[t]`,
c'est-à-dire **le rendement que la position `p_t` encaisse**. Le plan de `t` sait donc, en
partie, ce que `t` va rapporter.

La taille de l'effet est mesurable en lecture seule sur l'artefact publié, à 2 bp et H = 10 :

| symbole | corr(rhat[t+1], y[t]) | corr(rhat[t], y[t]) |
|---|---|---|
| BTCUSDT | **+0.3951** | +0.2285 |
| DOGEUSDT | **+0.2680** | +0.1455 |

L'écart entre les deux colonnes est le rendement de la fuite.

### 3.2 Le planificateur causal

Un planificateur honnête à `t` ne dispose que des prévisions **faites à `t`** des rendements
`r_t .. r_{t+H-1}`. Elles s'obtiennent par **rollout autorégressif** du world model depuis la
fenêtre de `t` (`mirage.wm.rollout`) : le modèle prédit le prochain état, cet état est réinjecté
comme dernière ligne de la fenêtre, et ainsi de suite. C'est la seule source d'information sur
le futur qui soit légitime à `t`.

- `R[t, k]` = composante `ret` du rollout depuis la fenêtre de `t`, pour `k = 0..H-1`.
  `R[t, 0]` est `rhat[t]` **au bit près** (test 6).
- `C[t, k]` = coût prévu = **persistance du demi-spread courant** `+` frais. C'est le choix le
  plus simple, sans degré de liberté, figé dans le prereg.
- `R = C = 0` **au-delà de la fin du jour de `t`**. Le masque est un **escalier** ligne par
  ligne : la ligne `i` d'un jour de longueur `L` est nulle à partir de la colonne `L - i`.
  C'est la règle « aucune somme par-dessus la nuit », et elle a demandé une correction (un
  premier jet annulait un tronçon final unique, laissant la ligne `i` lire des barres après
  minuit).
- `plan_positions_causal(R, C, days)` résout la **programmation dynamique à horizon
  glissant** exacte sur l'espace d'états `{-1, 0, +1}`, coût `c_t * |p_t - p_{t-1}|` payé au
  changement, et **chaque journée repart à plat**.

La fonction est **bit à bit équivalente** à `plan_positions` quand on lui donne la vérité
(`R[t,k] = r[t+k]`) : c'est le test 1, et c'est ce qui garantit que le seul changement est la
source des prévisions, pas la mécanique d'optimisation.

### 3.3 Le pré-enregistrement

`configs/phase2b_crypto_prereg.yaml`, commité **avant** tout calcul. Il fixe : les 44 dates,
les 5 symboles (un couple manquant est un échec bloquant), le monde (LinearWM, lookback 16, 5
folds, `min_train_frac` 0.4, embargo 16), l'agent causal (H = 10 primaire, robustesse à
H ∈ {1, 5, 20}), les frais (0 / 2 / 5.5 bp, primaire 2.0), le bootstrap (2000 tirages, graine
0, blocs de 3 journées, apparié), les règles et leurs seuils, et la liste littérale des 44
dates neuves de l'extension. Ses `interdits` proscrivent de changer H, les frais, les seuils,
le coût prévu ou la méthode de rollout après avoir vu un résultat, et de retirer un symbole ou
une journée.

### 3.4 Le harnais et ses deux contrôles bloquants

`scripts/crypto/phase2b.py` rejoue tout, symbole par symbole. Il applique deux contrôles
d'intégrité **avant** de publier quoi que ce soit ; si l'un casse, aucun résultat n'est écrit :

1. **le bras myope reproduit l'artefact publié au bit près** (max|diff| = 0 sur `y`, `day` et
   `pred` contre `experiments/crypto_lob_oos_<SYM>.npz`) ;
2. **`agent_fuite` reproduit le net publié de la Phase 2 à 1e-9** sur les trois frais et les
   deux natures (imaginaire et réelle).

Le second est le contrôle qui compte : il établit que le harnais causal sait reproduire le
chiffre publié quand on lui rend la fuite. Sans lui, un « le résultat disparaît » pourrait
être un bug.

Les diagnostics de confondant sont la factorisation de ceux de la Phase 2 (`diag_arrays`,
extraite de `agent_diag.diag_symbol` sans changer sa sortie publiée), portés sur l'agent
causal.

## 4. Résultats

### 4.1 Les contrôles bloquants sont verts

Sur DOGEUSDT puis sur les 5 symboles, les deux contrôles passent : le myope est identique au
bit près à l'artefact publié, et `agent_fuite` redonne le net publié à ≤ 7e-17 près sur ses
trois frais et ses deux natures (+0.238241 / +0.090477 / +0.009871 bp en réel à 0 / 2 / 5.5
bp). Le harnais causal reproduit donc exactement la Phase 2 quand on lui rend la fuite.

### 4.2 Le verdict principal

Net **réel** à 2 bp, en bp/barre, avec IC95 bootstrap par journées (2000 tirages, blocs de 3) :

| symbole | myope | agent_fuite (publié) | agent_causal (honnête) | IC95 du causal |
|---|---|---|---|---|
| BTCUSDT | -0.9865 | +0.0213 | **+0.0003** | [-0.0010, +0.0018] |
| DOGEUSDT | -1.5051 | +0.0905 | **-0.0056** | [-0.0129, -0.0005] |
| ETHUSDT | -1.2535 | +0.0418 | **-0.0016** | [-0.0033, +0.0001] |
| SOLUSDT | -1.5135 | +0.0553 | **-0.0021** | [-0.0049, -0.0001] |
| XRPUSDT | -1.5657 | +0.0551 | **-0.0020** | [-0.0043, -0.0002] |

`edge_reel_causal` (borne basse de l'IC95 > 0, seuil ≥ 4/5) : **NON, 0/5**. La borne basse
n'est jamais au-dessus de 0 ; elle est **entièrement négative** sur DOGE, SOL et XRP, et
contient 0 sur BTC et ETH. Le seul résultat positif du projet ne survit pas à la correction.

### 4.3 La fuite, mesurée

Écart apparié par journée entre le bras publié et l'agent honnête (`agent_fuite@2 −
agent_causal@2`), en bp/barre :

| symbole | Δnet | IC95 | bloc-3 IC95 |
|---|---|---|---|
| BTCUSDT | +0.0209 | [+0.0128, +0.0313] | [+0.0126, +0.0327] |
| DOGEUSDT | +0.0960 | [+0.0702, +0.1279] | [+0.0717, +0.1364] |
| ETHUSDT | +0.0433 | [+0.0344, +0.0524] | [+0.0354, +0.0545] |
| SOLUSDT | +0.0574 | [+0.0338, +0.0878] | [+0.0363, +0.0977] |
| XRPUSDT | +0.0571 | [+0.0355, +0.0804] | [+0.0373, +0.0924] |

`fuite_significative` (borne basse > 0, seuil ≥ 4/5) : **OUI, 5/5**, et le bootstrap par
blocs de 3 journées ne change rien. L'ampleur de la fuite (+0.021 à +0.096) est du même ordre
que le net publié lui-même (+0.021 à +0.090) : **la fuite explique le résultat publié, pas
seulement une part**.

### 4.4 Ce qui tombe avec la fuite, et ce qui survit

**Ce qui tombe.**

- Le contrôle du **bruit**. En Phase 2, un agent nourri de bruit, avec les mêmes coûts,
  perdait sur 5/5 : c'était la preuve que le gain n'était pas un artefact de coût. Avec
  l'agent honnête, ce contrôle ne tient plus que sur **2/5** (DOGE, XRP). C'est attendu et
  instructif : la permutation intra-journée des lignes de `R` détruit la fuite **en même temps
  que** le signal, elle ne pouvait donc pas la voir. Le contrôle n'était pas faux, il était
  aveugle à ce qui portait réellement le résultat.
- L'**écart d'exploitation** de la Phase 2, négatif et significatif sur 5/5 (§4 de
  `PHASE2_AGENT.md` et §10 : le modèle imaginé promettait plus qu'il ne livrait). Avec le
  planificateur causal il vaut -0.0003 à +0.0084 bp et son IC95 **contient 0 ou est > 0 sur
  5/5**. La prédiction pré-enregistrée de l'étape B est confirmée : l'écart négatif avait une
  explication mécanique, la fuite, et il disparaît avec elle.

**Ce qui survit.** L'agent honnête **bat le myope sur 5/5** (`agent_causal_bat_le_myope` :
OUI), de +0.99 à +1.56 bp/barre, IC95 entièrement > 0. Mais la décomposition est sans
ambiguïté :

| symbole | Δnet | Δbrut | coût évité |
|---|---|---|---|
| BTCUSDT | +0.9868 | **-0.1633** | +1.1501 |
| DOGEUSDT | +1.4995 | **-0.2672** | +1.7667 |
| ETHUSDT | +1.2520 | **-0.2315** | +1.4835 |
| SOLUSDT | +1.5114 | **-0.2722** | +1.7837 |
| XRPUSDT | +1.5637 | **-0.2205** | +1.7842 |

Le brut directionnel est **négatif sur 5/5**, et le net positif vient à 100 % du coût évité.
L'agent ne lit pas mieux le marché : il constate qu'il n'y a rien à y gagner et reste à plat.
Le turnover le confirme, en changements de position par journée :

| symbole | agent causal | myope |
|---|---|---|
| BTCUSDT | 10.8 | 48 400 |
| DOGEUSDT | 262.2 | 64 736 |
| ETHUSDT | 34.3 | 62 066 |
| SOLUSDT | 8.4 | 65 435 |
| XRPUSDT | 134.3 | 60 514 |

L'agent causal est plat 12 % à 78 % du temps selon le symbole, et change de position 8 à 262
fois par jour contre 48 000 à 65 000 pour le myope. Sa part plate (`frac_plat`) va de 0.12
(ETH) à 0.78 (SOL).

### 4.5 La robustesse à l'horizon

Net réel à 2 bp, agent causal, pour H ∈ {1, 5, 10, 20} :

| symbole | H = 1 | H = 5 | H = 10 | H = 20 |
|---|---|---|---|---|
| BTCUSDT | +0.0003 | +0.0004 | +0.0003 | +0.0004 |
| DOGEUSDT | -0.0001 | -0.0056 | -0.0056 | -0.0046 |
| ETHUSDT | +0.0011 | -0.0008 | -0.0016 | -0.0014 |
| SOLUSDT | +0.0012 | -0.0002 | -0.0021 | -0.0025 |
| XRPUSDT | +0.0006 | -0.0028 | -0.0020 | -0.0026 |

Rien de positif ne subsiste à aucun horizon : les valeurs restent à ±0.006 bp, sans
signification. Il est notable que le résultat **croissait** avec H dans la Phase 2 : c'est
exactement la signature de la fuite, dont la taille augmente avec le nombre de pas futurs
lisibles.

### 4.6 Calibration : la pente vaut 1

Contrôle de calibration du bras myope publié (lecture seule de l'artefact, sans
ré-entraînement) : pente β de `y` sur `pred`, IC95 par journée.

| symbole | β | IC95 | β (décile supérieur de \|pred\|) | E[y] (bp) | E[pred] (bp) |
|---|---|---|---|---|---|
| BTCUSDT | 1.022 | [0.958, 1.085] | 0.951 | +0.0002 | -0.0007 |
| DOGEUSDT | 0.667 | [0.494, 0.899] | 0.465 | +0.0005 | -0.0085 |
| ETHUSDT | 1.061 | [1.016, 1.102] | 0.939 | +0.0009 | -0.0023 |
| SOLUSDT | 1.009 | [0.954, 1.068] | 1.011 | +0.0002 | +0.0018 |
| XRPUSDT | 0.695 | [0.526, 0.939] | 0.575 | -0.0002 | -0.0059 |

La pente vaut 1 (IC95 contenant 1 sur 4 symboles, au-dessus sur ETH), et elle **ne s'effondre
pas** dans le décile supérieur de `|pred|` : ni sur-confiance, ni contraction. Cela **réfute**
l'hypothèse « contraction Ridge » avancée au §10 de la Phase 2 pour expliquer l'écart
d'exploitation négatif. La calibration du myope est bonne ; ce n'est pas là qu'était le
problème.

### 4.7 Le plafond du clairvoyant

Pour situer l'échelle, l'agent **clairvoyant** (prévision parfaite, même mécanique de coût)
net à 2 bp : +0.12 (BTC), +0.39 (DOGE), +0.21 (ETH), +0.31 (SOL), +0.25 (XRP) bp/barre. Le
bras publié (+0.02 à +0.09) en captait donc une fraction notable ; l'agent honnête, rien.

## 5. Interprétation

**Le résultat de la Phase 2 était un artefact.** La démonstration se tient par les deux
bouts : l'écart apparié entre le bras publié et l'agent causal est significatif sur 5/5 et
son ampleur couvre le net publié ; et le contrôle d'intégrité montre que le harnais reproduit
le chiffre publié au chiffre près quand on lui rend la fuite. Ce n'est pas une correction
d'estimation, c'est une attribution.

**Le contrôlait qui validait le résultat ne pouvait pas le mettre en cause.** Les trois
diagnostics de confondant de la Phase 2 (bruit, dérive, décomposition) avaient été conçus pour
écarter les artefacts de coût et de dérive. Aucun n'était sensible à une fuite construite dans
l'alignement des cibles. La permutation intra-journée du bruit, en particulier, détruit la
fuite en même temps que le signal : elle ne pouvait que confirmer. La leçon méthodologique est
là, et elle vaut plus que le résultat : **un contrôle de cohérence, pas une intuition, a
démoli le résultat le plus positif du projet.**

**Ce qui reste vrai.** La Phase 2 avait raison sur un point, et il n'est pas anodin : le net
de l'agent est très supérieur à celui du myope, et il l'est par le coût évité. Sur des frais de
2 bp et un horizon d'une seconde, la bonne décision est presque toujours **ne pas trader**.
Un planificateur exact, même dans un world model médiocre, le trouve. Mais ce n'est pas un
edge : c'est l'absence de perte. La Phase 1 avait montré qu'il n'y a pas d'edge net de coûts ;
la Phase 2b montre qu'il n'y en a pas davantage quand on planifie dans le modèle.

**Pourquoi la fuite croissait avec H.** C'est le mécanisme le plus instructif de l'épisode.
Avec un coût négligeable devant l'échelle de la prévision, la récursion du planificateur exact
donne une valeur **identique pour les trois actions** (`F_k[a, s]` = somme des `|r|`), et le
planificateur **dégénère en `sign(r_t)`** : l'horizon ne sert plus à rien, donc la fuite non
plus. Mesuré sur un processus AR(1) : avec `c = 0`, 0 décision sur 280 bascule quand on
perturbe un état futur ; avec `c = 2 × std(rhat)`, 40 sur 280 basculent. C'est le **coût
relatif très élevé des données réelles** (quelques bp de spread contre une fraction de bp de
prévision) qui fait vivre l'horizon, et donc la fuite. Cela explique, chiffres en main,
pourquoi le net publié croissait avec H : plus d'horizon, plus de pas futurs lisibles.

## 6. Limites

- **La mesure de la fuite est une attribution, pas une décomposition exacte.** On mesure
  `agent_fuite − agent_causal` ; on ne prétend pas que toute la différence est due à la seule
  composante `rhat[t+1]`. L'écart est significatif et du bon ordre de grandeur, c'est ce qui
  est affirmé.
- **Le planificateur causal dépend du rollout, donc de la qualité du monde.** Un rollout à 10
  pas d'un LinearWM est une pièce imparfaite ; il est possible qu'un meilleur modèle rende
  l'agent causal rentable. Ce n'est pas testé ici, et ce serait un autre pré-enregistrement.
- **Le coût prévu = persistance du demi-spread** est une simplification. Un coût prévu
  dépendant de la position serait plus fin, mais introduirait un degré de liberté que le
  prereg refuse.
- **L'extension à 88 journées et la cible cumulée (A2) ne sont pas encore exécutées** (voir
  §7). Le présent rapport porte sur les 44 journées publiées.
- Comme partout dans ce projet, **aucun ordre réel n'est jamais passé** : tout le net est
  calculé sur des positions et des coûts simulés.

## 7. Étapes non encore exécutées

Deux étapes du plan ont leur code en place mais pas encore leurs résultats au moment de
l'écriture, et seront ajoutées ici :

- **2b-3 (A2, cible cumulée)** : world model entraîné sur la somme glissante `yH` à H = 10,
  embargo 26, bras `agent_direct@2` (prévision scalaire étalée sur l'horizon) comparé au bras
  `agent_causal_A2@2` recalculé sur le même échantillon. Exploratoire.
- **2b-5 (extension à 88 journées)** : union des 44 journées publiées et de 44 journées
  neuves, sous-ensemble `neuf` **confirmatoire**. Sur ≈ 53 journées hors échantillon, un
  résultat négatif est plus solide que sur 27.

## 8. Reproduire

```powershell
# Les six tests de causalité, qui doivent être verts AVANT tout run
.\.venv\Scripts\python.exe -m pytest tests/test_plan_causal.py -q

# Re-mesure causale : 5 symboles, frais 0/2/5.5, écrit experiments_2b/ (~20 min)
.\.venv\Scripts\python.exe scripts\crypto\phase2b.py --out experiments_2b

# Diagnostics de confondant (bruit 39 permutations, dérive 400 décalages), ~9 min/symbole
.\.venv\Scripts\python.exe scripts\crypto\phase2b.py --diag --perm 39 --shift 400

# Calibration (B), lecture seule des npz
.\.venv\Scripts\python.exe scripts\crypto\calib_diag.py
```

Protocole figé : `configs/phase2b_crypto_prereg.yaml`. Sorties : `experiments_2b/`
(`p2b_net.csv`, `p2b_compare.csv`, `p2b_exploit.csv`, `p2b_diag_*.csv`, `calib_*.csv`,
`p2b_oos_<SYM>.npz`, `phase2b_results.json`). Le planificateur causal est
`mirage.plan.plan_positions_causal` ; ses chemins d'entrée, `mirage.plan.causal_paths`.

Erratum correspondant dans la Phase 2 : [`PHASE2_AGENT.md`](PHASE2_AGENT.md). Le bras publié
de la Phase 2 reste dans `experiments/` ; la re-mesure vit dans `experiments_2b/`, et les
deux ne sont jamais confondus.
