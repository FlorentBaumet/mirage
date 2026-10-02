# Phase 1c (crypto) - OFI événementiel : l'apport n'est pas établi

> **TL;DR.** La phase 1 laissait une porte ouverte : son R²_OOS est un **plancher**, car
> les barres 1 s ne conservent que l'état du carnet à la fin de chaque seconde et jettent
> tout le flux intra-seconde. La phase 1c ajoute au vecteur d'état l'**OFI événementiel**
> (Order Flow Imbalance de Cont, Kukanov & Stoikov au meilleur niveau, calculé événement
> par événement sur le flux brut Bybit puis sommé par seconde) - la seule information
> réellement absente des barres. Le résultat est **négatif, et publié tel quel** :
> l'ajout de l'OFI ne modifie pas la prévision du rendement (ΔR²_OOS apparié compris entre
> **−0.00062 et +0.00004** selon le symbole, IC95 appariés de largeur 0.00039 à 0.00219),
> la borne basse n'est positive sur **0/5** symbole pour le seuil pré-enregistré de ≥ 4/5
> (règle `apport` : **APPORT NON ÉTABLI**), et le verdict économique est **inchangé** :
> signal réel 5/5, mirage confirmé, 0/5 exploitable. L'information intra-seconde au
> meilleur niveau n'est pas exploitable par ce modèle linéaire - ou elle est déjà contenue
> dans l'état.

---

## 1. Question

La phase 1 a montré un signal de microstructure **réel** (R²_OOS poolé 0.016–0.052, IC95
excluant 0 sur 5/5 symboles) mais **non exploitable** (net négatif dès 2 bp, mort sans
frais sur XRP). Elle notait explicitement que son R²_OOS est un **plancher** : la métrique
d'erreur quadratique pénalise un prédicteur bruité, et surtout les barres 1 s ne sont
qu'une **représentation** du carnet - elles ne contiennent pas tout le flux.

La phase 1c repose une seule question, changée par un seul paramètre : apporter au modèle
une information que les barres 1 s ne contiennent pas améliore-t-il la prévision du
rendement, et cet apport suffit-il à changer le verdict économique ?

**Pourquoi l'OFI événementiel, et pas un OFI recalculé sur les barres.** Un OFI estimé par
la variation nette du carnet entre deux snapshots à 1 s serait une quasi-fonction de la
fenêtre d'entrée déjà fournie au modèle (imbalances, micro-price, rendements) : le test
serait quasi vide - on n'ajouterait pas d'information, seulement une combinaison linéaire
de features existantes. Seul l'OFI **événementiel**, calculé sur le flux brut
(snapshot + deltas ~10 ms) puis agrégé par seconde, apporte quelque chose que la barre ne
capture pas : l'ordre et l'ampleur des modifications du meilleur niveau **à l'intérieur**
de la seconde. C'est le test décisif pour savoir si la borne basse du R²_OOS venait d'une
information manquante ou du modèle.

**Un résultat négatif est un résultat.** Le pré-enregistrement l'écrit noir sur blanc : si
l'OFI n'apporte rien, il est publié tel quel. Ce rapport ne cherche aucune lecture de
rattrapage.

## 2. Données

Mêmes données que la phase 1, **au couple (symbole, date) près** : c'est ce qui rend la
comparaison appariée valide. Rien n'a été ajouté ni retiré au jeu ; seule la représentation
change.

| | |
|---|---|
| Source | Bybit L2, `quote-saver.bycsi.com/orderbook/linear`, dumps `ob500` (500 niveaux) |
| Symboles | BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT |
| Journées | **44**, échelonnées du 2023-01-20 au 2025-08-06 |
| Couples OFI construits | **220** = 44 dates × 5 symboles, **tous présents** (44 × 5 = 220) |
| Volume par symbole | 3,80 M échantillons |
| **Fenêtre testée (OOS)** | **27 journées, 2024-02-06 → 2025-08-06**, 2,28 M points/symbole → 11,4 M au total |

**Couverture vérifiée sur disque.** `data/raw/crypto_ofi/` contient **220 fichiers**
`.pkl`, un par couple (symbole, date), soit exactement ce qu'exige le pré-enregistrement
(`couverture_exigee`). Aucun couple manquant : il n'y a donc **pas** d'échec bloquant à
rapporter, et la comparaison appariée porte bien sur des échantillons identiques entre les
deux bras.

**Définition de la feature.** Pour un événement qui fait passer le carnet de `n-1` à `n`,
avec `(P^b, q^b)` le meilleur bid et `(P^a, q^a)` le meilleur ask (Cont, Kukanov &
Stoikov) :

```
e_n = 1{P^b_n >= P^b_{n-1}} q^b_n - 1{P^b_n <= P^b_{n-1}} q^b_{n-1}
    - 1{P^a_n <= P^a_{n-1}} q^a_n + 1{P^a_n >= P^a_{n-1}} q^a_{n-1}

OFI(seconde t) = somme des e_n des événements horodatés dans t
```

Implémentation : `src/mirage/data/bybit_ofi.py`. **Alignement** : la barre d'indice `T`
contient l'état du carnet *après* les événements de la seconde `T` ; `OFI(T)` est donc la
somme des contributions de la seconde `T`, contemporaine de `ret(T) = log(mid_T) −
log(mid_{T-1})`. Les secondes sans événement au meilleur niveau valent `0` : un flux nul
n'est pas une donnée manquante. **Causalité** : `OFI(T)` ne dépend que des événements de la
seconde `T`, propriété de préfixe vérifiée par `tests/test_ofi_causal.py` (ajouter du futur
ne modifie pas le passé).

## 3. Protocole

### 3.1 Pré-enregistrement

Le fichier [`configs/phase1c_crypto_prereg.yaml`](../configs/phase1c_crypto_prereg.yaml) a
été **figé et commité avant toute exécution** de l'évaluation enrichie (invariant I3). Il
fixe la liste des dates, l'état de base et l'état enrichi, le placement de la nouvelle
dimension, le modèle (identique à la phase 1), la grille de frais, le dispositif de
bootstrap (unités, B, graine, variante de robustesse) et **les quatre règles de décision**.
Rien n'a été retouché après avoir vu un chiffre.

### 3.2 Le seul changement : une colonne en dernière position

| | Dimensions |
|---|---|
| Base (= phase 1) | `ret, spread_rel, imb1, depth_imb, micro_dev` |
| Enrichi | `ret, spread_rel, imb1, depth_imb, micro_dev, ofi` |

`ofi` est **ajoutée en dernière position**. Conséquence voulue : `RET_IDX` reste `0` et les
cinq premières dimensions gardent leur sens exact, donc tout l'aval (export `.npz`,
bootstrap, position = signe de la prédiction de `ret`) est inchangé. Modèle, walks-forward,
folds, embargo, cible et métrique sont **strictement identiques** à la phase 1 - c'est la
condition de comparabilité.

### 3.3 Bootstrap apparié

Les deux bras partagent tout sauf une colonne : mêmes barres, mêmes journées, mêmes folds,
mêmes cibles, même nombre d'échantillons. Le dispositif n'est donc **pas** deux intervalles
de confiance calculés séparément - ce serait traiter deux modèles presque identiques comme
deux expériences indépendantes, élargir artificiellement l'IC de la différence et risquer
de rater un apport réel.

Le bootstrap est **apparié** : à chaque réplique, les **mêmes journées** tirées avec remise
servent à recalculer `R²_OOS(base)` et `R²_OOS(enrichi)` (idem pour `net_bp`), et la
statistique est leur **différence**. C'est la loi de la différence sur échantillons
appariés. Le tirage est celui de `bootstrap_signif.py` (même fonction `boot_mult`), pour que
les deux verdicts ne divergent pas par la seule mécanique du tirage. Les deux bras sont
vérifiés **appariables** avant tout calcul (`paired_ofi.py`, `assert_apparies`) : si une
colonne ajoutée avait décalé la fenêtre ou fait disparaître des lignes, la comparaison
échouerait bruyamment au lieu d'être silencieusement fausse.

Paramètres inchangés : unité = la journée, B = 2000, graine 0, block bootstrap circulaire,
rééchantillonnage de journées entières avec remise ; variante pré-enregistrée en blocs
contigus de 3 journées.

### 3.4 Les quatre règles de décision

| Règle | Énoncé |
|---|---|
| `apport` | L'OFI **apporte une information réelle** ssi l'IC95 apparié de `ΔR²_OOS(ret)` a une borne basse strictement positive pour **≥ 4 des 5** symboles. |
| `mirage` | Le mirage est **confirmé** ssi, dans le bras enrichi, l'IC95 de `net_bp` à 2 bp est **entièrement strictement négatif pour chaque symbole**. |
| `bascule` | Une **bascule** de verdict exige que la borne basse de l'IC95 de `net_bp` à 2 bp soit **strictement positive** pour un symbole du bras enrichi. Aucune autre lecture (net moyen > 0, borne haute > 0, amélioration du seul R²) ne vaut bascule. |
| `amelioration_non_concluante` | Si l'OFI améliore le R²_OOS sans que le net à 2 bp cesse d'être négatif, le résultat est rapporté comme **amélioration statistique sans bascule économique**. |

## 4. Résultats

### 4.1 - Couverture réelle

**220 couples (symbole, date) sur 220** ont un OFI. Les 44 dates et 5 symboles du
pré-enregistrement sont tous couverts ; il n'y a aucun couple manquant, donc aucun échec
bloquant et aucun retrait silencieux. Les deux bras portent exactement 2 280 492 points
OOS (BTC), 2 280 495 (ETH), 2 280 453 (SOL), 2 280 200 (XRP), 2 278 628 (DOGE), appariés.

### 4.2 - R²_OOS 1-step par dimension

R²_OOS moyen (5 folds × 5 symboles), modèle **linéaire**, base vs enrichi. La ligne `ofi`
n'existe que dans le bras enrichi : c'est la dimension ajoutée elle-même.

| Dim | base (5 dims) | enrichi (6 dims) | Δ |
|---|---|---|---|
| `ret` (prix) | 0.05299 | **0.05280** | **−0.00019** |
| `spread_rel` | 0.41797 | 0.41761 | −0.00035 |
| `imb1` | 0.26572 | 0.26602 | +0.00030 |
| `depth_imb` | 0.19375 | 0.19424 | +0.00049 |
| `micro_dev` | 0.29069 | 0.29292 | +0.00223 |
| **`ofi`** | **absent de la base** | **0.43568** | - |

Lecture : la cible `ret` **ne bouge pas** (elle recule même très légèrement). Les dimensions
d'état `imb1`, `depth_imb` et `micro_dev` sont marginalement mieux prévues (+0.0003 à
+0.0022), mais ce n'est pas la question posée : la cible pré-enregistrée est le rendement.
La dimension `ofi` est, elle, très prévisible 1-step (R² 0.436, la deuxième plus prédictible
après `spread_rel`) - mais une feature très prévisible n'est utile que si elle aide à prédire
*autre chose*.

### 4.3 - ΔR²_OOS(ret) par symbole, IC95 apparié

`ΔR² = enrichi − base`, sur la cible `ret`, bootstrap apparié par journées (B = 2000,
graine 0) :

| Symbole | R² base | R² enrichi | ΔR² | IC95 apparié | p(ΔR² ≤ 0) |
|---|---|---|---|---|---|
| BTCUSDT | 0.05217 | 0.05171 | −0.00046 | [−0.00163, +0.00056] | 0.758 |
| ETHUSDT | 0.04964 | 0.04903 | −0.00061 | **[−0.00117, −0.00004]** | 0.995 |
| SOLUSDT | 0.03700 | 0.03704 | **+0.00004** | [−0.00020, +0.00019] | 0.401 |
| XRPUSDT | 0.02125 | 0.02089 | −0.00036 | [−0.00092, +0.00044] | 0.807 |
| DOGEUSDT | 0.01585 | 0.01523 | −0.00062 | [−0.00145, +0.00014] | 0.888 |

Le seul ΔR² positif est celui de SOL (+0.00004), et son IC95 encadre zéro de très près.
Quatre symboles sur cinq ont un ΔR² **négatif** : le point d'ETH a même un IC95 **entièrement
sous zéro** ([−0.00117, −0.00004]) - ajouter l'OFI dégrade très légèrement, mais de façon
significative au sens apparié. La largeur des IC95 appariés de ΔR² va de **0.00039** (SOL, le
plus étroit) à **0.00219** (BTC, le plus large) : le dispositif est assez fin pour détecter
un apport de l'ordre de quelques dix-millièmes de R², et il n'en détecte aucun.

**Robustesse, blocs contigus de 3 journées** (variante pré-enregistrée) :

| Symbole | ΔR² | IC95 apparié (blocs de 3 j) |
|---|---|---|
| BTCUSDT | −0.00046 | [−0.00141, +0.00036] |
| ETHUSDT | −0.00061 | [−0.00127, +0.00001] |
| SOLUSDT | +0.00004 | [−0.00019, +0.00017] |
| XRPUSDT | −0.00036 | [−0.00086, +0.00033] |
| DOGEUSDT | −0.00062 | [−0.00138, +0.00020] |

Sous la variante la plus conservatrice, aucun IC95 n'a de borne basse positive non plus. Le
seuil pré-enregistré (≥ 4/5) échoue de la même façon, avec ou sans blocs.

### 4.4 - Tableau économique complet, base vs enrichi

Net par barre, en points de base ; coût à chaque changement de position = demi-spread
réellement mesuré + frais taker. Base = `experiments/`, enrichi = `experiments_ofi/`.

| Symbole | bras | gross/barre | turnover | net 0 bp | net 2 bp | net 5,5 bp |
|---|---|---|---|---|---|---|
| BTCUSDT | base | +0.1639 | 0.287 | +0.1596 | −0.9865 | −2.9921 |
| BTCUSDT | enrichi | +0.1645 | 0.289 | +0.1601 | −0.9964 | −3.0204 |
| ETHUSDT | base | +0.2308 | 0.367 | +0.2161 | −1.2535 | −3.8254 |
| ETHUSDT | enrichi | +0.2313 | 0.368 | +0.2166 | −1.2554 | −3.8313 |
| SOLUSDT | base | +0.2705 | 0.387 | +0.0360 | −1.5135 | −4.2251 |
| SOLUSDT | enrichi | +0.2704 | 0.388 | +0.0354 | −1.5166 | −4.2325 |
| XRPUSDT | base | +0.2223 | 0.358 | −0.1326 | −1.5657 | −4.0736 |
| XRPUSDT | enrichi | +0.2214 | 0.357 | −0.1317 | −1.5597 | −4.0588 |
| DOGEUSDT | base | +0.2705 | 0.384 | +0.0291 | −1.5051 | −4.1898 |
| DOGEUSDT | enrichi | +0.2705 | 0.384 | +0.0278 | −1.5072 | −4.1936 |

Les deux bras sont **visuellement indiscernables** : gross et turnover à ±0.002, net à ±0.02
bp. L'OFI ne déplace ni le gain brut, ni la fréquence de trading, ni le coût. Le net à 0 bp
reste positif sur BTC/ETH/SOL/DOGE et négatif sur XRP dans les deux bras ; à 2 bp et 5,5 bp,
les cinq symboles sont perdants dans les deux bras. (Les « PnL cumulés » de
`crypto_lob_economic.csv` sont la même somme arithmétique de rendements par barre que dans
la phase 1, sans composition - chiffre à ne pas lire comme un rendement de compte.)

### 4.5 - net_bp à 2 bp par symbole et par bras, IC95, et bascule

| Symbole | net base @2 bp [IC95] | net enrichi @2 bp [IC95] | Δnet apparié [IC95] | bascule |
|---|---|---|---|---|
| BTCUSDT | −0.9865 [−1.0576, −0.9164] | −0.9964 [−1.0694, −0.9256] | −0.0099 [−0.016, −0.005] | non |
| ETHUSDT | −1.2535 [−1.3135, −1.1978] | −1.2554 [−1.3154, −1.1984] | −0.0019 [−0.005, +0.001] | non |
| SOLUSDT | −1.5135 [−1.5879, −1.4425] | −1.5166 [−1.5939, −1.4435] | −0.0031 [−0.007, +0.000] | non |
| XRPUSDT | −1.5657 [−1.7005, −1.4468] | −1.5597 [−1.6900, −1.4438] | +0.0060 [−0.002, +0.014] | non |
| DOGEUSDT | −1.5051 [−1.6113, −1.4041] | −1.5072 [−1.6153, −1.4040] | −0.0022 [−0.007, +0.001] | non |

Dans le bras enrichi, la borne basse de l'IC95 de net à 2 bp est négative pour les **cinq**
symboles : aucune bascule n'est réalisable. Le Δnet apparié est nul à l'échelle de la mesure
(≤ 0.01 bp) et son IC95 encadre 0 sur quatre symboles. Le seul déplacement notable, XRP
(+0.0060 bp), reste du bruit : sa borne basse est négative.

### 4.6 - Bootstrap du bras enrichi

IC95 par bootstrap de journées (B = 2000, graine 0), bras enrichi :

| Symbole | R²_OOS [IC95] | net 0 bp [IC95] | net 2 bp [IC95] |
|---|---|---|---|
| BTCUSDT | 0.0517 [0.0346, 0.0791] | +0.1601 [+0.141, +0.179] | −0.9964 [−1.069, −0.926] |
| ETHUSDT | 0.0490 [0.0353, 0.0676] | +0.2166 [+0.203, +0.231] | −1.2554 [−1.315, −1.198] |
| SOLUSDT | 0.0370 [0.0212, 0.0635] | +0.0354 [+0.001, +0.069] | −1.5166 [−1.594, −1.443] |
| XRPUSDT | 0.0209 [0.0081, 0.0402] | −0.1317 [−0.202, −0.063] | −1.5597 [−1.690, −1.444] |
| DOGEUSDT | 0.0152 [0.0008, 0.0472] | +0.0278 [−0.009, +0.064] | −1.5072 [−1.615, −1.404] |

Verdicts mécaniques du bras enrichi (`bootstrap_log.txt`) : **signal 5/5 → SIGNAL RÉEL** ;
**mirage → MIRAGE CONFIRMÉ** ; **exploitable 0/5 → AUCUN SYMBOLE EXPLOITABLE**. Identiques
à ceux du bras de base. À 0 frais, net moyen `[0.16, 0.217, 0.035, −0.132, 0.028]` bp, borne
basse > 0 sur 3/5 - c'est la borne optimiste (frais nuls irréalistes), qui ne vaut pas edge.
Sous blocs de 3 journées, DOGE encadre 0 (R²_OOS IC95 [−0.0014, 0.0571]) dans le bras enrichi
comme dans le bras de base : le point le plus fragile de la phase 1 le reste.

### 4.7 - Verdicts des quatre règles, appliqués mécaniquement

| Règle | Énoncé résumé | Verdict |
|---|---|---|
| `apport` | borne basse de l'IC95 apparié de ΔR²_OOS(ret) > 0 sur ≥ 4/5 | **APPORT NON ÉTABLI** (0/5) |
| `mirage` | IC95 de net à 2 bp entièrement < 0 pour chaque symbole (bras enrichi) | **MIRAGE CONFIRMÉ** |
| `bascule` | borne basse de l'IC95 de net à 2 bp > 0 sur un symbole (bras enrichi) | **AUCUNE BASCULE** (0/5) |
| `amelioration_non_concluante` | OFI améliore le R²_OOS sans lever le net à 2 bp | **NON DÉCLENCHÉE** (l'OFI n'améliore pas le R²_OOS `ret`) |

La dernière ligne mérite d'être explicitée : la branche « amélioration statistique sans
bascule économique » n'est même pas atteinte. L'OFI n'améliore pas la prévision du
rendement, il ne la dégrade pas de façon exploitable non plus - il ne fait **rien** à la
cible. Il n'y a ni déplacement statistique, ni déplacement économique.

## 5. Interprétation

L'ajout de l'OFI événementiel **ne change pas le verdict**, à aucun niveau : ΔR²_OOS(ret)
apparié entre −0.0006 et +0.0000, net à 2 bp déplacé de moins de 0.01 bp, mêmes verdicts
statistiques et économiques que le bras de base. Le plancher de R²_OOS noté par la phase 1
ne venait donc **pas** d'une information intra-seconde manquante au meilleur niveau : cette
information, une fois fournie explicitement et causalement au modèle, ne lui permet pas de
mieux prévoir le rendement suivant. Deux lectures, non tranchées par ce dispositif :

- **Soit** l'information de flux intra-seconde au meilleur niveau est déjà **contenue** dans
  l'état que les barres 1 s portent (imbalances, micro-price, rendements) - auquel cas l'OFI
  n'ajoute rien parce qu'il n'y a rien à ajouter au meilleur niveau sur cet horizon de 1 s.
- **Soit** elle est réelle mais **non exploitable par le modèle linéaire** testé : la règle
  `apport` a été évaluée sur `LinearWM`, comme pré-enregistré. Que la dimension `ofi` soit
  elle-même très prévisible 1-step (R² 0.436) sans aider la cible `ret` est cohérent avec
  une information redondante pour cette cible, ou avec une relation non linéaire que la
  régression ne capte pas.

Le résultat économique tranche, lui, la question posée par le pré-enregistrement : même si
l'OFI avait apporté un gain statistique, il fallait qu'il **fasse basculer** le net à 2 bp
au-dessus de zéro pour compter. Ce n'est pas le cas. Le signal de la phase 1 reste ce qu'il
était : réel, significatif, et noyé par le coût d'exécution.

**Pourquoi ce résultat négatif est intéressant.** Il ferme une échappatoire classique du
discours sur les world models : « le R²_OOS est un plancher, il manque une information ».
Ici l'information manquante a été identifiée, construite correctement (événementielle,
causale, alignée), vérifiée sur 220/220 couples, fournie au modèle dans une comparaison
appariée - et elle n'apporte rien. Le plancher n'était pas un défaut de représentation au
meilleur niveau ; le rendement à +1 s ne porte pas d'edge supplémentaire que cette
information révèle dans ce modèle.

## 6. Limites

**Ce que l'OFI événementiel ne capture pas** (précisé dans `src/mirage/data/bybit_ofi.py`) :

- **Le flux au-delà du meilleur niveau.** `_contribution` ne regarde que le meilleur bid et
  le meilleur ask. Toute l'activité sur les niveaux 2 à 500 - ajouts, retraits, exécutions
  profonds - est invisible. Un déséquilibre profond qui précéderait le mouvement de mid
  n'est pas mesuré.
- **Les ordres cachés et iceberg.** Le flux L2 ne montre que la taille affichée du carnet ;
  les ordres cachés ou iceberg sont par construction absents de la donnée, donc de l'OFI.
- **Les annulations ne sont pas distinguées des exécutions.** Le flux Bybit L2 est un flux
  de *modifications* : une baisse de taille au meilleur prix inchangé est lue comme un
  retrait (`−q_b`/`−q_a`) et un franchissement de prix comme une exécution agressive, mais
  le flux ne porte **aucun drapeau d'exécution**. Un cancel et un fill de même taille au
  même niveau produisent la même contribution. L'OFI est donc un flux *signé net*, pas un
  flux *d'agression identifiée*.
- **La séquence intra-seconde est écrasée.** OFI(t) est une **somme** sur la seconde : elle
  est insensible à l'ordre des événements à l'intérieur de la seconde. Le modèle voit un
  total, pas une trajectoire de flux.
- **Aucune distinction taille/granularité.** Un ordre de 10 et dix ordres de 1 contribuent
  identiquement. Le nombre d'ordres et leur taille individuelle ne sont pas des features.

**Limites de portée, héritées de la phase 1** :

- **27 journées effectives**, pas 2,3 M d'observations ; échantillonnage en peigne (44
  journées sur ~947), donc couverture de régimes et non série continue ; **2023 n'est jamais
  testé** (il reste dans le train initial).
- Stratégie **taker naïve** (signe, sans sizing, sans filtre, sans maker) ; frais =
  hypothèses (2 / 5,5 bp) ; top-10 niveaux pour l'état.
- **Le R²_OOS est une métrique d'erreur quadratique** : un prédicteur directionnellement
  informatif mais bruité y score mal. Le verdict `apport` porte sur le **modèle linéaire**
  (`LinearWM`), seul bras exporté pour le test apparié ; un modèle non linéaire n'a pas été
  testé pour cette règle.
- **Aucune figure pour le bras enrichi** : les figures de `reports/figures/` sont celles du
  bras de base. Ce rapport n'en présente aucune, et n'en référence aucune.

## 7. Reproduire

```powershell
# 1. OFI evenementiel par (symbole, date) -> data/raw/crypto_ofi/  (220 fichiers .pkl)
#    Les dates sont lues dans le pre-enregistrement, pas passees en ligne de commande.
.\.venv\Scripts\python.exe scripts\crypto\fetch_ofi.py

# 2. Bras enrichi : meme walk-forward, etat = base + ofi en derniere position
#    (refuse d'ecrire dans experiments/ : le bras de base publie ne doit pas etre touche)
.\.venv\Scripts\python.exe scripts\crypto\crypto_lob.py --out experiments_ofi --state ofi

# 3. IC95 par jour du bras enrichi + verdicts pre-enregistres
.\.venv\Scripts\python.exe scripts\crypto\bootstrap_signif.py --oos-dir experiments_ofi

# 4. Comparaison APPARIEE base vs enrichi (delta R2, delta net, dimension ofi)
#    suppose que experiments/ (bras de base) existe deja
.\.venv\Scripts\python.exe scripts\crypto\paired_ofi.py

# 5. Causalite de l'OFI
.\.venv\Scripts\python.exe -m pytest -q tests/test_ofi_causal.py
```

Sorties du bras enrichi : `experiments_ofi/` - `crypto_lob_1step.csv`,
`crypto_lob_bootstrap.csv`, `crypto_lob_economic.csv`, `crypto_lob_rollout.csv`,
`crypto_lob_nsample.json`, les `.npz` de prédictions OOS, `crypto_ofi_paired.csv` et les
logs `paired_log.txt` / `bootstrap_log.txt`.
