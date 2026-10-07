# Phase 2 - un agent qui planifie dans le world model

> **ERRATUM - 2026-10-08. Le verdict de ce rapport est SUSPENDU.**
>
> Le planificateur de la Phase 2 (`plan_by_run` sur la série des prédictions à 1 pas) lit, à
> la barre t, les prédictions `rhat[t+1..t+H-1]`. Or `make_supervised` aligne
> X[k] = états k..k+L-1 et Y[k] = état k+L : `rhat[t+1]` est donc calculée sur une fenêtre
> qui **contient Y[t]**, c'est-à-dire le rendement que la position `p_t` encaisse. Un
> planificateur honnête à t ne dispose que des prévisions **faites à t** des rendements
> `r_t .. r_{t+H-1}`. Contrôle en lecture seule sur l'artefact publié, à 2 bp et H = 10 :
> `corr(rhat[t+1], y[t]) = +0.2680` (DOGE) et `+0.3951` (BTC), contre `+0.1455` et `+0.2285`
> pour la corrélation légitime. L'écart entre les deux est le rendement de la fuite.
>
> **Ce qui est suspendu** : `edge_reel` (OUI sur 5/5 ci-dessous) et `agent_bat_le_myope`
> (OUI sur 5/5). Le rapport reste publié tel quel - il documente exactement ce qui a été
> mesuré, et le reste de son contenu est intact - mais **aucune de ces deux conclusions
> positives ne doit être citée sans cette réserve**.
>
> **Ce qui n'est pas touché** : le bras myope, le clairvoyant, les Phases 0 à 1e, et l'agent
> **H = 1** de la Phase 2, où `F = 0` et donc où il n'y a rien à lire dans le futur (IC95
> contenant 0 sur 5/5). Les trois diagnostics de confondant gardent leur **méthode** ; mais
> celui sur le bruit ne pouvait pas voir la fuite, la permutation intra-journée détruisant la
> fuite en même temps que le signal.
>
> Re-mesure **pré-enregistrée avant tout calcul** : `configs/phase2b_crypto_prereg.yaml`,
> avec un planificateur causal par rollout (`plan_positions_causal`). Le résultat y sera
> écrit, quel qu'il soit.

**Verdict publié le 2026-10-07 (suspendu depuis) : premier résultat positif du projet, et il
survit aux trois contrôles de
confondant - mais il faut dire précisément ce qu'il est.** Un planificateur exact, qui
choisit ses positions en maximisant la récompense **imaginée** par le world model appris et
qui **paie le coût dans son plan**, dégage un net **réel** positif à 2 bp sur **5/5**
symboles (`edge_reel` : OUI) et bat le bras publié sur **5/5** (`agent_bat_le_myope` : OUI).
Ce n'est ni de la dérive directionnelle, ni un artefact de coût : un agent nourri de bruit,
avec la même conscience du coût et les mêmes coûts, **perd** de l'argent sur 5/5.

Ce qu'il n'est pas : **l'agent ne prédit pas mieux que le bras publié.** Son brut
directionnel est **inférieur** à celui du myope sur 5/5. Tout son avantage sur le myope vient
du coût qu'il ne paie plus. Et à H = 1, il ne reste rien.

Protocole figé et commité avant toute exécution : `configs/phase2_crypto_prereg.yaml`.

## 1. La question

La spec du projet définit le Stage 2 comme « un agent basé-modèle qui planifie **dans** le
world model, cherche un edge, et qu'on valide sur données réelles tenues à part pour
distinguer un **vrai edge** d'une **model exploitation** ». Les phases 1 à 1e ont établi
qu'une règle myope - position = signe de la prédiction, toujours en position, ignorante du
coût - perd de l'argent dès 2 bp partout.

La question restée ouverte est donc : une politique qui **planifie plusieurs pas** dans le
modèle, et qui **tient compte du coût**, fait-elle mieux ? Et si elle gagne dans le modèle,
gagne-t-elle dans le réel ?

C'est la question du mirage posée une dernière fois, au niveau de l'agent.

## 2. Le protocole, en une page

**Le monde appris est inchangé** par rapport aux phases 1/1c/1d/1e : régression Ridge sur
l'état `[ret, spread_rel, imb1, depth_imb, micro_dev]`, lookback 16, cible = rendement du mid
à +1 s. Mêmes 44 journées, 5 symboles, mêmes folds (walk-forward purgé, 5 folds, embargo 16,
`min_train_frac` 0.4). L'agent ne doit pas gagner parce qu'on aurait amélioré le modèle en
même temps.

**L'agent est un planificateur exact, pas une politique apprise.** C'est un choix de méthode,
pas une facilité : tout échec doit être imputable au **monde appris**, jamais à la variance
d'un optimiseur. `mirage.plan.plan_positions` résout exactement

```
maximiser   Σ_t p_t · r̂_t  −  Σ_t |p_t − p_{t−1}| · c_t ,     p_t ∈ {−1, 0, +1}
```

où `r̂` est le rendement **prédit** et `c_t` le demi-spread **réellement mesuré** à la barre de
décision, plus les frais. On planifie H = 10 barres, on exécute la première, on avance d'une
barre, on replanifie ; la position de départ d'un plan est la position **réellement exécutée**
au pas précédent. Un plan ne traverse jamais une frontière de journée, et chaque journée
repart à plat - la convention de coût de `mirage.backtest.position_changes`.

**La mesure centrale.** L'agent choisit ses positions en n'optimisant **que** la récompense
imaginée. On rejoue ensuite **ces mêmes positions** sur les rendements vrais, avec les mêmes
coûts. L'écart `net_imaginé − net_réel` est la model exploitation, quantifiée.

**Références.** `myope` (le bras publié, à battre), `plat` (net = 0 exactement), `oracle`
(`p = signe(r)` vrai), et `clairvoyant` : **le même planificateur DP**, nourri des rendements
vrais. Le clairvoyant est la borne utile - l'oracle, lui, retourne sa position à chaque barre
et le coût mange sa prévoyance parfaite.

## 3. Contrôles

**Contrôle d'intégrité.** Le bras `myope` de cette phase reproduit le bras publié. Vérifié au
niveau du bit par `scripts/crypto/agent_plan.py --check` (`max|diff| = 0.000e+00` sur `y`,
`pred`, `day`, `R2_OOS(ret) = 0.01584620`), et re-vérifié ici au niveau du net journalier :
le diagnostic recalcule `−0.986484 bp` là où le harnais écrit `−0.986484 bp`, écart `1.1e−16`.
La chaîne de données de la phase 2 est donc appariable à celle des phases 1/1c/1d/1e.

**Contrôle anti-bug.** Le net **imaginé** de l'agent doit être ≥ le net imaginé du myope : le
DP maximise exactement cette quantité, une violation serait un bug d'implémentation, jamais
un résultat.

| Symbole | agent imaginé | myope imaginé | |
|---|---|---|---|
| BTCUSDT | +0.0032 | −0.9921 | |
| DOGEUSDT | +0.0342 | −1.4369 | |
| ETHUSDT | +0.0051 | −1.2759 | |
| SOLUSDT | +0.0124 | −1.5092 | |
| XRPUSDT | +0.0167 | −1.5004 | |

**→ 5/5.**

## 4. La mesure centrale - net imaginé vs net réel

*(bp par barre, 27 journées réellement en test, 2,28 M barres par symbole, frais 2 bp)*

| Symbole | net imaginé | **net réel** | IC95 (réel) | écart d'exploitation | IC95 (écart) |
|---|---|---|---|---|---|
| BTCUSDT | +0.0032 | **+0.0213** | [+0.0131, +0.0318] | −0.0181 | [−0.0278, −0.0104] |
| DOGEUSDT | +0.0342 | **+0.0905** | [+0.0623, +0.1239] | −0.0563 | [−0.0864, −0.0308] |
| ETHUSDT | +0.0051 | **+0.0418** | [+0.0325, +0.0526] | −0.0367 | [−0.0466, −0.0281] |
| SOLUSDT | +0.0124 | **+0.0553** | [+0.0352, +0.0837] | −0.0429 | [−0.0701, −0.0235] |
| XRPUSDT | +0.0167 | **+0.0551** | [+0.0345, +0.0784] | −0.0384 | [−0.0617, −0.0192] |

**Les deux faits contre-intuitifs de cette table :**

1. **Le net réel de l'agent est POSITIF partout**, cinq fois, avec un IC95 dont la borne basse
   est largement au-dessus de zéro - là où le bras publié perd 1 à 1,5 bp par barre.
2. **L'écart d'exploitation est NÉGATIF et significatif sur 5/5.** Le modèle **sous-promet** :
   les positions que l'agent choisit rapportent *plus* que ce que le modèle leur promettait.
   Ce n'est pas de l'exploitation du modèle, c'est l'inverse. Lecture la plus simple : le
   Ridge est **contracté** (il sous-estime l'amplitude - c'est le prix de la régularisation),
   donc l'agent est sous-confiant, pas sur-confiant.

À comparer au myope, dont l'écart d'exploitation est **positif** : il suit `signe(r̂)` sans
regarder le coût, sur-promet, et perd. Les deux bras ont des écarts de signes **opposés**, ce
qui est un bon signe de cohérence interne : chacun se trompe dans le sens de sa propre règle.

## 5. Net réel par bras et par frais

*(bp par barre. Un agent « @f » est un agent qui a **planifié** dans le régime de frais f :
ses positions sont donc différentes d'un régime à l'autre, comme le pré-enregistrement
l'exige.)*

| Symbole | plat | myope @2 | agent @0 | agent @2 | agent @5,5 | clairvoyant @2 | oracle @2 |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 0 | −0.9865 | +0.1652 | **+0.0213** | +0.0025 | +0.1195 | −0.4770 |
| DOGEUSDT | 0 | −1.5051 | +0.2382 | **+0.0905** | +0.0099 | +0.3933 | −0.5347 |
| ETHUSDT | 0 | −1.2535 | +0.2303 | **+0.0418** | +0.0026 | +0.2098 | −0.5791 |
| SOLUSDT | 0 | −1.5135 | +0.2415 | **+0.0553** | +0.0043 | +0.3077 | −0.6461 |
| XRPUSDT | 0 | −1.5657 | +0.1662 | **+0.0551** | +0.0041 | +0.2491 | −0.5325 |

**L'oracle perd de l'argent** (−0.48 à −0.65 bp par barre) : `signe(r_true)` retourne la
position à chaque barre et le coût mange la prévoyance parfaite. La « prévoyance parfaite »
n'est donc **pas** une borne supérieure de net - c'est le **clairvoyant** qui l'est, parce
qu'il est conscient du coût. Il faut l'écrire noir sur blanc : dans ce dispositif, **savoir
quand ne pas trader vaut plus qu'avoir raison à chaque barre**.

Le clairvoyant réalise +0.12 à +0.39 ; l'agent en capte **18 à 23 %**. La marge au-dessus de
`plat` (= 0 exactement) est donc réelle mais modeste.

## 6. Robustesse par horizon

*(diagnostics, jamais une grille de réglage : H = 10 est l'horizon primaire et il le reste)*

| Symbole | H = 1 | H = 5 | **H = 10** | H = 20 |
|---|---|---|---|---|
| BTCUSDT | +0.0003 [−0.0002, +0.0014] | +0.0067 | **+0.0213** | +0.0449 |
| DOGEUSDT | −0.0001 [−0.0026, +0.0026] | +0.0429 | **+0.0905** | +0.1193 |
| ETHUSDT | +0.0011 [−0.0002, +0.0027] | +0.0105 | **+0.0418** | +0.0765 |
| SOLUSDT | +0.0012 [−0.0001, +0.0033] | +0.0147 | **+0.0553** | +0.0891 |
| XRPUSDT | +0.0005 [−0.0006, +0.0020] | +0.0204 | **+0.0551** | +0.0793 |

**Le résultat croît avec l'horizon, et à H = 1 il n'y a rien.** Sur les cinq symboles, l'IC95
de H = 1 contient zéro - trois bornes basses sont même négatives. Autrement dit : le gain ne
vient pas d'une meilleure lecture de la barre suivante, il vient de tenir une position sur
plusieurs secondes. C'est cohérent avec le mécanisme identifié en §9 (éviter le churn), et
c'est une limite à assumer : **le résultat dépend fortement de H**, et H est un paramètre
pré-enregistré, pas un fait de marché.

## 7. Verdicts - appliqués mécaniquement

| Règle (pré-enregistrée) | Résultat | Verdict |
|---|---|---|
| `planificateur_coherent` (contrôle anti-bug) | 5/5 | |
| `exploitation_du_modele` | 0/5 | aucun symbole ne gagne dans le modèle en perdant dans le réel |
| `agent_bat_le_myope` (≥ 4/5) | **5/5** | **OUI** |
| `edge_reel` (≥ 4/5) | **5/5** | **OUI** |

## 8. Les trois confondants - ce que le rapport a le droit d'affirmer

Les règles figées donnent OUI sur `edge_reel`. Un OUI mécanique ne suffit pas : trois
confondants pouvaient produire un faux positif. Ils ont été instrumentés **après** le run,
sur l'artefact publié lui-même (`experiments/crypto_lob_oos_<SYM>.npz`, qui contient `pred`,
`y`, `half` et `day`), sans ré-entraînement ni ré-implémentation - et **sans toucher aux
règles**. Script : `scripts/crypto/agent_diag.py`.

### 8.1 Le gain est-il le coût ou le signal ?

*Instrument : rejouer le **même** planificateur sur une prédiction **bruitée** (r̂ permuté
dans le temps à l'intérieur de chaque journée) avec les **mêmes** coûts, 39 fois par symbole.*

| Symbole | agent réel (observé) | agent nourri de bruit : moyenne | IC95 | **maximum sur 39 tirages** | p |
|---|---|---|---|---|---|
| BTCUSDT | **+0.0213** | −0.0023 | [−0.0037, −0.0013] | −0.0008 | ≤ 0.025 |
| DOGEUSDT | **+0.0905** | −0.0437 | [−0.0459, −0.0420] | −0.0403 | ≤ 0.025 |
| ETHUSDT | **+0.0418** | −0.0102 | [−0.0118, −0.0089] | −0.0086 | ≤ 0.025 |
| SOLUSDT | **+0.0553** | −0.0206 | [−0.0231, −0.0186] | −0.0177 | ≤ 0.025 |
| XRPUSDT | **+0.0551** | −0.0289 | [−0.0306, −0.0271] | −0.0263 | ≤ 0.025 |

**L'agent nourri de bruit perd de l'argent sur 5/5, et le net observé dépasse le maximum des
39 tirages sur 5/5.** La borne conservatrice - valable même si les cinq symboles étaient
parfaitement dépendants - est p ≈ 1/40 = 0.025. Le gain **n'est pas** un artefact mécanique
de coût : un planificateur également économe, mais nourri de bruit, ne gagne pas.

Deux précisions d'honnêteté :

- Le turnover de l'agent de bruit est **plus faible** que celui du vrai agent (0.0012 contre
  0.0049 sur BTC ; 0.0193 contre 0.0311 sur DOGE). Il paie donc **moins** de coûts que
  l'agent réel : le test est indulgent dans le sens qui aurait pu produire un faux positif,
  et il reste négatif partout.
- Son net **imaginé** est négatif lui aussi (−0.0023 à −0.0420). Le modèle nourri de bruit ne
  promet rien. Le net imaginé n'est donc pas structurellement positif : il redevient positif
  quand - et seulement quand - l'entrée porte du signal.

### 8.2 Confondant de dérive directionnelle

*Instrument : la décomposition **exacte** `Σ p_t·r_t = p̄·Σ r_t + Σ (p_t − p̄)·r_t`, plus un
bras achat-et-conservation, plus le rejeu des **mêmes** positions sur des rendements décalés
**circulairement** dans chaque journée - un décalage circulaire est une permutation du jour,
il **préserve donc exactement la dérive** et ne détruit que l'alignement.*

| Symbole | position moyenne | dérive de la fenêtre | contribution **dérive** | IC95 | contribution **timing** | IC95 | achat-conservation |
|---|---|---|---|---|---|---|---|
| BTCUSDT | −0.0271 | +520 bp | **−0.0000** | [−0.0000, +0.0000] | **+0.0310** | [+0.0222, +0.0420] | +0.0002 |
| DOGEUSDT | −0.0769 | +1192 bp | **−0.0000** | [−0.0002, +0.0001] | **+0.1622** | [+0.1298, +0.1997] | +0.0005 |
| ETHUSDT | −0.0479 | +2018 bp | **−0.0000** | [−0.0002, +0.0000] | **+0.0649** | [+0.0542, +0.0766] | +0.0009 |
| SOLUSDT | +0.0250 | +481 bp | **+0.0000** | [−0.0000, +0.0001] | **+0.0918** | [+0.0699, +0.1210] | +0.0002 |
| XRPUSDT | −0.0398 | −405 bp | **+0.0000** | [−0.0001, +0.0001] | **+0.1014** | [+0.0743, +0.1301] | −0.0002 |

**La contribution de dérive est nulle à la quatrième décimale sur 5/5, l'intervalle est
[0, 0], et toute la marge est dans le terme de timing.** Le mécanisme est visible : la
position moyenne de l'agent vaut entre −0.08 et +0.03, c'est-à-dire **quasi nulle** - l'agent
est long 48 % du temps et short 51 %, il n'est pas exposé à la dérive de l'actif. Et le
décalage circulaire le confirme : sur **400** décalages aléatoires par symbole (dérive
préservée à l'identique), le brut observé dépasse les 400 sur 5/5 (percentile 100 %), la
moyenne des décalages valant +0.0001/−0.0000/+0.0002/+0.0001/−0.0000 bp.

Le bras **achat-et-conservation** rapporte lui aussi ≈ 0 (+0.0009 au mieux sur ETH) : la
dérive de la fenêtre, étalée sur 2,28 M barres, est négligeable à l'échelle des nets mesurés.
**Le confondant de dérive directionnelle est écarté, exactement et non par argument.**

### 8.3 Ordre de grandeur, turnover et part de l'extractible

| Symbole | changer/barre agent | changer/barre myope | changements **par jour** agent | myope | coût bp/barre agent | myope |
|---|---|---|---|---|---|---|
| BTCUSDT | 0.0049 | 0.5730 | 410 | 48 400 | 0.0098 | 1.1504 |
| DOGEUSDT | 0.0311 | 0.7671 | 2 629 | 64 736 | 0.0717 | 1.7756 |
| ETHUSDT | 0.0114 | 0.7348 | 966 | 62 066 | 0.0231 | 1.4844 |
| SOLUSDT | 0.0159 | 0.7747 | 1 345 | 65 435 | 0.0365 | 1.7840 |
| XRPUSDT | 0.0199 | 0.7166 | 1 677 | 60 514 | 0.0463 | 1.7880 |

Part de temps en position : long 0.46–0.51, short 0.49–0.54, **plat 0.00–0.01**. L'agent
n'est donc **pas** un agent qui reste dehors : il est en position **99 % du temps**. Ce qu'il
ne fait pas, c'est **changer** - 410 à 2 629 changements par jour contre 48 000 à 65 000 pour
le myope, soit **20 à 120 fois moins**.

**La décomposition qui tranche.** `net(agent) − net(myope) = Δbrut + coût économisé` :

| Symbole | Δnet | IC95 | **Δbrut (directionnel)** | **coût économisé** |
|---|---|---|---|---|
| BTCUSDT | +1.0078 | [+0.9244, +1.0949] | **−0.1329** | +1.1406 |
| DOGEUSDT | +1.5956 | [+1.4429, +1.7938] | **−0.1084** | +1.7039 |
| ETHUSDT | +1.2953 | [+1.2174, +1.3739] | **−0.1660** | +1.4613 |
| SOLUSDT | +1.5688 | [+1.4642, +1.6712] | **−0.1787** | +1.7476 |
| XRPUSDT | +1.6208 | [+1.4617, +1.8299] | **−0.1208** | +1.7417 |

**Le brut directionnel de l'agent est INFÉRIEUR à celui du myope sur 5/5.** Tout l'avantage
sur le bras publié - et au-delà - vient du coût qui n'est plus payé. Ce n'est pas une
contradiction avec le §8.1 : les deux instruments ne répondent pas à la même question.

- Face au **myope** : « l'agent lit-il mieux le marché ? » **Non.** Il lit un peu moins bien,
  et il gagne parce qu'il ne paie plus 1,15 à 1,79 bp par barre de demi-spread.
- Face au **bruit** : « le net positif de l'agent peut-il s'expliquer sans signal ? »
  **Non.** Un planificateur aussi économe nourri de bruit perd de l'argent.

Les deux tiennent ensemble parce que le myope est dans un régime absurde : son net brut est
positif (+0.16 à +0.27 bp) mais il paie 1,15 à 1,79 bp pour l'obtenir. L'agent, en
n'acceptant une position que si le gain espéré couvre le coût, laisse filer une partie du
brut - et garde tout le reste.

## 9. Ce que l'agent ne fait pas

- **Pas d'impact de ses propres ordres** : il est un petit trader sans impact sur le marché,
  même convention que les phases 1/1c/1d/1e. Le coût de ses propres ordres est un autre
  chantier, chiffré en phase 1b (**1,7–3 bp** pour le plus petit ordre).
- **Pas de maker / post-only** : il paie le demi-spread au marché ; le risque de
  non-exécution d'un ordre passif n'est pas modélisé.
- **Pas de politique apprise** : le planificateur est exact, donc le verdict est imputable au
  monde appris et à rien d'autre. Un policy gradient ajouterait une instabilité qui rendrait
  le résultat inattribuable.
- **Pas de meilleur modèle** : à aucun moment le monde appris n'a été modifié pour cette
  phase. C'est ce qui permet de dire que le gain vient de la **planification**, pas d'une
  meilleure prévision - et c'est cohérent avec le fait que son brut directionnel soit
  *inférieur* à celui du myope.
- **Rien n'a été réglé pour obtenir ce résultat** : H = 10, 2 bp, ≥ 4/5 et les cinq
  définitions de règles étaient figés dans `configs/phase2_crypto_prereg.yaml` avant la
  première ligne de code. Le run n'a pas été relancé avec d'autres paramètres.

## 10. Limites

- **Les niveaux ne sont pas du P&L crédible.** Tous les nets sont exprimés en bp **par barre
  de 1 s**, la convention des phases 1 à 1e, et cumulés sur 2,28 M barres ils donnent des
  ordres de grandeur qui n'ont pas de sens économique (le myope « perd » −1,5 bp par barre).
  Ce qui porte l'information est le **signe**, la **comparaison appariée** et la position
  relative aux références et aux nuls - pas le niveau.
- **Le résultat dépend fortement de H.** À H = 1 l'IC95 contient zéro sur 5/5 ; à H = 20 le
  net est 2 à 3 fois celui de H = 10. H = 10 est pré-enregistré, donc ce n'est pas un
  ajustement - mais un résultat qui dépend d'un horizon n'est pas un fait de marché.
- **La marge est faible en absolu** : +0.02 à +0.09 bp par barre contre +0.12 à +0.39 pour le
  clairvoyant. L'agent capte 18–23 % de l'extractible. À 5,5 bp de frais il ne reste que
  +0.0026 à +0.0099 bp - positif, mais à la limite du mesurable.
- **L'agent n'est pas un meilleur prévisionniste.** Son brut directionnel est inférieur à
  celui du myope sur 5/5 (§8.3). Le résultat est un résultat d'**exécution** : refuser de
  payer le spread quand le gain espéré ne le couvre pas. Il ne faut pas le lire comme « le
  world model prédit le prix ».
- **L'écart d'exploitation négatif a une explication plausible mais non démontrée** : la
  contraction Ridge. Elle n'a pas été testée ici (une variante non régularisée aurait été un
  nouveau réglage).
- **27 journées OOS** (2024-02-06 → 2025-08-06), 5 symboles, échantillonnage en peigne. IC
  larges, couverture de régimes bornée par construction. Les diagnostics de confondant
  tournent sur les mêmes 27 journées : ils écartent des mécanismes, ils n'élargissent pas
  l'échantillon.
- **Cible `ret` uniquement, 1 s.** Un horizon plus long ou une autre cible n'ont pas été
  testés.

## 11. Reproduire

```powershell
# Le run complet (5 symboles, ~10 min) : entrainement, planification DP, CSV + npz
.\.venv\Scripts\python.exe scripts\crypto\agent_plan.py --out experiments_agent
# Controle d'integrite seul : le bras myope doit reproduire le bras publie au bit pres
.\.venv\Scripts\python.exe scripts\crypto\agent_plan.py --symbols DOGEUSDT --check

# Les trois diagnostics de confondant (~9 min par symbole ; les 5 en parallele)
.\.venv\Scripts\python.exe scripts\crypto\agent_diag.py --symbols BTCUSDT --perm 39 --shift 400
.\.venv\Scripts\python.exe scripts\crypto\agent_diag.py --perm 39 --shift 400
```

Sorties du harnais (`experiments_agent/`, gitignored) : `agent_net.csv`,
`agent_compare.csv`, `agent_exploit.csv`, `agent_oos_<SYM>.npz`.
Sorties des diagnostics : `diag_ordre.csv`, `diag_confondants.csv`, `diag_vs_myope.csv`
(+ un fichier par symbole, pour permettre de lancer les symboles en processus séparés).
Tous les chiffres de ce rapport proviennent de ces fichiers, sans sélection.
