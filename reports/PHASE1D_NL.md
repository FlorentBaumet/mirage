# Phase 1d - l'OFI est-il exploitable par un modèle non linéaire ?

**Verdict : non. `0/5` symboles.** À état strictement identique, un MLP n'extrait pas de
l'OFI événementiel l'information que le modèle linéaire manquait. La lecture « l'information
est déjà contenue dans l'état » sort **renforcée** de ce test : elle vaut maintenant pour
deux classes de modèles, pas une seule.

Protocole figé et commité avant toute exécution : `configs/phase1d_crypto_prereg.yaml`.

## 1. La question, et pourquoi elle était ouverte

La phase 1c ([`PHASE1C_OFI.md`](PHASE1C_OFI.md)) a mesuré l'apport de l'OFI événementiel
**sur le seul modèle linéaire**, comme son pré-enregistrement l'exigeait. Son verdict
(`APPORT NON ÉTABLI`, 0/5) laissait explicitement **deux lectures non tranchées** :

- soit l'information intra-seconde est **déjà contenue** dans l'état de barres ;
- soit elle est **réelle mais hors de portée** du modèle linéaire.

Ces deux lectures prédisent la même chose sous `LinearWM`. Elles divergent sous un modèle
capable de combinaisons non linéaires. C'est cette divergence qu'on teste.

**Une seule variable change** : la classe du modèle. Les données, l'état, les folds, la
cible, les frais et le dispositif de bootstrap sont identiques à la phase 1c, à l'octet -
ce sont les mêmes fichiers sur disque.

## 2. Contrôle de cohérence avec la phase 1c

Le bras linéaire de cette phase doit reproduire la phase 1c. Il la reproduit **exactement** :

| Symbole | ΔR² 1d, linéaire (`ofi − base`) | Publié en 1c |
|---|---|---|
| BTCUSDT | −0.00046 | |
| ETHUSDT | −0.00061 | |
| SOLUSDT | +0.00004 | |
| XRPUSDT | −0.00036 | |
| DOGEUSDT | −0.00062 | |

L'intervalle `[−0.00062, +0.00004]` est celui publié par la phase 1c, au chiffre près. Le
dispositif n'a donc pas dérivé, et les chiffres ci-dessous sont comparables à ceux de la 1c.

Contrôle d'intégrité complémentaire, sur DOGEUSDT : le bras `base`+`linear` reproduit
`experiments/crypto_lob_oos_DOGEUSDT.npz` avec `pred` **bit à bit identique**
(`max|diff| = 0.000e+00`) et `R2_OOS(ret) = 0.01584620` exact.

## 3. Le résultat - ΔR²_OOS(ret) apparié, MLP et linéaire côte à côte

Bootstrap apparié, B = 2000, graine 0, unité = la journée (27 journées OOS).
**Une seule série de journées tirée par réplique sert aux deux bras** ; la statistique est
leur différence. C'est la comparaison qui répond à la question.

| Symbole | **MLP** : `ofi − base` | IC95 | **Linéaire** : `ofi − base` | IC95 |
|---|---|---|---|---|
| BTCUSDT | −0.00197 | [−0.00591, 0.00165] | −0.00046 | [−0.00163, 0.00056] |
| ETHUSDT | −0.00110 | [−0.00366, 0.00155] | −0.00061 | [−0.00117, −0.00004] |
| SOLUSDT | −0.00470 | [−0.00733, −0.00065] | +0.00004 | [−0.00020, 0.00019] |
| XRPUSDT | −0.02981 | [−0.04143, −0.01477] | −0.00036 | [−0.00092, 0.00044] |
| DOGEUSDT | −0.06019 | [−0.09662, 0.00663] | −0.00062 | [−0.00145, 0.00014] |

**Aucune borne basse n'est strictement positive : 0/5.** Il n'y a pas de symbole où le MLP
tirerait de l'OFI un gain que le linéaire manque.

Et le résultat n'est pas « neutre ». Sur **SOL et XRP, l'IC95 est entièrement négatif** :
ajouter l'OFI **dégrade significativement** la prévision. Sur DOGE, le point estimate est
franchement négatif (−0.060) - le R²_OOS(ret) du MLP passe de **+0.00478 à −0.05540**, soit
une prévision devenue *pire que la marche aléatoire*. L'intervalle est large, donc on ne
conclut pas sur DOGE seule, mais la direction est cohérente avec SOL et XRP : le MLP ne
trouve pas d'information dans l'OFI, il y trouve du bruit qu'il sur-ajuste.

Robustesse (blocs contigus de 3 jours) : les intervalles s'élargissent, les conclusions ne
changent pas - `ofi_mlp − base_mlp` reste négatif partout, significativement sur SOL
([−0.00720, −0.00043]).

### La dimension `ofi` est très prévisible - et toujours inutile

| Symbole | R² 1-step de la dimension `ofi` (linéaire) | R² 1-step (MLP) |
|---|---|---|
| BTCUSDT | 0.40351 | 0.39467 |
| ETHUSDT | 0.41896 | 0.40398 |
| SOLUSDT | 0.44076 | 0.38847 |
| XRPUSDT | 0.47034 | 0.46618 |
| DOGEUSDT | 0.44481 | 0.44132 |

*(moyenne sur les folds)*

L'OFI est l'une des dimensions **les plus prévisibles** de tout l'état : R² ≈ 0.39–0.47.
Ce n'est pas une variable bruitée ou mal construite. Elle est simplement **sans pouvoir
prédictif sur le rendement** - ce que la phase 1c avait déjà montré, et que ce test confirme
sous une seconde classe de modèles.

## 4. Effet de la classe de modèle seule (question secondaire)

À état de base, le MLP bat-il le linéaire ? **Réponse : non, et le signe est mixte.**

| Symbole | `base_mlp − base_linear` | IC95 |
|---|---|---|
| BTCUSDT | **+0.00621** | [0.00332, 0.00902] |
| ETHUSDT | −0.00200 | [−0.00825, 0.00287] |
| SOLUSDT | −0.00478 | [−0.00943, 0.00119] |
| XRPUSDT | +0.00799 | [−0.00222, 0.02078] |
| DOGEUSDT | **−0.01106** | [−0.01486, −0.00376] |

Une seule borne basse strictement positive (BTC, 1/5) → **DIFFÉRENCE DE CLASSE NON ÉTABLIE**,
règle `modele_gagne_t-il` appliquée mécaniquement.

Le détail est instructif et va **contre** l'intuition naïve « le non linéaire est meilleur » :
sur BTC le MLP **gagne** significativement (+0.0062), sur DOGE il **perd** significativement
(−0.0111). Le non linéaire n'est pas un gain uniforme ; il paie en variance ce qu'il gagne
en souplesse. Ce point est rapporté ici **séparément**, conformément au pré-enregistrement :
il ne se confond pas avec l'apport de l'OFI.

## 5. Tableau économique - net à 2 bp par bras et par modèle

Position = signe de la prédiction OOS. Coût = demi-spread **réellement mesuré** + frais,
payé au changement de position, remise à plat à chaque frontière de journée.

| Symbole | base lin. | base MLP | ofi lin. | **ofi MLP** | IC95 (ofi MLP) |
|---|---|---|---|---|---|
| BTCUSDT | −0.9865 | −0.8957 | −0.9964 | **−0.8389** | [−0.926, −0.755] |
| ETHUSDT | −1.2535 | −1.1484 | −1.2554 | **−1.1907** | [−1.257, −1.129] |
| SOLUSDT | −1.5135 | −1.4602 | −1.5166 | **−1.4571** | [−1.526, −1.390] |
| XRPUSDT | −1.5657 | −1.4569 | −1.5597 | **−1.3945** | [−1.466, −1.328] |
| DOGEUSDT | −1.5051 | −1.3902 | −1.5072 | **−1.3969** | [−1.477, −1.318] |

*(bp par barre)*

Le MLP réduit la perte partout (−0.84 à −1.46 bp contre −0.99 à −1.57 bp) - mais **l'IC95 de
chaque bras est entièrement sous zéro**, sur les cinq symboles, pour les six combinaisons
bras × modèle. Le mirage ne se dissout pas en changeant de classe de modèle.

## 6. Verdicts - appliqués mécaniquement

| Règle (pré-enregistrée) | Résultat | Verdict |
|---|---|---|
| `apport_non_lineaire` (≥ 4/5) | **0/5** | **APPORT NON ÉTABLI** |
| `apport_non_lineaire_partiel` (1–3/5) | non atteint | - |
| `plancher_non_lineaire` | - | **APPLIQUÉ : la lecture « déjà contenue » est renforcée** |
| `modele_gagne_t-il` | 1/5 | DIFFÉRENCE DE CLASSE NON ÉTABLIE |
| `mirage` (bras ofi, MLP) | net < 0 partout à 2 bp | **MIRAGE CONFIRMÉ** |
| `bascule` | aucune borne basse > 0 | **AUCUNE BASCULE** |

Ce que ce résultat **démontre**, et c'est plus fort que la phase 1c : l'absence d'apport de
l'OFI n'est pas un artefact du modèle linéaire. Un MLP à deux couches cachées, entraîné sur
le même état apparié, ne trouve rien de plus - et sur deux symboles trouve activement du
bruit. La lecture « l'information intra-seconde au meilleur niveau est déjà contenue dans
les barres 1 s » tient maintenant pour deux classes de modèles.

## 7. Limites - ce que ce test ne prouve pas

- **Un MLP n'est pas n'importe quel modèle non linéaire.** Le résultat borne les modèles de
  cette famille et de cette capacité (2 couches, 64×32, `max_iter=300`, early stopping). Un
  GRU sur la fenêtre complète, ou un modèle à attention, n'est pas écarté - mais rien ici ne
  suggère qu'ils trouveraient autre chose, la dimension `ofi` étant déjà très prévisible
  linéairement (R² ≈ 0.44), donc peu susceptible de cacher une structure non linéaire forte.
- **Le test porte sur `ret`.** L'OFI pourrait aider à prévoir d'autres dimensions de l'état
  (le spread, la profondeur) sans aider sur le rendement. Non testé ici, et sans intérêt
  économique direct.
- **L'OFI reste au meilleur niveau.** Le flux des niveaux 2 à 500 n'est pas couvert - c'est
  l'objet de la phase 1e, et c'est la limite n°1 du rapport 1c.
- **Un résultat négatif ne dit rien de la direction opposée.** Il ne prouve pas qu'aucune
  information intra-seconde n'existe ; il prouve que *celle-ci*, sous *cette* forme, sur
  *ces* données, n'est pas exploitable pour le rendement à 1 s.
- **27 journées OOS** (2024-02-06 → 2025-08-06), 5 symboles, échantillonnage en peigne. La
  couverture de régimes est bornée par construction ; les IC sont larges en conséquence.

## 8. Reproduire

```powershell
# Les 5 symboles en un seul passage (le bras `base`+`linear` sert de contrôle d'intégrité)
.\.venv\Scripts\python.exe scripts\crypto\arm_eval.py --out experiments_arms
# Contrôle seul, sur un symbole : doit reproduire experiments/crypto_lob_oos_<SYM>.npz
.\.venv\Scripts\python.exe scripts\crypto\arm_eval.py --symbols DOGEUSDT --check
# Bootstrap apparié : ΔR² et net, IC95, verdicts mécaniques des deux pré-enregistrements
.\.venv\Scripts\python.exe scripts\crypto\paired_arms.py
```

Sorties : `experiments_arms/arms_1step.csv`, `paired_arms.csv`,
`paired_arms_net_by_arm.csv`, `paired_arms_log.txt`.
Tous les chiffres de ce rapport proviennent de `paired_arms_log.txt`, sans sélection.
