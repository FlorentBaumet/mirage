# Rapports

Chaque rapport suit la même structure : **Question, Données, Protocole, Résultats,
Interprétation, Limites, Reproduire**. Les protocoles correspondants sont figés dans
[`../configs/`](../configs) avant tout entraînement, et les figures citées sont dans
[`figures/`](figures).

## Par où commencer

Si tu ne lis qu'un document, lis le dernier.

| # | Rapport | Données | Ce qu'il établit |
|---|---|---|---|
| 1 | [`PHASE0.md`](PHASE0.md) | 5 actions LOBSTER, 1 journée, barres 1 s | Le rendement du mid n'est pas prédictible (**NO-GO**, R²_OOS ≤ 0 de 1 à 60 s). Le *signe*, lui, l'est sur les large-tick (80–83 % à 1 s) - mais pour un gain des dizaines de fois sous le demi-spread. Un edge statistique qui est un mirage net de coûts. |
| 2 | [`PHASE0_CRYPTO.md`](PHASE0_CRYPTO.md) | 4 cryptos, 1 an de klines 1 min | La question rejouée sur des mois : random walk à 1 min. Le signal sous-seconde a disparu avec la résolution. |
| 3 | [`PHASE1.md`](PHASE1.md) | LOBSTER | Prédire un **vecteur d'état** (5 dims) plutôt qu'un scalaire. Le prix reste un random walk ; la forme du carnet est légèrement prévisible linéairement. |
| 4 | [`PHASE1B.md`](PHASE1B.md) | LOBSTER | Action-conditionner le world model : impact de ses propres ordres. Le plus petit ordre coûte **1.7–3 bp** contre un edge prédictible **≤ 0.1 bp**. |
| 5 | [`PHASE1_CRYPTO.md`](PHASE1_CRYPTO.md) | 5 symboles × 44 journées, carnet Bybit L2, 19 M barres 1 s (11,4 M en test) | **Le cas d'école.** Le signal existe vraiment (R²_OOS 0.016–0.052, IC95 > 0 sur **5/5** symboles et positif sur 23 folds sur 25) et n'est toujours pas un edge : négatif dès 2 bp de frais avec un IC95 entièrement sous zéro, et déjà **négatif sans frais sur XRP**. |

## Le fil

La Phase 0 montre un cas où « pas d'edge » vient surtout de « pas de signal ». La Phase 1
crypto construit le cas inverse, celui qui compte : un signal **réel**, mesuré, robuste -
et une évaluation honnête qui refuse quand même de l'appeler un edge. Entre les deux, la
Phase 1b chiffre la raison de fond : le coût d'exécution.

Les rapports sont écrits pour être lus dans l'ordre ci-dessus, mais chacun tient seul.
