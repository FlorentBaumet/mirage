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
| 6 | [`PHASE1C_OFI.md`](PHASE1C_OFI.md) | idem PHASE1_CRYPTO, enrichi de l'OFI événementiel au meilleur niveau | **L'information en plus ne change rien.** Ajouter au world model le flux intra-seconde absent des barres 1 s (OFI événementiel) ne déplace pas la prévision du rendement (ΔR²_OOS(ret) apparié entre **−0.0006 et +0.0000**, apport non établi sur **0/5** au seuil ≥ 4/5) ni le verdict économique (net à 2 bp inchangé, mirage confirmé). Un résultat négatif publié tel quel. |
| 7 | [`PHASE1D_NL.md`](PHASE1D_NL.md) | idem PHASE1C, l'OFI re-testé sous **modèle non linéaire** (MLP), état identique | **Ce n'est pas un défaut du modèle.** La Phase 1c laissait ouvert : information « déjà contenue dans l'état » ou « réelle mais hors de portée du linéaire ». Un MLP tranche - il n'en tire rien (**0/5**) et la **dégrade** significativement sur SOL et XRP. Le contrôle linéaire reproduit la Phase 1c au chiffre près. La lecture « déjà contenue » vaut pour deux classes de modèles. |
| 8 | [`PHASE1E_PROFONDEUR.md`](PHASE1E_PROFONDEUR.md) | idem PHASE1_CRYPTO, + déséquilibre des niveaux 2–10 et asymétrie de pente du carnet | **Apport partiel, non promu.** La profondeur au-delà du meilleur niveau améliore significativement BTC et ETH mais **dégrade** XRP et DOGE (**2/5**, découverte exigée à ≥ 4/5), et le signe **s'inverse** sous MLP. La réserve pré-enregistrée (« `slope_asym` porte le seul axe orthogonal ») est **réfutée par les données** - publiée telle quelle. |

## Le fil

La Phase 0 montre un cas où « pas d'edge » vient surtout de « pas de signal ». La Phase 1
crypto construit le cas inverse, celui qui compte : un signal **réel**, mesuré, robuste -
et une évaluation honnête qui refuse quand même de l'appeler un edge. Entre les deux, la
Phase 1b chiffre la raison de fond : le coût d'exécution.

La Phase 1c ferme l'échappatoire qui restait ouverte après la Phase 1 : si le R²_OOS est
un **plancher**, c'est peut-être qu'il manque une information au modèle. L'information
manquante au meilleur niveau - le flux intra-seconde, absent des barres 1 s - a été
construite (OFI événementiel), fournie au modèle et testée en **apparié**. Elle n'apporte
rien : le plancher n'était pas un défaut de représentation. Un résultat négatif qui mérite
d'être publié comme tel.

Les Phases 1d et 1e ferment les deux échappatoires qui restaient après la 1c. D'abord le
**modèle** : et si l'OFI était hors de portée d'un modèle linéaire ? Un MLP pré-enregistré,
à état strictement identique, répond non - et dégrade même la prévision sur deux symboles.
Ensuite la **représentation** : et si l'information utile était plus profond dans le
carnet ? Deux dimensions de profondeur la capturent ; elles aident deux symboles sur cinq,
en dégradent deux, et leur signe dépend de la classe du modèle. Ni l'une ni l'autre de ces
deux portes ne rouvre la conclusion. Le mirage tient, et il est maintenant cerné de trois
côtés.

Les rapports sont écrits pour être lus dans l'ordre ci-dessus, mais chacun tient seul.
