"""Planificateur exact conscient du cout (Phase 2).

Un agent qui doit choisir sa position dans {-1, 0, +1} a chaque barre, en payant un cout
proportionnel a chaque changement de position, resout un probleme de commande optimale a
etats finis. Ce module le resout EXACTEMENT : aucune approximation, aucun hyperparametre
d'optimisation. Tout echec de l'agent reste donc imputable aux rendements anticipes qu'on
lui fournit, jamais a un optimiseur.

Fonctions pures sur deux tableaux :
  r[t] : rendement anticipe a la barre t (ce que le modele predit, ou la verite)
  c[t] : cout unitaire d'un changement de position a la barre t (demi-spread + frais)

Objectif maximise :  somme_t p_t*r_t - somme_t |p_t - p_{t-1}| * c_t,   p_t dans POS.
"""
from __future__ import annotations

import numpy as np

POS = np.array([-1.0, 0.0, 1.0])
D = np.abs(POS[:, None] - POS[None, :])     # D[j, j'] = |POS[j] - POS[j']|
PLAT = 1                                    # indice de la position nulle


def dp_plan(r, c, p_init: float = 0.0) -> np.ndarray:
    """Plan optimal exact, version lente et directement lisible.

    Sert de REFERENCE : `plan_positions` (vectorisee) doit rendre exactement le meme
    resultat. Deux implementations independantes qui s'accordent valent mieux qu'une.

    Programmation dynamique avant sur les 3 etats, puis remontee de la politique.
    """
    r = np.asarray(r, float)
    c = np.asarray(c, float)
    T = len(r)
    V = np.full(3, -np.inf)
    V[int(np.argmin(np.abs(POS - p_init)))] = 0.0
    choice = np.zeros((T, 3), dtype=np.int8)
    for t in range(T):
        Vn = np.full(3, -np.inf)
        for j in range(3):
            for i in range(3):
                if not np.isfinite(V[i]):
                    continue
                val = V[i] + POS[j] * r[t] - D[j, i] * c[t]
                if val > Vn[j]:
                    Vn[j] = val
                    choice[t, j] = i
        V = Vn
    j = int(np.argmax(V))
    out = np.empty(T)
    for t in range(T - 1, -1, -1):
        out[t] = POS[j]
        j = choice[t, j]
    return out


def plan_positions(r, c, horizon: int) -> np.ndarray:
    """Positions par horizon glissant : planifier `horizon` pas, executer le premier,
    avancer d'une barre, replanifier. La position de depart de chaque plan est la position
    REELLEMENT executee a la barre precedente.

    La valeur des pas restants est calculee en arriere sur TOUT le tableau d'un coup --
    c'est ce qui rend le cout O(horizon * n) en operations vectorisees au lieu de
    O(horizon * n) en boucles Python. La seule partie sequentielle est la chaine des
    positions reellement executees.
    """
    r = np.asarray(r, float)
    c = np.asarray(c, float)
    n = len(r)
    # F[s, j] = valeur optimale des pas s.. s+k-1 sachant qu'on detient POS[j] avant le pas s.
    F = np.zeros((3, n + 1))
    for _ in range(1, horizon):
        # G[j', s] = POS[j']*r_s + F_{k-1}[s+1, j']
        G = F[:, 1:] + POS[:, None] * r[None, :]
        # F_k[s, j] = max_{j'} G[j', s] - c_s * D[j, j']
        cand = G[None, :, :] - c[None, None, :] * D[:, :, None]
        F = np.concatenate([cand.max(axis=1), np.zeros((3, 1))], axis=1)
    pos = np.empty(n)
    i = PLAT
    for t in range(n):
        j = int(np.argmax(POS * r[t] + F[:, t + 1] - c[t] * D[:, i]))
        pos[t] = POS[j]
        i = j
    return pos


def plan_objective(pos, r, c, p_init: float = 0.0) -> float:
    """Valeur de l'objectif pour une suite de positions donnee (sert aux tests)."""
    pos = np.asarray(pos, float)
    r = np.asarray(r, float)
    c = np.asarray(c, float)
    prev = np.concatenate([[p_init], pos[:-1]])
    return float(np.sum(pos * r - np.abs(pos - prev) * c))
