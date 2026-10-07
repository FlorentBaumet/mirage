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

from .state import RET_IDX
from .wm import rollout

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


# --------------------------------------------------------------------------------------
# Planificateur CAUSAL (Phase 2b).
#
# `plan_positions` ci-dessus recoit UNE serie r a laquelle il applique un horizon glissant :
# la valeur des pas restants a la barre t est calculee sur r[t+1], r[t+2], ... Si r est la
# serie des predictions a 1 pas, ces valeurs-la sont des predictions FAITES PLUS TARD que t,
# et `make_supervised` (X[k] = etats k..k+L-1, Y[k] = etat k+L) fait que r[t+1] contient Y[t],
# donc le rendement que p_t encaisse. C'est la fuite de la Phase 2.
#
# Ici la valeur des pas restants a la barre t est lue dans la LIGNE t de R : les previsions
# faites A t. Aucune ligne t+1, t+2, ... n'entre dans une decision prise a t.
# --------------------------------------------------------------------------------------

def plan_positions_causal(R: np.ndarray, C: np.ndarray, days: np.ndarray,
                          chunk: int = 400_000) -> np.ndarray:
    """Planificateur a horizon glissant dont les anticipations sont toutes faites a t.

    R[t, k] = rendement attendu au pas t+k, PREVU A t (k = 0..H-1).
    C[t, k] = cout unitaire attendu au pas t+k, PREVU A t.
    days[t] = identifiant de journee : remise a plat a chaque frontiere.

    Renvoie p_t, chainee sur la position REELLEMENT executee. Meme objectif et meme ordre
    d'operations que `plan_positions` : avec R[t, k] = r[t+k] et C[t, k] = c[t+k] (nuls
    au-dela de la journee), les deux rendent le MEME plan, bloc par bloc.
    """
    R = np.asarray(R, float)
    C = np.asarray(C, float)
    days = np.asarray(days)
    if R.ndim != 2 or C.shape != R.shape:
        raise ValueError("R et C doivent etre deux tableaux (n, H) de meme forme.")
    n, H = R.shape
    if len(days) != n:
        raise ValueError("days doit avoir une entree par barre.")
    if H < 1:
        raise ValueError("H >= 1 requis.")

    # V[t, i] = valeur des pas t+k..t+H-1 du plan fait a t, sachant POS[i] detenu avant le pas.
    V = np.zeros((n, 3))
    for k in range(H - 1, 0, -1):
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            # Meme ordre que plan_positions : (valeur future + POS*r) - c*D, max sur la
            # position choisie.
            G = V[s:e] + POS[None, :] * R[s:e, k, None]
            V[s:e] = (G[:, None, :] - C[s:e, k, None, None] * D[None, :, :]).max(axis=2)

    # A[t, j] = valeur du plan fait a t s'il termine le PREMIER pas en POS[j].
    A = POS[None, :] * R[:, 0, None] + V
    J = np.empty((n, 3), dtype=np.int8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        J[s:e] = (A[s:e, None, :] - C[s:e, 0, None, None] * D[None, :, :]).argmax(axis=2)

    # Seule partie sequentielle : la chaine des positions reellement executees.
    pos = np.empty(n)
    POSJ = POS[J].ravel()
    Jf = J.ravel()
    brk = np.flatnonzero(days[1:] != days[:-1]) + 1 if n > 1 else np.zeros(0, dtype=int)
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [n]])
    for s, e in zip(starts, ends, strict=True):
        i = PLAT                                   # chaque journee repart a plat
        for t in range(int(s), int(e)):
            i = int(Jf[3 * t + i])                 # int() : sinon 3*t+i deborde en int8
            pos[t] = POSJ[3 * t + i]
    return pos


def causal_paths(model, X: np.ndarray, half: np.ndarray, fee: float, horizon: int,
                 days: np.ndarray, lookback: int, d: int, ret_idx: int = RET_IDX,
                 chunk: int = 250_000) -> tuple[np.ndarray, np.ndarray]:
    """(R, C) du planificateur causal : un rollout par fenetre, et rien d'autre.

    R[t, k] = composante `ret` du rollout AUTOREGRESSIF du world model depuis la fenetre de
    t, et seulement elle. R[t, 0] est identique au bit pres a `model.predict(X[t])[:, ret]`.
    C[t, k] = demi-spread mesure a t + frais : le cout futur est TENU a sa valeur de t
    (persistance). Choix fige : le plus simple, sans fuite, sans degre de liberte.

    R et C sont mis a ZERO au-dela de la fin de la journee de t : un plan ne traverse jamais
    une nuit, et l'agent connait le calendrier (ce n'est pas une fuite).
    """
    X = np.asarray(X)
    n = len(X)
    R = np.empty((n, horizon))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        win = np.ascontiguousarray(X[s:e], float).reshape(-1, lookback, d)
        R[s:e] = rollout(model, win, horizon)[:, :, ret_idx]
    C = np.repeat((np.asarray(half, float) + fee * 1e-4)[:, None], horizon, axis=1)

    brk = np.flatnonzero(days[1:] != days[:-1]) + 1 if n > 1 else np.zeros(0, dtype=int)
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [n]])
    for s, e in zip(starts, ends, strict=True):
        reste = int(e - s)
        if reste < horizon:
            R[s:e, reste:] = 0.0
            C[s:e, reste:] = 0.0
    return R, C
