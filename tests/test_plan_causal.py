"""Tests du planificateur causal (Phase 2b, regle `causalite_verifiee` du prereg).

Ces six tests doivent etre verts AVANT tout run de la Phase 2b : ils sont la seule
justification de l'affirmation "aucune information posterieure a t n'entre dans la
decision prise a t".
"""
from __future__ import annotations

import numpy as np

from mirage.plan import (
    POS,
    causal_paths,
    dp_plan,
    plan_positions,
    plan_positions_causal,
)
from mirage.state import RET_IDX
from mirage.wm import LinearWM, make_supervised

H = 10


def _blocs(days):
    """Bornes (debut, fin) de chaque journee."""
    n = len(days)
    brk = np.flatnonzero(days[1:] != days[:-1]) + 1 if n > 1 else np.zeros(0, dtype=int)
    return [(int(s), int(e)) for s, e in
            zip(np.concatenate([[0], brk]), np.concatenate([brk, [n]]), strict=True)]


def _verite(r, c, days, horizon):
    """R, C construits sur la VERITE : R[t, k] = r[t+k], nuls au-dela de la journee de t.

    C'est l'entree qui doit faire rendre au planificateur causal EXACTEMENT le plan du
    planificateur a serie unique applique journee par journee.
    """
    n = len(r)
    R = np.zeros((n, horizon))
    C = np.zeros((n, horizon))
    for s, e in _blocs(days):
        for t in range(s, e):
            k = min(horizon, e - t)
            R[t, :k] = r[t:t + k]
            C[t, :k] = c[t:t + k]
    return R, C


# --- 1. equivalence bit a bit avec le planificateur par blocs ---------------------------

def test_equivalence_avec_le_planificateur_par_blocs():
    """R[t, k] = r[t+k] doit redonner plan_positions, journee par journee, au bit pres."""
    for T in (17, 60, 250):
        rng = np.random.default_rng(1234 + T)
        r = rng.normal(0, 1e-3, T)
        c = np.abs(rng.normal(0, 4e-4, T))
        # journees coupees a des endroits qui ne sont PAS des multiples de l'horizon
        days = np.searchsorted([13, 40, 137], np.arange(T))
        R, C = _verite(r, c, days, H)
        attendu = np.empty(T)
        for s, e in _blocs(days):
            attendu[s:e] = plan_positions(r[s:e], c[s:e], H)
        assert np.array_equal(plan_positions_causal(R, C, days), attendu), f"T={T}"


# --- 2. H = 1 : le planificateur causal se reduit au myope ------------------------------

def test_h1_redonne_le_myope():
    """A H = 1 il n'y a aucun pas futur : le causal et l'agent H = 1 de la Phase 2
    coincident, et les deux doivent s'accorder avec l'enumeration ecrite a la main.

    H = 1 est glouton enchainé (on redecide a chaque barre depuis la position tenue), pas
    l'optimum global d'une journee : c'est dp_plan qui donne l'optimum global, et c'est le
    test 3 qui le verifie, pas ici.
    """
    rng = np.random.default_rng(7)
    T = 120
    r = rng.normal(0, 1e-3, T)
    c = np.abs(rng.normal(0, 4e-4, T))
    days = np.repeat([0, 1, 2], 40)
    R, C = _verite(r, c, days, 1)
    assert R.shape == (T, 1)
    got = plan_positions_causal(R, C, days)
    for s, e in _blocs(days):
        assert np.array_equal(got[s:e], plan_positions(r[s:e], c[s:e], 1))
        prev = 0.0
        for t in range(s, e):
            vals = [POS[j] * r[t] - c[t] * abs(POS[j] - prev) for j in range(len(POS))]
            assert got[t] == POS[int(np.argmax(vals))], f"t={t}"
            prev = got[t]


# --- 3. chaque decision est celle du plan optimal exact ---------------------------------

def test_chaque_decision_est_celle_du_plan_optimal():
    """A chaque barre, la decision doit egaler le PREMIER pas du plan optimal exact a H pas
    calcule sur la seule ligne t (reference independante : dp_plan)."""
    rng = np.random.default_rng(99)
    T = 60
    r = rng.normal(0, 1e-3, T)
    c = np.abs(rng.normal(0, 4e-4, T))
    days = np.zeros(T, dtype=int)
    R, C = _verite(r, c, days, H)
    pos = plan_positions_causal(R, C, days)
    prev = 0.0
    for t in range(T):
        attendu = dp_plan(R[t], C[t], p_init=prev)[0]
        assert pos[t] == attendu, f"desaccord a t={t} : {pos[t]} contre {attendu}"
        prev = pos[t]


# --- 4. causalite generique : un etat posterieur a la fenetre ne change rien -------------

def test_le_plan_causal_ignore_les_etats_posterieurs():
    """Perturber S[t + L] ne doit pas modifier une seule position jusqu'a t inclus."""
    n, lookback, d = 400, 16, 3
    rng = np.random.default_rng(0)
    S = rng.normal(0, 1, (n, d))
    X, Y, _, _ = make_supervised(S, lookback)
    model = LinearWM().fit(X[:200], Y[:200])

    t = 120
    S2 = S.copy()
    S2[t + lookback, RET_IDX] += 5.0          # etat POSTERIEUR a la fenetre de t
    X2, _, _, _ = make_supervised(S2, lookback)

    rhat = model.predict(X)[:, RET_IDX]
    rhat2 = model.predict(X2)[:, RET_IDX]
    assert rhat[t] == rhat2[t]                # la fenetre de t est intacte
    assert rhat[t + 1] != rhat2[t + 1]        # celle de t+1 a change

    half = np.full(len(X), 1e-4)
    days = np.zeros(len(X), dtype=int)
    R1, C1 = causal_paths(model, X, half, 2.0, H, days, lookback, d)
    R2, C2 = causal_paths(model, X2, half, 2.0, H, days, lookback, d)
    assert np.array_equal(R1[t], R2[t])

    a = plan_positions_causal(R1, C1, days)
    b = plan_positions_causal(R2, C2, days)
    assert np.array_equal(a[:t + 1], b[:t + 1])
    assert a[t] == b[t]


# --- 5. le planificateur publie de la Phase 2, lui, lit le futur ------------------------

def test_published_planner_reads_the_future():
    """Demonstration en laboratoire de la fuite de la Phase 2.

    Le planificateur publie consomme rhat[t+1..t+H-1] pour decider a t. On perturbe un etat
    POSTERIEUR a la fenetre de t, S[t + L] : la fenetre de t est intacte, donc rhat[t] aussi,
    et pourtant la decision prise a t doit changer. C'est le mecanisme ; l'ampleur mesuree
    sur les donnees reelles est dans configs/phase2b_crypto_prereg.yaml.

    Deux ingredients sont necessaires, et ce test les documente :

    * des rendements persistants (AR(1)), sans quoi les coefficients Ridge sont ~0 et
      perturber S[t+L] ne bouge rhat[t+1] que d'une fraction de point de base ;
    * un cout NON negligeable devant l'echelle de la prevision. Avec c = 0, la recursion
      donne F_k[a, s] = somme des |r[s..s+k-1]|, IDENTIQUE pour les trois actions : le
      planificateur degenere en sign(r_t), l'horizon ne sert plus a rien, donc la fuite non
      plus. C'est le tres fort cout relatif des donnees reelles (quelques bp contre une
      fraction de bp) qui fait vivre l'horizon -- et c'est pour cela que la fuite mesuree
      croit avec H.
    """
    n, lookback, d = 300, 16, 3
    rng = np.random.default_rng(3)
    eps = rng.normal(0, 1, (n, d))
    S = np.zeros((n, d))
    for k in range(1, n):
        S[k] = 0.9 * S[k - 1] + eps[k]

    X, Y, _, _ = make_supervised(S, lookback)
    model = LinearWM().fit(X[:150], Y[:150])

    rhat = model.predict(X)[:, RET_IDX]
    c = np.full(len(X), 2.0 * rhat.std())
    pos = plan_positions(rhat, c, H)

    flips = 0
    for t in range(60, 200):
        for delta in (20.0, -20.0):
            S2 = S.copy()
            S2[t + lookback, RET_IDX] += delta
            X2 = make_supervised(S2, lookback)[0]
            rhat2 = model.predict(X2)[:, RET_IDX]
            assert rhat2[t] == rhat[t], f"fenetre de t={t} modifiee"
            assert rhat2[t + 1] != rhat[t + 1], f"rhat[t+1] inchange a t={t}"
            if plan_positions(rhat2, c, H)[t] != pos[t]:
                flips += 1
    assert flips >= 5, f"{flips}/280 seulement : le mecanisme de fuite n'est pas reproduit"


# --- 6. ce que fabrique causal_paths ----------------------------------------------------

def test_causal_paths_est_le_rollout_et_s_annule_en_fin_de_journee():
    n, lookback, d = 200, 16, 3
    rng = np.random.default_rng(5)
    S = rng.normal(0, 1, (n, d))
    X, Y, _, _ = make_supervised(S, lookback)
    model = LinearWM().fit(X[:100], Y[:100])
    m = len(X)
    half = np.full(m, 3e-4)
    days = np.concatenate([np.zeros(6, dtype=int), np.ones(m - 6, dtype=int)])

    R, C = causal_paths(model, X, half, 2.0, H, days, lookback, d)

    rhat = model.predict(X)[:, RET_IDX]
    assert np.array_equal(R[:, 0], rhat)                       # premier pas = rhat, au bit pres
    assert np.array_equal(C[:, 0], half + 2.0 * 1e-4)          # cout courant = demi-spread + frais

    # La journee 0 ne fait que 6 barres : la ligne i ne dispose plus que de 6 - i pas. Le
    # masque est donc un ESCALIER (6, 5, 4, 3, 2, 1), pas un troncon final unique : la
    # derniere ligne de la journee n'a plus qu'un seul pas utilisable.
    for i, reste in enumerate((6, 5, 4, 3, 2, 1)):
        assert np.all(R[i, reste:] == 0.0), i
        assert np.all(C[i, reste:] == 0.0), i
    # ... la journee 1, plus longue que l'horizon, n'est touchee que sur ses H - 1 dernieres
    # lignes : le masque ne mange pas la journee, il n'en mange que la fin.
    assert not np.any(R[6:m - H, :] == 0.0)
    assert not np.any(C[6:m - H, :] == 0.0)
    assert np.all(R[m - 1, 1:] == 0.0)          # derniere ligne : un seul pas utilisable
