"""Le planificateur de la Phase 2 doit etre EXACT, pas seulement raisonnable.

Trois niveaux de verification, du plus faible au plus fort :

1. des cas ou la reponse est connue analytiquement (sans rendement on reste plat, sans
   cout on achete et on tient, un cout trop eleve interdit d'entrer) ;
2. l'accord entre les deux implementations (`dp_plan` lente et lisible, `plan_positions`
   vectorisee) -- deux codes independants qui tombent d'accord ;
3. la comparaison a une enumeration exhaustive de toutes les suites de positions
   possibles : c'est la seule qui prouve l'OPTIMALITE et non la coherence.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from mirage.plan import POS, dp_plan, plan_objective, plan_positions


def _best_by_enumeration(r, c, p_init=0.0):
    T = len(r)
    best, best_pos = -np.inf, None
    for combo in itertools.product(POS, repeat=T):
        v = plan_objective(combo, r, c, p_init)
        if v > best:
            best, best_pos = v, combo
    return best, np.array(best_pos)


def test_dp_atteint_l_optimum_par_enumeration():
    """Le coeur : l'objectif du plan rendu doit EGALER le maximum enumere."""
    rng = np.random.default_rng(0)
    for essai, T in enumerate((6, 8, 9)):
        r = rng.normal(0.0, 1e-3, T)
        c = np.abs(rng.normal(0.0, 5e-4, T))
        best, _ = _best_by_enumeration(r, c)
        got = plan_objective(dp_plan(r, c), r, c)
        assert got == pytest.approx(best, abs=1e-15), f"essai {essai}, T={T}"


def test_dp_optimal_depuis_une_position_non_plate():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 1e-3, 8)
    c = np.abs(rng.normal(0.0, 5e-4, 8))
    for p_init in POS:
        best, _ = _best_by_enumeration(r, c, p_init)
        got = plan_objective(dp_plan(r, c, p_init), r, c, p_init)
        assert got == pytest.approx(best, abs=1e-15)


def test_position_vectorisee_egale_position_de_reference():
    """`plan_positions` avec un horizon couvrant toute la sequence doit rendre le meme
    plan que la DP complete : c'est la meme optimisation, calculee autrement."""
    rng = np.random.default_rng(3)
    for T in (7, 12, 25):
        r = rng.normal(0.0, 1e-3, T)
        c = np.abs(rng.normal(0.0, 4e-4, T))
        lent = dp_plan(r, c)
        vite = plan_positions(r, c, horizon=T)
        assert np.array_equal(lent, vite), f"desaccord a T={T}"


def test_horizon_un_egale_la_regle_gloutonne():
    """A H=1 le planificateur n'a plus qu'un pas devant lui : sa decision doit coincider
    avec la regle gloutonne ecrite ici independamment (max de p*r - |p-prev|*c)."""
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 1e-3, 30)
    c = np.abs(rng.normal(0.0, 4e-4, 30))
    got = plan_positions(r, c, horizon=1)
    prev = 0.0
    for t in range(len(r)):
        valeurs = [p * r[t] - abs(p - prev) * c[t] for p in POS]
        attendu = POS[int(np.argmax(valeurs))]
        assert got[t] == attendu, f"desaccord a la barre {t}"
        prev = got[t]


def test_planifier_moins_loin_ne_fait_jamais_mieux():
    """Invariant d'optimalite : la valeur du plan a horizon court ne peut pas depasser
    celle du plan a horizon complet, puisqu'on maximise le meme objectif."""
    rng = np.random.default_rng(23)
    r = rng.normal(0.0, 1e-3, 40)
    c = np.abs(rng.normal(0.0, 4e-4, 40))
    reference = plan_objective(dp_plan(r, c), r, c)
    for H in (1, 2, 5, 20, 40):
        v = plan_objective(plan_positions(r, c, horizon=H), r, c)
        assert v <= reference + 1e-15, f"H={H} fait mieux que l'optimum"
        if H >= len(r):
            assert v == pytest.approx(reference, abs=1e-15)


def test_sans_rendement_on_reste_plat():
    r = np.zeros(20)
    c = np.full(20, 1e-4)
    assert np.all(dp_plan(r, c) == 0.0)
    assert np.all(plan_positions(r, c, 5) == 0.0)


def test_sans_cout_on_achete_et_on_tient():
    r = np.full(20, 1e-3)
    c = np.zeros(20)
    pos = dp_plan(r, c)
    assert np.all(pos == 1.0)
    assert plan_objective(pos, r, c) == pytest.approx(20 * 1e-3)


def test_sans_cout_on_suit_chaque_alternance():
    r = np.tile([+1e-3, -1e-3], 10)
    c = np.zeros(20)
    pos = dp_plan(r, c)
    attendu = np.tile([+1.0, -1.0], 10)
    assert np.array_equal(pos, attendu)
    assert plan_objective(pos, r, c) == pytest.approx(20 * 1e-3)


def test_un_cout_trop_eleve_interdit_d_entrer():
    """Le cout n'est paye QU'AU CHANGEMENT : entrer coute c une fois, pas T fois.
    Gain cumule 10*eps contre un cout d'entree 20*eps : rester plat est optimal."""
    r = np.full(10, 1e-3)
    c = np.full(10, 2e-2)
    assert np.all(dp_plan(r, c) == 0.0)
    assert plan_objective(np.zeros(10), r, c) == 0.0


def test_on_entre_quand_le_gain_cumule_paie_le_cout():
    """Gain 10*eps, cout d'entree 3*eps : acheter et tenir vaut 7*eps, plat vaut 0."""
    r = np.full(10, 1e-3)
    c = np.full(10, 3e-4)
    pos = dp_plan(r, c)
    assert np.all(pos == 1.0)
    assert plan_objective(pos, r, c) == pytest.approx(10 * 1e-3 - 3e-4)


def test_un_bref_retournement_ne_paie_pas_ses_deux_couts():
    """Un seul pas perdant au milieu d'une tendance : sortir puis rentrer coute 2 couts
    pour economiser un pas. Le plan doit tenir la position."""
    r = np.concatenate([np.full(5, 1e-2), [-1e-3], np.full(5, 1e-2)])
    c = np.full(11, 5e-3)
    pos = dp_plan(r, c)
    assert np.all(pos == 1.0)
    tenir = plan_objective(pos, r, c)
    sortir_rentrer = np.full(11, 1.0)
    sortir_rentrer[4:6] = 0.0
    assert tenir > plan_objective(sortir_rentrer, r, c)
