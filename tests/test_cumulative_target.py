"""Tests de la cible a horizon (Phase 2b, etape 2b-3)."""
from __future__ import annotations

import numpy as np
import pytest

from mirage.wm import cumulative_target


def test_somme_glissante_et_exclusion_de_la_fin():
    Y = np.zeros((6, 2))
    Y[:, 0] = np.arange(6, dtype=float)            # colonne 0 = 'ret'
    yH, idx = cumulative_target(Y, 3, ret_idx=0)
    assert np.array_equal(idx, [0, 1, 2, 3])
    assert np.allclose(yH, [0 + 1 + 2, 1 + 2 + 3, 2 + 3 + 4, 3 + 4 + 5])
    # les horizon - 1 = 2 derniers echantillons sont exclus
    assert len(yH) == 6 - 3 + 1


def test_h1_est_la_cible_a_un_pas():
    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, (20, 4))
    yH, idx = cumulative_target(Y, 1, ret_idx=2)
    assert np.array_equal(idx, np.arange(20))
    assert np.allclose(yH, Y[:, 2])               # difference de cumsum : pas bit-exacte


def test_journee_trop_courte():
    Y = np.zeros((3, 4))
    yH, idx = cumulative_target(Y, 10, ret_idx=0)
    assert len(yH) == 0 and len(idx) == 0


def test_aligne_sur_les_fenetres_du_world_model():
    """yH[k] doit couvrir exactement les rendements que le modele predit, un par pas."""
    Y = (1e-4 * np.random.default_rng(1).normal(size=(30, 3)))
    H = 10
    yH, idx = cumulative_target(Y, H, ret_idx=0)
    for k in idx[:5]:
        assert np.isclose(yH[k], Y[k:k + H, 0].sum())


def test_horizon_invalide():
    with pytest.raises(ValueError):
        cumulative_target(np.zeros((5, 3)), 0)
