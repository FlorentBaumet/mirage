"""Dimensions de profondeur (Phase 1e) : justesse, causalite, additivite.

Quatre proprietes, dont les deux dernieres portent l'interpretation :

  1. JUSTESSE - `imb_deep` et `slope_asym` valent exactement ce qu'on calcule a la main
     sur un carnet jouet.
  2. CAUSALITE - la valeur d'une barre ne depend que de cette barre (propriete de
     prefixe : ajouter du futur ne modifie pas le passe).
  3. ADDITIVITE - ajouter `deep=True` ne change NI le nombre de lignes NI les cinq
     premieres dimensions. Sans cela, les trois bras ne seraient pas appariables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mirage.state import DEEP_COLS, OFI_COL, RET_IDX, STATE_COLS, build_state

NIVEAUX = 10


def _barres(n: int, levels: int = NIVEAUX) -> pd.DataFrame:
    """Carnet jouet : tailles variables par niveau et par barre (non constantes)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="s")
    bars = pd.DataFrame(index=idx)
    bars["bid_price_1"] = [100.0 + 0.1 * i for i in range(n)]
    bars["ask_price_1"] = [101.0 + 0.1 * i for i in range(n)]
    for lvl in range(1, levels + 1):
        bars[f"bid_size_{lvl}"] = [1.0 + 0.5 * lvl + i for i in range(n)]
        bars[f"ask_size_{lvl}"] = [2.0 + 0.25 * lvl + 0.5 * i for i in range(n)]
    return bars


def test_formules_calcul_ala_main():
    """imb_deep et slope_asym sur un carnet jouet, calculees a la main.

    bid_size_1 = 5, bid_size_2..10 = 2 ; ask_size_1 = 1, ask_size_2..10 = 3.
      somme bid = 5 + 9*2 = 23 ; somme ask = 1 + 9*3 = 28
      imb_deep  = (18 - 27) / (18 + 27) = -0.2
      slope_asym = 5/23 - 1/28
    """
    n = 3
    bars = _barres(n)
    for lvl in range(1, NIVEAUX + 1):
        bars[f"bid_size_{lvl}"] = [5.0 if lvl == 1 else 2.0] * n
        bars[f"ask_size_{lvl}"] = [1.0 if lvl == 1 else 3.0] * n

    S, _ = build_state(bars, deep=True)
    row = S.iloc[0]
    assert row["imb_deep"] == pytest.approx(-9.0 / 45.0)
    assert row["slope_asym"] == pytest.approx(5.0 / 23.0 - 1.0 / 28.0)
    # et sur toute la serie (colonnes constantes ici)
    assert np.allclose(S["imb_deep"].to_numpy(), -0.2)
    assert np.allclose(S["slope_asym"].to_numpy(), 5.0 / 23.0 - 1.0 / 28.0)


def test_causalite_prefixe():
    """Ajouter des barres FUTURES ne doit rien changer aux barres DEJA closes."""
    bars = _barres(8)
    plein, _ = build_state(bars, deep=True)
    prefixe, _ = build_state(bars.iloc[:5], deep=True)
    # `prefixe` couvre les barres 1..4 (la barre 0 est retiree : ret = NaN).
    pd.testing.assert_frame_equal(plein.loc[prefixe.index, DEEP_COLS], prefixe[DEEP_COLS])


def test_nombre_de_lignes_identique():
    """Aucune ligne ne doit disparaitre : condition de l'appariement base/enrichi."""
    bars = _barres(6)
    base, _ = build_state(bars)
    deep, _ = build_state(bars, deep=True)
    assert len(deep) == len(base)
    assert deep.index.equals(base.index)
    assert not deep[DEEP_COLS].isna().any().any()


def test_nan_impossible_meme_carnet_degenere():
    """Un carnet a tailles nulles ne doit pas fabriquer de NaN (denominateur = EPS)."""
    n = 4
    bars = _barres(n)
    for lvl in range(1, NIVEAUX + 1):
        bars[f"bid_size_{lvl}"] = [0.0] * n
        bars[f"ask_size_{lvl}"] = [0.0] * n
    deep, _ = build_state(bars, deep=True)
    assert not deep[DEEP_COLS].isna().any().any()
    assert np.isfinite(deep[DEEP_COLS].to_numpy()).all()


def test_ret_idx_et_cinq_premieres_colonnes_inchangees():
    bars = _barres(6)
    base, _ = build_state(bars)
    deep, _ = build_state(bars, deep=True)
    assert RET_IDX == 0
    assert list(deep.columns) == STATE_COLS + DEEP_COLS
    assert list(deep.columns)[: len(STATE_COLS)] == STATE_COLS
    pd.testing.assert_frame_equal(deep[STATE_COLS], base[STATE_COLS])


def test_ordre_des_colonnes_avec_ofi():
    """Ordre final impose : base 5, puis deep 2, puis ofi."""
    bars = _barres(6)
    ofi = pd.Series(1.0, index=bars.index, name=OFI_COL)
    S, _ = build_state(bars, ofi=ofi, deep=True)
    assert list(S.columns) == STATE_COLS + DEEP_COLS + [OFI_COL]
