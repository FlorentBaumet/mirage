"""Tests anti-lookahead — la garantie d'honnêteté (I3).

Vérifient :
  1. alignement features/cible (la cible est bien le FUTUR ; ret_lag_1 le présent),
  2. walk-forward : pas de chevauchement train/test + embargo respecté,
  3. standardisation : stats calculées sur le TRAIN uniquement,
  4. shuffle test : mélanger le temps ne doit pas aider (sinon fuite/aucun signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mirage.features import build_features
from mirage.splits import walk_forward_splits
from mirage.eval import standardize


def _toy_bars(n=30, seed=0):
    """Carnet jouet L1 avec un mid connu (marche aléatoire douce)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-06-21 09:30:00", periods=n, freq="1s")
    mid = 50.0 + np.cumsum(rng.normal(0, 0.01, n))
    df = pd.DataFrame(index=idx)
    df["ask_price_1"] = mid + 0.01
    df["ask_size_1"] = rng.integers(1, 100, n)
    df["bid_price_1"] = mid - 0.01
    df["bid_size_1"] = rng.integers(1, 100, n)
    return df, mid


def test_target_is_future_and_lag_is_present():
    bars, _ = _toy_bars()
    X, y, persist = build_features(bars, levels=1, lags=(1, 2), use_ofi=True, horizon=1)

    mid = (bars["ask_price_1"] + bars["bid_price_1"]) / 2.0
    r = np.log(mid).diff()
    exp_target = r.shift(-1)   # r_{t+1}
    exp_lag1 = r               # r_t

    for ts in X.index:
        assert np.isclose(y.loc[ts], exp_target.loc[ts]), "cible != rendement futur"
        assert np.isclose(X.loc[ts, "ret_lag_1"], exp_lag1.loc[ts]), "ret_lag_1 != r_t"
        assert np.isclose(persist.loc[ts], exp_lag1.loc[ts]), "persist@h=1 != r_t"


def test_horizon_target_alignment():
    """Cible à horizon h = log(mid_{t+h}) - log(mid_t) ; persist = passé sur h."""
    bars, _ = _toy_bars()
    h = 4
    X, y, persist = build_features(bars, levels=1, lags=(1,), use_ofi=False, horizon=h)
    mlog = np.log((bars["ask_price_1"] + bars["bid_price_1"]) / 2.0)
    exp_target = mlog.shift(-h) - mlog
    exp_persist = mlog - mlog.shift(h)
    for ts in X.index:
        assert np.isclose(y.loc[ts], exp_target.loc[ts]), "cible h mal alignée (lookahead ?)"
        assert np.isclose(persist.loc[ts], exp_persist.loc[ts]), "persist h mal aligné"


def test_walk_forward_no_overlap_and_embargo():
    n, embargo = 200, 10
    for tr, te in walk_forward_splits(n, n_folds=5, embargo=embargo,
                                      min_train_frac=0.4, scheme="expanding"):
        assert set(tr).isdisjoint(set(te)), "chevauchement train/test"
        # le test est strictement dans le futur, avec un trou d'embargo
        assert tr.max() < te.min(), "train pas avant test"
        assert te.min() - tr.max() >= embargo, "embargo non respecté"


def test_standardize_uses_train_stats_only():
    df = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2})
    train, test = df.iloc[:60], df.iloc[60:]
    tr_s, te_s = standardize(train, test)
    # le train standardisé est centré-réduit sur lui-même
    assert np.allclose(tr_s.mean(), 0, atol=1e-9)
    # le test utilise mu/sd du TRAIN -> sa moyenne n'est PAS 0 (data ascendante)
    assert te_s["a"].mean() > 1.0


def test_shuffle_destroys_or_not_signal():
    """Sanity : sur des features aléatoires non liées à la cible, le R²_OOS d'un
    linéaire reste <= 0 (pas de signal fabriqué)."""
    from sklearn.linear_model import Ridge
    from mirage.metrics import r2_oos

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = rng.normal(size=300)  # indépendant de X
    m = Ridge().fit(X[:200], y[:200])
    pred = m.predict(X[200:])
    assert r2_oos(y[200:], pred) <= 0.05, "signal fantôme : fuite probable"


def test_state_supervised_alignment():
    """World model (Phase 1) : la fenêtre d'entrée est strictement causale et la
    cible est l'état suivant."""
    from mirage.state import STATE_COLS, build_state
    from mirage.wm import make_supervised

    bars, _ = _toy_bars(40)
    S, _ = build_state(bars, levels=1)
    lookback = 4
    X, Y, pos, d = make_supervised(S.values, lookback)
    assert d == len(STATE_COLS)
    k = 5
    assert np.allclose(Y[k], S.values[pos[k]]), "cible != état suivant"
    win = X[k].reshape(lookback, d)
    assert np.allclose(win, S.values[pos[k] - lookback:pos[k]]), "fenêtre mal alignée"
    assert np.allclose(win[-1], S.values[pos[k] - 1]), "fenêtre contient le futur (fuite)"
