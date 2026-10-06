"""La garde d'appariement doit pouvoir ECHOUER.

Regression sur un defaut reel : `arm_eval._per_day` reduisait tous les bras avec la cible
du bras de reference, donc `sse_b_{tag}` etait identique par construction et
`paired_arms.assert_apparies` ne pouvait jamais lever. Une garde qui ne peut pas echouer
n'affiche pas une securite, elle affiche une securite inexistante.

Ces tests passent par les VRAIES fonctions (aucune reimplementation) : ils echoueraient
si `_per_day` revenait a une reduction depuis la cible de reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "crypto"))

import arm_eval  # noqa: E402
import paired_arms  # noqa: E402


def _entry(y, pred, day, dpos=None, half=None) -> dict:
    """Une entree de `arrays` telle que `eval_symbol` la produit."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    day = np.asarray(day)
    if dpos is None:
        dpos = np.zeros_like(y)
    if half is None:
        half = np.zeros_like(y)
    return dict(pred=pred, y=y, day=day, gross=np.sign(pred) * y,
                dpos=np.asarray(dpos, float), half=np.asarray(half, float))


def test_per_day_reduces_each_arm_from_its_own_target():
    """Deux bras a cibles differentes doivent produire des empreintes differentes."""
    day = np.array([0, 0, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    arrays = {
        ("base", "linear"): _entry(y, y, day),
        ("ofi", "linear"): _entry(y * 2.0, y, day),  # cible reellement differente
    }
    npz: dict = {}
    arm_eval._per_day(npz, arrays)
    assert not np.array_equal(npz["sse_b_base_linear"], npz["sse_b_ofi_linear"])
    # et l'empreinte est bien la somme par jour des y^2 du bras considere
    np.testing.assert_allclose(npz["sse_b_base_linear"], [1.0 + 4.0, 9.0 + 16.0])
    np.testing.assert_allclose(npz["sse_b_ofi_linear"], [4.0 + 16.0, 36.0 + 64.0])


def test_assert_apparies_detects_mismatch():
    """Le coeur du defaut : la garde DOIT lever sur des echantillons differents."""
    day = np.array([0, 0, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    arrays = {
        ("base", "linear"): _entry(y, y, day),
        ("ofi", "linear"): _entry(y * 2.0, y, day),
    }
    npz: dict = {}
    arm_eval._per_day(npz, arrays)
    with pytest.raises(SystemExit):
        paired_arms.assert_apparies("SYNTH", npz, "ofi_linear", "base_linear")


def test_assert_apparies_passes_when_arms_are_paired():
    """Contre-epreuve : sur des bras reellement apparies, la garde reste silencieuse."""
    day = np.array([0, 0, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    arrays = {
        ("base", "linear"): _entry(y, y, day),
        ("ofi", "linear"): _entry(y, y + 0.5, day),  # meme cible, prediction differente
    }
    npz: dict = {}
    arm_eval._per_day(npz, arrays)
    paired_arms.assert_apparies("SYNTH", npz, "ofi_linear", "base_linear")


def test_per_day_rejects_arms_on_different_days():
    """Un bras qui ne porte pas les memes journees doit faire echouer la reduction."""
    good = _entry([1.0, 2.0], [1.0, 2.0], np.array([0, 0]))
    bad = _entry([1.0, 2.0], [1.0, 2.0], np.array([0, 1]))
    with pytest.raises(SystemExit):
        arm_eval._per_day({}, {("base", "linear"): good, ("ofi", "linear"): bad})


def test_net_day_matches_the_cost_formula():
    """net = brut - demi-spread - frais, le x1e4 etant applique apres le bootstrap."""
    npz = {"gross_b_l": np.array([10.0]), "dposhalf_b_l": np.array([3.0]),
           "dpos_b_l": np.array([2.0])}
    got = paired_arms.net_day(npz, "b", "l", 2.0)
    assert got[0] == pytest.approx(10.0 - 3.0 - 2.0 * 1e-4 * 2.0)
