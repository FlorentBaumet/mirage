"""Phase 2b : ce qui protege l'echantillon publie, et le masque de fin de journee.

Deux choses que la phase 2b a cassees ou rendues fragiles, et qu'on veut voir echouer ici
plutot qu'en pleine nuit au milieu d'un run :

  - `crypto_lob.load_cached` listait TOUT `data/raw/crypto_lob`. Deposer les 44 journees de
    l'extension 2b changeait donc en silence l'echantillon des phases 1 et 2, et leurs
    controles d'integrite tombaient pour une raison sans rapport avec le code. La liste des
    44 journees publiees est maintenant nommee, et une journee manquante leve.
  - le masque `R = C = 0 au-dela de la fin du jour` est calcule UNE fois sur le rollout
    maitre (H = 20) puis decoupe en `[:, :H]`. Toute l'economie du harnais tient a cette
    equivalence : masquer a 20 puis decouper doit donner exactement le masque d'un rollout
    d'horizon H. C'est ce que verifie `test_masque_a_hmax_decoupe_egale_masque_direct`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "crypto"))

import crypto_lob  # noqa: E402
import phase2b  # noqa: E402

from mirage.plan import mask_day_end  # noqa: E402


def _jours(len_days: list[int]) -> np.ndarray:
    return np.concatenate([np.full(n, i) for i, n in enumerate(len_days)])


def test_dates_publiees_est_celle_du_prereg():
    """La liste nommee doit etre EXACTEMENT celle du pre-enregistrement 1c, dans l'ordre."""
    cfg = yaml.safe_load((ROOT / "configs" / "phase1c_crypto_prereg.yaml").read_text(
        encoding="utf-8"))
    dates = cfg["donnees"]["dates"]
    assert len(dates) == 44
    assert list(crypto_lob.DATES_PUBLIEES_1C) == dates


def test_date_of_lit_la_journee_dans_le_nom():
    assert crypto_lob.date_of("data/raw/crypto_lob/2024-03-22_ETHUSDT_1s_book.pkl") == \
        "2024-03-22"


def test_load_cached_filtre_et_leve_sur_une_journee_absente(tmp_path, monkeypatch):
    """Le cache disque ne doit plus decider tout seul de l'echantillon."""
    for nom in ("2023-01-20_BTCUSDT_1s_book.pkl", "2023-02-11_BTCUSDT_1s_book.pkl",
                "2099-01-01_BTCUSDT_1s_book.pkl"):
        (tmp_path / nom).write_bytes(b"")
    monkeypatch.setattr(crypto_lob, "DIR", str(tmp_path))

    cache = crypto_lob.load_cached(["2023-01-20", "2023-02-11"])
    assert list(cache) == ["BTCUSDT"]
    assert [crypto_lob.date_of(p) for p in cache["BTCUSDT"]] == ["2023-01-20", "2023-02-11"]

    # une journee demandee et absente est une erreur, pas un echantillon silencieusement
    # amputee ; le defaut (les 44 publiees) est donc inexistant dans ce faux cache
    with pytest.raises(SystemExit, match="absentes"):
        crypto_lob.load_cached(["2023-01-20", "2024-03-22"])
    with pytest.raises(SystemExit, match="absentes"):
        crypto_lob.load_cached()


def test_masque_zero_apres_la_fin_du_jour():
    """Jour de longueur L : la ligne i est nulle a partir de la colonne L - i, et pas avant."""
    dd = _jours([5, 3, 7])
    A = mask_day_end(np.ones((15, 4)), dd)
    attendu = np.ones((15, 4))
    for s, L in ((0, 5), (5, 3), (8, 7)):
        for i in range(L):
            attendu[s + i, max(L - i, 0):] = 0.0
    assert np.array_equal(A, attendu)
    # Dans une journee plus longue que l'horizon, seules ses DERNIERES lignes sont touchees :
    # ici L = 7, H = 4, donc i = 0..3 disposent encore de 4 pas ou plus.
    assert np.all(A[8:12, :] != 0.0)


def test_masque_a_hmax_decoupe_egale_masque_direct():
    """Masquer a H_MAX puis decouper == masquer directement a H. Toute l'economie du harnais."""
    dd = _jours([5, 40, 3, 21])
    rng = np.random.default_rng(0)
    for H_MAX, H in ((20, 1), (20, 5), (20, 10), (20, 20)):
        full = rng.normal(size=(len(dd), H_MAX))
        gauche = mask_day_end(full.copy(), dd)[:, :H]
        droite = mask_day_end(full[:, :H].copy(), dd)
        assert np.array_equal(gauche, droite), f"H_MAX={H_MAX} H={H}"


def test_causal_planner_ignore_le_cout_scalaire_et_slice():
    """Le planificateur du diagnostic ne lit que `X` et la matrice de couts, jamais `c`."""
    from mirage.plan import plan_positions_causal

    dd = _jours([40, 30])
    rng = np.random.default_rng(3)
    R = rng.normal(size=(len(dd), 10)) * 1e-4
    C = np.abs(rng.normal(size=(len(dd), 10))) * 1e-4
    planner = phase2b.causal_planner(C)
    attendu = plan_positions_causal(R[:, :5], C[:, :5], dd)
    assert np.array_equal(planner(R, np.zeros(len(dd)), dd, 5), attendu)
    # ... et il DEPEND bien de la matrice de couts : sinon le test ci-dessus serait vide
    autre = phase2b.causal_planner(C * 50.0)(R, np.zeros(len(dd)), dd, 5)
    assert not np.array_equal(autre, attendu)


# --- etape A2 (cible cumulee, exploratoire) --------------------------------------------

def test_embargo_a2_est_celui_du_prereg():
    """L'embargo A2 doit couvrir le chevauchement de la cible : LOOKBACK + H, soit 26."""
    cfg = yaml.safe_load((ROOT / "configs" / "phase2b_crypto_prereg.yaml").read_text(
        encoding="utf-8"))
    assert phase2b.EMBARGO_A2 == cfg["regles_A2"]["embargo"] == 26
    assert phase2b.EMBARGO_A2 >= phase2b.LOOKBACK + phase2b.HORIZON


def test_r2_ci_donne_le_r2_naif_et_un_intervalle():
    """Le point de `_r2_ci` doit etre exactement le R² naif, et l'IC l'encadrer."""
    rng = np.random.default_rng(11)
    dd = np.repeat(np.arange(9), 30)
    y = rng.normal(0, 1e-3, len(dd))
    p = 0.6 * y + rng.normal(0, 1e-3, len(dd))
    r2, lo, hi = phase2b._r2_ci(y, p, dd, 9, blk=1)
    assert abs(r2 - (1.0 - ((y - p) ** 2).sum() / (y ** 2).sum())) < 1e-15
    assert lo < r2 < hi

    # Piege du tirage circulaire par blocs : avec des blocs au moins aussi longs que le
    # pool, chaque replication contient TOUTES les journees (une permutation circulaire),
    # donc la loi bootstrap est degeneree et l'IC se reduit au point. Ce n'est pas un bug
    # de `_r2_ci`, c'est la convention de `boot_mult` ; sur les 27-44 journees du harnais
    # (blocs de 3) elle ne mord pas, mais elle rendrait un IC vide sur un petit pool.
    r2b, lob, hib = phase2b._r2_ci(y, p, dd, 9, blk=9)
    assert (lob, hib) == (r2b, r2b)


def test_verdicts_a2_compte_les_bonnes_lignes():
    """Les deux regles A2 se lisent sur `lo > 0`, sur les seules lignes du bon bras."""
    syms = ["A", "B", "C", "D", "E"]
    r2 = pd.DataFrame([{"symbol": s, "type": "direct_yH", "h": 10,
                        "lo": 1e-4 if s != "E" else -1e-4} for s in syms])
    cmp_ = pd.DataFrame([{"symbol": s, "gain": "agent_direct@2", "ref": "agent_causal_A2@2",
                          "nature": "reel", "lo": 1e-4 if s != "E" else -1e-4} for s in syms]
                        + [{"symbol": s, "gain": "agent_direct@2", "ref": "myope",
                            "nature": "reel", "lo": -1.0} for s in syms])
    v = phase2b.verdicts_a2(pd.DataFrame(), cmp_, r2)
    assert v["signal_horizon"]["n"] == "4/5" and v["signal_horizon"]["ok"]
    assert v["direct_bat_rollout"]["n"] == "4/5" and v["direct_bat_rollout"]["ok"]
    assert v["direct_bat_rollout"]["detail"]["E"] is False


def test_a2_tourne_de_bout_en_bout_sur_des_journees_synthetiques():
    """`eval_symbol_a2` doit aller au bout, et ses deux chemins doivent tomber d'accord.

    Ce test existe parce que le chemin A2 a ete ecrit et commite SANS avoir jamais ete
    execute. Au premier run reel il est mort sur `mc.predict(X2[te])[:, 0]` : sklearn >= 1.9
    aplatit la prediction d'un modele mono-sortie, donc l'indexation en colonne levait
    IndexError. Un aller-retour sur des journees synthetiques assez longues pour produire
    NFOLDS plis l'aurait vu tout de suite. Il verifie aussi le controle bloquant du harnais
    (le R² du rollout recalcule a partir de `_r2_ci` doit egaler celui de `rollout_r2`), qui
    n'aurait sinon ete exerce qu'en pleine nuit sur les vraies donnees.
    """
    rng = np.random.default_rng(0)
    from mirage.state import RET_IDX, STATE_COLS

    d = len(STATE_COLS)
    spread_idx = STATE_COLS.index("spread_rel")
    days = []
    for _ in range(6):
        s = rng.normal(0.0, 1.0, (400, d))
        s[:, RET_IDX] = rng.normal(0.0, 1e-4, 400)   # rendements en relatif
        s[:, spread_idx] = 2e-4                      # demi-spread ~ 1 bp
        days.append(s)

    npz, rows = phase2b.eval_symbol_a2("SYNTH", days)

    assert npz["n_oos"] > 0
    assert npz["symbole"] == "SYNTH"
    # Les agregats par jour de chaque bras focal du prereg A2, et du myope.
    for bras in ("myope", "agent_causal_A2@2", "agent_direct@2"):
        for prefix in ("gi_", "gr_", "dp_", "dph_"):
            assert prefix + bras in npz, f"{prefix}{bras} manquant"
    # Les trois lignes de R² (rollout a 1, 5 et H pas, plus la cible cumulee).
    types = [r["type"] for r in rows]
    assert types.count("rollout_1pas") == 3 and types.count("direct_yH") == 1
    for r in rows:
        assert np.isfinite(r["r2"]) and r["lo"] <= r["r2"] <= r["hi"]
