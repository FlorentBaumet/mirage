"""OFI evenementiel Bybit : justesse du calcul et causalite.

Deux proprietes sont testees, et la seconde est la raison d'etre du fichier :

  1. JUSTESSE - sur un flux synthetique dont chaque contribution est calculable a la
     main, la serie produite doit etre exactement celle qu'on a calculee.
  2. CAUSALITE (anti-lookahead) - tronquer le flux apres l'instant T ne doit RIEN
     changer aux secondes <= T. C'est la definition meme de l'absence de fuite : si
     ajouter des evenements futurs modifiait le passe, une feature batie dessus
     utiliserait de l'information non encore disponible.
"""
from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from mirage.data.bybit_ofi import load_ofi
from mirage.features import _ofi_best_level


def _line(ts: int, b, a, typ: str = "delta") -> str:
    return json.dumps({"topic": "orderbook.500.TESTUSDT", "type": typ, "ts": ts,
                       "data": {"s": "TESTUSDT", "b": b, "a": a}})


# Flux de reference. Chaque etape est commentee avec la contribution OFI attendue.
STREAM = [
    _line(0, [[100, 1]], [[101, 1]], typ="snapshot"),   # etablit l'etat initial
    _line(100, [[100, 2]], []),      # bid 100 : 1 -> 2, meilleur inchange  => +1
    _line(1500, [], [[101, 0.5]]),   # ask 101 : 1 -> 0.5, meilleur inchange => +0.5
    _line(1600, [], [[100.5, 1]]),   # nouvel ask 100.5 : vente agressive   => -1
    _line(2000, [[100.5, 4]], []),   # nouvel bid 100.5, meilleur monte     => +4
    _line(3000, [[100.5, 0], [100, 5]], []),  # meilleur bid retire, recalcul => -4
    _line(3100, [[95, 7]], []),      # niveau PROFOND : ne touche pas le meilleur => 0
]

ATTENDU = {0: 1.0, 1: -0.5, 2: 4.0, 3: -4.0}


def _write_zip(path, lines):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("2024-01-01_TESTUSDT.data", "\n".join(lines) + "\n")
    return str(path)


def test_ofi_valeurs_calculees_a_la_main(tmp_path):
    z = _write_zip(tmp_path / "a.zip", STREAM)
    s = load_ofi(z)
    obtenu = {int(ts.timestamp()): v for ts, v in s.items()}
    assert obtenu == pytest.approx(ATTENDU)


def test_evenement_profond_ne_contribue_pas(tmp_path):
    """Le dernier evenement du flux touche le niveau 95 et doit etre sans effet."""
    sans = load_ofi(_write_zip(tmp_path / "b.zip", STREAM[:-1]))
    avec = load_ofi(_write_zip(tmp_path / "c.zip", STREAM))
    assert avec.iloc[-1] == sans.iloc[-1] == -4.0


def test_causalite_prefixe(tmp_path):
    """Ajouter des evenements FUTURS ne doit rien changer aux secondes deja closes.

    Le prefixe s'arrete a t=1600, donc a la frontiere de la seconde 1 : les deux
    premieres secondes sont completes de part et d'autre, et doivent coincider.
    """
    complet = load_ofi(_write_zip(tmp_path / "d.zip", STREAM))
    prefixe = load_ofi(_write_zip(tmp_path / "e.zip", STREAM[:4]))  # secondes 0 et 1 closes
    pd.testing.assert_series_equal(complet.reindex(prefixe.index), prefixe)


def test_troncature_en_milieu_de_seconde(tmp_path):
    """Tronquer AU MILIEU d'une seconde ne peut affecter que cette seconde-la.

    C'est la meme propriete vue depuis l'autre bout, et elle merite d'etre explicite :
    une seconde est l'unite indivisible du calcul. Tronquer pendant la seconde 1
    modifie legitimement la seconde 1 (des evenements manquent) mais doit laisser la
    seconde 0 rigoureusement intacte.
    """
    complet = load_ofi(_write_zip(tmp_path / "h.zip", STREAM))
    partiel = load_ofi(_write_zip(tmp_path / "i.zip", STREAM[:3]))  # seconde 1 incomplete
    assert partiel.loc[partiel.index[0]] == complet.loc[complet.index[0]] == 1.0


# --- Non-regression de la formule de Cont elle-meme --------------------------
#
# La version de `mirage.features._ofi_best_level` soustrayait le cote ask au lieu de
# l'additionner. Elle n'etait donc correcte que lorsque le prix ask NE BOUGEAIT PAS :
# une asymetrie invisible dans un score global, ou un OFI partiellement inverse reste
# informatif. Les cas ci-dessous ont ete calcules a la main depuis la formule de Cont,
# et incluent deliberement les deux cas ou le prix ask change.

CAS = [
    # (bid_avant, ask_avant, bid_apres, ask_apres, e_n attendu, motif)
    ((100, 5), (101, 3), (100, 5), (100.5, 2), -2.0, "ask qui DESCEND = vente agressive"),
    ((100, 5), (101, 3), (100, 5), (102, 4), +3.0, "ask qui monte = retrait"),
    ((100, 5), (101, 3), (100.5, 7), (101, 3), +7.0, "bid qui monte = achat agressif"),
    ((100, 5), (101, 3), (99, 2), (101, 3), -5.0, "bid qui descend = retrait"),
    ((100, 5), (101, 3), (100, 7), (101, 4), +1.0, "prix stables, tailles variables"),
]


@pytest.mark.parametrize("b0,a0,b1,a1,attendu,motif", CAS)
def test_formule_cont_bars(b0, a0, b1, a1, attendu, motif):
    """OFI sur barres (LOBSTER) : la formule de Cont, cas par cas."""
    bars = pd.DataFrame({
        "bid_price_1": [b0[0], b1[0]], "bid_size_1": [b0[1], b1[1]],
        "ask_price_1": [a0[0], a1[0]], "ask_size_1": [a0[1], a1[1]],
    })
    assert _ofi_best_level(bars).iloc[1] == pytest.approx(attendu), motif


def _delta(avant, apres):
    """Un delta ne porte QUE les niveaux modifies : il faut retirer l'ancien prix."""
    if avant[0] == apres[0]:
        return [list(apres)]
    return [[avant[0], 0], list(apres)]


@pytest.mark.parametrize("b0,a0,b1,a1,attendu,motif", CAS)
def test_formule_cont_evenementiel(tmp_path, b0, a0, b1, a1, attendu, motif):
    """Meme formule, version evenementielle : les deux doivent coincider."""
    lignes = [_line(100, [list(b0)], [list(a0)]),
              _line(200, _delta(b0, b1), _delta(a0, a1))]
    s = load_ofi(_write_zip(tmp_path / "k.zip", lignes))
    assert s.iloc[0] == pytest.approx(attendu), motif


def test_instantane_n_emet_pas_de_flux(tmp_path):
    """Un snapshot etablit l'etat sans produire de contribution."""
    lignes = [_line(0, [[100, 1]], [[101, 1]], typ="snapshot"),
              _line(200, [[100, 1]], [[101, 1]], typ="snapshot")]
    s = load_ofi(_write_zip(tmp_path / "f.zip", lignes))
    assert len(s) == 0


def test_index_alignable_sur_les_barres(tmp_path):
    """L'index doit etre un DatetimeIndex pose sur des frontieres de seconde."""
    s = load_ofi(_write_zip(tmp_path / "g.zip", STREAM))
    assert isinstance(s.index, pd.DatetimeIndex)
    assert (s.index.microsecond == 0).all()
    assert (s.index.nanosecond == 0).all()
