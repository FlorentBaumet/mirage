"""Order Flow Imbalance (Cont, Kukanov & Stoikov) au meilleur niveau, EVENEMENTIEL.

Pourquoi ce module existe : les barres 1 s ne conservent que l'ETAT du carnet a la fin
de chaque seconde. Tout l'ecoulement d'ordres INTRA-seconde est perdu - or c'est
precisement ce que l'OFI mesure. Un OFI recalcule sur les barres (variation nette entre
deux snapshots) serait une quasi-fonction des features deja presentes ; il faut rejouer
le flux brut pour obtenir un OFI qui apporte une information nouvelle.

Formule (Cont et al. 2014). Pour un evenement n qui fait passer le carnet de n-1 a n,
avec (P^b, q^b) le meilleur bid et (P^a, q^a) le meilleur ask :

    e_n = 1{P^b_n >= P^b_{n-1}} q^b_n - 1{P^b_n <= P^b_{n-1}} q^b_{n-1}
        - 1{P^a_n <= P^a_{n-1}} q^a_n + 1{P^a_n >= P^a_{n-1}} q^a_{n-1}

    OFI(seconde t) = somme des e_n des evenements dont l'horodatage tombe dans t.

La meme formule est implementee pour LOBSTER dans `mirage.features._ofi_best_level`.
ATTENTION : cette derniere etait erronee des que le prix ask bougeait (voir le test de
non-regression dans tests/test_ofi_causal.py) ; elle est corrigee en meme temps.

CAUSALITE : OFI(t) ne depend que des evenements de la seconde t. Aucun evenement
posterieur n'entre dans la serie - la propriete est verifiee par
`tests/test_ofi_causal.py`.

Alignement : la barre d'indice T contient l'etat du carnet APRES les evenements de la
seconde T (c'est la convention de `bybit_lob.load_book_bars`). OFI(T) est donc la somme
des e_n de la seconde T - le flux qui a produit cet etat, contemporain du rendement
ret(T) = log(mid_T) - log(mid_{T-1}).
"""
from __future__ import annotations

import json

import pandas as pd

from mirage.data.bybit_lob import _iter_lines, _levels


def _best_bid(book: dict[float, float], cur, hi_added):
    """Meilleur bid apres application, sans rescanner le carnet si `cur` a survecu."""
    if cur is not None and cur in book:
        return hi_added if (hi_added is not None and hi_added > cur) else cur
    return max(book) if book else None


def _best_ask(book: dict[float, float], cur, lo_added):
    if cur is not None and cur in book:
        return lo_added if (lo_added is not None and lo_added < cur) else cur
    return min(book) if book else None


def _contribution(pb, qb, nb, nq, is_bid: bool) -> float:
    """Contribution SIGNEE d'un cote a e_n (Cont et al.). 0 si un etat est indefini.

    ATTENTION : ce n'est PAS la convention de `mirage.features._ofi_best_level`, qui
    est erronee des que le prix ask bouge (cf. le test de non-regression dans
    tests/test_ofi_causal.py). Ici on applique la formule de Cont telle quelle, et le
    site d'appel ADDITIONNE les deux cotes.
    """
    if pb is None or nb is None:
        return 0.0
    if is_bid:
        if nb > pb:            # le bid monte   -> achat agressif
            return nq
        if nb == pb:
            return nq - qb
        return -qb             # le bid descend -> retrait
    if nb < pb:                # l'ask descend  -> vente agressive
        return -nq
    if nb == pb:
        return qb - nq
    return qb                  # l'ask monte    -> retrait


def load_ofi(data_zip: str, freq_s: int = 1) -> pd.Series:
    """OFI evenementiel agrege par seconde, indexe comme les barres 1 s.

    Renvoie une Series (float) dont l'index est un DatetimeIndex aux frontieres de
    seconde, alignable tel quel sur la sortie de `load_book_bars`.
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    acc: dict[int, float] = {}
    bp = bq = ap = aq = None            # meilleur bid/ask (prix, taille) avant l'evenement

    for line in _iter_lines(data_zip):
        o = json.loads(line)
        typ, ts, b, a = _levels(o)
        bucket = ts // (1000 * freq_s)

        hi_bid = None
        for p, q in b:
            p, q = float(p), float(q)
            if q == 0:
                bids.pop(p, None)
            else:
                bids[p] = q
                if hi_bid is None or p > hi_bid:
                    hi_bid = p
        lo_ask = None
        for p, q in a:
            p, q = float(p), float(q)
            if q == 0:
                asks.pop(p, None)
            else:
                asks[p] = q
                if lo_ask is None or p < lo_ask:
                    lo_ask = p

        nbp = _best_bid(bids, bp, hi_bid)
        nap = _best_ask(asks, ap, lo_ask)
        nbq = bids.get(nbp) if nbp is not None else None
        naq = asks.get(nap) if nap is not None else None

        # Un evenement qui ne bouge pas le meilleur a une contribution nulle : rien a
        # accumuler. Le test evite surtout le cout du calcul sur l'ecrasante majorite
        # des evenements, qui portent sur des niveaux profonds.
        if typ != "snapshot" and (nbp, nbq, nap, naq) != (bp, bq, ap, aq):
            e = (_contribution(bp, bq, nbp, nbq, True)
                 + _contribution(ap, aq, nap, naq, False))
            acc[bucket] = acc.get(bucket, 0.0) + e

        bp, bq, ap, aq = nbp, nbq, nap, naq

    buckets = sorted(acc)
    return pd.Series(
        [acc[b] for b in buckets],
        index=pd.to_datetime([b * freq_s for b in buckets], unit="s"),
        dtype=float,
        name="ofi",
    )
