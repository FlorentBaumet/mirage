"""Batch carnet Bybit : telecharge -> reconstruit les barres 1 s -> supprime le zip.

Traite chaque (symbole, date) séquentiellement pour que le pic disque reste ~300 Mo
au lieu de plusieurs Go : on ne conserve que le cache .pkl (~28 Mo/jour-symbole).
Idempotent : saute tout (symbole, date) dont le .pkl existe déjà.

Les dates peuvent venir d'un pré-enregistrement (`--prereg KEY`) au lieu de la ligne de
commande : une liste de dates écrite dans un fichier figé ne peut pas être ajustée après
avoir vu un résultat, ce qui est le point.

`--with-ofi` extrait EN PLUS l'OFI événementiel du MÊME zip : les deux artefacts coûtent
un seul téléchargement (les deux passes de reconstruction restent, elles, distinctes).

Téléchargement en `.part` puis `os.replace` : un arrêt brutal laisse un `.part` qui ne
sera jamais pris pour un zip complet (l'ancienne version testait `os.path.exists(zip)`
et pouvait donc prendre un zip tronqué pour complet).

    python scripts/crypto/fetch_bybit_batch.py --symbols BTCUSDT ETHUSDT SOLUSDT \
        --dates 2025-05-08 2025-05-22 2025-06-02 2025-06-18 2025-07-09 2025-08-06
    python scripts/crypto/fetch_bybit_batch.py --prereg configs/phase2b_crypto_prereg.yaml \
        --prereg-key extension_88_jours.dates_neuves --with-ofi
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request

import yaml

from mirage.data.bybit_lob import load_book_bars
from mirage.data.bybit_ofi import load_ofi

BASE = "https://quote-saver.bycsi.com/orderbook/linear"


def _dig(cfg: dict, key: str):
    """Lit une clé pointée ('a.b.c') dans un dict YAML chargé."""
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise SystemExit(f"Cle '{key}' absente du prereg (bloc manquant : '{part}').")
        cur = cur[part]
    if not isinstance(cur, list) or not cur:
        raise SystemExit(f"Cle '{key}' : liste de dates attendue.")
    return [str(d) for d in cur]


def head_check(todo: list[tuple[str, str]]) -> int:
    """HEAD sur chaque URL : echoue tot si la couverture demandee n'existe pas."""
    manquants = []
    for i, (sym, date) in enumerate(todo, 1):
        stem = f"{date}_{sym}_ob500.data.zip"
        req = urllib.request.Request(f"{BASE}/{sym}/{stem}", method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=30):
                pass
        except urllib.error.HTTPError as e:
            manquants.append((date, sym, e.code))
        except urllib.error.URLError as e:
            manquants.append((date, sym, f"reseau: {e.reason}"))
        if i % 25 == 0:
            print(f"  HEAD {i}/{len(todo)}...", flush=True)
    if manquants:
        for date, sym, code in manquants:
            print(f"  ABSENT ({code}) : {date} {sym}", flush=True)
        raise SystemExit(f"Couverture incomplete : {len(manquants)} couple(s) absent(s). "
                         f"Un couple manquant est un ECHEC BLOQUANT, jamais un retrait "
                         f"silencieux.")
    print(f"  HEAD : {len(todo)} URL presentes.", flush=True)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--dates", nargs="+", default=None)
    ap.add_argument("--prereg", default=None, help="fichier YAML contenant la liste de dates")
    ap.add_argument("--prereg-key", default="donnees.dates",
                    help="cle pointee de la liste de dates dans --prereg")
    ap.add_argument("--out", default=os.path.join("data", "raw", "crypto_lob"))
    ap.add_argument("--with-ofi", action="store_true",
                    help="extrait aussi l'OFI evenementiel du meme zip")
    ap.add_argument("--ofi-out", default=os.path.join("data", "raw", "crypto_ofi"))
    ap.add_argument("--head", action="store_true",
                    help="verifie d'abord (HEAD) que les 220 URL existent")
    ap.add_argument("--keep-zip", action="store_true",
                    help="conserver le zip brut (par defaut il est supprime apres reconstruction)")
    args = ap.parse_args()

    if args.prereg:
        with open(args.prereg, encoding="utf-8") as f:
            dates = _dig(yaml.safe_load(f), args.prereg_key)
        print(f"Preregre {args.prereg} -> {args.prereg_key} : {len(dates)} dates", flush=True)
    elif args.dates:
        dates = list(args.dates)
    else:
        raise SystemExit("Fournir --dates ou --prereg.")

    os.makedirs(args.out, exist_ok=True)
    if args.with_ofi:
        os.makedirs(args.ofi_out, exist_ok=True)

    todo = [(s, d) for s in args.symbols for d in dates]
    print(f"{len(args.symbols)} symboles x {len(dates)} dates = {len(todo)} couples", flush=True)
    if args.head:
        head_check(todo)

    done = skipped = failed = 0
    for i, (sym, date) in enumerate(todo, 1):
        pkl = os.path.join(args.out, f"{date}_{sym}_1s_book.pkl")
        opkl = os.path.join(args.ofi_out, f"{date}_{sym}_ofi.pkl")
        if os.path.exists(pkl) and (not args.with_ofi or os.path.exists(opkl)):
            print(f"[{i}/{len(todo)}] skip (cache present) : {date} {sym}", flush=True)
            skipped += 1
            continue

        stem = f"{date}_{sym}_ob500.data.zip"
        zpath = os.path.join(args.out, stem)
        part = zpath + ".part"
        t0 = time.time()
        if not os.path.exists(zpath):
            try:
                urllib.request.urlretrieve(f"{BASE}/{sym}/{stem}", part)
                os.replace(part, zpath)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                code = getattr(e, "code", "reseau")
                print(f"[{i}/{len(todo)}] ABSENT ({code}) : {date} {sym}", flush=True)
                failed += 1
                for p in (part, zpath):
                    if os.path.exists(p):
                        os.remove(p)
                continue
        mb = os.path.getsize(zpath) / 1e6

        if not os.path.exists(pkl):
            bars = load_book_bars(zpath, levels=10, freq_s=1)
            bars.to_pickle(pkl)
            mid = (bars["ask_price_1"] + bars["bid_price_1"]) / 2
            spr = (bars["ask_price_1"] - bars["bid_price_1"]) / mid * 1e4
            nbar, midm, sprm = len(bars), float(mid.mean()), float(spr.mean())
            del bars, mid, spr
        else:
            nbar, midm, sprm = 0, float("nan"), float("nan")

        ofi_msg = ""
        if args.with_ofi and not os.path.exists(opkl):
            ofi = load_ofi(zpath)
            ofi.to_pickle(opkl)
            ofi_msg = f" | OFI {len(ofi)} s actives"
            del ofi

        if not args.keep_zip:
            os.remove(zpath)                      # le .pkl suffit ; le zip est re-telechargeable

        print(f"[{i}/{len(todo)}] OK {date} {sym} : {nbar} barres 1s | "
              f"zip {mb:.0f} Mo | mid~{midm:.2f} | spread~{sprm:.3f} bp{ofi_msg} | "
              f"{time.time() - t0:.0f}s", flush=True)
        done += 1

    print(f"\n=== Batch termine : {done} construits, {skipped} deja en cache, "
          f"{failed} absents ===", flush=True)


if __name__ == "__main__":
    main()
