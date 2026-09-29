# Données — LOBSTER (Phase 0)

`data/raw/` est **gitignored** : on ne commit jamais les données.

## Récupérer le sample gratuit (réel)
1. Aller sur https://lobsterdata.com/info/DataSamples.php
2. Télécharger, pour la date **2012-06-21**, ce qu'attend `configs/phase0.yaml` :
   - **AMZN, GOOG, INTC, MSFT** — journée complète, niveau **10** ;
   - **AAPL** — niveau **50**, sur **09:30–10:30 seulement** (c'est le seul extrait
     L50 d'une heure du sample gratuit).
3. Dézipper dans `data/raw/` (fichiers à plat). Les noms attendus sont
   `{ticker}_{date}_{start_ms}_{end_ms}_message_{niveaux}.csv` (idem `_orderbook_`),
   soit dix fichiers :
   - `AMZN_2012-06-21_34200000_57600000_message_10.csv` / `..._orderbook_10.csv`
   - `GOOG_2012-06-21_34200000_57600000_message_10.csv` / `..._orderbook_10.csv`
   - `INTC_2012-06-21_34200000_57600000_message_10.csv` / `..._orderbook_10.csv`
   - `MSFT_2012-06-21_34200000_57600000_message_10.csv` / `..._orderbook_10.csv`
   - `AAPL_2012-06-21_34200000_37800000_message_50.csv` / `..._orderbook_50.csv`

Ces chemins sont reconstruits par `mirage.eval` depuis `data.raw_dir`, `data.date` et la
liste `data.instruments` (`ticker`, `start_ms`, `end_ms`, `file_level`) de
[`../configs/phase0.yaml`](../configs/phase0.yaml) : ne rien renommer, le config doit
tourner tel quel.

## Format LOBSTER (rappel)
**message** (6 colonnes) : `time, event_type, order_id, size, price, direction`
- `time` = secondes après minuit (la session 09:30→16:00 = 34200→57600 s).
- `event_type` : 1=new LO, 2=cancel partiel, 3=delete, 4=exec visible, 5=exec cachée, 6=cross, 7=halt.
- `price` = **dollars × 10000** (entier) → on divise par 10000.
- `direction` : 1=bid (buy), -1=ask (sell).

**orderbook** (4×niveaux colonnes) : pour 10 niveaux →
`ask_price_1, ask_size_1, bid_price_1, bid_size_1, ask_price_2, ...` (prix ×10000).
Chaque ligne = état du carnet **immédiatement après** l'event correspondant du message file
(les deux fichiers sont alignés 1:1, ligne à ligne).

## Pas de données sous la main ?
Génère un échantillon **synthétique** aux mêmes noms, sessions et profondeurs que le
config (les cinq instruments ci-dessus), depuis la racine du dépôt :
```powershell
.\.venv\Scripts\python.exe scripts\lobster\make_synthetic_lobster.py
.\.venv\Scripts\python.exe -m mirage.eval --config configs\phase0.yaml --raw-dir data\raw\synthetic
```
Le script écrit dans `data/raw/synthetic/` ; l'option `--raw-dir` de `mirage.eval`
pointe dessus sans dupliquer le protocole. C'est un **banc d'essai du code**, pas une
mesure de marché : les résultats obtenus sur ces données aléatoires n'ont **aucune
signification**. L'évaluateur y rend d'ailleurs NO-GO, ce qui est le comportement attendu.

## Autres jeux de données
- Klines Binance (OHLCV) : `scripts/crypto/download_binance.py`.
- Carnets Bybit L2 : `scripts/crypto/fetch_bybit_batch.py`, caches `.pkl` dans `data/raw/crypto_lob/`.

## Limite honnête (I3)
Le sample LOBSTER gratuit = **une seule journée (2012-06-21), 5 tickers**. Donc
out-of-sample limité à l'**intraday** (walk-forward sur fenêtres de la journée). Aucune
prétention à généraliser : c'est un MVP.
