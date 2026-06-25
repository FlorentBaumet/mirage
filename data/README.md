# Données — LOBSTER (Phase 0)

> ⚠️ `data/raw/` est **gitignored** (on ne commit jamais les données).

## Récupérer le sample gratuit (réel)
1. Aller sur https://lobsterdata.com/info/DataSamples.php
2. Télécharger le sample **AAPL** (ou AMZN/GOOG/INTC/MSFT), date **2012-06-21**, niveau **10**.
3. Dézipper dans `data/raw/`. Tu dois obtenir deux CSV (sans en-tête) :
   - `AAPL_2012-06-21_34200000_57600000_message_10.csv`
   - `AAPL_2012-06-21_34200000_57600000_orderbook_10.csv`

Les chemins attendus sont dans [`../configs/phase0.yaml`](../configs/phase0.yaml) (`data.message_file` / `data.orderbook_file`).

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
Génère un échantillon **synthétique** au même format (pour tester le pipeline / les tests
anti-fuite, **pas** pour conclure quoi que ce soit) :
```powershell
..\.venv\Scripts\python.exe ..\scripts\make_synthetic_lobster.py
```

## Limite honnête (I3)
Le sample = **1 seule journée, 1 ticker, 2012**. Donc out-of-sample limité au *intraday*
(walk-forward sur fenêtres de la journée). Aucune prétention à généraliser : c'est un MVP.
