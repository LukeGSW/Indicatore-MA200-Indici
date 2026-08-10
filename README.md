# 📡 Breadth Monitor — Kriterion Quant

Dashboard Streamlit per il monitoraggio della **breadth di mercato**:
percentuale di costituenti sopra la media mobile a 200 periodi per S&P 500, Nasdaq 100 e DAX 40.

---

## Struttura repository

```
breadth-monitor/
├── app.py                    # Entry point Streamlit
├── requirements.txt
├── .streamlit/
│   ├── config.toml           # Tema dark
│   └── secrets.toml          # API key (NON committare — vedi .gitignore)
├── src/
│   ├── __init__.py
│   ├── config.py             # Soglie, ticker, costanti
│   ├── constituents.py       # Costituenti da fonti pubbliche gratuite
│   ├── constituents_snapshot.py  # Lista congelata (rete di sicurezza offline)
│   ├── data_fetcher.py       # Fetch prezzi EODHD con parallelismo e caching
│   ├── calculations.py       # Breadth, drawdown, regime, segnali
│   └── charts.py             # Grafici Plotly regime-colored
├── tools/
│   └── refresh_snapshot.py   # Rigenera lo snapshot dei costituenti
└── README.md
```

---

## Setup locale

### 1. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 2. Configura la chiave API EODHD

Serve per i **prezzi** (titoli e indici), non per i costituenti: è sufficiente il piano
EOD Historical Data, senza Fundamentals.

Crea il file `.streamlit/secrets.toml` (non committare mai):

```toml
EODHD_API_KEY = "la-tua-chiave-eodhd"
```

### 3. Avvia la dashboard

```bash
streamlit run app.py
```

---

## Deploy su Streamlit Cloud

1. Fai il push del repository su GitHub (`.streamlit/secrets.toml` è in `.gitignore`)
2. Vai su [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Connetti il repository e seleziona `app.py`
4. In **Settings → Secrets** incolla:
   ```toml
   EODHD_API_KEY = "la-tua-chiave-eodhd"
   ```
5. Clicca **Deploy**

---

## Indici monitorati

| Indice       | Ticker EODHD | Soglia | Estrema |
|-------------|-------------|--------|---------|
| S&P 500     | GSPC.INDX   | 13%    | 6.5%    |
| Nasdaq 100  | NDX.INDX    | 18%    | 9.0%    |
| DAX 40      | GDAXI.INDX  | 7%     | 3.5%    |

Le soglie sono **backtestrate su tutto lo storico disponibile** e non modificabili dall'interfaccia.

---

## Da dove arrivano i costituenti

L'endpoint EODHD `fundamentals/{INDEX}.INDX` (filter=Components) richiede il piano
**Fundamentals Data** e non è incluso nell'abbonamento EOD in uso. I costituenti
arrivano quindi da fonti pubbliche gratuite, **senza alcuna chiave API**; la chiave
EODHD resta necessaria per tutti i prezzi (titoli e indici `.INDX`, entrambi inclusi
nel piano EOD).

Ogni indice ha una catena di fonti indipendenti: si usa la prima che supera i
controlli di plausibilità ed è aggiornata, con fallback automatico alle successive.

| Indice | Primaria | Fallback |
|---|---|---|
| **S&P 500** | CSV GitHub `chinobing/historical_sp500_constituents` (rigenerato ogni 12h, **dichiara la propria data**) | CSV `datasets/s-and-p-500-companies` → Wikipedia EN (wikitext) |
| **Nasdaq 100** | Wikipedia EN `List of NASDAQ-100 companies` (wikitext) | `api.nasdaq.com` → holdings iShares Nasdaq 100 UCITS |
| **DAX 40** | Holdings iShares Core DAX UCITS (sito DE) | mirror CH dello stesso fondo → Wikipedia DE (REST) → Wikipedia DE (HTML) |

Verificate con download reale il **10/08/2026**: tutte le fonti di ciascun indice
restituiscono liste **identiche** (503 / 102 / 40), pur avendo lineage diversi
(CSV GitHub, Wikipedia, API Nasdaq, holdings BlackRock).

### Scelte di progetto

- **Nessuna dipendenza aggiuntiva.** Solo `requests` + stdlib: niente `lxml` o
  `beautifulsoup4`, quindi `requirements.txt` resta invariato. Wikipedia viene letta
  come wikitext/HTML con regex, non con `pandas.read_html`.
- **User-Agent obbligatorio, e diverso per host.** Wikimedia risponde `403` senza uno
  User-Agent descrittivo (per questo `pandas.read_html(url)` diretto non funziona mai);
  `api.nasdaq.com` fa invece l'opposto e resetta la connessione se lo User-Agent
  contiene una email o un URL.
- **Colonne lette per nome, mai per posizione.** Una fonte usa `symbol`, un'altra
  `Symbol`: leggere la prima colonna restituirebbe i nomi delle società al posto dei
  ticker, con conteggio corretto e nessun errore.
- **Controllo di staleness.** Il conteggio non rivela una lista vecchia: una lista di
  due mesi fa ha comunque 503 righe. Dove la fonte dichiara una data la si legge, e
  oltre `CONSTITUENTS_MAX_STALE_DAYS` (10 giorni) la fonte viene declassata.
- **Wikipedia EN `DAX` è esclusa di proposito:** restituisce 40 righe (quindi passa
  qualunque controllo sul conteggio) ma è sbagliata — riporta `PAH3` invece di `HOT`.
- **Snapshot di emergenza.** Se tutte le fonti falliscono si usa
  `src/constituents_snapshot.py`, versionato nel repo, e la dashboard segnala la
  modalità degradata. Per rigenerarlo: `python -m tools.refresh_snapshot`
  (accetta una lista solo se almeno due fonti indipendenti concordano).

### Conversione al formato ticker EODHD

- **USA** → `SIMBOLO.US`, con le share class scritte col **trattino**:
  `BRK.B` → `BRK-B.US`, `BF.B` → `BF-B.US` (gli unici due casi su 518 simboli).
- **DAX** → `SIMBOLO.XETRA`, nessuna eccezione. I suffissi numerici (`DB1`, `HEN3`,
  `MUV2`, `VOW3`…) fanno parte del simbolo e non vanno normalizzati. Anche Airbus e
  QIAGEN vanno presi su Xetra (`AIR.XETRA`, `QIA.XETRA`): usare `AIR.PA` / `QIA.AS`
  introdurrebbe un calendario di borsa disallineato dall'indice.

Se una società cambia simbolo e EODHD resta indietro, il titolo compare nella tabella
**"Ticker senza prezzi"** in dashboard: basta aggiungere una voce a `EODHD_ALIASES`
in `src/config.py` e viene ritentato automaticamente.

---

## Note tecniche

- **Survivorship bias:** i costituenti correnti vengono applicati retroattivamente (approccio standard per breadth real-time)
- **Denominatore:** i titoli senza 200 barre di storico (IPO recenti) sono esclusi dal calcolo, non contati come "sotto la MA200"
- **Copertura:** la dashboard dichiara quanti costituenti hanno effettivamente restituito prezzi ed elenca i mancanti
- **Cache:** 24h per prezzi costituenti e costituenti, 1h per prezzo indice
- **Parallelismo:** `ThreadPoolExecutor` con 12 worker per il fetch dei prezzi storici
- **Prima esecuzione:** ~2-5 minuti per il download completo; le successive sono istantanee
