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
│   ├── data_fetcher.py       # Fetch EODHD con parallelismo e caching
│   ├── calculations.py       # Breadth, drawdown, regime, segnali
│   └── charts.py             # Grafici Plotly regime-colored
└── README.md
```

---

## Setup locale

### 1. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 2. Configura la chiave API EODHD

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

## Note tecniche

- **Costituenti:** lista corrente da endpoint EODHD `{INDEX}.INDX` fundamentals/Components
- **Survivorship bias:** i costituenti correnti vengono applicati retroattivamente (approccio standard per breadth real-time)
- **Cache:** 24h per prezzi costituenti, 1h per prezzo indice
- **Parallelismo:** `ThreadPoolExecutor` con 12 worker per il fetch dei prezzi storici
- **Prima esecuzione:** ~2-5 minuti per il download completo; le successive sono istantanee
