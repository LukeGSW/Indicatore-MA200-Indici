"""
config.py — Configurazione centralizzata del Breadth Monitor.

Tutti i parametri fissi (soglie backtestati, ticker indici, costanti)
sono definiti qui. Non modificare le soglie senza un nuovo backtest.
"""

# ── Configurazione indici ────────────────────────────────────────────────────
# index_code   : codice base per l'endpoint EODHD fundamentals (.INDX)
# price_ticker : ticker EODHD per lo storico prezzi dell'indice
# threshold    : soglia breadth % backtestata (zona di acquisto se sotto)
# extreme_mult : moltiplicatore per zona "estrema" (threshold * extreme_mult)

INDEX_CONFIG = {
    "SP500": {
        "label":        "S&P 500",
        "index_code":   "GSPC",
        "price_ticker": "GSPC.INDX",
        "threshold":    13.0,
        "extreme_mult": 0.5,          # zona estrema: < 6.5%
        "tab_icon":     "🇺🇸",
        "description": (
            "Il S&P 500 monitora le 500 aziende a maggiore capitalizzazione "
            "quotate negli USA. La soglia del **13%** identifica storicamente "
            "le fasi di ipervenduto estremo, con alta probabilità statistica "
            "di recupero nei 12 mesi successivi."
        ),
    },
    "NASDAQ": {
        "label":        "Nasdaq 100",
        "index_code":   "NDX",
        "price_ticker": "NDX.INDX",
        "threshold":    18.0,
        "extreme_mult": 0.5,          # zona estrema: < 9%
        "tab_icon":     "💻",
        "description": (
            "Il Nasdaq 100 include le 100 maggiori aziende non-finanziarie "
            "quotate al Nasdaq. La soglia del **18%** riflette la maggiore "
            "volatilità strutturale del comparto tecnologico rispetto all'S&P 500."
        ),
    },
    "DAX": {
        "label":        "DAX 40",
        "index_code":   "GDAXI",
        "price_ticker": "GDAXI.INDX",
        "threshold":    7.0,
        "extreme_mult": 0.5,          # zona estrema: < 3.5%
        "tab_icon":     "🇩🇪",
        "description": (
            "Il DAX 40 è il principale indice azionario tedesco, composto dalle "
            "40 blue chip di Deutsche Börse. La soglia del **7%** è più restrittiva "
            "per via della minore volatilità media e della struttura ciclica dell'indice."
        ),
    },
}

# ── Parametri tecnici ────────────────────────────────────────────────────────

MA_PERIOD      = 200          # periodi media mobile
HISTORY_START  = "1980-01-01" # data di inizio fetch (EODHD restituirà dal disponibile)
MAX_WORKERS    = 12           # thread paralleli per il fetch dei costituenti
CACHE_TTL_DAY  = 86_400       # TTL cache dati (24h) — breadth cambia lentamente
CACHE_TTL_HOUR = 3_600        # TTL cache prezzi indice (1h)
REQUEST_TIMEOUT = 25          # timeout singola chiamata EODHD (secondi)
MAX_RETRIES    = 3            # tentativi su errori transitori (429, 5xx)
RETRY_DELAY    = 2.0          # secondi di attesa base tra retry

# ── Palette colori ───────────────────────────────────────────────────────────

COLORS = {
    "background": "#0E0E1A",
    "surface":    "#1A1A2E",
    "surface2":   "#22223A",
    "text":       "#E0E0E0",
    "subtext":    "#9E9E9E",
    "grid":       "#2A2A4A",
    "healthy":    "#4CAF50",   # verde   — breadth > soglia
    "caution":    "#F44336",   # rosso   — threshold/2 < breadth ≤ soglia
    "extreme":    "#1565C0",   # blu     — breadth ≤ soglia/2 (zona acquisto estrema)
    "breadth":    "#CE93D8",   # viola   — linea breadth %
    "threshold":  "#42A5F5",   # azzurro — linea soglia tratteggiata
    "primary":    "#2196F3",
    "accent":     "#FF9800",
}
