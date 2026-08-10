"""
config.py — Configurazione centralizzata del Breadth Monitor.

Tutti i parametri fissi (soglie backtestati, ticker indici, costanti)
sono definiti qui. Non modificare le soglie senza un nuovo backtest.
"""

# ── Configurazione indici ────────────────────────────────────────────────────
# index_code       : codice base dell'indice (storicamente l'endpoint EODHD .INDX)
# price_ticker     : ticker EODHD per lo storico prezzi dell'indice
# threshold        : soglia breadth % backtestata (zona di acquisto se sotto)
# extreme_mult     : moltiplicatore per zona "estrema" (threshold * extreme_mult)
# expected_count   : range di costituenti plausibile — se la fonte esce da qui,
#                    la lista viene scartata e si passa alla fonte successiva
# sentinel_tickers : titoli che DEVONO comparire; la loro assenza smaschera un
#                    parser che ha letto la tabella sbagliata restituendo comunque
#                    un conteggio credibile

INDEX_CONFIG = {
    "SP500": {
        "label":        "S&P 500",
        "index_code":   "GSPC",
        "price_ticker": "GSPC.INDX",
        "threshold":    13.0,
        "extreme_mult": 0.5,          # zona estrema: < 6.5%
        "expected_count":   (480, 520),
        "sentinel_tickers": ("AAPL.US", "MSFT.US", "NVDA.US", "BRK-B.US"),
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
        # Non fissare 100: le doppie classi (GOOGL/GOOG, FOXA/FOX) portano il
        # numero di *titoli* sopra il numero di *società* dichiarato dall'indice.
        "expected_count":   (95, 110),
        "sentinel_tickers": ("AAPL.US", "MSFT.US", "NVDA.US", "GOOGL.US"),
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
        "expected_count":   (40, 40),
        "sentinel_tickers": ("SAP.XETRA", "SIE.XETRA", "ALV.XETRA"),
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


# ── Costituenti da fonti pubbliche gratuite ──────────────────────────────────
# L'endpoint EODHD `fundamentals/{INDEX}.INDX` richiede il piano Fundamentals
# Data e non è incluso nell'abbonamento in uso: i costituenti arrivano ora dalle
# catene di fonti definite in src/constituents.py. Vedi il README per l'elenco.

CONSTITUENTS_TIMEOUT        = 20   # timeout singola fonte costituenti (secondi)
CONSTITUENTS_MAX_STALE_DAYS = 10   # oltre questa età la fonte è declassata

# Rete di sicurezza per le rinomine di ticker. EODHD può restare indietro (o
# avanti) di qualche settimana rispetto alle liste pubbliche quando una società
# cambia simbolo. Se un ticker non restituisce prezzi si ritenta con l'alias.
#
# Oggi la mappa è VUOTA di proposito: al 10/08/2026 i simboli delle fonti
# coincidono con quelli di EODHD anche sulle rinomine recenti (MRSH, FISV, BNY,
# ECHO, PSKY, XYZ). Popolala solo quando la tabella di copertura in dashboard
# segnala un titolo mancante, nella forma  "TICKER.US": "ALTERNATIVA.US".
EODHD_ALIASES: dict[str, str] = {}

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
