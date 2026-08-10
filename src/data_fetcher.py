"""
data_fetcher.py — Fetch dati EODHD con caching e parallelismo.

Funzioni principali:
  - fetch_index_components()  : lista costituenti da fonti pubbliche gratuite
  - fetch_all_closes()        : prezzi close storici di tutti i costituenti (parallelo)
  - fetch_index_price()       : serie storica del prezzo dell'indice stesso

Nota sui costituenti: la lista NON arriva più da EODHD. L'endpoint
`fundamentals/{INDEX}.INDX` (filter=Components) richiede il piano *Fundamentals
Data* e non è incluso nell'abbonamento in uso; il recupero è stato spostato su
una catena di fonti pubbliche gratuite in src/constituents.py. La chiave EODHD
resta necessaria per tutti i prezzi, titoli e indici.
"""

import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .config import (
    HISTORY_START, MA_PERIOD, MAX_WORKERS,
    CACHE_TTL_DAY, CACHE_TTL_HOUR,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    EODHD_ALIASES,
)
from .constituents import ComponentsResult, get_index_components


# ═══════════════════════════════════════════════════════════════════════════════
# UTILS — retry e request base
# ═══════════════════════════════════════════════════════════════════════════════

def _get(url: str, params: dict, retries: int = MAX_RETRIES) -> requests.Response:
    """
    GET con retry automatico su errori transitori (429, 500, 502, 503).

    Args:
        url:     URL endpoint EODHD
        params:  Parametri query string
        retries: Numero massimo di tentativi

    Returns:
        Response con status 200

    Raises:
        requests.HTTPError per errori non transitori o tentativi esauriti
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 500, 502, 503) and attempt < retries:
                wait = RETRY_DELAY * attempt
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.RequestException:
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)
                continue
            raise
    raise RuntimeError(f"Esauriti {retries} tentativi per {url}")


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH COSTITUENTI INDICE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_DAY, show_spinner=False)
def fetch_index_components(index_key: str) -> ComponentsResult:
    """
    Recupera la lista dei costituenti correnti di un indice da fonti gratuite.

    Non richiede la chiave EODHD. Delega a src/constituents.py, che percorre una
    catena di fonti indipendenti (ETF holdings, CSV su GitHub, Wikipedia, API
    Nasdaq) e ripiega su uno snapshot versionato nel repo se sono tutte
    irraggiungibili.

    Args:
        index_key: Chiave di INDEX_CONFIG ('SP500', 'NASDAQ', 'DAX')

    Returns:
        ComponentsResult: ticker in formato EODHD ('AAPL.US', 'SAP.XETRA'),
        più fonte usata, data dichiarata dalla fonte ed eventuali avvisi.

    Raises:
        ComponentsError: se nessuna fonte è utilizzabile
    """
    return get_index_components(index_key)


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH STORICO SINGOLO TITOLO
# ═══════════════════════════════════════════════════════════════════════════════

def _download_close(ticker: str, start: str, api_key: str) -> Optional[pd.Series]:
    """Scarica adjusted_close per un ticker; None se assente o su errore."""
    url = f"https://eodhd.com/api/eod/{ticker}"
    try:
        resp = _get(url, params={
            "from":        start,
            "period":      "d",
            "api_token":   api_key,
            "fmt":         "json",
        })
        data = resp.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Preferisce adjusted_close; se assente usa close
        col = "adjusted_close" if "adjusted_close" in df.columns else "close"
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        return series if not series.empty else None

    except Exception:
        return None


def _fetch_close_single(ticker: str, start: str, api_key: str) -> tuple[str, Optional[pd.Series]]:
    """
    Scarica la serie storica dei prezzi adjusted_close per un singolo ticker.

    Wrapper non-cached usato internamente dal fetch parallelo.

    Se il ticker non restituisce nulla e in config è definito un alias, ritenta
    con quello: serve a coprire lo sfasamento temporale fra le liste pubbliche di
    costituenti e l'anagrafica EODHD quando una società cambia simbolo. La serie
    resta comunque etichettata con il ticker originale.

    Args:
        ticker:  Ticker EODHD (es. 'AAPL.US')
        start:   Data inizio YYYY-MM-DD
        api_key: Chiave EODHD

    Returns:
        Tupla (ticker, pd.Series con index DatetimeIndex) oppure (ticker, None) su errore
    """
    series = _download_close(ticker, start, api_key)

    if series is None and ticker in EODHD_ALIASES:
        series = _download_close(EODHD_ALIASES[ticker], start, api_key)

    return ticker, series


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH PARALLELO DI TUTTI I COSTITUENTI
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_DAY, show_spinner=False)
def fetch_all_closes(
    tickers: tuple[str, ...],   # tuple (hashable per cache)
    api_key: str,
    start: str = HISTORY_START,
) -> pd.DataFrame:
    """
    Scarica i prezzi adjusted_close storici di tutti i costituenti in parallelo.

    Usa ThreadPoolExecutor con MAX_WORKERS thread. Ogni colonna del DataFrame
    risultante corrisponde a un ticker; l'indice è DatetimeIndex comune.

    Args:
        tickers: Tupla di ticker EODHD (hashable per st.cache_data)
        api_key: Chiave EODHD
        start:   Data di inizio storico (default: HISTORY_START da config)

    Returns:
        DataFrame (date × ticker) con prezzi adjusted_close; NaN dove assente
    """
    results: dict[str, pd.Series] = {}
    total = len(tickers)

    progress = st.progress(0, text="Caricamento prezzi costituenti...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_fetch_close_single, t, start, api_key): t
            for t in tickers
        }
        done = 0
        for future in as_completed(future_map):
            ticker, series = future.result()
            if series is not None:
                results[ticker] = series
            done += 1
            pct = int(done / total * 100)
            progress.progress(pct, text=f"Caricati {done}/{total} costituenti...")

    progress.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH PREZZO INDICE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_HOUR, show_spinner=False)
def fetch_index_price(price_ticker: str, api_key: str, start: str = HISTORY_START) -> pd.Series:
    """
    Scarica lo storico prezzi dell'indice stesso (es. GSPC.INDX).

    Args:
        price_ticker: Ticker EODHD dell'indice (es. 'GSPC.INDX')
        api_key:      Chiave EODHD
        start:        Data di inizio YYYY-MM-DD

    Returns:
        pd.Series con index DatetimeIndex e valori close dell'indice
    """
    url = f"https://eodhd.com/api/eod/{price_ticker}"
    resp = _get(url, params={
        "from":      start,
        "period":    "d",
        "api_token": api_key,
        "fmt":       "json",
    })
    data = resp.json()
    if not data:
        return pd.Series(dtype=float)

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    series.name = price_ticker
    return series
