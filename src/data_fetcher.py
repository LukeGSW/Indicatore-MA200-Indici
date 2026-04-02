"""
data_fetcher.py — Fetch dati EODHD con caching e parallelismo.

Funzioni principali:
  - fetch_index_components()  : lista costituenti da {INDEX}.INDX fundamentals
  - fetch_all_closes()        : prezzi close storici di tutti i costituenti (parallelo)
  - fetch_index_price()       : serie storica del prezzo dell'indice stesso
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
)


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
def fetch_index_components(index_code: str, api_key: str) -> list[str]:
    """
    Recupera la lista dei costituenti correnti di un indice da EODHD.

    Usa l'endpoint fundamentals/{index_code}.INDX con filter=Components.
    Restituisce ticker in formato EODHD (es. 'AAPL.US', 'SAP.XETRA').

    Args:
        index_code: Codice base dell'indice (es. 'GSPC', 'NDX', 'GDAXI')
        api_key:    Chiave API EODHD

    Returns:
        Lista di ticker EODHD dei costituenti correnti
    """
    url = f"https://eodhd.com/api/fundamentals/{index_code}.INDX"
    resp = _get(url, params={"api_token": api_key, "filter": "Components", "fmt": "json"})
    raw = resp.json()

    tickers = []
    if isinstance(raw, dict):
        for comp in raw.values():
            if not isinstance(comp, dict):
                continue
            code     = comp.get("Code", "").strip()
            exchange = comp.get("Exchange", "").strip()
            if code and exchange:
                tickers.append(f"{code}.{exchange}")

    return sorted(set(tickers))


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH STORICO SINGOLO TITOLO
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_close_single(ticker: str, start: str, api_key: str) -> tuple[str, Optional[pd.Series]]:
    """
    Scarica la serie storica dei prezzi adjusted_close per un singolo ticker.

    Wrapper non-cached usato internamente dal fetch parallelo.

    Args:
        ticker:  Ticker EODHD (es. 'AAPL.US')
        start:   Data inizio YYYY-MM-DD
        api_key: Chiave EODHD

    Returns:
        Tupla (ticker, pd.Series con index DatetimeIndex) oppure (ticker, None) su errore
    """
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
            return ticker, None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Preferisce adjusted_close; se assente usa close
        col = "adjusted_close" if "adjusted_close" in df.columns else "close"
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        return ticker, series if not series.empty else None

    except Exception:
        return ticker, None


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
