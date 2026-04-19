"""
calculations.py — Calcoli quantitativi puri: breadth, drawdown, segnali.

Tutte le funzioni sono pure (nessun effetto collaterale, nessun fetch API)
e ricevono DataFrame/Series già puliti da data_fetcher.py.
"""

import numpy as np
import pandas as pd

from .config import MA_PERIOD


# ═══════════════════════════════════════════════════════════════════════════════
# BREADTH: % COSTITUENTI SOPRA LA 200 MA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_breadth(closes: pd.DataFrame, ma_period: int = MA_PERIOD) -> pd.Series:
    """
    Calcola la percentuale giornaliera di costituenti con close > MA(ma_period).
    Include filtri per gestire dati API parziali e asincroni.
    """
    if closes.empty:
        return pd.Series(dtype=float)

    # 1. Filtro dati incompleti: calcola quanti ticker hanno un dato valido oggi
    daily_counts = closes.notna().sum(axis=1)
    
    # Trova il massimo dei titoli attivi negli ultimi 10 giorni
    recent_max = daily_counts.rolling(window=10, min_periods=1).max()
    
    # Mantieni solo i giorni in cui ha scambiato almeno l'80% dei titoli "normali"
    valid_days = daily_counts >= (recent_max * 0.8)
    closes_clean = closes.loc[valid_days].copy()

    # 2. Imputazione: Forward fill per gestire gap fisiologici o ritardi di 1-2 giorni
    closes_clean = closes_clean.ffill(limit=3)

    # Calcolo della Breadth
    sma = closes_clean.rolling(window=ma_period, min_periods=ma_period).mean()
    above = (closes_clean > sma) & closes_clean.notna() & sma.notna()
    valid_count = sma.notna().sum(axis=1)

    breadth = above.sum(axis=1) / valid_count.replace(0, np.nan) * 100
    breadth.name = "breadth_pct"
    
    return breadth.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# DRAWDOWN DELL'INDICE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_drawdown(index_price: pd.Series) -> pd.Series:
    """
    Calcola il drawdown percentuale dal massimo rolling dell'indice.

    Drawdown(t) = (Price(t) / max(Price[0..t]) - 1) * 100

    Args:
        index_price: pd.Series con prezzi dell'indice

    Returns:
        pd.Series con valori ≤ 0 (percentuale di calo dal picco)
    """
    if index_price.empty:
        return pd.Series(dtype=float)

    rolling_max = index_price.cummax()
    dd = (index_price / rolling_max - 1.0) * 100.0
    dd.name = "drawdown_pct"
    return dd


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DI MERCATO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_regime(
    breadth: pd.Series,
    threshold: float,
    extreme_mult: float = 0.5,
) -> pd.Series:
    """
    Classifica ogni data in uno dei 3 regimi basati sulla breadth %.

    Regimi:
      - 'healthy'  → breadth > threshold            (mercato sano, verde)
      - 'caution'  → threshold/2 < breadth ≤ threshold  (attenzione, rosso)
      - 'extreme'  → breadth ≤ threshold * extreme_mult  (zona acquisto estrema, blu)

    Args:
        breadth:      Serie breadth % giornaliera
        threshold:    Soglia principale backtestata
        extreme_mult: Moltiplicatore per la soglia estrema (default 0.5)

    Returns:
        pd.Series di stringhe ('healthy', 'caution', 'extreme')
    """
    extreme_threshold = threshold * extreme_mult
    regime = pd.Series("healthy", index=breadth.index, dtype=str)
    regime[breadth <= threshold]          = "caution"
    regime[breadth <= extreme_threshold]  = "extreme"
    return regime


# ═══════════════════════════════════════════════════════════════════════════════
# SEGNALI: CROSSING SOTTO SOGLIA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(breadth: pd.Series, threshold: float) -> pd.DataFrame:
    """
    Individua tutti i periodi in cui la breadth è scesa sotto la soglia.

    Per ogni episodio (crossing sotto → crossing sopra) calcola:
      - data_entry:      primo giorno sotto soglia
      - data_exit:       primo giorno sopra soglia (o NaT se ancora attivo)
      - duration_days:   durata in giorni calendario
      - min_breadth:     minimo della breadth durante l'episodio
      - min_breadth_date: data del minimo

    Args:
        breadth:   Serie breadth %
        threshold: Soglia di segnale

    Returns:
        DataFrame con una riga per episodio, ordinato per data_entry
    """
    if breadth.empty:
        return pd.DataFrame()

    below = breadth <= threshold
    # Individua le transizioni: True = nuovo episodio inizia
    crossings = below.astype(int).diff().fillna(0)

    entries = breadth.index[crossings == 1].tolist()
    exits   = breadth.index[crossings == -1].tolist()

    # Se il primo dato è già sotto soglia, aggiungi come entry
    if below.iloc[0]:
        entries = [breadth.index[0]] + entries

    records = []
    for entry in entries:
        # Trova la prima exit dopo l'entry
        future_exits = [e for e in exits if e > entry]
        exit_date = future_exits[0] if future_exits else None

        # Slice dell'episodio
        if exit_date:
            episode = breadth.loc[entry:exit_date]
        else:
            episode = breadth.loc[entry:]

        min_val  = episode.min()
        min_date = episode.idxmin()

        records.append({
            "data_entry":       entry,
            "data_exit":        exit_date,
            "duration_days":    (exit_date - entry).days if exit_date else None,
            "min_breadth":      round(min_val, 2),
            "min_breadth_date": min_date,
            "attivo":           exit_date is None,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# KPI SOMMARIO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kpis(
    breadth: pd.Series,
    index_price: pd.Series,
    threshold: float,
) -> dict:
    """
    Calcola le metriche chiave per il KPI row della dashboard.

    Args:
        breadth:     Serie breadth %
        index_price: Serie prezzi indice
        threshold:   Soglia breadth

    Returns:
        Dizionario con le metriche principali
    """
    current_breadth = breadth.iloc[-1]
    prev_breadth    = breadth.iloc[-2] if len(breadth) > 1 else current_breadth
    delta_breadth   = current_breadth - prev_breadth

    current_price = index_price.iloc[-1] if not index_price.empty else np.nan
    ytd_start     = index_price[index_price.index.year == index_price.index[-1].year].iloc[0]
    ytd_return    = (current_price / ytd_start - 1) * 100 if not np.isnan(current_price) else np.nan

    signals = compute_signals(breadth, threshold)
    avg_duration = signals["duration_days"].dropna().mean() if not signals.empty else np.nan
    last_entry   = signals["data_entry"].iloc[-1] if not signals.empty else None
    is_active    = signals["attivo"].iloc[-1] if not signals.empty else False

    # Percentuale di tempo sotto soglia
    pct_time_below = (breadth <= threshold).mean() * 100

    return {
        "current_breadth": current_breadth,
        "delta_breadth":   delta_breadth,
        "threshold":       threshold,
        "is_below":        current_breadth <= threshold,
        "is_active":       is_active,
        "current_price":   current_price,
        "ytd_return":      ytd_return,
        "avg_duration":    avg_duration,
        "last_entry":      last_entry,
        "pct_time_below":  pct_time_below,
        "n_signals":       len(signals),
    }
