"""
optimizer.py — Ottimizzazione automatica delle soglie breadth.

Metodologia in 3 fasi:

  1. SCAN — per ogni soglia candidata in un range definito per indice,
     calcola le statistiche complete di backtest (hit rate, edge, p-value,
     segnali). Usa bootstrap a 2.000 iterazioni (rapido) nella fase di scan.

  2. SCORING COMPOSITO — ogni soglia riceve un punteggio [0-100] su 5 criteri
     pesati: hit rate 12M (25%), hit rate 24M (20%), edge normalizzato 12M (20%),
     significatività statistica (20%), qualità del conteggio segnali (15%).

  3. WALK-FORWARD IS/OOS — la soglia ottima (top score) viene rivalidata su
     un split IS 70% / OOS 30%: se il segnale è robusto l'OOS deve replicare
     il comportamento IS senza crollare. Questo testa la robustezza e il
     rischio di overfitting sulla storia passata.

Output: soglia ottima per ogni indice, metrics IS/OOS, ranking completo.
Nessuna chiamata API: lavora sui dati già in cache.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats

from .config import CACHE_TTL_DAY
from .calculations import compute_signals
from .backtest import (
    compute_signal_forward_returns,
    compute_unconditional_returns,
    build_backtest_stats,
    HORIZONS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE SCAN
# ═══════════════════════════════════════════════════════════════════════════════

# Range di scan per ogni indice: (min%, max%, step%)
SCAN_RANGES: dict[str, tuple] = {
    "SP500":  (4.0,  30.0, 1.0),
    "NASDAQ": (5.0,  35.0, 1.0),
    "DAX":    (2.0,  25.0, 1.0),
}

# Pesi del punteggio composito
SCORE_WEIGHTS = {
    "hit_12M":    0.25,   # hit rate orizzonte 12M
    "hit_24M":    0.20,   # hit rate orizzonte 24M
    "edge_12M":   0.20,   # edge normalizzato 12M (mean_signal - mean_uncond)
    "sig_score":  0.20,   # % orizzonti statisticamente significativi (p<0.05)
    "count_score": 0.15,  # qualità del conteggio segnali
}

# Zona "Goldilocks" per il numero di segnali
MIN_SIGNALS_PEAK = 10   # sotto = segnale troppo raro (fragile statisticamente)
MAX_SIGNALS_PEAK = 25   # sopra = segnale troppo frequente (noisy)
ABS_MIN_SIGNALS  = 5    # sotto = candidato scartato

N_BOOTSTRAP_SCAN  = 2_000   # bootstrap durante lo scan (rapido)
N_BOOTSTRAP_FINAL = 10_000  # bootstrap per la validazione finale IS/OOS
WALK_FORWARD_IS_PCT = 0.70  # 70% in-sample, 30% out-of-sample


# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_candidates(index_key: str) -> list[float]:
    """
    Genera la lista di soglie candidate per un dato indice.

    Args:
        index_key: Chiave indice ('SP500', 'NASDAQ', 'DAX')

    Returns:
        Lista di valori percentuali arrotondati al decimale
    """
    lo, hi, step = SCAN_RANGES.get(index_key, (4.0, 30.0, 1.0))
    return [round(v, 2) for v in np.arange(lo, hi + step * 0.1, step)]


def _count_score(n: int) -> float:
    """
    Punteggio [0-100] per il numero di segnali.

    Funzione a campana con picco nel range [MIN_SIGNALS_PEAK, MAX_SIGNALS_PEAK].
    Penalizza fortemente troppo pochi o troppi segnali.
    """
    if MIN_SIGNALS_PEAK <= n <= MAX_SIGNALS_PEAK:
        return 100.0
    elif n < MIN_SIGNALS_PEAK:
        return max(0.0, 60.0 - (MIN_SIGNALS_PEAK - n) * 12.0)
    else:
        return max(0.0, 80.0 - (n - MAX_SIGNALS_PEAK) * 2.0)


def _sig_score(stats_df: pd.DataFrame) -> float:
    """Percentuale di orizzonti con p-value < 0.05 (0-100)."""
    if stats_df.empty or "p-value vs incond." not in stats_df.columns:
        return 0.0
    col = stats_df["p-value vs incond."]
    return float((col < 0.05).mean() * 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN SOGLIE
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_single(
    breadth: pd.Series,
    index_price: pd.Series,
    threshold: float,
    uncond_fwd: pd.DataFrame,
    horizons: dict = HORIZONS,
    n_bootstrap: int = N_BOOTSTRAP_SCAN,
) -> dict | None:
    """
    Calcola le statistiche di backtest per una singola soglia candidata.

    Returns:
        Dict con le metriche chiave, oppure None se segnali insufficienti.
    """
    signals = compute_signals(breadth, threshold)
    n = len(signals)
    if n < ABS_MIN_SIGNALS:
        return None

    sig_fwd   = compute_signal_forward_returns(index_price, signals, horizons)
    stats_df  = build_backtest_stats(sig_fwd, uncond_fwd, horizons)

    if stats_df.empty:
        return None

    row: dict = {"threshold": threshold, "n_signals": n}

    for _, s in stats_df.iterrows():
        h = s["Orizzonte"]
        row[f"hit_{h}"]   = s["Hit rate (%)"]
        row[f"edge_{h}"]  = s["Media (%)"] - s["Media incond. (%)"]
        row[f"mean_{h}"]  = s["Media (%)"]
        row[f"pval_{h}"]  = s["p-value vs incond."]
        row[f"ci_lo_{h}"] = s["CI 95% inf (%)"]
        row[f"ci_hi_{h}"] = s["CI 95% sup (%)"]

    row["sig_score"] = _sig_score(stats_df)

    return row


@st.cache_data(ttl=CACHE_TTL_DAY, show_spinner=False)
def run_threshold_scan(
    breadth_values: tuple,
    breadth_index: tuple,
    price_values: tuple,
    price_index: tuple,
    index_key: str,
) -> pd.DataFrame:
    """
    Scansiona tutte le soglie candidate per un indice e restituisce il DataFrame
    con le statistiche e i punteggi compositi.

    Accetta tuple (hashable) per compatibilità con st.cache_data.

    Args:
        breadth_values / breadth_index: valori e index della serie breadth
        price_values / price_index:     valori e index del prezzo indice
        index_key:                      chiave indice per il range di scan

    Returns:
        DataFrame ordinato per composite_score decrescente
    """
    # Ricostruisce le Series dalle tuple
    breadth     = pd.Series(breadth_values, index=pd.DatetimeIndex(breadth_index))
    index_price = pd.Series(price_values,   index=pd.DatetimeIndex(price_index))

    candidates  = generate_candidates(index_key)
    uncond_fwd  = compute_unconditional_returns(index_price, HORIZONS)

    records = []
    for thr in candidates:
        result = _scan_single(breadth, index_price, thr, uncond_fwd,
                              n_bootstrap=N_BOOTSTRAP_SCAN)
        if result is not None:
            records.append(result)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return _score_and_rank(df)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING E RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def _score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica il punteggio composito e ordina per score decrescente.

    Normalizza l'edge 12M su [0, 100] prima di pesarlo,
    così da non favorire automaticamente le soglie con ritorni assoluti più alti.
    """
    df = df.copy()

    # Normalizzazione edge 12M
    e_col = "edge_12M"
    if e_col in df.columns:
        e_min, e_max = df[e_col].min(), df[e_col].max()
        span = e_max - e_min if e_max > e_min else 1.0
        df["edge_12M_norm"] = (df[e_col] - e_min) / span * 100.0
    else:
        df["edge_12M_norm"] = 50.0

    # Punteggio conteggio segnali
    df["count_score"] = df["n_signals"].apply(_count_score)

    # Punteggio composito
    hit_12m = df.get("hit_12M",      pd.Series(50.0, index=df.index))
    hit_24m = df.get("hit_24M",      pd.Series(50.0, index=df.index))
    edge_n  = df["edge_12M_norm"]
    sig_s   = df.get("sig_score",    pd.Series(0.0,  index=df.index))
    cnt_s   = df["count_score"]

    df["composite_score"] = (
        hit_12m * SCORE_WEIGHTS["hit_12M"]   +
        hit_24m * SCORE_WEIGHTS["hit_24M"]   +
        edge_n  * SCORE_WEIGHTS["edge_12M"]  +
        sig_s   * SCORE_WEIGHTS["sig_score"] +
        cnt_s   * SCORE_WEIGHTS["count_score"]
    ).round(2)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def get_optimal_threshold(scan_df: pd.DataFrame) -> float:
    """
    Restituisce la soglia con il punteggio composito più alto.

    Args:
        scan_df: Output di run_threshold_scan()

    Returns:
        Soglia ottima (float) oppure NaN se scan vuoto
    """
    if scan_df.empty or "composite_score" not in scan_df.columns:
        return float("nan")
    return float(scan_df.iloc[0]["threshold"])


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD IS/OOS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_DAY, show_spinner=False)
def run_walk_forward(
    breadth_values: tuple,
    breadth_index: tuple,
    price_values: tuple,
    price_index: tuple,
    optimal_threshold: float,
    in_sample_pct: float = WALK_FORWARD_IS_PCT,
) -> dict:
    """
    Valida la soglia ottima su un periodo out-of-sample.

    Split:
      - IS (In-Sample):  primi {in_sample_pct*100:.0f}% della storia breadth
      - OOS (Out-of-Sample): restanti {(1-in_sample_pct)*100:.0f}%

    Se la soglia è robusta, hit rate e edge OOS devono restare
    vicini ai valori IS (± ~15%). Un crollo nell'OOS segnala overfitting.

    Returns:
        Dict con keys: split_date, is_stats, oos_stats, is_signals, oos_signals
    """
    breadth     = pd.Series(breadth_values, index=pd.DatetimeIndex(breadth_index))
    index_price = pd.Series(price_values,   index=pd.DatetimeIndex(price_index))

    # Split
    split_pos  = int(len(breadth) * in_sample_pct)
    split_date = breadth.index[split_pos]

    b_is  = breadth.iloc[:split_pos]
    b_oos = breadth.iloc[split_pos:]
    p_is  = index_price.loc[index_price.index <= split_date]
    p_oos = index_price.loc[index_price.index >  split_date]

    # IS
    sigs_is    = compute_signals(b_is, optimal_threshold)
    uncond_is  = compute_unconditional_returns(p_is, HORIZONS)
    fwd_is     = compute_signal_forward_returns(p_is, sigs_is, HORIZONS)
    stats_is   = build_backtest_stats(fwd_is, uncond_is, HORIZONS)

    # OOS
    sigs_oos   = compute_signals(b_oos, optimal_threshold)
    uncond_oos = compute_unconditional_returns(p_oos, HORIZONS)
    fwd_oos    = compute_signal_forward_returns(p_oos, sigs_oos, HORIZONS)
    stats_oos  = build_backtest_stats(fwd_oos, uncond_oos, HORIZONS)

    return {
        "split_date":  split_date,
        "is_end":      b_is.index[-1],
        "oos_start":   b_oos.index[0],
        "oos_end":     b_oos.index[-1],
        "is_stats":    stats_is,
        "oos_stats":   stats_oos,
        "n_is":        len(sigs_is),
        "n_oos":       len(sigs_oos),
        "threshold":   optimal_threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SOMMARIO MULTI-INDICE
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer_summary(
    scan_results:  dict[str, pd.DataFrame],
    wf_results:    dict[str, dict],
    config_thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Costruisce la tabella riepilogativa con soglia default vs ottima per ogni indice.

    Args:
        scan_results:       {index_key: scan_df}
        wf_results:         {index_key: wf_dict}
        config_thresholds:  {index_key: threshold_default}

    Returns:
        DataFrame display-ready con una riga per indice
    """
    rows = []
    for key, scan_df in scan_results.items():
        if scan_df.empty:
            continue
        opt_thr   = get_optimal_threshold(scan_df)
        top_score = float(scan_df.iloc[0]["composite_score"])
        top_n     = int(scan_df.iloc[0]["n_signals"])
        top_hit12 = float(scan_df.iloc[0].get("hit_12M", float("nan")))
        top_hit24 = float(scan_df.iloc[0].get("hit_24M", float("nan")))
        top_sig   = float(scan_df.iloc[0].get("sig_score", float("nan")))

        wf   = wf_results.get(key, {})
        is_s = wf.get("is_stats", pd.DataFrame())
        oos_s= wf.get("oos_stats", pd.DataFrame())

        def _h(df, orz, col):
            if df.empty: return float("nan")
            m = df[df["Orizzonte"] == orz][col]
            return float(m.iloc[0]) if not m.empty else float("nan")

        rows.append({
            "Indice":              key,
            "Soglia default (%)":  config_thresholds.get(key, float("nan")),
            "Soglia ottima (%)":   opt_thr,
            "Score composito":     round(top_score, 1),
            "N segnali":           top_n,
            "Hit rate 12M (%)":    round(top_hit12, 1),
            "Hit rate 24M (%)":    round(top_hit24, 1),
            "Sign. orizzonti":     f"{top_sig:.0f}%",
            "Hit 12M IS (%)":      round(_h(is_s,  "12M", "Hit rate (%)"), 1),
            "Hit 12M OOS (%)":     round(_h(oos_s, "12M", "Hit rate (%)"), 1),
            "Edge 12M IS (%)":     round(_h(is_s,  "12M", "Media (%)") - _h(is_s,  "12M", "Media incond. (%)"), 1),
            "Edge 12M OOS (%)":    round(_h(oos_s, "12M", "Media (%)") - _h(oos_s, "12M", "Media incond. (%)"), 1),
        })

    return pd.DataFrame(rows)
