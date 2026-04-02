"""
backtest.py — Motore statistico per il backtest del segnale breadth.

Metodologia:
  1. Forward Return Analysis: per ogni segnale, rendimento indice a
     1M / 3M / 6M / 12M / 24M (trading days).
  2. Distribuzione incondizionata: rendimenti su TUTTI i periodi possibili
     (null distribution, vettorializzata).
  3. Test di Welch: il rendimento medio condizionato al segnale è
     statisticamente diverso da quello incondizionato?
  4. Bootstrap CI: intervallo di confidenza al 95% sulla media del segnale
     (10.000 ricampionamenti, robusto con pochi segnali).
  5. Max Adverse Excursion (MAE): drawdown massimo da ogni entry al minimo
     dell'episodio, prima del recupero sopra soglia.

Nessuna chiamata API: tutto lavora sui dati già in cache.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════════════════

# Orizzonti in trading days → etichetta leggibile
HORIZONS: dict[str, int] = {
    "1M":  21,
    "3M":  63,
    "6M":  126,
    "12M": 252,
    "24M": 504,
}

N_BOOTSTRAP = 10_000   # ricampionamenti bootstrap
BOOTSTRAP_SEED = 42    # riproducibilità


# ═══════════════════════════════════════════════════════════════════════════════
# RENDIMENTI FORWARD DAI SEGNALI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signal_forward_returns(
    index_price: pd.Series,
    signals_df: pd.DataFrame,
    horizons: dict = HORIZONS,
) -> pd.DataFrame:
    """
    Per ogni segnale (entry breadth sotto soglia), calcola il rendimento
    dell'indice ai vari orizzonti forward.

    Gestisce automaticamente:
      - date di entry non presenti nel price index (usa la prima data ≥ entry)
      - segnali con dati forward insufficienti (→ NaN, esclusi dalle stats)

    Args:
        index_price: pd.Series prezzi indice con DatetimeIndex
        signals_df:  Output di calculations.compute_signals()
        horizons:    Dict {label: trading_days}

    Returns:
        DataFrame con colonne: signal_date, actual_entry, 1M, 3M, 6M, 12M, 24M
    """
    if signals_df.empty or index_price.empty:
        return pd.DataFrame()

    price     = index_price.dropna().sort_index()
    price_idx = price.index
    price_arr = price.values
    records   = []

    for _, sig in signals_df.iterrows():
        entry_date = sig["data_entry"]

        # Prima data disponibile nel price index >= entry_date
        future = price_idx[price_idx >= entry_date]
        if future.empty:
            continue
        actual_entry = future[0]
        pos          = price_idx.get_loc(actual_entry)
        entry_px     = price_arr[pos]

        row = {"signal_date": entry_date, "actual_entry": actual_entry}
        for label, h in horizons.items():
            exit_pos = pos + h
            if exit_pos < len(price_arr):
                row[label] = (price_arr[exit_pos] / entry_px - 1.0) * 100.0
            else:
                row[label] = np.nan  # dati forward insufficienti

        records.append(row)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUZIONE INCONDIZIONATA (NULL DISTRIBUTION)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_unconditional_returns(
    index_price: pd.Series,
    horizons: dict = HORIZONS,
) -> pd.DataFrame:
    """
    Calcola i rendimenti su TUTTI i possibili periodi di partenza (benchmark).

    Implementazione vettorializzata: per ogni orizzonte h,
    entry[i] = price[i], exit[i] = price[i+h], i = 0..N-h-1.

    Questa è la distribuzione "null": cosa ci aspettiamo senza segnale?

    Args:
        index_price: pd.Series prezzi indice
        horizons:    Dict {label: trading_days}

    Returns:
        DataFrame con una colonna per orizzonte, una riga per ogni data valida
    """
    if index_price.empty:
        return pd.DataFrame()

    price = index_price.dropna().sort_index().values
    n     = len(price)

    results: dict[str, np.ndarray] = {}
    for label, h in horizons.items():
        if n > h:
            results[label] = (price[h:] / price[:n - h] - 1.0) * 100.0

    if not results:
        return pd.DataFrame()

    min_len = min(len(v) for v in results.values())
    return pd.DataFrame({k: v[:min_len] for k, v in results.items()})


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════════════

def _bootstrap_mean_ci(
    data: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, np.ndarray]:
    """
    Intervallo di confidenza bootstrap sulla media (percentile method).

    Con pochi segnali (tipicamente 10-15 per indice), il bootstrap è
    preferibile al t-interval classico perché non assume normalità.

    Args:
        data:   Array 1D dei rendimenti del segnale
        n_iter: Numero ricampionamenti (default 10.000)
        ci:     Livello di confidenza (default 0.95)
        seed:   Seed per riproducibilità

    Returns:
        (ci_lower, ci_upper, array_means_bootstrap)
    """
    if len(data) < 3:
        return np.nan, np.nan, np.array([])

    rng     = np.random.default_rng(seed=seed)
    samples = rng.choice(data, size=(n_iter, len(data)), replace=True)
    means   = samples.mean(axis=1)
    alpha   = (1.0 - ci) / 2.0
    lo      = float(np.percentile(means, alpha * 100.0))
    hi      = float(np.percentile(means, (1.0 - alpha) * 100.0))
    return lo, hi, means


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICHE COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════

def build_backtest_stats(
    signal_fwd: pd.DataFrame,
    uncond_fwd: pd.DataFrame,
    horizons: dict = HORIZONS,
) -> pd.DataFrame:
    """
    Tabella statistica completa per ogni orizzonte forward.

    Per ogni orizzonte calcola:
      - Statistiche descrittive del segnale (media, mediana, std, min, max)
      - Confronto con distribuzione incondizionata
      - Hit rate (% rendimenti positivi)
      - Outperformance rate (% segnale > media incondizionata)
      - Welch t-test vs incondizionato (p-value)
      - t-test a un campione vs 0 (p-value)
      - Bootstrap CI 95% sulla media del segnale

    Args:
        signal_fwd: Output di compute_signal_forward_returns()
        uncond_fwd: Output di compute_unconditional_returns()
        horizons:   Dict {label: trading_days}

    Returns:
        DataFrame display-ready (una riga per orizzonte)
    """
    rows = []

    for label in horizons:
        if label not in signal_fwd.columns or label not in uncond_fwd.columns:
            continue

        sig = signal_fwd[label].dropna().values
        unc = uncond_fwd[label].dropna().values

        if len(sig) < 3:
            continue

        # Welch t-test: μ_segnale ≠ μ_incondizionato?
        t_vs_unc, p_vs_unc = stats.ttest_ind(sig, unc, equal_var=False)

        # t-test a un campione: μ_segnale > 0?
        t_vs_0, p_vs_0 = stats.ttest_1samp(sig, popmean=0.0)

        # Bootstrap CI sulla media del segnale
        ci_lo, ci_hi, _ = _bootstrap_mean_ci(sig)

        rows.append({
            "Orizzonte":           label,
            "N segnali":           len(sig),
            "Media (%)":           round(float(sig.mean()), 2),
            "Mediana (%)":         round(float(np.median(sig)), 2),
            "Std (%)":             round(float(sig.std(ddof=1)), 2),
            "Min (%)":             round(float(sig.min()), 2),
            "Max (%)":             round(float(sig.max()), 2),
            "Media incond. (%)":   round(float(unc.mean()), 2),
            "Hit rate (%)":        round(float((sig > 0).mean() * 100.0), 1),
            "Outperf. rate (%)":   round(float((sig > unc.mean()).mean() * 100.0), 1),
            "CI 95% inf (%)":      round(ci_lo, 2) if not np.isnan(ci_lo) else np.nan,
            "CI 95% sup (%)":      round(ci_hi, 2) if not np.isnan(ci_hi) else np.nan,
            "p-value vs incond.":  round(float(p_vs_unc), 4),
            "p-value vs 0":        round(float(p_vs_0), 4),
            "Sign. 5%":            "✅" if p_vs_unc < 0.05 else "❌",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUZIONI RAW PER I GRAFICI
# ═══════════════════════════════════════════════════════════════════════════════

def get_return_distributions(
    signal_fwd: pd.DataFrame,
    uncond_fwd: pd.DataFrame,
    horizons: dict = HORIZONS,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Restituisce le distribuzioni raw (numpy arrays) per i grafici Plotly.

    Returns:
        Dict {horizon_label: {"signal": array, "uncond": array}}
    """
    result = {}
    for label in horizons:
        sig = signal_fwd[label].dropna().values if label in signal_fwd.columns else np.array([])
        unc = uncond_fwd[label].dropna().values if label in uncond_fwd.columns else np.array([])
        if len(sig) > 0:
            result[label] = {"signal": sig, "uncond": unc}
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAX ADVERSE EXCURSION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mae(
    index_price: pd.Series,
    signals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Max Adverse Excursion (MAE) per ogni episodio segnale.

    MAE = (min_price_nell_episodio / entry_price - 1) * 100

    Misura il drawdown massimo sofferto dall'entry prima del recupero sopra
    soglia. Utile per valutare il rischio immediato del segnale e per
    dimensionare correttamente le posizioni in accumulo.

    Args:
        index_price: pd.Series prezzi indice
        signals_df:  Output di compute_signals()

    Returns:
        DataFrame con colonne: signal_date, mae_pct, duration_days, exit_date
    """
    if index_price.empty or signals_df.empty:
        return pd.DataFrame()

    price   = index_price.dropna().sort_index()
    records = []

    for _, sig in signals_df.iterrows():
        entry_date = sig["data_entry"]
        exit_date  = sig.get("data_exit")

        # Slice dell'episodio
        if pd.notna(exit_date):
            episode = price.loc[entry_date:exit_date]
        else:
            episode = price.loc[entry_date:]

        if len(episode) < 2:
            continue

        entry_px = float(episode.iloc[0])
        min_px   = float(episode.min())
        mae_pct  = (min_px / entry_px - 1.0) * 100.0

        records.append({
            "signal_date":   entry_date,
            "mae_pct":       round(mae_pct, 2),
            "duration_days": sig.get("duration_days"),
            "exit_date":     exit_date,
            "attivo":        sig.get("attivo", False),
        })

    return pd.DataFrame(records) if records else pd.DataFrame()
