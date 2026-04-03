"""
app.py — Breadth Monitor | Kriterion Quant

Dashboard Streamlit per il monitoraggio della breadth di mercato:
percentuale di costituenti sopra la media mobile a 200 periodi per
S&P 500, Nasdaq 100 e DAX 40.

Deployment: Streamlit Cloud — API key via st.secrets["EODHD_API_KEY"]
Locale:     .streamlit/secrets.toml → EODHD_API_KEY = "..."
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import INDEX_CONFIG, MA_PERIOD, COLORS
from src.data_fetcher import fetch_index_components, fetch_all_closes, fetch_index_price
from src.calculations import (
    compute_breadth, compute_drawdown, compute_regime,
    compute_signals, compute_kpis,
)
from src.charts import build_breadth_chart, build_price_chart, build_drawdown_chart
from src.backtest import (
    compute_signal_forward_returns, compute_unconditional_returns,
    build_backtest_stats, get_return_distributions, compute_mae,
    HORIZONS,
)
from src.backtest_charts import (
    build_box_comparison, build_hit_rate_chart,
    build_mean_bar_chart, build_mae_histogram, build_pvalue_heatmap,
    build_score_vs_threshold_chart, build_is_oos_comparison,
)
from src.optimizer import (
    run_threshold_scan, run_walk_forward,
    get_optimal_threshold, build_optimizer_summary,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Breadth Monitor | Kriterion Quant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="metric-container"] {
      background-color: #1A1A2E;
      border: 1px solid #2A2A4A;
      border-radius: 8px;
      padding: 12px 16px;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }
  .signal-badge {
      display: inline-block;
      background-color: #1565C0;
      color: #E0E0E0;
      padding: 5px 16px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 10px;
  }
  .signal-badge-ok { background-color: #1B5E20; }
  .opt-info-box {
      background-color: #1A2744;
      border: 1px solid #2196F3;
      border-radius: 8px;
      padding: 10px 16px;
      margin-bottom: 12px;
      font-size: 13px;
      color: #90CAF9;
  }
  .breadth-footer {
      text-align: center;
      color: #555577;
      font-size: 12px;
      margin-top: 40px;
      padding-top: 16px;
      border-top: 1px solid #2A2A4A;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# API KEY
# ═══════════════════════════════════════════════════════════════════════════════

try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except Exception:
    st.error(
        "❌ **Chiave API mancante.**\n\n"
        "Aggiungi `EODHD_API_KEY` in `.streamlit/secrets.toml` (locale) "
        "o in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — soglie ottimali (override dinamico)
# ═══════════════════════════════════════════════════════════════════════════════

if "optimal_thresholds" not in st.session_state:
    st.session_state["optimal_thresholds"] = {}    # {index_key: float}

if "optimizer_ran" not in st.session_state:
    st.session_state["optimizer_ran"] = False


def _get_threshold(index_key: str) -> float:
    """Restituisce la soglia ottima (se applicata) o quella di default."""
    opt = st.session_state["optimal_thresholds"]
    if index_key in opt:
        return opt[index_key]
    return INDEX_CONFIG[index_key]["threshold"]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📡 Breadth Monitor")
    st.markdown("**Kriterion Quant**")
    st.divider()
    st.markdown(f"**Indicatore:** % sopra **{MA_PERIOD} MA**")
    st.markdown("**Fonte:** EODHD Historical Data")
    st.markdown("**Cache:** 24 ore")
    st.divider()

    if st.button("🔄 Svuota cache e ricarica", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.session_state["optimal_thresholds"] = {}
        st.session_state["optimizer_ran"] = False
        st.rerun()

    st.divider()
    st.markdown("##### Soglie attive")
    for key, cfg in INDEX_CONFIG.items():
        thr_active  = _get_threshold(key)
        thr_default = cfg["threshold"]
        ext          = thr_active * cfg["extreme_mult"]
        is_custom    = key in st.session_state["optimal_thresholds"]
        label_extra  = " ✨" if is_custom else ""
        st.markdown(
            f"**{cfg['label']}**{label_extra} &nbsp; "
            f"<span style='color:#42A5F5'>{thr_active:.1f}%</span>"
            + (f" <span style='color:#9E9E9E'>(default {thr_default:.0f}%)</span>" if is_custom else "")
            + f" | estrema <span style='color:#1565C0'>{ext:.1f}%</span>",
            unsafe_allow_html=True,
        )

    if st.session_state["optimizer_ran"]:
        st.divider()
        if st.button("↩️ Ripristina soglie default", use_container_width=True, type="secondary"):
            st.session_state["optimal_thresholds"] = {}
            st.session_state["optimizer_ran"] = False
            st.rerun()

    st.divider()
    st.caption(f"Apertura: {datetime.now().strftime('%d/%m/%Y %H:%M')}")


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.title("📡 Breadth Monitor — % Costituenti sopra la 200 MA")
st.markdown("""
La dashboard monitora la **salute interna** di S&P 500, Nasdaq 100 e DAX 40 misurando
la quota di costituenti che trattano al di sopra della propria media mobile a 200 periodi.

> **Come leggere:** quando la percentuale scende sotto la linea tratteggiata (soglia backtestata),
> il mercato entra in **zona di potenziale accumulo**. Il colore **blu** indica la zona di stress estremo.
""")

# Banner se le soglie ottimali sono attive
if st.session_state["optimizer_ran"] and st.session_state["optimal_thresholds"]:
    thr_str = " | ".join(
        f"{INDEX_CONFIG[k]['label']}: {v:.1f}%"
        for k, v in st.session_state["optimal_thresholds"].items()
    )
    st.markdown(
        f'<div class="opt-info-box">✨ <b>Soglie ottimali attive</b> — {thr_str}</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# FUNZIONE RENDERING TAB INDICE
# ═══════════════════════════════════════════════════════════════════════════════

def render_index_tab(index_key: str, cfg: dict, api_key: str) -> None:
    """
    Renderizza il contenuto completo di un tab indice.

    Usa la soglia ottima da session_state se disponibile, altrimenti
    la soglia di default da INDEX_CONFIG.

    Flusso:
      1. Fetch costituenti → fetch prezzi parallelo → fetch prezzo indice
      2. Calcolo breadth, regime, drawdown, segnali, KPI
      3. Badge stato → KPI row → 3 grafici → tabella segnali → download → metodologia
    """
    label      = cfg["label"]
    index_code = cfg["index_code"]
    price_tick = cfg["price_ticker"]
    threshold  = _get_threshold(index_key)          # soglia attiva (default o ottima)
    ext_mult   = cfg["extreme_mult"]
    ext_thr    = threshold * ext_mult
    is_custom  = index_key in st.session_state["optimal_thresholds"]

    # ── 1. Fetch costituenti ─────────────────────────────────────────────────
    with st.spinner(f"📋 Caricamento costituenti {label}..."):
        try:
            tickers = fetch_index_components(index_code, api_key)
        except Exception as e:
            st.error(f"❌ Errore fetch costituenti {label}: {e}")
            return

    if not tickers:
        st.error(f"Nessun costituente trovato per {label}. Controlla il codice indice '{index_code}'.")
        return

    st.caption(f"Costituenti correnti caricati: **{len(tickers)}**")

    # ── 2. Fetch prezzi storici (parallelo, cache 24h) ───────────────────────
    with st.spinner(f"📥 Download storico prezzi {label} ({len(tickers)} titoli) — prima esecuzione: ~2-3 min..."):
        try:
            closes = fetch_all_closes(tuple(sorted(tickers)), api_key)
        except Exception as e:
            st.error(f"❌ Errore fetch prezzi {label}: {e}")
            return

    if closes.empty:
        st.error(f"Nessun dato prezzi disponibile per i costituenti di {label}.")
        return

    # ── 3. Fetch prezzo indice ───────────────────────────────────────────────
    with st.spinner(f"📈 Caricamento prezzo {label}..."):
        try:
            index_price = fetch_index_price(price_tick, api_key)
        except Exception as e:
            st.warning(f"⚠️ Prezzo indice non disponibile: {e}")
            index_price = pd.Series(dtype=float)

    # ── 4. Calcoli ───────────────────────────────────────────────────────────
    breadth  = compute_breadth(closes)
    drawdown = compute_drawdown(index_price) if not index_price.empty else pd.Series(dtype=float)
    regime   = compute_regime(breadth, threshold, ext_mult)
    signals  = compute_signals(breadth, threshold)
    kpis     = compute_kpis(breadth, index_price, threshold)

    # ── 5. Badge stato segnale ───────────────────────────────────────────────
    current_pct = kpis["current_breadth"]
    if current_pct <= ext_thr:
        badge_txt   = f"🔵 ZONA ESTREMA — {label}: {current_pct:.1f}% &lt; {ext_thr:.1f}%"
        badge_class = "signal-badge"
    elif current_pct <= threshold:
        badge_txt   = f"🔴 SOTTO SOGLIA — {label}: {current_pct:.1f}% &lt; {threshold:.0f}%"
        badge_class = "signal-badge"
    else:
        badge_txt   = f"✅ MERCATO SANO — {label}: {current_pct:.1f}% &gt; {threshold:.0f}%"
        badge_class = "signal-badge signal-badge-ok"

    st.markdown(f'<span class="{badge_class}">{badge_txt}</span>', unsafe_allow_html=True)

    if is_custom:
        default_thr = cfg["threshold"]
        st.markdown(
            f'<span style="color:#90CAF9; font-size:12px;">✨ Soglia ottimizzata attiva: '
            f'<b>{threshold:.1f}%</b> (default: {default_thr:.0f}%)</span>',
            unsafe_allow_html=True,
        )

    # ── 6. KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Breadth attuale",
        f"{current_pct:.1f}%",
        delta=f"{kpis['delta_breadth']:+.1f}% (1g)",
        delta_color="off",
    )
    c2.metric(
        "Soglia segnale",
        f"{threshold:.1f}%",
        delta=f"Estrema: {ext_thr:.1f}%",
        delta_color="off",
    )
    c3.metric(
        "Prezzo indice",
        f"{kpis['current_price']:,.0f}" if pd.notna(kpis["current_price"]) else "N/D",
        delta=(
            f"YTD: {kpis['ytd_return']:+.1f}%"
            if pd.notna(kpis.get("ytd_return", np.nan)) else None
        ),
    )
    c4.metric("N° segnali storici", str(kpis["n_signals"]))
    c5.metric(
        "Durata media segnale",
        f"{kpis['avg_duration']:.0f} gg" if pd.notna(kpis.get("avg_duration", np.nan)) else "N/D",
    )
    c6.metric(
        "% tempo sotto soglia",
        f"{kpis['pct_time_below']:.1f}%",
    )

    st.divider()

    # ── 7. Descrizione ───────────────────────────────────────────────────────
    st.markdown(cfg["description"])
    st.markdown(
        f"Lo storico copre **{closes.index[0].strftime('%b %Y')}** → "
        f"**{closes.index[-1].strftime('%b %Y')}** "
        f"({len(breadth):,} osservazioni giornaliere)."
    )

    # ── 8. Grafico breadth % ─────────────────────────────────────────────────
    st.subheader("📊 Breadth — % costituenti sopra la 200 MA")
    fig_breadth = build_breadth_chart(breadth, threshold, ext_mult, label)
    st.plotly_chart(fig_breadth, width="stretch", key=f"breadth_{index_key}")

    # ── 9. Grafico prezzo log ─────────────────────────────────────────────────
    if not index_price.empty:
        st.subheader("📈 Prezzo indice (scala log) — colore per regime breadth")
        fig_price = build_price_chart(index_price, regime, label)
        st.plotly_chart(fig_price, width="stretch", key=f"price_{index_key}")

    # ── 10. Grafico drawdown ──────────────────────────────────────────────────
    if not drawdown.empty:
        st.subheader("📉 Drawdown dal massimo storico — colore per regime breadth")
        fig_dd = build_drawdown_chart(drawdown, regime, label)
        st.plotly_chart(fig_dd, width="stretch", key=f"dd_{index_key}")

    st.divider()

    # ── 11. Tabella storica segnali ───────────────────────────────────────────
    st.subheader(f"📋 Storico segnali — {label}")
    st.markdown(
        f"Tutti gli episodi in cui la breadth è scesa sotto la soglia del **{threshold:.1f}%**."
    )

    if signals.empty:
        st.info("Nessun segnale nel periodo disponibile.")
    else:
        df_sig = signals.copy()
        df_sig["Inizio"]      = df_sig["data_entry"].dt.strftime("%d/%m/%Y")
        df_sig["Fine"]        = df_sig["data_exit"].apply(
            lambda x: x.strftime("%d/%m/%Y") if pd.notna(x) else "🔴 Attivo"
        )
        df_sig["Durata"]      = df_sig["duration_days"].apply(
            lambda x: f"{int(x)} gg" if pd.notna(x) else "—"
        )
        df_sig["Min breadth"] = df_sig["min_breadth"].apply(lambda x: f"{x:.1f}%")
        df_sig["Data minimo"] = df_sig["min_breadth_date"].dt.strftime("%d/%m/%Y")
        df_sig["Attivo"]      = df_sig["attivo"].apply(lambda x: "✅" if x else "")

        st.dataframe(
            df_sig[["Inizio", "Fine", "Durata", "Min breadth", "Data minimo", "Attivo"]],
            width="stretch",
            hide_index=True,
        )

    # ── 12. Download CSV ──────────────────────────────────────────────────────
    export_df = pd.DataFrame({"breadth_pct": breadth, "regime": regime})
    if not index_price.empty:
        export_df["index_price"] = index_price.reindex(export_df.index)
    if not drawdown.empty:
        export_df["drawdown_pct"] = drawdown.reindex(export_df.index)

    st.download_button(
        label=f"⬇️ Scarica dati {label} (CSV)",
        data=export_df.to_csv().encode("utf-8"),
        file_name=f"breadth_{index_key.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key=f"dl_{index_key}",
    )

    # ── 13. Metodologia ───────────────────────────────────────────────────────
    with st.expander("ℹ️ Metodologia e note tecniche"):
        st.markdown(f"""
**Calcolo:** per ogni giorno di trading, si conta il numero di costituenti di {label}
con prezzo adjusted_close > SMA({MA_PERIOD}) e si esprime la quota percentuale sul totale
di costituenti con SMA valida in quella data.

**Costituenti:** lista corrente scaricata dall'endpoint EODHD `{index_code}.INDX` (fundamentals/Components).
I costituenti correnti vengono applicati retroattivamente a tutto lo storico disponibile
(*survivorship bias* — approccio standard per indicatori breadth in tempo reale).

**Soglie (backtestrate su tutto lo storico):**
- Principale: **{threshold:.1f}%** → zona di potenziale accumulo{'  ✨ *ottimizzata*' if is_custom else ''}
- Estrema: **{ext_thr:.1f}%** → stress massimo (crisi 2002, 2008-09, 2020)

**Regime colori (prezzo e drawdown):**
- 🟢 Verde: breadth > {threshold:.1f}%
- 🔴 Rosso: {ext_thr:.1f}% < breadth ≤ {threshold:.1f}%
- 🔵 Blu: breadth ≤ {ext_thr:.1f}%

**Fonte dati:** EODHD Historical Data API (prezzi adjusted_close giornalieri).
**Cache:** 24h. Per aggiornare: sidebar → "Svuota cache e ricarica".
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNZIONE RENDERING TAB BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def render_backtest_tab(api_key: str) -> None:
    """
    Rendering della tab Backtest per tutti e 3 gli indici.

    Struttura:
      - Sezione Ottimizzatore: scan soglie, score composito, walk-forward IS/OOS
      - Heatmap p-value cross-indice (visione d'insieme)
      - Sub-tab per ogni indice con 4 grafici + tabella statistica + tabella MAE

    Riutilizza i dati già in cache (breadth + prezzi) — nessuna chiamata API aggiuntiva.
    """
    st.markdown("""
**Metodologia:** Event Study con Forward Return Analysis.
Per ogni segnale (crossing breadth sotto soglia), si misurano i rendimenti dell'indice
a **1M / 3M / 6M / 12M / 24M**. Si confrontano con la distribuzione incondizionata
(tutti i periodi possibili) tramite **Welch t-test** e **bootstrap CI 95%** (10.000 iterazioni).
Il **Max Adverse Excursion (MAE)** quantifica il drawdown massimo dall'entry prima del recupero.
""")
    st.divider()

    # ── Raccolta dati per tutti gli indici ───────────────────────────────────
    all_stats:  dict[str, pd.DataFrame] = {}
    index_data: dict[str, dict]         = {}

    for index_key, cfg in INDEX_CONFIG.items():
        label      = cfg["label"]
        index_code = cfg["index_code"]
        price_tick = cfg["price_ticker"]
        threshold  = _get_threshold(index_key)
        ext_mult   = cfg["extreme_mult"]

        with st.spinner(f"Caricamento dati {label} per backtest..."):
            try:
                tickers = fetch_index_components(index_code, api_key)
                closes  = fetch_all_closes(tuple(sorted(tickers)), api_key)
                i_price = fetch_index_price(price_tick, api_key)
            except Exception as e:
                st.warning(f"⚠️ Dati {label} non disponibili: {e}")
                continue

        if closes.empty or i_price.empty:
            st.warning(f"⚠️ Dati incompleti per {label}, indice saltato.")
            continue

        breadth  = compute_breadth(closes)
        signals  = compute_signals(breadth, threshold)
        sig_fwd  = compute_signal_forward_returns(i_price, signals, HORIZONS)
        unc_fwd  = compute_unconditional_returns(i_price, HORIZONS)
        stats_df = build_backtest_stats(sig_fwd, unc_fwd, HORIZONS)
        distrib  = get_return_distributions(sig_fwd, unc_fwd, HORIZONS)
        mae_df   = compute_mae(i_price, signals)

        all_stats[label]  = stats_df
        index_data[label] = {
            "index_key": index_key,
            "cfg":       cfg,
            "breadth":   breadth,
            "i_price":   i_price,
            "signals":   signals,
            "sig_fwd":   sig_fwd,
            "unc_fwd":   unc_fwd,
            "stats":     stats_df,
            "distrib":   distrib,
            "mae":       mae_df,
            "threshold": threshold,
        }

    if not index_data:
        st.error("Nessun dato disponibile per il backtest.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # SEZIONE OTTIMIZZATORE
    # ═══════════════════════════════════════════════════════════════════════════

    st.subheader("🎯 Ottimizzatore automatico delle soglie")
    st.markdown("""
L'ottimizzatore scansiona un range di soglie candidate per ogni indice e assegna un
**punteggio composito [0-100]** su 5 criteri: hit rate 12M (25%), hit rate 24M (20%),
edge normalizzato 12M (20%), significatività statistica (20%), qualità conteggio segnali (15%).

La soglia ottima viene poi validata con **walk-forward IS/OOS (70%/30%)** per verificare
la robustezza out-of-sample ed evitare overfitting sulla storia passata.
""")

    if st.button(
        "🚀 Esegui ottimizzazione soglie (scan + walk-forward)",
        type="primary",
        use_container_width=True,
        key="btn_run_optimizer",
    ):
        _run_optimizer(index_data)

    if st.session_state["optimizer_ran"]:
        _render_optimizer_results(index_data)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # HEATMAP P-VALUE CROSS-INDICE
    # ═══════════════════════════════════════════════════════════════════════════

    st.subheader("🗺️ Mappa significatività statistica — tutti gli indici")
    st.markdown(
        "P-value del Welch t-test (rendimento segnale vs incondizionato) per ogni indice e orizzonte. "
        "**Verde = statisticamente significativo a 5%.**"
    )
    fig_heatmap = build_pvalue_heatmap(all_stats)
    st.plotly_chart(fig_heatmap, width="stretch", key="pvalue_heatmap")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SUB-TAB PER OGNI INDICE
    # ═══════════════════════════════════════════════════════════════════════════

    sub_tab_labels = [
        f"{INDEX_CONFIG[k]['tab_icon']} {INDEX_CONFIG[k]['label']}"
        for k in INDEX_CONFIG
        if INDEX_CONFIG[k]["label"] in index_data
    ]
    sub_tabs = st.tabs(sub_tab_labels)

    for sub_tab, (label, data) in zip(sub_tabs, index_data.items()):
        with sub_tab:
            _render_single_backtest(label, data)


def _run_optimizer(index_data: dict) -> None:
    """
    Esegue la scansione delle soglie e il walk-forward per tutti gli indici
    e salva i risultati in session_state.
    """
    scan_results:  dict = {}
    wf_results:    dict = {}
    new_thresholds: dict = {}

    progress = st.progress(0.0, text="Ottimizzazione in corso...")

    items = list(index_data.items())
    n     = len(items)

    for i, (label, data) in enumerate(items):
        index_key = data["index_key"]
        breadth   = data["breadth"]
        i_price   = data["i_price"]

        progress.progress(
            (i / n) * 0.85,
            text=f"Scan soglie {label} ({i+1}/{n})...",
        )

        # Scan
        scan_df = run_threshold_scan(
            breadth_values=tuple(breadth.values.tolist()),
            breadth_index=tuple(breadth.index.tolist()),
            price_values=tuple(i_price.values.tolist()),
            price_index=tuple(i_price.index.tolist()),
            index_key=index_key,
        )
        scan_results[index_key] = scan_df

        if scan_df.empty:
            continue

        opt_thr = get_optimal_threshold(scan_df)
        new_thresholds[index_key] = opt_thr

        # Walk-forward
        progress.progress(
            (i / n) * 0.85 + 0.1,
            text=f"Walk-forward {label}...",
        )
        wf = run_walk_forward(
            breadth_values=tuple(breadth.values.tolist()),
            breadth_index=tuple(breadth.index.tolist()),
            price_values=tuple(i_price.values.tolist()),
            price_index=tuple(i_price.index.tolist()),
            optimal_threshold=opt_thr,
        )
        wf_results[index_key] = wf

    progress.progress(1.0, text="Ottimizzazione completata ✅")
    progress.empty()

    # Salva risultati in session_state
    st.session_state["scan_results"]  = scan_results
    st.session_state["wf_results"]    = wf_results
    st.session_state["opt_new_thr"]   = new_thresholds
    st.session_state["optimizer_ran"] = True

    st.success(
        f"✅ Ottimizzazione completata per {len(scan_results)} indici. "
        "Vedi i risultati qui sotto, poi clicca **Applica soglie ottimali** per aggiornare la dashboard."
    )


def _render_optimizer_results(index_data: dict) -> None:
    """
    Mostra i risultati dell'ottimizzazione (score, confronto IS/OOS, tabella riepilogativa)
    e offre il pulsante per applicare le soglie ottimali.
    """
    scan_results  = st.session_state.get("scan_results",  {})
    wf_results    = st.session_state.get("wf_results",    {})
    new_thresholds = st.session_state.get("opt_new_thr",  {})

    if not scan_results:
        return

    # ── Tabella riepilogativa ────────────────────────────────────────────────
    config_thresholds = {k: cfg["threshold"] for k, cfg in INDEX_CONFIG.items()}
    summary_df = build_optimizer_summary(scan_results, wf_results, config_thresholds)

    if not summary_df.empty:
        st.markdown("#### 📊 Riepilogo ottimizzazione — confronto default vs ottima")
        st.dataframe(summary_df, width="stretch", hide_index=True)

    # ── Pulsante applica ─────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button(
            "✨ Applica soglie ottimali",
            type="primary",
            use_container_width=True,
            key="btn_apply_thresholds",
        ):
            st.session_state["optimal_thresholds"] = new_thresholds.copy()
            st.success(
                "Soglie ottimali applicate! La dashboard si aggiornerà con le nuove soglie. "
                "Naviga su una tab indice per vedere i grafici aggiornati."
            )
            st.rerun()

    with col_info:
        if new_thresholds:
            thr_str = " | ".join(
                f"{INDEX_CONFIG[k]['label']}: **{v:.1f}%** "
                f"(default: {INDEX_CONFIG[k]['threshold']:.0f}%)"
                for k, v in new_thresholds.items()
            )
            st.markdown(f"Soglie ottimali: {thr_str}")

    st.divider()

    # ── Dettaglio per indice: score + IS/OOS ────────────────────────────────
    st.markdown("#### 🔍 Dettaglio per indice")

    opt_tabs = st.tabs([
        f"{INDEX_CONFIG[k]['tab_icon']} {INDEX_CONFIG[k]['label']}"
        for k in scan_results
        if not scan_results[k].empty
    ])

    valid_keys = [k for k in scan_results if not scan_results[k].empty]

    for tab, index_key in zip(opt_tabs, valid_keys):
        with tab:
            scan_df   = scan_results[index_key]
            wf_result = wf_results.get(index_key, {})
            opt_thr   = new_thresholds.get(index_key, float("nan"))
            label     = INDEX_CONFIG[index_key]["label"]

            col_sc, col_wf = st.columns(2)

            with col_sc:
                fig_score = build_score_vs_threshold_chart(scan_df, opt_thr, label)
                st.plotly_chart(fig_score, width="stretch", key=f"score_{index_key}")

            with col_wf:
                if wf_result:
                    fig_isoos = build_is_oos_comparison(wf_result, label)
                    st.plotly_chart(fig_isoos, width="stretch", key=f"isoos_{index_key}")
                else:
                    st.info("Walk-forward non disponibile per questo indice.")

            # Tabella top-10 candidati
            with st.expander(f"📋 Top-10 soglie candidate — {label}"):
                display_cols = [
                    c for c in [
                        "threshold", "composite_score", "n_signals",
                        "hit_12M", "hit_24M", "edge_12M", "sig_score",
                    ]
                    if c in scan_df.columns
                ]
                top10 = scan_df.head(10)[display_cols].copy()
                rename_map = {
                    "threshold":       "Soglia (%)",
                    "composite_score": "Score",
                    "n_signals":       "N segnali",
                    "hit_12M":         "Hit 12M (%)",
                    "hit_24M":         "Hit 24M (%)",
                    "edge_12M":        "Edge 12M (%)",
                    "sig_score":       "Sign. (%)",
                }
                top10.rename(columns={k: v for k, v in rename_map.items() if k in top10.columns}, inplace=True)
                st.dataframe(top10, width="stretch", hide_index=True)


def _render_single_backtest(label: str, data: dict) -> None:
    """
    Rendering backtest per un singolo indice all'interno della sub-tab.

    Mostra:
      1. KPI sintetici del backtest
      2. Box plot segnale vs incondizionato
      3. Media con CI 95%
      4. Hit rate e outperformance rate
      5. Distribuzione MAE
      6. Tabella statistica completa
      7. Tabella dettaglio segnali con forward returns
    """
    stats_df  = data["stats"]
    distrib   = data["distrib"]
    mae_df    = data["mae"]
    sig_fwd   = data["sig_fwd"]
    signals   = data["signals"]
    threshold = data["threshold"]

    if stats_df.empty:
        st.warning(f"Statistiche non disponibili per {label} (segnali insufficienti).")
        return

    # ── KPI sintetici ────────────────────────────────────────────────────────
    best_row = stats_df.loc[stats_df["Media (%)"].idxmax()]
    n_sig    = int(stats_df["N segnali"].iloc[0])
    n_sign   = int((stats_df["Sign. 5%"] == "✅").sum())
    best_hr  = float(stats_df["Hit rate (%)"].max())
    avg_mae  = float(mae_df["mae_pct"].mean()) if not mae_df.empty else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("N° segnali storici",    str(n_sig))
    c2.metric("Orizzonti sign. 5%",    f"{n_sign} / {len(stats_df)}")
    c3.metric("Best media (segnale)",  f"{best_row['Media (%)']:.1f}% @ {best_row['Orizzonte']}")
    c4.metric("Hit rate massimo",      f"{best_hr:.1f}%")
    c5.metric("MAE medio",             f"{avg_mae:.1f}%" if not np.isnan(avg_mae) else "N/D")

    st.divider()

    # ── Grafici in griglia 2×2 ────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Box plot: segnale vs incondizionato")
        fig_box = build_box_comparison(distrib, label)
        st.plotly_chart(fig_box, width="stretch", key=f"box_{label}")

    with col_r:
        st.markdown("##### Media rendimento + CI 95% bootstrap")
        fig_mean = build_mean_bar_chart(stats_df, label)
        st.plotly_chart(fig_mean, width="stretch", key=f"mean_{label}")

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("##### Hit rate e Outperformance rate")
        fig_hr = build_hit_rate_chart(stats_df, label)
        st.plotly_chart(fig_hr, width="stretch", key=f"hr_{label}")

    with col_r2:
        st.markdown("##### Distribuzione Max Adverse Excursion (MAE)")
        fig_mae = build_mae_histogram(mae_df, label)
        st.plotly_chart(fig_mae, width="stretch", key=f"mae_{label}")

    st.divider()

    # ── Tabella statistica completa ───────────────────────────────────────────
    st.markdown("##### 📊 Tabella statistica completa")
    st.markdown(
        "Media, mediana, CI 95% bootstrap, hit rate, p-value Welch test e "
        "t-test vs 0 per ogni orizzonte forward."
    )
    display_cols = [
        "Orizzonte", "N segnali", "Media (%)", "Mediana (%)", "Std (%)",
        "Media incond. (%)", "Hit rate (%)", "Outperf. rate (%)",
        "CI 95% inf (%)", "CI 95% sup (%)", "p-value vs incond.", "p-value vs 0", "Sign. 5%",
    ]
    st.dataframe(
        stats_df[[c for c in display_cols if c in stats_df.columns]],
        width="stretch", hide_index=True,
    )

    # ── Tabella segnali con forward returns ───────────────────────────────────
    st.markdown("##### 📋 Dettaglio segnali con rendimenti realizzati")
    if not sig_fwd.empty:
        sig_display = sig_fwd.copy()
        sig_display["Inizio"]     = sig_display["signal_date"].dt.strftime("%d/%m/%Y")
        sig_display["Entry eff."] = sig_display["actual_entry"].dt.strftime("%d/%m/%Y")
        for h in HORIZONS:
            if h in sig_display.columns:
                sig_display[h] = sig_display[h].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
                )
        show_cols = ["Inizio", "Entry eff."] + [h for h in HORIZONS if h in sig_display.columns]
        st.dataframe(sig_display[show_cols], width="stretch", hide_index=True)
    else:
        st.info("Nessun segnale con dati forward disponibili.")

    # ── Download ──────────────────────────────────────────────────────────────
    csv = stats_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Scarica statistiche {label} (CSV)",
        data=csv,
        file_name=f"backtest_{label.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key=f"dl_bt_{label}",
    )

    # ── Metodologia ───────────────────────────────────────────────────────────
    with st.expander("ℹ️ Dettagli metodologici"):
        st.markdown(f"""
**Forward Return Analysis:** rendimento dell'indice {label} da ogni entry segnale
a orizzonte fisso (in trading days: 1M=21, 3M=63, 6M=126, 12M=252, 24M=504).

**Distribuzione incondizionata:** rendimenti su TUTTI i possibili periodi di partenza
nell'intero storico disponibile (null distribution, implementazione vettorializzata).

**Welch t-test** (varianze diseguali): H₀ = media segnale = media incondizionata.
p-value < 0.05 → si rigetta H₀ al 5% → il segnale ha un rendimento medio significativamente diverso.

**Bootstrap CI 95%:** {10_000:,} ricampionamenti con sostituzione sulla distribuzione del segnale.
Più robusto del t-interval classico con N segnali ridotto (tipicamente 10-20 per indice).

**Max Adverse Excursion:** drawdown massimo dall'entry segnale al minimo realizzato
durante l'episodio (dalla data di crossing-down alla data di crossing-up sopra soglia {threshold:.1f}%).
Risponde a: *"quanto peggio può andare prima del recupero?"*

**Survivorship bias:** si usano i costituenti correnti su tutto lo storico (approccio standard per breadth real-time).
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS — rendering dei 3 indici + backtest
# ═══════════════════════════════════════════════════════════════════════════════

tab_labels = [f"{cfg['tab_icon']} {cfg['label']}" for cfg in INDEX_CONFIG.values()] + ["🔬 Backtest"]
tabs = st.tabs(tab_labels)
*index_tabs, backtest_tab = tabs

for tab_obj, (index_key, cfg) in zip(index_tabs, INDEX_CONFIG.items()):
    with tab_obj:
        render_index_tab(index_key, cfg, EODHD_API_KEY)

with backtest_tab:
    st.subheader("🔬 Backtest statistico del segnale breadth — tutti gli indici")
    render_backtest_tab(EODHD_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="breadth-footer">'
    f"Kriterion Quant — Breadth Monitor v2.0 &nbsp;|&nbsp; "
    f"Dati: EODHD &nbsp;|&nbsp; "
    f"Aggiornato: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    f"</div>",
    unsafe_allow_html=True,
)
