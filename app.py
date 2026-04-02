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

from src.config import INDEX_CONFIG, MA_PERIOD
from src.data_fetcher import fetch_index_components, fetch_all_closes, fetch_index_price
from src.calculations import (
    compute_breadth, compute_drawdown, compute_regime,
    compute_signals, compute_kpis,
)
from src.charts import build_breadth_chart, build_price_chart, build_drawdown_chart


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
        st.rerun()

    st.divider()
    st.markdown("##### Soglie backtestrate")
    for key, cfg in INDEX_CONFIG.items():
        thr = cfg["threshold"]
        ext = thr * cfg["extreme_mult"]
        st.markdown(
            f"**{cfg['label']}** &nbsp; "
            f"<span style='color:#42A5F5'>{thr:.0f}%</span> | "
            f"estrema <span style='color:#1565C0'>{ext:.1f}%</span>",
            unsafe_allow_html=True,
        )
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
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# FUNZIONE RENDERING TAB — definita prima delle tab per evitare errori
# ═══════════════════════════════════════════════════════════════════════════════

def render_index_tab(index_key: str, cfg: dict, api_key: str) -> None:
    """
    Renderizza il contenuto completo di un tab indice.

    Flusso:
      1. Fetch costituenti → fetch prezzi parallelo → fetch prezzo indice
      2. Calcolo breadth, regime, drawdown, segnali, KPI
      3. Badge stato → KPI row → 3 grafici → tabella segnali → download → metodologia

    Args:
        index_key: Chiave in INDEX_CONFIG (es. 'SP500')
        cfg:       Dict di configurazione dell'indice
        api_key:   Chiave EODHD
    """
    label      = cfg["label"]
    index_code = cfg["index_code"]
    price_tick = cfg["price_ticker"]
    threshold  = cfg["threshold"]
    ext_mult   = cfg["extreme_mult"]
    ext_thr    = threshold * ext_mult

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
        f"{threshold:.0f}%",
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
        f"Tutti gli episodi in cui la breadth è scesa sotto la soglia del **{threshold:.0f}%**."
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
- Principale: **{threshold:.0f}%** → zona di potenziale accumulo
- Estrema: **{ext_thr:.1f}%** → stress massimo (crisi 2002, 2008-09, 2020)

**Regime colori (prezzo e drawdown):**
- 🟢 Verde: breadth > {threshold:.0f}%
- 🔴 Rosso: {ext_thr:.1f}% < breadth ≤ {threshold:.0f}%
- 🔵 Blu: breadth ≤ {ext_thr:.1f}%

**Fonte dati:** EODHD Historical Data API (prezzi adjusted_close giornalieri).
**Cache:** 24h. Per aggiornare: sidebar → "Svuota cache e ricarica".
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS — rendering dei 3 indici
# ═══════════════════════════════════════════════════════════════════════════════

tab_labels = [f"{cfg['tab_icon']} {cfg['label']}" for cfg in INDEX_CONFIG.values()]
tabs = st.tabs(tab_labels)

for tab_obj, (index_key, cfg) in zip(tabs, INDEX_CONFIG.items()):
    with tab_obj:
        render_index_tab(index_key, cfg, EODHD_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="breadth-footer">'
    f"Kriterion Quant — Breadth Monitor v1.0 &nbsp;|&nbsp; "
    f"Dati: EODHD &nbsp;|&nbsp; "
    f"Aggiornato: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    f"</div>",
    unsafe_allow_html=True,
)
