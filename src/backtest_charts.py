"""
backtest_charts.py — Grafici Plotly per la tab backtest.

Quattro grafici per ogni indice:
  1. build_box_comparison()      → Box plot segnale vs incondizionato per orizzonte
  2. build_hit_rate_chart()      → Hit rate e outperformance rate per orizzonte
  3. build_mean_bar_chart()      → Media segnale vs incondizionata + CI 95%
  4. build_mae_histogram()       → Distribuzione MAE (max adverse excursion)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COLORS


# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def _base_layout(title: str = "", height: int = 400) -> dict:
    """Layout dark condiviso."""
    return dict(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"]), x=0.5, xanchor="center"),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, color=COLORS["text"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"],
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=55, b=50),
        height=height,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BOX PLOT: SEGNALE VS INCONDIZIONATO
# ═══════════════════════════════════════════════════════════════════════════════

def build_box_comparison(
    distributions: dict[str, dict[str, np.ndarray]],
    index_label: str,
) -> go.Figure:
    """
    Box plot comparativo: rendimenti condizionati al segnale (blu)
    vs rendimenti incondizionati (grigio) per ogni orizzonte.

    Permette di vedere visivamente se il segnale concentra i rendimenti
    in una zona migliore rispetto al caso base.

    Args:
        distributions: Output di backtest.get_return_distributions()
                       Dict {horizon: {"signal": array, "uncond": array}}
        index_label:   Nome dell'indice per il titolo

    Returns:
        go.Figure con box plot affiancati per orizzonte
    """
    fig = go.Figure()

    horizons = list(distributions.keys())

    # Trace "incondizionato" — un box per orizzonte
    for h in horizons:
        unc = distributions[h]["uncond"]
        fig.add_trace(go.Box(
            y=unc,
            name=h,
            legendgroup="incond",
            legendgrouptitle_text="Incondizionato",
            showlegend=(h == horizons[0]),
            marker_color=COLORS["subtext"],
            fillcolor="rgba(158,158,158,0.15)",
            line=dict(color=COLORS["subtext"], width=1),
            boxmean=True,
            boxpoints=False,
            offsetgroup="incond",
            hovertemplate=(
                f"<b>Incondizionato — {h}</b><br>"
                "Media: %{mean:.2f}%<br>"
                "Mediana: %{median:.2f}%<br>"
                "<extra></extra>"
            ),
        ))

    # Trace "segnale" — un box per orizzonte, sovrapposto
    for h in horizons:
        sig = distributions[h]["signal"]
        fig.add_trace(go.Box(
            y=sig,
            name=h,
            legendgroup="signal",
            legendgrouptitle_text="Segnale",
            showlegend=(h == horizons[0]),
            marker_color=COLORS["primary"],
            fillcolor="rgba(33,150,243,0.20)",
            line=dict(color=COLORS["primary"], width=1.5),
            boxmean=True,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            marker=dict(size=5, opacity=0.6),
            offsetgroup="signal",
            hovertemplate=(
                f"<b>Segnale — {h}</b><br>"
                "Media: %{mean:.2f}%<br>"
                "Mediana: %{median:.2f}%<br>"
                "N: " + str(len(sig)) + "<br>"
                "<extra></extra>"
            ),
        ))

    # Linea dello zero
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["subtext"], line_width=0.8)

    layout = _base_layout(
        title=f"Rendimenti condizionati al segnale vs incondizionati — {index_label}",
        height=440,
    )
    layout["xaxis"].update(title="Orizzonte forward")
    layout["yaxis"].update(title="Rendimento (%)", ticksuffix="%")
    layout["barmode"] = "group"
    fig.update_layout(**layout, boxmode="group")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HIT RATE E OUTPERFORMANCE RATE
# ═══════════════════════════════════════════════════════════════════════════════

def build_hit_rate_chart(stats_df: pd.DataFrame, index_label: str) -> go.Figure:
    """
    Grafico a barre doppie: hit rate (% rendimenti > 0) e outperformance rate
    (% rendimenti segnale > media incondizionata) per ogni orizzonte.

    Linea tratteggiata al 50% come riferimento neutralità.

    Args:
        stats_df:    Output di backtest.build_backtest_stats()
        index_label: Nome indice

    Returns:
        go.Figure
    """
    if stats_df.empty:
        return go.Figure()

    horizons = stats_df["Orizzonte"].tolist()
    hit      = stats_df["Hit rate (%)"].tolist()
    outperf  = stats_df["Outperf. rate (%)"].tolist()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=horizons, y=hit,
        name="Hit rate (ret > 0%)",
        marker_color=COLORS["healthy"],
        opacity=0.85,
        text=[f"{v:.1f}%" for v in hit],
        textposition="outside",
        hovertemplate="Hit rate %{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=horizons, y=outperf,
        name="Outperf. rate (ret > media incond.)",
        marker_color=COLORS["primary"],
        opacity=0.85,
        text=[f"{v:.1f}%" for v in outperf],
        textposition="outside",
        hovertemplate="Outperf. rate %{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))

    # Linea 50% (neutralità)
    fig.add_hline(
        y=50, line_dash="dash", line_color=COLORS["subtext"], line_width=1.2,
        annotation_text="50% (neutro)", annotation_position="bottom right",
        annotation_font=dict(color=COLORS["subtext"], size=10),
    )

    layout = _base_layout(
        title=f"Hit rate e Outperformance rate per orizzonte — {index_label}",
        height=380,
    )
    layout["xaxis"].update(title="Orizzonte forward")
    layout["yaxis"].update(title="Percentuale (%)", ticksuffix="%", range=[0, 110])
    fig.update_layout(**layout, barmode="group")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MEDIA SEGNALE VS INCONDIZIONATA + CI 95%
# ═══════════════════════════════════════════════════════════════════════════════

def build_mean_bar_chart(stats_df: pd.DataFrame, index_label: str) -> go.Figure:
    """
    Grafico a barre: media rendimento condizionato al segnale (blu) con
    error bar CI 95% bootstrap, vs media rendimento incondizionato (grigio).

    Evidenzia visivamente il vantaggio statistico del segnale.

    Args:
        stats_df:    Output di backtest.build_backtest_stats()
        index_label: Nome indice

    Returns:
        go.Figure
    """
    if stats_df.empty:
        return go.Figure()

    horizons    = stats_df["Orizzonte"].tolist()
    sig_mean    = stats_df["Media (%)"].tolist()
    unc_mean    = stats_df["Media incond. (%)"].tolist()
    ci_lo       = stats_df["CI 95% inf (%)"].tolist()
    ci_hi       = stats_df["CI 95% sup (%)"].tolist()
    sig_n       = stats_df["N segnali"].tolist()

    # Error bar: distanza dal CI al valore medio
    err_lo = [m - lo if not (np.isnan(m) or np.isnan(lo)) else 0
              for m, lo in zip(sig_mean, ci_lo)]
    err_hi = [hi - m if not (np.isnan(m) or np.isnan(hi)) else 0
              for m, hi in zip(sig_mean, ci_hi)]

    fig = go.Figure()

    # Barre incondizionate
    fig.add_trace(go.Bar(
        x=horizons, y=unc_mean,
        name="Media incondizionata",
        marker_color=COLORS["subtext"],
        opacity=0.6,
        hovertemplate="Incond. %{x}: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # Barre segnale con CI
    fig.add_trace(go.Bar(
        x=horizons, y=sig_mean,
        name="Media segnale (CI 95% bootstrap)",
        marker_color=COLORS["primary"],
        opacity=0.9,
        error_y=dict(
            type="data",
            symmetric=False,
            array=err_hi,
            arrayminus=err_lo,
            color=COLORS["accent"],
            thickness=2,
            width=6,
        ),
        text=[f"N={n}" for n in sig_n],
        textposition="outside",
        hovertemplate=(
            "Segnale %{x}: <b>%{y:.2f}%</b><br>"
            "CI 95%: [%{error_y.arrayminus:.2f}% / +%{error_y.array:.2f}%]"
            "<extra></extra>"
        ),
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["subtext"], line_width=0.8)

    layout = _base_layout(
        title=f"Media rendimento segnale vs incondizionato + CI 95% — {index_label}",
        height=380,
    )
    layout["xaxis"].update(title="Orizzonte forward")
    layout["yaxis"].update(title="Rendimento medio (%)", ticksuffix="%")
    fig.update_layout(**layout, barmode="group")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DISTRIBUZIONE MAE
# ═══════════════════════════════════════════════════════════════════════════════

def build_mae_histogram(mae_df: pd.DataFrame, index_label: str) -> go.Figure:
    """
    Istogramma della distribuzione del Max Adverse Excursion (MAE).

    Il MAE misura il drawdown massimo dall'entry segnale al minimo dell'episodio,
    prima del recupero sopra soglia. Risponde a: "quanto peggio va prima di migliorare?"

    Linee di riferimento a -5%, -10%, -20% per inquadrare il rischio.

    Args:
        mae_df:      Output di backtest.compute_mae()
        index_label: Nome indice

    Returns:
        go.Figure
    """
    if mae_df.empty or "mae_pct" not in mae_df.columns:
        return go.Figure()

    mae_vals = mae_df["mae_pct"].dropna().values
    if len(mae_vals) == 0:
        return go.Figure()

    mean_mae   = float(mae_vals.mean())
    median_mae = float(np.median(mae_vals))

    # Colore per ogni barra: rosso se < -10%, arancio se < -5%, verde altrimenti
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=mae_vals,
        nbinsx=max(8, len(mae_vals) // 2),
        name="MAE (%)",
        marker=dict(
            color=COLORS["caution"],
            opacity=0.85,
            line=dict(color=COLORS["surface"], width=0.5),
        ),
        hovertemplate="MAE: %{x:.1f}%<br>Frequenza: %{y}<extra></extra>",
    ))

    # Linee di riferimento verticali
    for level, label, color in [
        (-5,  "-5%",  COLORS["accent"]),
        (-10, "-10%", COLORS["caution"]),
        (-20, "-20%", COLORS["extreme"]),
    ]:
        fig.add_vline(
            x=level, line_dash="dash", line_color=color, line_width=1.2,
            annotation_text=label,
            annotation_position="top",
            annotation_font=dict(color=color, size=10),
        )

    # Media e mediana
    fig.add_vline(
        x=mean_mae, line_dash="solid", line_color=COLORS["primary"], line_width=1.8,
        annotation_text=f"Media: {mean_mae:.1f}%",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["primary"], size=11),
    )
    fig.add_vline(
        x=median_mae, line_dash="dot", line_color=COLORS["healthy"], line_width=1.5,
        annotation_text=f"Mediana: {median_mae:.1f}%",
        annotation_position="bottom right",
        annotation_font=dict(color=COLORS["healthy"], size=11),
    )

    layout = _base_layout(
        title=f"Distribuzione Max Adverse Excursion da entry segnale — {index_label}",
        height=360,
    )
    layout["xaxis"].update(title="MAE (%)", ticksuffix="%")
    layout["yaxis"].update(title="Frequenza (N segnali)")
    layout.pop("hovermode", None)
    fig.update_layout(**layout)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFICO RIASSUNTIVO — HEATMAP P-VALUE
# ═══════════════════════════════════════════════════════════════════════════════

def build_pvalue_heatmap(
    stats_all: dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Heatmap dei p-value (vs incondizionato) per tutti gli indici × orizzonti.

    Celle verdi = significativo a 5%; celle rosse = non significativo.
    Permette di confrontare la qualità del segnale tra indici e orizzonti.

    Args:
        stats_all: Dict {index_label: stats_df} da build_backtest_stats()

    Returns:
        go.Figure heatmap
    """
    if not stats_all:
        return go.Figure()

    # Costruisce matrice p-value: righe = indici, colonne = orizzonti
    indices  = list(stats_all.keys())
    horizons = stats_all[indices[0]]["Orizzonte"].tolist() if indices else []

    matrix = []
    for idx_label in indices:
        df = stats_all[idx_label]
        row = []
        for h in horizons:
            match = df[df["Orizzonte"] == h]["p-value vs incond."]
            row.append(float(match.iloc[0]) if not match.empty else np.nan)
        matrix.append(row)

    z      = np.array(matrix, dtype=float)
    text   = [[f"{v:.4f}" if not np.isnan(v) else "N/D" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=horizons,
        y=indices,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12, color=COLORS["text"]),
        colorscale=[
            [0.0,  "#1B5E20"],   # verde scuro = p < 0 (non esiste ma serve range)
            [0.05, "#4CAF50"],   # verde = significativo
            [0.10, "#FF9800"],   # arancio = borderline
            [0.50, "#F44336"],   # rosso = non significativo
            [1.0,  "#B71C1C"],   # rosso scuro
        ],
        zmid=0.05,
        zmin=0.0,
        zmax=0.5,
        colorbar=dict(
            title=dict(text="p-value", font=dict(color=COLORS["text"])),
            tickvals=[0.0, 0.05, 0.10, 0.20, 0.50],
            ticktext=["0", "0.05*", "0.10", "0.20", "0.50"],
            tickfont=dict(color=COLORS["text"]),
        ),
        hovertemplate="<b>%{y} — %{x}</b><br>p-value: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Heatmap p-value (Welch test segnale vs incondizionato)<br>"
                 "<sup>Verde = significativo a 5% | Rosso = non significativo</sup>",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], size=12),
        height=280,
        margin=dict(l=100, r=40, t=80, b=50),
        xaxis=dict(title="Orizzonte forward", color=COLORS["text"]),
        yaxis=dict(title="Indice", color=COLORS["text"]),
    )

    return fig
