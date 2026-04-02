"""
charts.py — Grafici Plotly con colorazione per regime di mercato.

Tre grafici principali per ogni indice:
  1. build_breadth_chart()   → % costituenti sopra 200 MA (linea viola + soglia)
  2. build_price_chart()     → prezzo indice scala log, colorato per regime
  3. build_drawdown_chart()  → drawdown dal massimo, colorato per regime

La colorazione del prezzo e del drawdown rispecchia il regime breadth:
  verde  = breadth sana (> soglia)
  rosso  = breadth sotto soglia
  blu    = breadth in zona estrema (< soglia/2)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COLORS, MA_PERIOD


# ═══════════════════════════════════════════════════════════════════════════════
# UTILS INTERNI
# ═══════════════════════════════════════════════════════════════════════════════

def _base_layout(title: str = "", height: int = 380) -> dict:
    """Layout Plotly dark condiviso da tutti i grafici."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=15, color=COLORS["text"]),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(
            showgrid=True, gridcolor=COLORS["grid"], gridwidth=1,
            zeroline=False, color=COLORS["text"],
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=COLORS["grid"], gridwidth=1,
            zeroline=False, color=COLORS["text"],
            tickfont=dict(size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"],
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=55, b=50),
        height=height,
    )


def _get_regime_color(regime_val: str) -> str:
    """Mappa nome regime → colore."""
    return {
        "healthy": COLORS["healthy"],
        "caution": COLORS["caution"],
        "extreme": COLORS["extreme"],
    }.get(regime_val, COLORS["subtext"])


def _add_colored_line(
    fig: go.Figure,
    x: pd.Index,
    y: pd.Series,
    regime: pd.Series,
    showlegend_map: dict,
    row: int = None,
    col: int = None,
    width: float = 1.4,
) -> None:
    """
    Aggiunge una linea colorata per segmenti in base al regime.

    Ogni segmento continuo dello stesso regime viene tracciato come una
    singola trace; i segmenti si sovrappongono di 1 punto per evitare gap.
    La legenda è mostrata solo alla prima occorrenza di ogni regime.

    Args:
        fig:           Figura Plotly
        x:             Index DatetimeIndex
        y:             Valori da tracciare
        regime:        Serie di etichette regime ('healthy', 'caution', 'extreme')
        showlegend_map: Dict {regime: bool} per controllare se mostrare legenda
        row, col:      Posizione subplot
        width:         Spessore linea
    """
    REGIME_LABELS = {
        "healthy": "Breadth sana",
        "caution": "Sotto soglia",
        "extreme": "Zona estrema",
    }

    # Allinea regime sull'indice y
    regime_aligned = regime.reindex(x).ffill().bfill()

    # Individua i breakpoint di cambio regime
    changes = (regime_aligned != regime_aligned.shift()).fillna(True)
    change_indices = list(regime_aligned.index[changes]) + [regime_aligned.index[-1]]

    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end   = change_indices[i + 1]

        # Slice con +1 punto di overlap per continuità visiva
        mask = (x >= start) & (x <= end)
        seg_x = x[mask]
        seg_y = y[mask]

        if seg_x.empty:
            continue

        r_val   = regime_aligned.loc[start]
        color   = _get_regime_color(r_val)
        label   = REGIME_LABELS.get(r_val, r_val)
        show_lg = showlegend_map.get(r_val, True)

        # row/col solo per figure make_subplots; None = figura semplice
        trace_kwargs = {}
        if row is not None:
            trace_kwargs["row"] = row
        if col is not None:
            trace_kwargs["col"] = col

        fig.add_trace(
            go.Scatter(
                x=seg_x, y=seg_y,
                mode="lines",
                line=dict(color=color, width=width),
                name=label,
                legendgroup=r_val,
                showlegend=show_lg,
                hoverinfo="skip",
            ),
            **trace_kwargs,
        )
        if show_lg:
            showlegend_map[r_val] = False  # mostra solo la prima volta


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFICO 1 — BREADTH %
# ═══════════════════════════════════════════════════════════════════════════════

def build_breadth_chart(
    breadth: pd.Series,
    threshold: float,
    extreme_mult: float,
    index_label: str,
) -> go.Figure:
    """
    Grafico della percentuale di costituenti sopra la 200 MA.

    Elementi:
      - Linea viola: breadth %
      - Linea tratteggiata azzurra: soglia principale
      - Area riempita sotto la soglia (trasparente rossa)
      - Annotazione del valore corrente e della soglia

    Args:
        breadth:       Serie breadth % giornaliera
        threshold:     Soglia principale (linea tratteggiata)
        extreme_mult:  Moltiplicatore soglia estrema
        index_label:   Nome indice per il titolo

    Returns:
        go.Figure
    """
    extreme_threshold = threshold * extreme_mult
    fig = go.Figure()

    # Area riempita sotto la soglia (zona di rischio)
    fig.add_trace(go.Scatter(
        x=breadth.index, y=breadth.values,
        fill="tozeroy",
        fillcolor="rgba(244, 67, 54, 0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
        name="_fill",
    ))

    # Linea breadth principale
    fig.add_trace(go.Scatter(
        x=breadth.index, y=breadth.values,
        mode="lines",
        line=dict(color=COLORS["breadth"], width=1.5),
        name=f"% sopra 200 MA",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))

    # Soglia principale
    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color=COLORS["threshold"], line_width=1.5,
        annotation_text=f"Soglia: {threshold:.0f}%",
        annotation_position="bottom right",
        annotation_font=dict(color=COLORS["threshold"], size=11),
    )

    # Soglia estrema
    fig.add_hline(
        y=extreme_threshold, line_dash="dot",
        line_color=COLORS["extreme"], line_width=1.2,
        annotation_text=f"Estrema: {extreme_threshold:.1f}%",
        annotation_position="bottom right",
        annotation_font=dict(color=COLORS["extreme"], size=10),
    )

    layout = _base_layout(
        title=f"Percentuale costituenti sopra la 200 MA — {index_label}",
        height=360,
    )
    layout["yaxis"].update(title="Percentuale (%)", ticksuffix="%", range=[-2, 102])
    layout["xaxis"].update(title="Data")
    fig.update_layout(**layout)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFICO 2 — PREZZO INDICE (SCALA LOG) COLORATO PER REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def build_price_chart(
    index_price: pd.Series,
    regime: pd.Series,
    index_label: str,
) -> go.Figure:
    """
    Grafico del prezzo dell'indice in scala logaritmica, colorato per regime breadth.

    La colorazione segmenta la serie di prezzi in base al regime di mercato:
    verde (sano), rosso (sotto soglia), blu (zona estrema).

    Args:
        index_price: Serie prezzi indice
        regime:      Serie regime ('healthy', 'caution', 'extreme')
        index_label: Nome indice per il titolo

    Returns:
        go.Figure con scala y logaritmica
    """
    # Allinea regime all'indice prezzi
    regime_aligned = regime.reindex(index_price.index).ffill().bfill()
    # Se il regime non copre tutto il prezzo, usa "healthy" come default
    regime_aligned = regime_aligned.fillna("healthy")

    fig = go.Figure()
    showlegend_map = {"healthy": True, "caution": True, "extreme": True}

    _add_colored_line(
        fig, index_price.index, index_price,
        regime_aligned, showlegend_map, width=1.5,
    )

    layout = _base_layout(
        title=f"Prezzo {index_label} (scala log) — colorato per regime breadth",
        height=360,
    )
    layout["yaxis"].update(title="Prezzo (log)", type="log")
    layout["xaxis"].update(title="Data")
    fig.update_layout(**layout)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFICO 3 — DRAWDOWN COLORATO PER REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def build_drawdown_chart(
    drawdown: pd.Series,
    regime: pd.Series,
    index_label: str,
) -> go.Figure:
    """
    Grafico del drawdown dal massimo storico, colorato per regime breadth.

    Args:
        drawdown:    Serie drawdown % (valori ≤ 0)
        regime:      Serie regime ('healthy', 'caution', 'extreme')
        index_label: Nome indice per il titolo

    Returns:
        go.Figure
    """
    regime_aligned = regime.reindex(drawdown.index).ffill().bfill().fillna("healthy")

    fig = go.Figure()
    showlegend_map = {"healthy": True, "caution": True, "extreme": True}

    _add_colored_line(
        fig, drawdown.index, drawdown,
        regime_aligned, showlegend_map, width=1.4,
    )

    # Linea dello zero come riferimento
    fig.add_hline(y=0, line_color=COLORS["subtext"], line_width=0.8, line_dash="dot")

    layout = _base_layout(
        title=f"Drawdown {index_label} (%) — colorato per regime breadth",
        height=320,
    )
    layout["yaxis"].update(title="Drawdown (%)", ticksuffix="%")
    layout["xaxis"].update(title="Data")
    fig.update_layout(**layout)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFICO COMBINATO (3 pannelli sovrapposti — opzionale per export)
# ═══════════════════════════════════════════════════════════════════════════════

def build_combined_chart(
    breadth: pd.Series,
    index_price: pd.Series,
    drawdown: pd.Series,
    regime: pd.Series,
    threshold: float,
    extreme_mult: float,
    index_label: str,
) -> go.Figure:
    """
    Crea un grafico combinato a 3 pannelli verticali con asse X condiviso.

    Pannelli dall'alto:
      1. Breadth % con soglie
      2. Prezzo indice (log) colorato per regime
      3. Drawdown colorato per regime

    Utile per export immagine o confronto visivo sincrono.

    Args:
        breadth, index_price, drawdown, regime: Series calcolate
        threshold:     Soglia principale
        extreme_mult:  Moltiplicatore estrema
        index_label:   Nome indice

    Returns:
        go.Figure con 3 subplot
    """
    extreme_threshold = threshold * extreme_mult

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.38, 0.24],
        subplot_titles=[
            f"% Costituenti sopra 200 MA — {index_label}",
            f"Prezzo {index_label} (log)",
            f"Drawdown {index_label} (%)",
        ],
    )

    # ── Pannello 1: breadth ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=breadth.index, y=breadth.values,
        mode="lines", line=dict(color=COLORS["breadth"], width=1.5),
        name="% sopra 200 MA", legendgroup="breadth",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color=COLORS["threshold"], line_width=1.4,
        row=1, col=1,
    )
    fig.add_hline(
        y=extreme_threshold, line_dash="dot",
        line_color=COLORS["extreme"], line_width=1.1,
        row=1, col=1,
    )

    # ── Pannello 2: prezzo log ───────────────────────────────────────────────
    regime_price = regime.reindex(index_price.index).ffill().bfill().fillna("healthy")
    showlegend_map = {"healthy": True, "caution": True, "extreme": True}
    _add_colored_line(fig, index_price.index, index_price, regime_price,
                      showlegend_map, row=2, col=1, width=1.5)

    # ── Pannello 3: drawdown ─────────────────────────────────────────────────
    regime_dd = regime.reindex(drawdown.index).ffill().bfill().fillna("healthy")
    showlegend_map2 = {"healthy": False, "caution": False, "extreme": False}
    _add_colored_line(fig, drawdown.index, drawdown, regime_dd,
                      showlegend_map2, row=3, col=1, width=1.4)
    fig.add_hline(y=0, line_color=COLORS["subtext"], line_width=0.7,
                  line_dash="dot", row=3, col=1)

    # ── Layout globale ───────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif", size=12),
        height=880,
        hovermode="x unified",
        margin=dict(l=65, r=20, t=60, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"],
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
    )

    # Griglia e stile assi
    for i in range(1, 4):
        fig.update_xaxes(
            showgrid=True, gridcolor=COLORS["grid"],
            zeroline=False, color=COLORS["text"], row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=COLORS["grid"],
            zeroline=False, color=COLORS["text"], row=i, col=1,
        )

    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=3, col=1)
    fig.update_xaxes(title_text="Data", row=3, col=1)

    return fig
