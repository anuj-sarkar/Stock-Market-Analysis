"""Plotly interactive financial charts module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    overlays: dict[str, bool],
) -> go.Figure:
    """Create a high-fidelity 2-row subplot with Candlesticks, MA overlays, and Volume.

    Args:
        df: Enriched OHLCV DataFrame
        ticker: Asset symbol
        overlays: Dictionary of booleans for indicator visibility

    Returns:
        go.Figure: Interactive Plotly figure
    """
    has_volume = overlays.get("volume", True) and "Volume" in df.columns

    if has_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{ticker} Price",
            increasing_line_color="#00E676",
            increasing_fillcolor="#00E676",
            decreasing_line_color="#FF5252",
            decreasing_fillcolor="#FF5252",
        ),
        row=1,
        col=1,
    )

    # 2. Moving Average Overlays
    if overlays.get("sma_20") and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="SMA 20",
                line=dict(color="#00E5FF", width=1.5),
            ),
            row=1,
            col=1,
        )

    if overlays.get("sma_50") and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="SMA 50",
                line=dict(color="#FFD600", width=1.5),
            ),
            row=1,
            col=1,
        )

    if overlays.get("sma_200") and "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                name="SMA 200",
                line=dict(color="#FF9100", width=2.0),
            ),
            row=1,
            col=1,
        )

    if overlays.get("ema_12") and "EMA_12" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_12"],
                name="EMA 12",
                line=dict(color="#E040FB", width=1.2, dash="dot"),
            ),
            row=1,
            col=1,
        )

    if overlays.get("ema_26") and "EMA_26" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_26"],
                name="EMA 26",
                line=dict(color="#7C4DFF", width=1.2, dash="dot"),
            ),
            row=1,
            col=1,
        )

    # 3. Bollinger Bands Overlay
    if overlays.get("bollinger") and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(0, 229, 255, 0.4)", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="Bollinger Bands (20, 2σ)",
                line=dict(color="rgba(0, 229, 255, 0.4)", width=1),
                fill="tonexty",
                fillcolor="rgba(0, 229, 255, 0.05)",
            ),
            row=1,
            col=1,
        )

    # 4. VWAP Overlay
    if overlays.get("vwap") and "VWAP" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["VWAP"],
                name="VWAP",
                line=dict(color="#B388FF", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # 5. Volume Subplot
    if has_volume:
        candle_colors = np.where(df["Close"] >= df["Open"], "#00E676", "#FF5252")
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=candle_colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

    # Layout styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=620,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="#1E293B",
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor="#1E293B",
            side="right",
        ),
    )

    if has_volume:
        fig.update_yaxes(title="Volume", gridcolor="#1E293B", row=2, col=1, side="right")
        fig.update_xaxes(gridcolor="#1E293B", row=2, col=1)

    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create Relative Strength Index (RSI) oscillator chart."""
    fig = go.Figure()

    if "RSI_14" not in df.columns:
        return fig

    # RSI Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI_14"],
            name="RSI (14)",
            line=dict(color="#A855F7", width=2),
        )
    )

    # Overbought / Oversold threshold lines
    fig.add_hline(
        y=70,
        line=dict(color="#FF5252", dash="dash", width=1),
        annotation_text="Overbought (70)",
    )
    fig.add_hline(
        y=30,
        line=dict(color="#00E676", dash="dash", width=1),
        annotation_text="Oversold (30)",
    )
    fig.add_hline(y=50, line=dict(color="#64748B", dash="dot", width=1))

    # Shaded neutral area
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(168, 85, 247, 0.05)", line_width=0)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=240,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(
            title="RSI",
            range=[0, 100],
            gridcolor="#1E293B",
            side="right",
        ),
        xaxis=dict(gridcolor="#1E293B"),
    )
    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """Create MACD line, signal line, and histogram chart."""
    fig = go.Figure()

    if "MACD" not in df.columns:
        return fig

    # Histogram
    hist_colors = np.where(df["MACD_Hist"] >= 0, "#00E676", "#FF5252")
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_Hist"],
            name="Histogram",
            marker_color=hist_colors,
            opacity=0.8,
        )
    )

    # MACD Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD (12, 26)",
            line=dict(color="#00E5FF", width=1.5),
        )
    )

    # Signal Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_Signal"],
            name="Signal (9)",
            line=dict(color="#FF9100", width=1.5),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=240,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        yaxis=dict(title="MACD", gridcolor="#1E293B", side="right"),
        xaxis=dict(gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_cumulative_returns_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create cumulative return growth area chart."""
    from src.analytics.metrics import calculate_cumulative_returns

    cum_returns = calculate_cumulative_returns(df)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns,
            name=f"{ticker} Growth (%)",
            line=dict(color="#00E676", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 230, 118, 0.08)",
        )
    )

    fig.add_hline(y=0, line=dict(color="#64748B", dash="dash", width=1))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        yaxis=dict(title="Cumulative Return (%)", gridcolor="#1E293B", side="right"),
        xaxis=dict(gridcolor="#1E293B"),
    )
    return fig


def create_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Create underwater drawdown area chart."""
    from src.analytics.metrics import calculate_max_drawdown

    _, dd_series = calculate_max_drawdown(df)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dd_series.index,
            y=dd_series,
            name="Drawdown (%)",
            line=dict(color="#FF5252", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 82, 82, 0.15)",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        yaxis=dict(title="Drawdown (%)", gridcolor="#1E293B", side="right"),
        xaxis=dict(gridcolor="#1E293B"),
    )
    return fig
