"""Backtesting and portfolio comparison Plotly chart generators."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.backtesting.strategy import BacktestResult


def create_equity_curve_chart(result: BacktestResult, ticker: str) -> go.Figure:
    """Create Strategy vs Benchmark Equity Curve growth chart."""
    fig = go.Figure()

    # 1. Strategy Portfolio Value
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name=f"{result.strategy_name} Portfolio",
            line=dict(color="#00E676", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 230, 118, 0.06)",
        )
    )

    # 2. Buy & Hold Benchmark Portfolio Value
    fig.add_trace(
        go.Scatter(
            x=result.benchmark_curve.index,
            y=result.benchmark_curve.values,
            name=f"Buy & Hold {ticker}",
            line=dict(color="#38BDF8", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        hovermode="x unified",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(title="Portfolio Value ($)", gridcolor="#1E293B", side="right"),
    )

    return fig


def create_signals_price_chart(result: BacktestResult, ticker: str) -> go.Figure:
    """Create price chart with Buy and Sell trade execution markers."""
    fig = go.Figure()
    signals = result.signals_df

    # 1. Underlying Asset Price
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals["Close"],
            name=f"{ticker} Close",
            line=dict(color="#94A3B8", width=1.5),
        )
    )

    # 2. Strategy Overlays (e.g. SMA Fast/Slow)
    if "SMA_Fast" in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals["SMA_Fast"],
                name="Fast SMA",
                line=dict(color="#00E5FF", width=1.2),
            )
        )
    if "SMA_Slow" in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals["SMA_Slow"],
                name="Slow SMA",
                line=dict(color="#FFD600", width=1.2),
            )
        )

    # 3. Buy Signals (Green Triangles)
    buys = signals[signals["Buy_Signal"]]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Close"],
                mode="markers",
                name="BUY Entry",
                marker=dict(symbol="triangle-up", size=11, color="#00E676"),
            )
        )

    # 4. Sell Signals (Red Triangles)
    sells = signals[signals["Sell_Signal"]]
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Close"],
                mode="markers",
                name="SELL Exit",
                marker=dict(symbol="triangle-down", size=11, color="#FF5252"),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        hovermode="x unified",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(title="Price (USD)", gridcolor="#1E293B", side="right"),
    )

    return fig


def create_multi_stock_comparison_chart(norm_df: pd.DataFrame) -> go.Figure:
    """Create multi-asset normalized percentage growth comparison chart."""
    fig = go.Figure()

    palette = ["#00E676", "#00E5FF", "#A855F7", "#FFD600", "#FF5252", "#FB7185", "#38BDF8"]

    for i, col in enumerate(norm_df.columns):
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=norm_df.index,
                y=norm_df[col],
                name=str(col),
                line=dict(color=color, width=2),
            )
        )

    fig.add_hline(y=0, line=dict(color="#64748B", dash="dash", width=1))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        hovermode="x unified",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(title="Cumulative Growth (%)", gridcolor="#1E293B", side="right"),
    )

    return fig


def create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Create annotated cross-asset correlation matrix heatmap."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_colorbar=dict(title="Correlation"),
    )

    return fig
