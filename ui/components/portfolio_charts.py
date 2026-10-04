"""Portfolio Optimization and Modern Portfolio Theory (MPT) interactive Plotly charts."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analytics.optimization import OptimizationResult


def create_efficient_frontier_chart(
    opt_result: OptimizationResult,
    show_simulation_cloud: bool = False,
) -> go.Figure:
    """Create clean interactive Markowitz Efficient Frontier risk vs return chart.

    Args:
        opt_result: OptimizationResult bundle
        show_simulation_cloud: Whether to overlay Monte Carlo simulated portfolios cloud

    Returns:
        go.Figure: Interactive Plotly chart
    """
    fig = go.Figure()

    sim_df = opt_result.simulated_portfolios
    rf_pct = opt_result.risk_free_rate * 100.0

    # 1. Optional Monte Carlo Cloud (off by default for a clean, simple chart)
    if show_simulation_cloud and not sim_df.empty:
        hover_text = [
            f"<b>Simulated Portfolio</b><br>"
            f"Expected Return: {r * 100:.1f}%<br>"
            f"Volatility: {v * 100:.1f}%<br>"
            f"Sharpe: {s:.2f}"
            for r, v, s in zip(sim_df["Return"], sim_df["Volatility"], sim_df["Sharpe"], strict=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=sim_df["Volatility"] * 100.0,
                y=sim_df["Return"] * 100.0,
                mode="markers",
                marker=dict(
                    size=4,
                    color=sim_df["Sharpe"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Sharpe Ratio", font=dict(color="#F8FAFC", size=11)),
                        thickness=12,
                        len=0.7,
                        tickfont=dict(color="#CBD5E1", size=10),
                    ),
                    opacity=0.45,
                ),
                text=hover_text,
                hoverinfo="text",
                name="Simulated Samples",
            )
        )

    # 2. Analytical Efficient Frontier Curve
    if opt_result.efficient_frontier_volatilities and opt_result.efficient_frontier_returns:
        ef_x = [v * 100.0 for v in opt_result.efficient_frontier_volatilities]
        ef_y = [r * 100.0 for r in opt_result.efficient_frontier_returns]

        fig.add_trace(
            go.Scatter(
                x=ef_x,
                y=ef_y,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#38BDF8", width=3),
                hoverinfo="skip",
            )
        )

    # 3. Capital Allocation Line (CAL) from (0, Rf) through Max Sharpe
    ms = opt_result.max_sharpe_portfolio
    ms_vol_pct = ms.volatility * 100.0
    ms_ret_pct = ms.expected_return * 100.0

    if ms_vol_pct > 0:
        max_x_cal = max(ms_vol_pct * 1.35, max(sim_df["Volatility"] * 100.0) * 0.95)
        cal_slope = (ms_ret_pct - rf_pct) / ms_vol_pct
        cal_y_end = rf_pct + cal_slope * max_x_cal

        fig.add_trace(
            go.Scatter(
                x=[0.0, ms_vol_pct, max_x_cal],
                y=[rf_pct, ms_ret_pct, cal_y_end],
                mode="lines",
                name=f"Capital Allocation Line (Rf={rf_pct:.1f}%)",
                line=dict(color="#F43F5E", width=1.8, dash="dot"),
                hoverinfo="skip",
            )
        )

    # 4. Individual Asset Scatter Points
    ann_rets = opt_result.annualized_returns
    cov_mat = opt_result.covariance_matrix
    for ticker in opt_result.tickers:
        asset_ret = float(ann_rets.get(ticker, 0.0)) * 100.0
        asset_vol = float(np.sqrt(cov_mat.loc[ticker, ticker])) * 100.0
        asset_sharpe = (asset_ret - rf_pct) / asset_vol if asset_vol > 0 else 0.0

        fig.add_trace(
            go.Scatter(
                x=[asset_vol],
                y=[asset_ret],
                mode="markers+text",
                name=ticker,
                text=[f"  {ticker}"],
                textposition="middle right",
                textfont=dict(color="#E2E8F0", size=11, family="monospace"),
                marker=dict(size=9, color="#94A3B8", symbol="circle"),
                hoverinfo="text",
                hovertext=(
                    f"<b>{ticker} (Standalone Asset)</b><br>"
                    f"Expected Return: {asset_ret:.2f}%<br>"
                    f"Volatility: {asset_vol:.2f}%<br>"
                    f"Sharpe Ratio: {asset_sharpe:.2f}"
                ),
                showlegend=False,
            )
        )

    # 5. Equal Weight Benchmark Marker
    ew = opt_result.equal_weight_portfolio
    fig.add_trace(
        go.Scatter(
            x=[ew.volatility * 100.0],
            y=[ew.expected_return * 100.0],
            mode="markers",
            name="Equal Weight Portfolio",
            marker=dict(
                size=13,
                color="#F59E0B",
                symbol="square",
                line=dict(color="#FFFFFF", width=1.5),
            ),
            hoverinfo="text",
            hovertext=(
                f"<b>Equal Weight Portfolio (1/N)</b><br>"
                f"Expected Return: {ew.expected_return * 100.0:.2f}%<br>"
                f"Volatility: {ew.volatility * 100.0:.2f}%<br>"
                f"Sharpe Ratio: {ew.sharpe_ratio:.2f}"
            ),
        )
    )

    # 6. Minimum Volatility Portfolio Marker
    mv = opt_result.min_volatility_portfolio
    fig.add_trace(
        go.Scatter(
            x=[mv.volatility * 100.0],
            y=[mv.expected_return * 100.0],
            mode="markers",
            name="Min Volatility Portfolio",
            marker=dict(
                size=15,
                color="#00E5FF",
                symbol="diamond",
                line=dict(color="#FFFFFF", width=1.5),
            ),
            hoverinfo="text",
            hovertext=(
                f"<b>Minimum Volatility Portfolio</b><br>"
                f"Expected Return: {mv.expected_return * 100.0:.2f}%<br>"
                f"Volatility: {mv.volatility * 100.0:.2f}%<br>"
                f"Sharpe Ratio: {mv.sharpe_ratio:.2f}"
            ),
        )
    )

    # 7. Maximum Sharpe Ratio Portfolio Marker
    fig.add_trace(
        go.Scatter(
            x=[ms_vol_pct],
            y=[ms_ret_pct],
            mode="markers",
            name="Max Sharpe Portfolio (Tangency)",
            marker=dict(
                size=18,
                color="#FFD700",
                symbol="star",
                line=dict(color="#000000", width=1.5),
            ),
            hoverinfo="text",
            hovertext=(
                f"<b>Max Sharpe Tangency Portfolio ⭐</b><br>"
                f"Expected Return: {ms_ret_pct:.2f}%<br>"
                f"Volatility: {ms_vol_pct:.2f}%<br>"
                f"Sharpe Ratio: {ms.sharpe_ratio:.2f}"
            ),
        )
    )

    # Layout styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=540,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Annualized Volatility / Risk (%)",
            gridcolor="#1E293B",
            zerolinecolor="#334155",
        ),
        yaxis=dict(
            title="Annualized Expected Return (%)",
            gridcolor="#1E293B",
            zerolinecolor="#334155",
        ),
        hovermode="closest",
    )

    return fig


def create_allocation_donut_charts(opt_result: OptimizationResult) -> go.Figure:
    """Create side-by-side donut charts showing asset allocations for Max Sharpe vs Min Volatility.

    Args:
        opt_result: OptimizationResult bundle

    Returns:
        go.Figure: Two-column subplot of allocation donuts
    """
    ms_weights = {k: v for k, v in opt_result.max_sharpe_portfolio.weights.items() if v > 0.001}
    mv_weights = {k: v for k, v in opt_result.min_volatility_portfolio.weights.items() if v > 0.001}

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Bold

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=[
            "⭐ Max Sharpe Portfolio Allocation",
            "🔷 Min Volatility Portfolio Allocation",
        ],
    )

    # Max Sharpe Donut
    fig.add_trace(
        go.Pie(
            labels=list(ms_weights.keys()),
            values=list(ms_weights.values()),
            hole=0.55,
            marker=dict(colors=palette),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            insidetextorientation="radial",
            title=dict(
                text=f"Sharpe<br>{opt_result.max_sharpe_portfolio.sharpe_ratio:.2f}",
                font=dict(size=13, color="#FFD700"),
            ),
        ),
        row=1,
        col=1,
    )

    # Min Volatility Donut
    fig.add_trace(
        go.Pie(
            labels=list(mv_weights.keys()),
            values=list(mv_weights.values()),
            hole=0.55,
            marker=dict(colors=palette),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            insidetextorientation="radial",
            title=dict(
                text=f"Vol<br>{opt_result.min_volatility_portfolio.volatility * 100:.1f}%",
                font=dict(size=13, color="#00E5FF"),
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    return fig


def create_portfolio_equity_curve_chart(equity_df: pd.DataFrame) -> go.Figure:
    """Create interactive equity curve growth chart ($) comparing portfolio allocation strategies.

    Args:
        equity_df: DataFrame of cumulative values over time for each strategy

    Returns:
        go.Figure: Plotly multi-line chart
    """
    fig = go.Figure()

    colors = {
        "Max Sharpe Portfolio": "#FFD700",
        "Min Volatility Portfolio": "#00E5FF",
        "Equal Weight (1/N)": "#F59E0B",
    }

    fallback_colors = px.colors.qualitative.Pastel

    for i, col in enumerate(equity_df.columns):
        line_color = colors.get(col, fallback_colors[i % len(fallback_colors)])
        line_width = 2.8 if col in colors else 1.4
        line_dash = "solid" if col in colors else "dot"

        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df[col],
                mode="lines",
                name=col,
                line=dict(color=line_color, width=line_width, dash=line_dash),
                hovertemplate=f"<b>{col}</b>: $%{{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        hovermode="x unified",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        xaxis=dict(title="Date", gridcolor="#1E293B"),
        yaxis=dict(title="Portfolio Value ($)", gridcolor="#1E293B", side="right"),
    )

    return fig
