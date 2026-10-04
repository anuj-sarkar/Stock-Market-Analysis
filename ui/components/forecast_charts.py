"""Forecasting visualizations and interactive Plotly chart generators."""

import pandas as pd
import plotly.graph_objects as go

from src.models.base import ForecastResult
from src.models.monte_carlo import MonteCarloGBM


def create_forecast_chart(
    result: ForecastResult,
    historical_series: pd.Series,
    ticker: str,
) -> go.Figure:
    """Create a unified time-series chart showing historical, in-sample test,
    and future forecast with confidence bands.

    Args:
        result: Standardized ForecastResult object
        historical_series: Historical price Series
        ticker: Asset symbol string

    Returns:
        go.Figure: Plotly chart figure
    """
    fig = go.Figure()

    # 1. Historical Prices
    fig.add_trace(
        go.Scatter(
            x=historical_series.index,
            y=historical_series.values,
            name=f"{ticker} Historical",
            line=dict(color="#38BDF8", width=2),
        )
    )

    # 2. In-sample test actuals vs predictions (if available)
    if not result.in_sample_predicted.empty:
        fig.add_trace(
            go.Scatter(
                x=result.in_sample_predicted.index,
                y=result.in_sample_predicted.values,
                name="In-Sample Test Fit",
                line=dict(color="#FFB74D", width=2, dash="dot"),
            )
        )

    # 3. 95% Confidence Interval Band
    fig.add_trace(
        go.Scatter(
            x=result.future_dates,
            y=result.upper_bound_95.values,
            name="95% Upper Bound",
            line=dict(color="rgba(0, 229, 255, 0.2)", width=1),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future_dates,
            y=result.lower_bound_95.values,
            name="95% Confidence Interval",
            line=dict(color="rgba(0, 229, 255, 0.2)", width=1),
            fill="tonexty",
            fillcolor="rgba(0, 229, 255, 0.10)",
        )
    )

    # 4. 80% Confidence Interval Band
    fig.add_trace(
        go.Scatter(
            x=result.future_dates,
            y=result.upper_bound_80.values,
            name="80% Upper Bound",
            line=dict(color="rgba(0, 229, 255, 0.3)", width=1),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future_dates,
            y=result.lower_bound_80.values,
            name="80% Confidence Interval",
            line=dict(color="rgba(0, 229, 255, 0.3)", width=1),
            fill="tonexty",
            fillcolor="rgba(0, 229, 255, 0.15)",
        )
    )

    # 5. Future Point Forecast Trajectory
    # Connect last historical point to first forecast point for seamless continuity
    last_hist_date = historical_series.index[-1]
    last_hist_price = historical_series.iloc[-1]
    extended_dates = [last_hist_date] + list(result.future_dates)
    extended_forecast = [last_hist_price] + list(result.forecast.values)

    fig.add_trace(
        go.Scatter(
            x=extended_dates,
            y=extended_forecast,
            name=f"{result.model_name} Forecast",
            line=dict(color="#00E676", width=2.5, dash="dash"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        hovermode="x unified",
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(title="Price (USD)", gridcolor="#1E293B", side="right"),
    )

    return fig


def create_monte_carlo_chart(
    simulator: MonteCarloGBM,
    historical_series: pd.Series,
    result: ForecastResult,
    ticker: str,
    max_paths_to_plot: int = 50,
) -> go.Figure:
    """Create a Monte Carlo simulation chart showing sample paths and quantile bounds."""
    fig = go.Figure()

    # 1. Historical Prices (recent 60 days)
    recent_hist = historical_series.tail(60)
    fig.add_trace(
        go.Scatter(
            x=recent_hist.index,
            y=recent_hist.values,
            name=f"{ticker} Recent Historical",
            line=dict(color="#38BDF8", width=2),
        )
    )

    # 2. Sample Stochastic Simulation Paths
    if simulator.simulated_paths is not None:
        last_date = historical_series.index[-1]
        last_price = float(historical_series.iloc[-1])
        all_dates = [last_date] + list(result.future_dates)

        n_paths = min(max_paths_to_plot, simulator.simulated_paths.shape[1])
        for i in range(n_paths):
            path_values = [last_price] + list(simulator.simulated_paths[:, i])
            fig.add_trace(
                go.Scatter(
                    x=all_dates,
                    y=path_values,
                    line=dict(color="rgba(148, 163, 184, 0.12)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # 3. Percentile Bounds
    fig.add_trace(
        go.Scatter(
            x=result.future_dates,
            y=result.upper_bound_95.values,
            name="95th Percentile",
            line=dict(color="#FFD600", width=1.5, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future_dates,
            y=result.lower_bound_95.values,
            name="5th Percentile",
            line=dict(color="#FF5252", width=1.5, dash="dot"),
        )
    )

    # 4. Median Forecast
    fig.add_trace(
        go.Scatter(
            x=result.future_dates,
            y=result.forecast.values,
            name="Median Trajectory (50th %)",
            line=dict(color="#00E676", width=2.5),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(title="Simulated Price (USD)", gridcolor="#1E293B", side="right"),
    )

    return fig
