"""Unit tests for UI components and Plotly chart generators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.technical import add_all_indicators
from ui.components.charts import (
    create_candlestick_chart,
    create_cumulative_returns_chart,
    create_drawdown_chart,
    create_macd_chart,
    create_rsi_chart,
)


@pytest.fixture
def enriched_test_df():
    """Create enriched sample dataframe for chart generation."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)

    close_prices = 150.0 + np.cumsum(np.random.randn(100) * 2.0)
    high_prices = close_prices + np.random.uniform(0.5, 3.0, 100)
    low_prices = close_prices - np.random.uniform(0.5, 3.0, 100)
    open_prices = low_prices + np.random.uniform(0.1, 2.0, 100)
    volumes = np.random.randint(500000, 10000000, 100)

    raw_df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
            "Daily_Return": pd.Series(close_prices).pct_change().values,
        },
        index=dates,
    )
    return add_all_indicators(raw_df)


def test_create_candlestick_chart(enriched_test_df):
    overlays = {
        "sma_20": True,
        "sma_50": True,
        "sma_200": True,
        "ema_12": True,
        "ema_26": True,
        "bollinger": True,
        "vwap": True,
        "volume": True,
    }
    fig = create_candlestick_chart(enriched_test_df, "AAPL", overlays)
    assert fig is not None
    assert len(fig.data) > 0


def test_create_rsi_chart(enriched_test_df):
    fig = create_rsi_chart(enriched_test_df)
    assert fig is not None
    assert len(fig.data) > 0


def test_create_macd_chart(enriched_test_df):
    fig = create_macd_chart(enriched_test_df)
    assert fig is not None
    assert len(fig.data) == 3  # Histogram, MACD line, Signal line


def test_create_cumulative_returns_chart(enriched_test_df):
    fig = create_cumulative_returns_chart(enriched_test_df, "AAPL")
    assert fig is not None
    assert len(fig.data) == 1


def test_create_drawdown_chart(enriched_test_df):
    fig = create_drawdown_chart(enriched_test_df)
    assert fig is not None
    assert len(fig.data) == 1


def test_portfolio_charts():
    from src.analytics.optimization import run_portfolio_optimization
    from ui.components.portfolio_charts import (
        create_allocation_donut_charts,
        create_efficient_frontier_chart,
        create_portfolio_equity_curve_chart,
    )

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "AAPL": 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100))),
            "MSFT": 200 * np.exp(np.cumsum(np.random.normal(0.0008, 0.015, 100))),
        },
        index=dates,
    )

    opt_res = run_portfolio_optimization(df, num_simulations=50)
    assert opt_res is not None

    ef_fig = create_efficient_frontier_chart(opt_res)
    assert ef_fig is not None
    assert len(ef_fig.data) > 0

    donut_fig = create_allocation_donut_charts(opt_res)
    assert donut_fig is not None
    assert len(donut_fig.data) == 2

    equity_df = pd.DataFrame({"Max Sharpe": [10000, 10500], "Min Vol": [10000, 10200]})
    growth_fig = create_portfolio_equity_curve_chart(equity_df)
    assert growth_fig is not None
    assert len(growth_fig.data) == 2

