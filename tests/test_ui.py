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
