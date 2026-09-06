"""Unit tests for strategy backtesting and portfolio comparison."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.portfolio import (
    calculate_correlation_matrix,
    calculate_normalized_returns,
)
from src.backtesting.strategy import (
    BacktestResult,
    backtest_macd_crossover,
    backtest_rsi_mean_reversion,
    backtest_sma_crossover,
)


@pytest.fixture
def sample_strategy_df():
    """Create a deterministic price dataframe for backtest verification."""
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)

    # Sine-wave oscillating price to guarantee multiple SMA and RSI cross events
    t = np.linspace(0, 4 * np.pi, 200)
    prices = 100.0 + (15.0 * np.sin(t)) + np.linspace(0, 20, 200)

    df = pd.DataFrame(
        {
            "Open": prices - 0.5,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, 200),
        },
        index=dates,
    )
    return df


def test_sma_crossover_backtest(sample_strategy_df):
    result = backtest_sma_crossover(sample_strategy_df, fast_window=10, slow_window=30)

    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == 200
    assert len(result.benchmark_curve) == 200
    assert result.total_trades > 0
    assert result.equity_curve.iloc[0] == 10000.0
    assert not result.equity_curve.isna().any()


def test_rsi_mean_reversion_backtest(sample_strategy_df):
    result = backtest_rsi_mean_reversion(sample_strategy_df, oversold=35.0, overbought=65.0)

    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == 200
    assert result.equity_curve.iloc[0] == 10000.0
    assert 0.0 <= result.win_rate_pct <= 100.0


def test_macd_crossover_backtest(sample_strategy_df):
    result = backtest_macd_crossover(sample_strategy_df, fast=8, slow=21, signal=7)

    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == 200
    assert result.total_trades > 0


def test_portfolio_normalized_returns():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "AAPL": [100.0, 105.0, 110.0, 108.0, 120.0],
            "MSFT": [200.0, 202.0, 210.0, 220.0, 230.0],
        },
        index=dates,
    )

    norm_df = calculate_normalized_returns(df)
    assert norm_df["AAPL"].iloc[0] == 0.0
    assert norm_df["MSFT"].iloc[0] == 0.0
    assert pytest.approx(norm_df["AAPL"].iloc[-1], rel=1e-3) == 20.0  # (120-100)/100 = 20%
    assert pytest.approx(norm_df["MSFT"].iloc[-1], rel=1e-3) == 15.0  # (230-200)/200 = 15%


def test_portfolio_correlation_matrix():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "AAPL": np.linspace(100, 200, 10),
            "MSFT": np.linspace(100, 200, 10),  # Perfectly correlated
        },
        index=dates,
    )

    corr_df = calculate_correlation_matrix(df)
    assert corr_df.shape == (2, 2)
    assert pytest.approx(corr_df.loc["AAPL", "MSFT"], rel=1e-3) == 1.0
