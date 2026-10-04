"""Unit tests for risk and return analytics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_returns,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_var,
    compute_performance_summary,
)


@pytest.fixture
def mock_price_series():
    """Create predictable linear price growth series."""
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    # Constant 10% annual gain (from 100 to 110) with small random oscillations
    prices = np.linspace(100, 110, 252)
    return pd.DataFrame({"Close": prices}, index=dates)


def test_cumulative_returns(mock_price_series):
    cum_returns = calculate_cumulative_returns(mock_price_series)
    assert cum_returns.iloc[0] == 0.0
    assert pytest.approx(cum_returns.iloc[-1], rel=1e-2) == 10.0


def test_annualized_return(mock_price_series):
    cagr = calculate_annualized_return(mock_price_series, periods_per_year=252)
    assert pytest.approx(cagr, rel=1e-2) == 10.0


def test_annualized_volatility(mock_price_series):
    vol = calculate_annualized_volatility(mock_price_series)
    assert vol >= 0.0


def test_sharpe_ratio(mock_price_series):
    # Risk-free rate 4%, return 10% -> Sharpe ratio must be positive
    sharpe = calculate_sharpe_ratio(mock_price_series, risk_free_rate=0.04)
    assert sharpe > 0.0


def test_max_drawdown():
    # Construct a known 20% drawdown: 100 -> 120 -> 96 -> 115
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Close": [100.0, 120.0, 96.0, 115.0]}, index=dates)
    max_dd, dd_series = calculate_max_drawdown(df)
    # (96 - 120) / 120 = -24 / 120 = -20%
    assert pytest.approx(max_dd, rel=1e-4) == -20.0


def test_var_calculation(mock_price_series):
    var_95 = calculate_var(mock_price_series, confidence_level=0.95)
    assert var_95 >= 0.0


def test_performance_summary(mock_price_series):
    summary = compute_performance_summary(mock_price_series)
    assert "Total Return (%)" in summary
    assert "Annualized Return (CAGR %)" in summary
    assert "Sharpe Ratio" in summary
    assert "Max Drawdown (%)" in summary
