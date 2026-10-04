"""Unit tests for Modern Portfolio Theory (MPT) and Markowitz optimization."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.optimization import (
    backtest_portfolio_allocations,
    calculate_efficient_frontier,
    calculate_equal_weight_portfolio,
    calculate_portfolio_performance,
    optimize_max_sharpe,
    optimize_min_volatility,
    run_monte_carlo_portfolio_simulation,
    run_portfolio_optimization,
)


@pytest.fixture
def mock_multi_asset_data():
    """Create synthetic historical prices for 3 distinct assets."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")

    # Asset A: High return (25%), High volatility (30%)
    ret_a = np.random.normal(0.25 / 252, 0.30 / np.sqrt(252), 252)
    price_a = 100 * np.exp(np.cumsum(ret_a))

    # Asset B: Moderate return (12%), Moderate volatility (15%)
    ret_b = np.random.normal(0.12 / 252, 0.15 / np.sqrt(252), 252)
    price_b = 50 * np.exp(np.cumsum(ret_b))

    # Asset C: Low return (6%), Low volatility (8%)
    ret_c = np.random.normal(0.06 / 252, 0.08 / np.sqrt(252), 252)
    price_c = 200 * np.exp(np.cumsum(ret_c))

    return pd.DataFrame(
        {
            "ASSET_A": price_a,
            "ASSET_B": price_b,
            "ASSET_C": price_c,
        },
        index=dates,
    )


def test_calculate_portfolio_performance(mock_multi_asset_data):
    daily_returns = mock_multi_asset_data.pct_change().dropna()
    mean_rets = daily_returns.mean() * 252
    cov_mat = daily_returns.cov() * 252

    weights = np.array([0.5, 0.3, 0.2])
    p_ret, p_vol, p_sharpe = calculate_portfolio_performance(
        weights, mean_rets, cov_mat, risk_free_rate=0.04
    )

    assert p_ret > 0.0
    assert p_vol > 0.0
    assert pytest.approx(p_sharpe, rel=1e-3) == (p_ret - 0.04) / p_vol


def test_optimize_max_sharpe(mock_multi_asset_data):
    daily_returns = mock_multi_asset_data.pct_change().dropna()
    mean_rets = daily_returns.mean() * 252
    cov_mat = daily_returns.cov() * 252

    ms = optimize_max_sharpe(mean_rets, cov_mat, risk_free_rate=0.04)
    ew = calculate_equal_weight_portfolio(mean_rets, cov_mat, risk_free_rate=0.04)

    # Weights must sum to 1.0 within tolerance
    total_weight = sum(ms.weights.values())
    assert pytest.approx(total_weight, rel=1e-3) == 1.0

    # Long-only: all weights >= 0
    for w in ms.weights.values():
        assert w >= -1e-5

    # Max Sharpe ratio must be greater than or equal to Equal Weight
    assert ms.sharpe_ratio >= ew.sharpe_ratio - 1e-4


def test_optimize_min_volatility(mock_multi_asset_data):
    daily_returns = mock_multi_asset_data.pct_change().dropna()
    mean_rets = daily_returns.mean() * 252
    cov_mat = daily_returns.cov() * 252

    mv = optimize_min_volatility(mean_rets, cov_mat, risk_free_rate=0.04)
    ew = calculate_equal_weight_portfolio(mean_rets, cov_mat, risk_free_rate=0.04)

    # Weights sum to 1.0
    total_weight = sum(mv.weights.values())
    assert pytest.approx(total_weight, rel=1e-3) == 1.0

    # Min volatility must be <= Equal Weight volatility
    assert mv.volatility <= ew.volatility + 1e-4


def test_calculate_efficient_frontier(mock_multi_asset_data):
    daily_returns = mock_multi_asset_data.pct_change().dropna()
    mean_rets = daily_returns.mean() * 252
    cov_mat = daily_returns.cov() * 252

    mv = optimize_min_volatility(mean_rets, cov_mat)
    max_ret = float(mean_rets.max())

    frontier_vols, frontier_rets = calculate_efficient_frontier(
        mean_rets, cov_mat, min_vol_ret=mv.expected_return, max_ret=max_ret, num_points=10
    )

    assert len(frontier_vols) > 0
    assert len(frontier_rets) == len(frontier_vols)
    # Check returns are non-decreasing
    assert frontier_rets[-1] >= frontier_rets[0]


def test_monte_carlo_portfolio_simulation(mock_multi_asset_data):
    daily_returns = mock_multi_asset_data.pct_change().dropna()
    mean_rets = daily_returns.mean() * 252
    cov_mat = daily_returns.cov() * 252

    num_samples = 300
    sim_df = run_monte_carlo_portfolio_simulation(
        mean_rets, cov_mat, num_portfolios=num_samples, risk_free_rate=0.04
    )

    assert len(sim_df) == num_samples
    assert "Return" in sim_df.columns
    assert "Volatility" in sim_df.columns
    assert "Sharpe" in sim_df.columns
    assert "Weight_ASSET_A" in sim_df.columns
    assert "Weight_ASSET_B" in sim_df.columns
    assert "Weight_ASSET_C" in sim_df.columns

    # Check simulated weights sum to 1 for all rows
    weight_cols = [c for c in sim_df.columns if c.startswith("Weight_")]
    row_sums = sim_df[weight_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_backtest_portfolio_allocations(mock_multi_asset_data):
    weights = {
        "Strategy_A": {"ASSET_A": 0.6, "ASSET_B": 0.4, "ASSET_C": 0.0},
        "Strategy_B": {"ASSET_A": 0.0, "ASSET_B": 0.5, "ASSET_C": 0.5},
    }
    capital = 10000.0
    equity_df = backtest_portfolio_allocations(
        mock_multi_asset_data, weights, initial_capital=capital
    )

    assert not equity_df.empty
    assert "Strategy_A" in equity_df.columns
    assert "Strategy_B" in equity_df.columns
    # Initial value should start near initial capital
    assert pytest.approx(equity_df["Strategy_A"].iloc[0], rel=1e-2) == capital


def test_run_portfolio_optimization_full(mock_multi_asset_data):
    result = run_portfolio_optimization(mock_multi_asset_data, risk_free_rate=0.04, num_simulations=100)

    assert result is not None
    assert result.max_sharpe_portfolio is not None
    assert result.min_volatility_portfolio is not None
    assert result.equal_weight_portfolio is not None
    assert len(result.tickers) == 3
    assert len(result.simulated_portfolios) == 100


def test_run_portfolio_optimization_edge_cases():
    # Empty DataFrame should return None
    assert run_portfolio_optimization(pd.DataFrame()) is None

    # Single asset should return None (need at least 2)
    single_df = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104, 105]})
    assert run_portfolio_optimization(single_df) is None
