"""Financial performance and risk analytics module."""

from typing import Any

import numpy as np
import pandas as pd


def calculate_cumulative_returns(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Calculate cumulative return series starting from 0 (or normalized base).

    Args:
        df: Price dataframe
        column: Price column name

    Returns:
        pd.Series: Cumulative return percentage
    """
    initial_price = df[column].iloc[0]
    return ((df[column] / initial_price) - 1.0) * 100.0


def calculate_annualized_return(
    df: pd.DataFrame,
    column: str = "Close",
    periods_per_year: int = 252,
) -> float:
    """Calculate Compound Annual Growth Rate (CAGR) / Annualized Return.

    Args:
        df: Price dataframe with DatetimeIndex or at least 2 rows
        column: Price column name
        periods_per_year: Number of trading periods in a year (252 for daily)

    Returns:
        float: Annualized return percentage (e.g., 15.2 for 15.2%)
    """
    n_periods = len(df)
    if n_periods < 2:
        return 0.0

    start_val = df[column].iloc[0]
    end_val = df[column].iloc[-1]

    if start_val <= 0:
        return 0.0

    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0

    cagr = ((end_val / start_val) ** (1.0 / years)) - 1.0
    return float(cagr * 100.0)


def calculate_annualized_volatility(
    df: pd.DataFrame,
    column: str = "Close",
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility from daily returns.

    Args:
        df: Price dataframe
        column: Price column name
        periods_per_year: Annualization factor (252 for daily)

    Returns:
        float: Annualized standard deviation percentage
    """
    daily_returns = df[column].pct_change().dropna()
    if daily_returns.empty:
        return 0.0

    vol = daily_returns.std() * np.sqrt(periods_per_year)
    return float(vol * 100.0)


def calculate_sharpe_ratio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    column: str = "Close",
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        df: Price dataframe
        risk_free_rate: Annual risk-free rate (e.g. 0.04 for 4%)
        column: Price column name
        periods_per_year: Number of trading periods in a year

    Returns:
        float: Sharpe ratio
    """
    daily_returns = df[column].pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = daily_returns - daily_rf
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    column: str = "Close",
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino Ratio (downside risk-adjusted return).

    Args:
        df: Price dataframe
        risk_free_rate: Annual risk-free rate
        column: Price column name
        periods_per_year: Trading periods per year

    Returns:
        float: Sortino ratio
    """
    daily_returns = df[column].pct_change().dropna()
    if daily_returns.empty:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = daily_returns - daily_rf

    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty:
        return 999.0  # Zero downside risk

    downside_std = np.sqrt((downside_returns ** 2).mean())
    if downside_std == 0:
        return 0.0

    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(
    df: pd.DataFrame,
    column: str = "Close",
) -> tuple[float, pd.Series]:
    """Calculate Maximum Drawdown percentage and full drawdown curve.

    Args:
        df: Price dataframe
        column: Price column name

    Returns:
        Tuple[float, pd.Series]: (Max Drawdown %, Drawdown Series %)
    """
    prices = df[column]
    cummax = prices.cummax()
    drawdown = ((prices - cummax) / cummax) * 100.0
    max_dd = float(drawdown.min())
    return max_dd, drawdown


def calculate_var(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    column: str = "Close",
) -> float:
    """Calculate historical Value at Risk (1-day VaR) percentage.

    Args:
        df: Price dataframe
        confidence_level: Confidence level (e.g. 0.95 for 95%)
        column: Price column name

    Returns:
        float: Daily VaR percentage (positive number representing potential loss)
    """
    daily_returns = df[column].pct_change().dropna()
    if daily_returns.empty:
        return 0.0

    cutoff = (1.0 - confidence_level) * 100.0
    var = np.percentile(daily_returns, cutoff)
    return float(abs(var) * 100.0)


def compute_performance_summary(
    df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    column: str = "Close",
) -> dict[str, Any]:
    """Compute comprehensive performance and risk summary dictionary.

    Args:
        df: Price dataframe
        risk_free_rate: Annual risk-free rate
        column: Price column name

    Returns:
        dict: Performance summary metrics
    """
    if df.empty or len(df) < 2:
        return {}

    total_return = float(((df[column].iloc[-1] / df[column].iloc[0]) - 1.0) * 100.0)
    cagr = calculate_annualized_return(df, column=column)
    volatility = calculate_annualized_volatility(df, column=column)
    sharpe = calculate_sharpe_ratio(df, risk_free_rate=risk_free_rate, column=column)
    sortino = calculate_sortino_ratio(df, risk_free_rate=risk_free_rate, column=column)
    max_dd, _ = calculate_max_drawdown(df, column=column)
    var_95 = calculate_var(df, confidence_level=0.95, column=column)
    daily_returns = df[column].pct_change().dropna()
    positive_days = (daily_returns > 0).sum()
    win_rate = (
        float((positive_days / len(daily_returns)) * 100.0) if len(daily_returns) > 0 else 0.0
    )

    return {
        "Total Return (%)": total_return,
        "Annualized Return (CAGR %)": cagr,
        "Annualized Volatility (%)": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_dd,
        "1-Day VaR (95%)": var_95,
        "Win Rate (Daily %)": win_rate,
        "Current Price": float(df[column].iloc[-1]),
        "Period High": float(df[column].max()),
        "Period Low": float(df[column].min()),
    }
