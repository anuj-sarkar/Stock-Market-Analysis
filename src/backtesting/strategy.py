"""Algorithmic trading strategy backtesting engine."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.indicators.technical import (
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)


@dataclass
class BacktestResult:
    """Standardized container for backtest equity curves, signals, and performance KPIs."""

    strategy_name: str
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    signals_df: pd.DataFrame
    total_return_pct: float
    benchmark_return_pct: float
    annualized_return_pct: float
    win_rate_pct: float
    total_trades: int
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float


def _compute_trade_metrics(
    daily_returns: pd.Series,
    strategy_returns: pd.Series,
    position: pd.Series,
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.04,
) -> dict[str, Any]:
    """Compute trade-by-trade metrics, equity curves, win rate, and drawdown."""
    # 1. Equity Curves
    strategy_equity = initial_capital * (1.0 + strategy_returns).cumprod()
    benchmark_equity = initial_capital * (1.0 + daily_returns).cumprod()

    # 2. Total & Annualized Returns
    total_return = float(((strategy_equity.iloc[-1] / initial_capital) - 1.0) * 100.0)
    bench_return = float(((benchmark_equity.iloc[-1] / initial_capital) - 1.0) * 100.0)

    n_days = len(daily_returns)
    years = n_days / 252.0 if n_days > 0 else 1.0
    if years > 0 and strategy_equity.iloc[-1] > 0:
        cagr_val = ((strategy_equity.iloc[-1] / initial_capital) ** (1.0 / years)) - 1.0
        cagr = float(cagr_val * 100.0)
    else:
        cagr = 0.0

    # 3. Maximum Drawdown
    cummax = strategy_equity.cummax()
    drawdown = ((strategy_equity - cummax) / cummax) * 100.0
    max_dd = float(drawdown.min())

    # 4. Strategy Sharpe Ratio
    strat_std = strategy_returns.std()
    if strat_std > 0:
        excess_ret = strategy_returns - (risk_free_rate / 252.0)
        sharpe = float((excess_ret.mean() / strat_std) * np.sqrt(252))
    else:
        sharpe = 0.0

    # 5. Trade Analysis (Individual round-trips)
    pos_diff = position.diff().fillna(0)
    entries = np.where(pos_diff == 1)[0]
    exits = np.where(pos_diff == -1)[0]

    trade_returns = []
    for entry_idx in entries:
        # Find next exit
        subsequent_exits = exits[exits > entry_idx]
        if len(subsequent_exits) > 0:
            exit_idx = subsequent_exits[0]
        else:
            exit_idx = len(daily_returns) - 1  # Open trade till end

        trade_cum_ret = (1.0 + strategy_returns.iloc[entry_idx + 1 : exit_idx + 1]).prod() - 1.0
        trade_returns.append(trade_cum_ret)

    total_trades = len(trade_returns)
    if total_trades > 0:
        trade_arr = np.array(trade_returns)
        wins = trade_arr[trade_arr > 0]
        losses = trade_arr[trade_arr < 0]

        win_rate = float((len(wins) / total_trades) * 100.0)
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 999.0
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return {
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "total_return": total_return,
        "benchmark_return": bench_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def backtest_sma_crossover(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """Backtest SMA Fast / Slow Moving Average Golden Cross Strategy."""
    data = df.copy()
    data["SMA_Fast"] = calculate_sma(data, window=fast_window)
    data["SMA_Slow"] = calculate_sma(data, window=slow_window)

    # Position: 1 when Fast > Slow, 0 otherwise
    data["Position"] = np.where(data["SMA_Fast"] > data["SMA_Slow"], 1, 0)
    data["Buy_Signal"] = (data["Position"] == 1) & (data["Position"].shift(1) == 0)
    data["Sell_Signal"] = (data["Position"] == 0) & (data["Position"].shift(1) == 1)

    daily_returns = data["Close"].pct_change().fillna(0)
    strategy_returns = data["Position"].shift(1).fillna(0) * daily_returns

    metrics = _compute_trade_metrics(
        daily_returns=daily_returns,
        strategy_returns=strategy_returns,
        position=data["Position"],
        initial_capital=initial_capital,
    )

    return BacktestResult(
        strategy_name=f"SMA Crossover ({fast_window}/{slow_window})",
        equity_curve=metrics["strategy_equity"],
        benchmark_curve=metrics["benchmark_equity"],
        signals_df=data[["Close", "SMA_Fast", "SMA_Slow", "Position", "Buy_Signal", "Sell_Signal"]],
        total_return_pct=metrics["total_return"],
        benchmark_return_pct=metrics["benchmark_return"],
        annualized_return_pct=metrics["cagr"],
        win_rate_pct=metrics["win_rate"],
        total_trades=metrics["total_trades"],
        profit_factor=metrics["profit_factor"],
        max_drawdown_pct=metrics["max_drawdown"],
        sharpe_ratio=metrics["sharpe"],
    )


def backtest_rsi_mean_reversion(
    df: pd.DataFrame,
    oversold: float = 30.0,
    overbought: float = 70.0,
    rsi_period: int = 14,
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """Backtest RSI Mean-Reversion Strategy (Buy < oversold, Exit > overbought)."""
    data = df.copy()
    data["RSI"] = calculate_rsi(data, period=rsi_period)

    positions = []
    current_pos = 0

    for rsi_val in data["RSI"]:
        if rsi_val < oversold:
            current_pos = 1
        elif rsi_val > overbought:
            current_pos = 0
        positions.append(current_pos)

    data["Position"] = positions
    data["Buy_Signal"] = (data["Position"] == 1) & (pd.Series(data["Position"]).shift(1) == 0)
    data["Sell_Signal"] = (data["Position"] == 0) & (pd.Series(data["Position"]).shift(1) == 1)

    daily_returns = data["Close"].pct_change().fillna(0)
    pos_series = pd.Series(data["Position"], index=data.index)
    strategy_returns = pos_series.shift(1).fillna(0) * daily_returns

    metrics = _compute_trade_metrics(
        daily_returns=daily_returns,
        strategy_returns=strategy_returns,
        position=pd.Series(data["Position"], index=data.index),
        initial_capital=initial_capital,
    )

    return BacktestResult(
        strategy_name=f"RSI Mean-Reversion ({oversold:.0f}/{overbought:.0f})",
        equity_curve=metrics["strategy_equity"],
        benchmark_curve=metrics["benchmark_equity"],
        signals_df=data[["Close", "RSI", "Position", "Buy_Signal", "Sell_Signal"]],
        total_return_pct=metrics["total_return"],
        benchmark_return_pct=metrics["benchmark_return"],
        annualized_return_pct=metrics["cagr"],
        win_rate_pct=metrics["win_rate"],
        total_trades=metrics["total_trades"],
        profit_factor=metrics["profit_factor"],
        max_drawdown_pct=metrics["max_drawdown"],
        sharpe_ratio=metrics["sharpe"],
    )


def backtest_macd_crossover(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """Backtest MACD Signal Line Crossover Strategy."""
    data = df.copy()
    macd_df = calculate_macd(data, fast=fast, slow=slow, signal=signal)
    data["MACD"] = macd_df["MACD"]
    data["MACD_Signal"] = macd_df["MACD_Signal"]

    data["Position"] = np.where(data["MACD"] > data["MACD_Signal"], 1, 0)
    data["Buy_Signal"] = (data["Position"] == 1) & (data["Position"].shift(1) == 0)
    data["Sell_Signal"] = (data["Position"] == 0) & (data["Position"].shift(1) == 1)

    daily_returns = data["Close"].pct_change().fillna(0)
    strategy_returns = data["Position"].shift(1).fillna(0) * daily_returns

    metrics = _compute_trade_metrics(
        daily_returns=daily_returns,
        strategy_returns=strategy_returns,
        position=data["Position"],
        initial_capital=initial_capital,
    )

    return BacktestResult(
        strategy_name="MACD Crossover (12/26/9)",
        equity_curve=metrics["strategy_equity"],
        benchmark_curve=metrics["benchmark_equity"],
        signals_df=data[["Close", "MACD", "MACD_Signal", "Position", "Buy_Signal", "Sell_Signal"]],
        total_return_pct=metrics["total_return"],
        benchmark_return_pct=metrics["benchmark_return"],
        annualized_return_pct=metrics["cagr"],
        win_rate_pct=metrics["win_rate"],
        total_trades=metrics["total_trades"],
        profit_factor=metrics["profit_factor"],
        max_drawdown_pct=metrics["max_drawdown"],
        sharpe_ratio=metrics["sharpe"],
    )
