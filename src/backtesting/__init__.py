"""Strategy backtesting package."""

from src.backtesting.strategy import (
    BacktestResult,
    backtest_macd_crossover,
    backtest_rsi_mean_reversion,
    backtest_sma_crossover,
)

__all__ = [
    "BacktestResult",
    "backtest_sma_crossover",
    "backtest_rsi_mean_reversion",
    "backtest_macd_crossover",
]
