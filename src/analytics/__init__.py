"""Financial analytics package."""

from src.analytics.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_returns,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    compute_performance_summary,
)
from src.analytics.portfolio import (
    calculate_correlation_matrix,
    calculate_normalized_returns,
    fetch_multiple_stocks,
)

__all__ = [
    "calculate_cumulative_returns",
    "calculate_annualized_return",
    "calculate_annualized_volatility",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_var",
    "compute_performance_summary",
    "fetch_multiple_stocks",
    "calculate_normalized_returns",
    "calculate_correlation_matrix",
]
