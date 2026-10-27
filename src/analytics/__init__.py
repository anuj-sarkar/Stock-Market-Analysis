"""Financial analytics package."""

from src.analytics.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_calmar_ratio,
    calculate_cumulative_returns,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    compute_performance_summary,
)
from src.analytics.optimization import (
    OptimizationResult,
    PortfolioMetrics,
    backtest_portfolio_allocations,
    calculate_efficient_frontier,
    calculate_equal_weight_portfolio,
    calculate_portfolio_performance,
    optimize_max_sharpe,
    optimize_min_volatility,
    run_monte_carlo_portfolio_simulation,
    run_portfolio_optimization,
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
    "calculate_calmar_ratio",
    "calculate_max_drawdown",
    "calculate_var",
    "compute_performance_summary",
    "fetch_multiple_stocks",
    "calculate_normalized_returns",
    "calculate_correlation_matrix",
    "PortfolioMetrics",
    "OptimizationResult",
    "calculate_portfolio_performance",
    "optimize_max_sharpe",
    "optimize_min_volatility",
    "calculate_equal_weight_portfolio",
    "calculate_efficient_frontier",
    "run_monte_carlo_portfolio_simulation",
    "backtest_portfolio_allocations",
    "run_portfolio_optimization",
]

