"""Modern Portfolio Theory (MPT) and Markowitz Efficient Frontier optimization."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioMetrics:
    """Performance metrics and asset weights for an allocated portfolio."""

    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]


@dataclass
class OptimizationResult:
    """Aggregated results from Modern Portfolio Theory optimization."""

    max_sharpe_portfolio: PortfolioMetrics
    min_volatility_portfolio: PortfolioMetrics
    equal_weight_portfolio: PortfolioMetrics
    efficient_frontier_returns: list[float]
    efficient_frontier_volatilities: list[float]
    simulated_portfolios: pd.DataFrame
    tickers: list[str]
    risk_free_rate: float
    annualized_returns: pd.Series
    covariance_matrix: pd.DataFrame


def calculate_portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> tuple[float, float, float]:
    """Calculate annualized expected return, volatility, and Sharpe ratio for portfolio weights.

    Args:
        weights: 1D array of portfolio weights (summing to 1)
        mean_returns: Annualized expected returns per asset
        cov_matrix: Annualized covariance matrix of asset returns
        risk_free_rate: Annual risk-free interest rate (e.g., 0.04 = 4%)

    Returns:
        tuple[float, float, float]: (annualized_return, annualized_volatility, sharpe_ratio)
    """
    weights = np.asarray(weights)
    p_return = float(np.sum(mean_returns * weights))
    p_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    if p_volatility <= 1e-8:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((p_return - risk_free_rate) / p_volatility)

    return p_return, p_volatility, sharpe_ratio


def optimize_max_sharpe(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioMetrics:
    """Find portfolio weights that maximize the Sharpe ratio (Tangency Portfolio).

    Args:
        mean_returns: Annualized expected returns per asset
        cov_matrix: Annualized covariance matrix
        risk_free_rate: Annual risk-free rate
        min_weight: Minimum weight per asset (long-only default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)

    Returns:
        PortfolioMetrics: Optimized portfolio details
    """
    num_assets = len(mean_returns)
    tickers = list(mean_returns.index)

    def neg_sharpe(weights: np.ndarray) -> float:
        _, _, sharpe = calculate_portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        return -sharpe

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    init_weights = np.ones(num_assets) / num_assets

    res = minimize(
        neg_sharpe,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    opt_weights = np.clip(res.x, 0.0, 1.0)
    opt_weights = opt_weights / np.sum(opt_weights)

    ret, vol, sharpe = calculate_portfolio_performance(
        opt_weights, mean_returns, cov_matrix, risk_free_rate
    )
    weights_dict = {ticker: float(round(w, 4)) for ticker, w in zip(tickers, opt_weights, strict=False)}

    return PortfolioMetrics(
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        weights=weights_dict,
    )


def optimize_min_volatility(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioMetrics:
    """Find portfolio weights that minimize total portfolio volatility.

    Args:
        mean_returns: Annualized expected returns per asset
        cov_matrix: Annualized covariance matrix
        risk_free_rate: Annual risk-free rate
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        PortfolioMetrics: Minimum volatility portfolio details
    """
    num_assets = len(mean_returns)
    tickers = list(mean_returns.index)

    def portfolio_vol(weights: np.ndarray) -> float:
        _, vol, _ = calculate_portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        return vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    init_weights = np.ones(num_assets) / num_assets

    res = minimize(
        portfolio_vol,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    opt_weights = np.clip(res.x, 0.0, 1.0)
    opt_weights = opt_weights / np.sum(opt_weights)

    ret, vol, sharpe = calculate_portfolio_performance(
        opt_weights, mean_returns, cov_matrix, risk_free_rate
    )
    weights_dict = {ticker: float(round(w, 4)) for ticker, w in zip(tickers, opt_weights, strict=False)}

    return PortfolioMetrics(
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        weights=weights_dict,
    )


def calculate_equal_weight_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> PortfolioMetrics:
    """Calculate metrics for a 1/N equal-weight benchmark portfolio."""
    num_assets = len(mean_returns)
    tickers = list(mean_returns.index)
    weights = np.ones(num_assets) / num_assets

    ret, vol, sharpe = calculate_portfolio_performance(
        weights, mean_returns, cov_matrix, risk_free_rate
    )
    weights_dict = {ticker: float(round(w, 4)) for ticker, w in zip(tickers, weights, strict=False)}

    return PortfolioMetrics(
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        weights=weights_dict,
    )


def calculate_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    min_vol_ret: float,
    max_ret: float,
    num_points: int = 40,
) -> tuple[list[float], list[float]]:
    """Compute risk and return coordinates along the Markowitz Efficient Frontier curve.

    Args:
        mean_returns: Annualized expected returns per asset
        cov_matrix: Annualized covariance matrix
        min_vol_ret: Return of the minimum volatility portfolio
        max_ret: Maximum individual asset return
        num_points: Number of frontier points to compute

    Returns:
        tuple[list[float], list[float]]: (frontier_volatilities, frontier_returns)
    """
    num_assets = len(mean_returns)
    target_returns = np.linspace(min_vol_ret, max_ret, num_points)

    frontier_returns = []
    frontier_vols = []
    init_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.sum(mean_returns * w) - t},
        ]

        def target_vol(weights: np.ndarray) -> float:
            return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

        res = minimize(
            target_vol,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-7, "maxiter": 500},
        )

        if res.success:
            frontier_vols.append(float(res.fun))
            frontier_returns.append(float(target))

    return frontier_vols, frontier_returns


def run_monte_carlo_portfolio_simulation(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_portfolios: int = 2500,
    risk_free_rate: float = 0.04,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate random asset allocations across the simplex to simulate the portfolio universe.

    Args:
        mean_returns: Annualized expected returns per asset
        cov_matrix: Annualized covariance matrix
        num_portfolios: Number of random portfolios to simulate
        risk_free_rate: Risk-free interest rate
        random_seed: Seed for reproducible simulation

    Returns:
        pd.DataFrame: DataFrame of simulated portfolios with Return, Volatility, Sharpe, and weights
    """
    rng = np.random.default_rng(random_seed)
    num_assets = len(mean_returns)
    tickers = list(mean_returns.index)

    # Dirichlet distribution samples uniformly from the simplex (sum(w) == 1, w >= 0)
    sim_weights = rng.dirichlet(np.ones(num_assets), size=num_portfolios)

    # Vectorized return calculation: shape (num_portfolios,)
    returns = np.dot(sim_weights, mean_returns.values)

    # Vectorized volatility: sqrt(diag(W * Cov * W.T))
    # cov_weights shape: (num_portfolios, num_assets)
    cov_weights = np.dot(sim_weights, cov_matrix.values)
    variances = np.sum(cov_weights * sim_weights, axis=1)
    volatilities = np.sqrt(np.maximum(variances, 1e-10))

    sharpe_ratios = (returns - risk_free_rate) / volatilities

    sim_df = pd.DataFrame(
        {
            "Return": returns,
            "Volatility": volatilities,
            "Sharpe": sharpe_ratios,
        }
    )

    for i, ticker in enumerate(tickers):
        sim_df[f"Weight_{ticker}"] = sim_weights[:, i]

    return sim_df


def backtest_portfolio_allocations(
    df_close: pd.DataFrame,
    weights_dict: dict[str, dict[str, float]],
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """Backtest historical cumulative portfolio growth for different allocation strategies.

    Args:
        df_close: DataFrame of historical daily closing prices
        weights_dict: Dictionary mapping strategy name to ticker weights dict
        initial_capital: Starting portfolio value in dollars

    Returns:
        pd.DataFrame: Daily portfolio equity curve ($) for each strategy
    """
    if df_close.empty:
        return pd.DataFrame()

    daily_returns = df_close.pct_change().fillna(0.0)
    equity_curves = pd.DataFrame(index=df_close.index)

    for strategy_name, weights in weights_dict.items():
        w_series = pd.Series(weights).reindex(df_close.columns).fillna(0.0)
        # Normalize weights if needed
        total_w = w_series.sum()
        if total_w > 0:
            w_series = w_series / total_w

        port_daily_ret = daily_returns.dot(w_series)
        port_cum_ret = (1.0 + port_daily_ret).cumprod()
        equity_curves[strategy_name] = initial_capital * port_cum_ret

    return equity_curves


def run_portfolio_optimization(
    df_close: pd.DataFrame,
    risk_free_rate: float = 0.04,
    num_simulations: int = 2500,
    periods_per_year: int = 252,
) -> OptimizationResult | None:
    """End-to-end execution of Modern Portfolio Theory optimization and Efficient Frontier.

    Args:
        df_close: Historical aligned closing prices (tickers as columns)
        risk_free_rate: Annual risk-free rate (e.g. 0.04)
        num_simulations: Number of Monte Carlo portfolios
        periods_per_year: Trading days per year (default: 252)

    Returns:
        OptimizationResult | None: Optimization data bundle or None if insufficient assets
    """
    if df_close.empty or len(df_close.columns) < 2:
        return None

    daily_returns = df_close.pct_change().dropna()
    if len(daily_returns) < 5:
        return None

    # Annualized mean returns and annualized covariance matrix
    mean_returns = daily_returns.mean() * periods_per_year
    cov_matrix = daily_returns.cov() * periods_per_year

    # 1. Optimal Portfolios
    max_sharpe_port = optimize_max_sharpe(
        mean_returns, cov_matrix, risk_free_rate=risk_free_rate
    )
    min_vol_port = optimize_min_volatility(
        mean_returns, cov_matrix, risk_free_rate=risk_free_rate
    )
    equal_weight_port = calculate_equal_weight_portfolio(
        mean_returns, cov_matrix, risk_free_rate=risk_free_rate
    )

    # 2. Monte Carlo Simulation Cloud
    sim_df = run_monte_carlo_portfolio_simulation(
        mean_returns,
        cov_matrix,
        num_portfolios=num_simulations,
        risk_free_rate=risk_free_rate,
    )

    # 3. Analytical Efficient Frontier
    max_asset_ret = float(mean_returns.max())
    min_vol_ret = float(min_vol_port.expected_return)
    # Ensure range has positive width
    if max_asset_ret <= min_vol_ret:
        max_asset_ret = min_vol_ret + 0.05

    frontier_vols, frontier_rets = calculate_efficient_frontier(
        mean_returns,
        cov_matrix,
        min_vol_ret=min_vol_ret,
        max_ret=max_asset_ret,
        num_points=40,
    )

    return OptimizationResult(
        max_sharpe_portfolio=max_sharpe_port,
        min_volatility_portfolio=min_vol_port,
        equal_weight_portfolio=equal_weight_port,
        efficient_frontier_returns=frontier_rets,
        efficient_frontier_volatilities=frontier_vols,
        simulated_portfolios=sim_df,
        tickers=list(df_close.columns),
        risk_free_rate=risk_free_rate,
        annualized_returns=mean_returns,
        covariance_matrix=cov_matrix,
    )
