"""Monte Carlo Geometric Brownian Motion (GBM) simulation engine."""

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster, ForecastResult
from src.models.evaluator import evaluate_forecast


class MonteCarloGBM(BaseForecaster):
    """Monte Carlo stochastic simulator using Geometric Brownian Motion."""

    def __init__(self, num_simulations: int = 1000):
        super().__init__(name="Monte Carlo GBM Simulator")
        self.num_simulations = num_simulations
        self.series: pd.Series | None = None
        self.test_series: pd.Series | None = None
        self.in_sample_pred: pd.Series | None = None
        self.metrics: dict[str, float] = {}
        self.mu: float = 0.0
        self.sigma: float = 0.0
        self.simulated_paths: np.ndarray | None = None

    def fit(self, series: pd.Series, test_size: float = 0.2) -> "MonteCarloGBM":
        """Estimate drift and volatility parameters from historical returns.

        Args:
            series: Historical price Series
            test_size: Proportion for evaluation split
        """
        clean_series = series.dropna().copy()
        if len(clean_series) < 20:
            raise ValueError("Need at least 20 historical points for Monte Carlo drift estimation.")

        self.series = clean_series
        log_returns = np.log(clean_series / clean_series.shift(1)).dropna()

        self.mu = float(log_returns.mean())
        self.sigma = float(log_returns.std())

        # Evaluation split
        split_idx = int(len(clean_series) * (1.0 - test_size))
        self.test_series = clean_series.iloc[split_idx:]

        # Deterministic drift baseline for in-sample evaluation
        s0 = clean_series.iloc[split_idx - 1]
        n_test = len(self.test_series)
        t = np.arange(1, n_test + 1)
        test_pred_vals = s0 * np.exp((self.mu - 0.5 * (self.sigma ** 2)) * t)
        self.in_sample_pred = pd.Series(test_pred_vals, index=self.test_series.index)
        self.metrics = evaluate_forecast(self.test_series, self.in_sample_pred)

        self.is_fitted = True
        return self

    def predict_future(self, steps: int = 30) -> ForecastResult:
        """Simulate stochastic paths and compute percentile confidence cones.

        Args:
            steps: Number of future days

        Returns:
            ForecastResult: Median trajectory and percentile cones
        """
        if not self.is_fitted or self.series is None:
            raise RuntimeError("Model must be fitted before running Monte Carlo simulation.")

        last_price = float(self.series.iloc[-1])
        last_date = self.series.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=steps, freq="B"
        )

        # Generate standard normal random variables (steps x num_simulations)
        np.random.seed(42)
        daily_shocks = np.random.normal(0, 1, size=(steps, self.num_simulations))

        # Geometric Brownian Motion step equation
        drift = self.mu - 0.5 * (self.sigma ** 2)
        daily_multipliers = np.exp(drift + (self.sigma * daily_shocks))

        # Accumulate price paths
        price_paths = np.zeros((steps + 1, self.num_simulations))
        price_paths[0] = last_price
        for t in range(1, steps + 1):
            price_paths[t] = price_paths[t - 1] * daily_multipliers[t - 1]

        self.simulated_paths = price_paths[1:, :]  # shape: (steps, num_simulations)

        median_forecast = pd.Series(np.median(self.simulated_paths, axis=1), index=future_dates)
        p05 = pd.Series(np.percentile(self.simulated_paths, 5, axis=1), index=future_dates)
        p95 = pd.Series(np.percentile(self.simulated_paths, 95, axis=1), index=future_dates)
        p10 = pd.Series(np.percentile(self.simulated_paths, 10, axis=1), index=future_dates)
        p90 = pd.Series(np.percentile(self.simulated_paths, 90, axis=1), index=future_dates)

        actual_test = self.test_series if self.test_series is not None else pd.Series()
        pred_test = self.in_sample_pred if self.in_sample_pred is not None else pd.Series()

        return ForecastResult(
            model_name=self.name,
            future_dates=future_dates,
            forecast=median_forecast,
            lower_bound_95=p05,
            upper_bound_95=p95,
            lower_bound_80=p10,
            upper_bound_80=p90,
            in_sample_actual=actual_test,
            in_sample_predicted=pred_test,
            metrics=self.metrics,
        )
