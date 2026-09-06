"""Holt-Winters Exponential Smoothing forecasting model with confidence intervals."""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.models.base import BaseForecaster, ForecastResult
from src.models.evaluator import evaluate_forecast


class ExponentialSmoothingForecaster(BaseForecaster):
    """Holt-Winters Exponential Smoothing forecaster with trend damping."""

    def __init__(self, trend: str = "add", damped_trend: bool = True):
        super().__init__(name="Holt-Winters Exponential Smoothing")
        self.trend = trend
        self.damped_trend = damped_trend
        self.series: pd.Series | None = None
        self.train_series: pd.Series | None = None
        self.test_series: pd.Series | None = None
        self.fitted_model = None
        self.in_sample_pred: pd.Series | None = None
        self.metrics: dict[str, float] = {}

    def fit(self, series: pd.Series, test_size: float = 0.2) -> "ExponentialSmoothingForecaster":
        """Fit Exponential Smoothing model with clean chronological train/test split.

        Args:
            series: Historical price Series with DatetimeIndex
            test_size: Proportion of historical data reserved for evaluation (e.g. 0.2 for 20%)
        """
        clean_series = series.dropna().copy()
        if len(clean_series) < 20:
            raise ValueError(
                f"Insufficient data points for forecasting ({len(clean_series)} provided, "
                "minimum 20 required). Please choose a longer timeframe in the sidebar."
            )

        self.series = clean_series
        split_idx = int(len(clean_series) * (1.0 - test_size))
        self.train_series = clean_series.iloc[:split_idx]
        self.test_series = clean_series.iloc[split_idx:]
        # Train on numpy values to avoid statsmodels index frequency validation
        # errors with irregular trading days
        eval_model = ExponentialSmoothing(
            np.asarray(self.train_series.values, dtype=float),
            trend=self.trend,
            damped_trend=self.damped_trend,
            initialization_method="estimated",
        ).fit()

        test_forecast_values = eval_model.forecast(len(self.test_series))
        self.in_sample_pred = pd.Series(test_forecast_values, index=self.test_series.index)
        self.metrics = evaluate_forecast(self.test_series, self.in_sample_pred)

        # Full fit on entire series for future out-of-sample projection
        self.fitted_model = ExponentialSmoothing(
            np.asarray(self.series.values, dtype=float),
            trend=self.trend,
            damped_trend=self.damped_trend,
            initialization_method="estimated",
        ).fit()

        self.is_fitted = True
        return self

    def predict_future(self, steps: int = 30) -> ForecastResult:
        """Generate future out-of-sample forecasts with analytical confidence intervals.

        Args:
            steps: Number of future business days to forecast

        Returns:
            ForecastResult: Forecast and confidence bounds
        """
        if not self.is_fitted or self.fitted_model is None or self.series is None:
            raise RuntimeError("Model must be fitted before predicting future values.")

        last_date = self.series.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=steps, freq="B"
        )

        point_forecast = np.asarray(self.fitted_model.forecast(steps), dtype=float)
        forecast_series = pd.Series(point_forecast, index=future_dates)

        # Residual volatility estimate for confidence intervals
        residuals = np.asarray(self.series.values, dtype=float) - self.fitted_model.fittedvalues
        sigma = float(np.std(residuals)) if len(residuals) > 0 else 1.0

        # Horizon scaling factor: std grows with sqrt(h)
        step_factors = np.sqrt(np.arange(1, steps + 1))
        margin_95 = 1.96 * sigma * step_factors
        margin_80 = 1.28 * sigma * step_factors

        lower_95 = pd.Series(
            np.maximum(0, forecast_series.values - margin_95), index=future_dates
        )
        upper_95 = pd.Series(forecast_series.values + margin_95, index=future_dates)
        lower_80 = pd.Series(
            np.maximum(0, forecast_series.values - margin_80), index=future_dates
        )
        upper_80 = pd.Series(forecast_series.values + margin_80, index=future_dates)

        actual_test = self.test_series if self.test_series is not None else pd.Series()
        pred_test = self.in_sample_pred if self.in_sample_pred is not None else pd.Series()

        return ForecastResult(
            model_name=self.name,
            future_dates=future_dates,
            forecast=forecast_series,
            lower_bound_95=lower_95,
            upper_bound_95=upper_95,
            lower_bound_80=lower_80,
            upper_bound_80=upper_80,
            in_sample_actual=actual_test,
            in_sample_predicted=pred_test,
            metrics=self.metrics,
        )
