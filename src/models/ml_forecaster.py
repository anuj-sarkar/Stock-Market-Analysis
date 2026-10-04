"""Feature-engineered Machine Learning multi-step forecaster."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseForecaster, ForecastResult
from src.models.evaluator import evaluate_forecast


class MLForecaster(BaseForecaster):
    """Multi-lag feature engineered Machine Learning price forecaster."""

    def __init__(self, lookback_lags: int = 10, alpha: float = 1.0):
        super().__init__(name="Feature-Engineered ML Forecaster")
        self.lookback_lags = lookback_lags
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)
        self.series: pd.Series | None = None
        self.test_series: pd.Series | None = None
        self.in_sample_pred: pd.Series | None = None
        self.metrics: dict[str, float] = {}
        self.residual_std: float = 1.0

    def _create_features(
        self, series: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Create tabular lag features from a time series."""
        values = series.values
        X, y, target_dates = [], [], []

        for i in range(self.lookback_lags, len(values)):
            X.append(values[i - self.lookback_lags : i])
            y.append(values[i])
            target_dates.append(series.index[i])

        return np.array(X), np.array(y), pd.DatetimeIndex(target_dates)

    def fit(self, series: pd.Series, test_size: float = 0.2) -> "MLForecaster":
        """Fit ML model with feature scaling and clean out-of-sample test split.

        Args:
            series: Historical price Series
            test_size: Proportion for evaluation split
        """
        clean_series = series.dropna().copy()
        min_required = self.lookback_lags + 10
        if len(clean_series) < min_required:
            raise ValueError(
                f"Insufficient data points for ML forecasting ({len(clean_series)} provided, "
                f"minimum {min_required} required). Please choose a longer timeframe."
            )

        self.series = clean_series
        X, y, dates = self._create_features(clean_series)

        split_idx = int(len(X) * (1.0 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        test_dates = dates[split_idx:]

        # Fit scaler strictly on training set only (No Data Leakage!)
        scaler_eval = StandardScaler()
        X_train_scaled = scaler_eval.fit_transform(X_train)
        X_test_scaled = scaler_eval.transform(X_test)

        eval_model = Ridge(alpha=self.alpha).fit(X_train_scaled, y_train)
        y_test_pred = eval_model.predict(X_test_scaled)

        self.test_series = pd.Series(y_test, index=test_dates)
        self.in_sample_pred = pd.Series(y_test_pred, index=test_dates)
        self.metrics = evaluate_forecast(self.test_series, self.in_sample_pred)

        # Full fit on all available data for future deployment
        X_full_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_full_scaled, y)

        # Compute residual standard deviation
        in_sample_residuals = y - self.model.predict(X_full_scaled)
        self.residual_std = (
            float(np.std(in_sample_residuals)) if len(in_sample_residuals) > 0 else 1.0
        )

        self.is_fitted = True
        return self

    def predict_future(self, steps: int = 30) -> ForecastResult:
        """Generate recursive multi-step future price forecast."""
        if not self.is_fitted or self.series is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        last_date = self.series.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=steps, freq="B"
        )

        current_window = list(self.series.values[-self.lookback_lags :])
        predictions = []

        for _ in range(steps):
            x_input = np.array(current_window[-self.lookback_lags :]).reshape(1, -1)
            x_input_scaled = self.scaler.transform(x_input)
            next_pred = float(self.model.predict(x_input_scaled)[0])
            predictions.append(next_pred)
            current_window.append(next_pred)

        forecast_series = pd.Series(predictions, index=future_dates)

        # Uncertainty intervals scaling with horizon
        step_factors = np.sqrt(np.arange(1, steps + 1))
        margin_95 = 1.96 * self.residual_std * step_factors
        margin_80 = 1.28 * self.residual_std * step_factors

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
