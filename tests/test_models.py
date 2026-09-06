"""Unit tests for forecasting models and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.models.base import ForecastResult
from src.models.evaluator import evaluate_forecast
from src.models.exponential_smoothing import ExponentialSmoothingForecaster
from src.models.ml_forecaster import MLForecaster
from src.models.monte_carlo import MonteCarloGBM


@pytest.fixture
def mock_price_history():
    """Create a realistic upward trending price series with noise."""
    dates = pd.date_range("2023-01-01", periods=150, freq="B")
    np.random.seed(42)
    trend = np.linspace(100, 180, 150)
    noise = np.random.normal(0, 2.0, 150)
    prices = trend + noise
    return pd.Series(prices, index=dates, name="Close")


def test_evaluator_metrics():
    actual = pd.Series([100.0, 105.0, 110.0, 115.0])
    predicted = pd.Series([101.0, 104.0, 111.0, 114.0])

    metrics = evaluate_forecast(actual, predicted)
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "MAPE (%)" in metrics
    assert "Directional Accuracy (%)" in metrics

    assert metrics["RMSE"] > 0
    assert metrics["MAE"] > 0
    assert metrics["MAPE (%)"] > 0
    assert metrics["Directional Accuracy (%)"] == 100.0


def test_exponential_smoothing_forecaster(mock_price_history):
    model = ExponentialSmoothingForecaster()
    model.fit(mock_price_history, test_size=0.2)

    assert model.is_fitted
    result = model.predict_future(steps=15)

    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == 15
    assert len(result.future_dates) == 15
    assert (result.upper_bound_95 >= result.lower_bound_95).all()
    assert (result.upper_bound_80 >= result.lower_bound_80).all()
    assert not result.forecast.isna().any()


def test_ml_forecaster(mock_price_history):
    model = MLForecaster(lookback_lags=5)
    model.fit(mock_price_history, test_size=0.2)

    assert model.is_fitted
    result = model.predict_future(steps=10)

    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == 10
    assert (result.upper_bound_95 >= result.lower_bound_95).all()
    assert not result.forecast.isna().any()


def test_monte_carlo_gbm(mock_price_history):
    model = MonteCarloGBM(num_simulations=100)
    model.fit(mock_price_history, test_size=0.2)

    assert model.is_fitted
    result = model.predict_future(steps=20)

    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == 20
    assert (result.upper_bound_95 >= result.lower_bound_95).all()
    assert model.simulated_paths.shape == (20, 100)
