"""Base interface and data classes for forecasting models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class ForecastResult:
    """Standardized container for model predictions and uncertainty intervals."""

    model_name: str
    future_dates: pd.DatetimeIndex
    forecast: pd.Series
    lower_bound_95: pd.Series
    upper_bound_95: pd.Series
    lower_bound_80: pd.Series
    upper_bound_80: pd.Series
    in_sample_actual: pd.Series
    in_sample_predicted: pd.Series
    metrics: dict[str, float]


class BaseForecaster(ABC):
    """Abstract Base Class for all time-series and ML forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, series: pd.Series, test_size: float = 0.2) -> "BaseForecaster":
        """Fit model on historical series with clean chronological train/test split.

        Args:
            series: Historical price Series with DatetimeIndex
            test_size: Proportion of data reserved for out-of-sample test evaluation
        """
        pass

    @abstractmethod
    def predict_future(self, steps: int = 30) -> ForecastResult:
        """Generate out-of-sample future price forecasts and confidence intervals.

        Args:
            steps: Number of future business days to project

        Returns:
            ForecastResult: Standardized forecast output
        """
        pass
