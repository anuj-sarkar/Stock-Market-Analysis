"""Accuracy evaluation metrics for time-series predictions."""


import numpy as np
import pandas as pd


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate standard statistical and financial forecasting error metrics.

    Args:
        actual: Ground truth price series
        predicted: Model predicted price series

    Returns:
        dict: RMSE, MAE, MAPE (%), and Directional Accuracy (%)
    """
    # Align and drop NaNs
    combined = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if combined.empty:
        return {
            "RMSE": 0.0,
            "MAE": 0.0,
            "MAPE (%)": 0.0,
            "Directional Accuracy (%)": 50.0,
        }

    y_true = combined["actual"].values
    y_pred = combined["predicted"].values

    # 1. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # 2. Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # 3. Mean Absolute Percentage Error (MAPE)
    non_zero = y_true != 0
    if np.any(non_zero):
        mape_arr = np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
        mape = float(np.mean(mape_arr) * 100.0)
    else:
        mape = 0.0

    # 4. Directional Accuracy (% of times predicted price change direction matches actual)
    if len(y_true) > 1:
        actual_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        direction_match = np.sign(actual_diff) == np.sign(pred_diff)
        dir_acc = float(np.mean(direction_match) * 100.0)
    else:
        dir_acc = 50.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": dir_acc,
    }
