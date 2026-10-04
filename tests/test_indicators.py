"""Unit tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.technical import (
    add_all_indicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_vwap,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create deterministic OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)

    close_prices = 100 + np.cumsum(np.random.randn(100) * 1.5)
    high_prices = close_prices + np.random.uniform(0.5, 3.0, 100)
    low_prices = close_prices - np.random.uniform(0.5, 3.0, 100)
    open_prices = low_prices + np.random.uniform(0.1, 2.0, 100)
    volumes = np.random.randint(100000, 5000000, 100)

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


def test_calculate_sma(sample_ohlcv_data):
    sma_20 = calculate_sma(sample_ohlcv_data, window=20)
    assert len(sma_20) == 100
    assert not sma_20.isna().all()
    # Check that the 20th point equals the average of the first 20 closes
    expected_20 = sample_ohlcv_data["Close"].iloc[0:20].mean()
    assert pytest.approx(sma_20.iloc[19], rel=1e-4) == expected_20


def test_calculate_ema(sample_ohlcv_data):
    ema_20 = calculate_ema(sample_ohlcv_data, span=20)
    assert len(ema_20) == 100
    assert not ema_20.isna().any()


def test_calculate_rsi(sample_ohlcv_data):
    rsi = calculate_rsi(sample_ohlcv_data, period=14)
    assert len(rsi) == 100
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()


def test_calculate_macd(sample_ohlcv_data):
    macd_df = calculate_macd(sample_ohlcv_data)
    assert "MACD" in macd_df.columns
    assert "MACD_Signal" in macd_df.columns
    assert "MACD_Hist" in macd_df.columns
    assert len(macd_df) == 100
    # Check relationship Hist = MACD - Signal
    np.testing.assert_allclose(
        macd_df["MACD_Hist"].values,
        (macd_df["MACD"] - macd_df["MACD_Signal"]).values,
        atol=1e-6,
    )


def test_calculate_bollinger_bands(sample_ohlcv_data):
    bb = calculate_bollinger_bands(sample_ohlcv_data, window=20, num_std=2.0)
    assert "BB_Upper" in bb.columns
    assert "BB_Middle" in bb.columns
    assert "BB_Lower" in bb.columns
    # Upper band should always be >= Lower band
    assert (bb["BB_Upper"] >= bb["BB_Lower"]).all()


def test_calculate_vwap(sample_ohlcv_data):
    vwap = calculate_vwap(sample_ohlcv_data)
    assert len(vwap) == 100
    assert not vwap.isna().any()


def test_calculate_atr(sample_ohlcv_data):
    atr = calculate_atr(sample_ohlcv_data, period=14)
    assert len(atr) == 100
    assert (atr >= 0.0).all()


def test_calculate_obv(sample_ohlcv_data):
    obv = calculate_obv(sample_ohlcv_data)
    assert len(obv) == 100
    assert isinstance(obv, pd.Series)


def test_add_all_indicators(sample_ohlcv_data):
    enriched = add_all_indicators(sample_ohlcv_data)
    expected_cols = [
        "SMA_20", "SMA_50", "SMA_100", "SMA_200",
        "EMA_12", "EMA_26", "EMA_50",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "BB_Pct",
        "ATR_14", "VWAP", "OBV"
    ]
    for col in expected_cols:
        assert col in enriched.columns
