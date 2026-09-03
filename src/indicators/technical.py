"""Technical analysis indicators implemented in pure NumPy and Pandas."""

import numpy as np
import pandas as pd


def calculate_sma(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    Args:
        df: Input dataframe containing the price column
        window: Moving average window period
        column: Price column name

    Returns:
        pd.Series: SMA values
    """
    return df[column].rolling(window=window, min_periods=1).mean()


def calculate_ema(df: pd.DataFrame, span: int = 20, column: str = "Close") -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Args:
        df: Input dataframe containing the price column
        span: EMA decay span
        column: Price column name

    Returns:
        pd.Series: EMA values
    """
    return df[column].ewm(span=span, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's exponential smoothing.

    Args:
        df: Input dataframe containing the price column
        period: Lookback period (default 14)
        column: Price column name

    Returns:
        pd.Series: RSI values bounded in [0, 100]
    """
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's Exponential Smoothing (alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fill NaN / division by zero edge cases
    rsi = rsi.fillna(50.0)
    return rsi.clip(lower=0.0, upper=100.0)


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD), Signal Line & Histogram.

    Args:
        df: Input dataframe containing the price column
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        column: Price column name

    Returns:
        pd.DataFrame: Columns ['MACD', 'MACD_Signal', 'MACD_Hist']
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": hist,
        },
        index=df.index,
    )


def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    column: str = "Close",
) -> pd.DataFrame:
    """Calculate Bollinger Bands (Upper, Middle, Lower, Bandwidth, %B).

    Args:
        df: Input dataframe containing the price column
        window: Moving average window period
        num_std: Number of standard deviations
        column: Price column name

    Returns:
        pd.DataFrame: Columns ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Pct']
    """
    middle = df[column].rolling(window=window, min_periods=1).mean()
    std = df[column].rolling(window=window, min_periods=1).std().fillna(0)

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    width = ((upper - lower) / middle.replace(0, np.nan)).fillna(0) * 100.0
    denom = (upper - lower).replace(0, np.nan)
    pct_b = ((df[column] - lower) / denom).fillna(0.5)

    return pd.DataFrame(
        {
            "BB_Upper": upper,
            "BB_Middle": middle,
            "BB_Lower": lower,
            "BB_Width": width,
            "BB_Pct": pct_b,
        },
        index=df.index,
    )


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume-Weighted Average Price (VWAP).

    Args:
        df: Input dataframe with High, Low, Close, Volume

    Returns:
        pd.Series: VWAP values
    """
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return df["Close"]

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_tp_vol = (typical_price * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.ffill().fillna(df["Close"])


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) for volatility estimation.

    Args:
        df: Input dataframe with High, Low, Close
        period: ATR smoothing period

    Returns:
        pd.Series: ATR values
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.fillna(tr1)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume (OBV).

    Args:
        df: Input dataframe with Close and Volume

    Returns:
        pd.Series: OBV values
    """
    if "Volume" not in df.columns:
        return pd.Series(0, index=df.index)

    close_diff = df["Close"].diff()
    direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
    obv = (direction * df["Volume"]).cumsum()
    return pd.Series(obv, index=df.index, name="OBV")


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich input OHLCV dataframe with a comprehensive set of technical indicators.

    Args:
        df: OHLCV DataFrame

    Returns:
        pd.DataFrame: Enriched DataFrame with all indicator columns
    """
    data = df.copy()

    # Moving Averages
    data["SMA_20"] = calculate_sma(data, window=20)
    data["SMA_50"] = calculate_sma(data, window=50)
    data["SMA_100"] = calculate_sma(data, window=100)
    data["SMA_200"] = calculate_sma(data, window=200)

    data["EMA_12"] = calculate_ema(data, span=12)
    data["EMA_26"] = calculate_ema(data, span=26)
    data["EMA_50"] = calculate_ema(data, span=50)

    # Momentum
    data["RSI_14"] = calculate_rsi(data, period=14)

    macd_df = calculate_macd(data, fast=12, slow=26, signal=9)
    data["MACD"] = macd_df["MACD"]
    data["MACD_Signal"] = macd_df["MACD_Signal"]
    data["MACD_Hist"] = macd_df["MACD_Hist"]

    # Volatility
    bb_df = calculate_bollinger_bands(data, window=20, num_std=2.0)
    data["BB_Upper"] = bb_df["BB_Upper"]
    data["BB_Middle"] = bb_df["BB_Middle"]
    data["BB_Lower"] = bb_df["BB_Lower"]
    data["BB_Width"] = bb_df["BB_Width"]
    data["BB_Pct"] = bb_df["BB_Pct"]

    data["ATR_14"] = calculate_atr(data, period=14)

    # Volume
    data["VWAP"] = calculate_vwap(data)
    data["OBV"] = calculate_obv(data)

    return data
