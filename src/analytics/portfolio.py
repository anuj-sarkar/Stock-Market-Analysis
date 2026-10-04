"""Multi-stock comparison, normalized returns, and correlation analytics."""


import pandas as pd
import yfinance as yf

try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=3600, show_spinner=False)
except Exception:
    def cache_decorator(func):
        return func


@cache_decorator
def fetch_multiple_stocks(
    tickers: list[str],
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch and align closing prices for multiple tickers.

    Args:
        tickers: List of ticker strings
        period: Historical period
        start: Optional start date
        end: Optional end date

    Returns:
        pd.DataFrame: Aligned DataFrame with tickers as columns
    """
    clean_tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not clean_tickers:
        return pd.DataFrame()

    kwargs = {"auto_adjust": True, "progress": False}
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = yf.download(clean_tickers, **kwargs)

    if df.empty:
        return pd.DataFrame()

    # If multi-ticker, yfinance provides multi-index columns: ('Close', 'AAPL'), etc.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            close_df = df["Close"].copy()
        else:
            close_df = df.xs("Close", axis=1, level=0).copy()
    else:
        # Single ticker result
        close_df = df[["Close"]].copy()
        close_df.columns = [clean_tickers[0]]

    # Drop all-NaN rows/columns
    close_df = close_df.dropna(how="all").ffill().bfill()
    return close_df


def calculate_normalized_returns(df_close: pd.DataFrame) -> pd.DataFrame:
    """Normalize multi-asset closing prices to base 100 (% return from start).

    Args:
        df_close: DataFrame of aligned closing prices

    Returns:
        pd.DataFrame: Percentage return series starting from 0%
    """
    if df_close.empty:
        return pd.DataFrame()

    initial_prices = df_close.iloc[0]
    normalized = ((df_close / initial_prices) - 1.0) * 100.0
    return normalized


def calculate_correlation_matrix(df_close: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix of daily percentage returns.

    Args:
        df_close: DataFrame of aligned closing prices

    Returns:
        pd.DataFrame: Symmetric correlation matrix
    """
    if df_close.empty:
        return pd.DataFrame()

    daily_returns = df_close.pct_change().dropna()
    return daily_returns.corr()
