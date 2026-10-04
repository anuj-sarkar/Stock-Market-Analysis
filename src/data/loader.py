"""Data ingestion module with caching and robust error handling."""

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# Try to use Streamlit cache if in Streamlit environment; fallback gracefully otherwise
try:
    import streamlit as st
    cache_data_decorator = st.cache_data(ttl=3600, show_spinner=False)
except Exception:
    def cache_data_decorator(func):
        return func


@cache_data_decorator
def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker using yfinance.

    Args:
        ticker: Symbol string, e.g., 'AAPL', 'BTC-USD', '^GSPC'
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        start: Optional start date string 'YYYY-MM-DD'
        end: Optional end date string 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: Cleaned OHLCV dataframe with DatetimeIndex and Daily Return
    """
    cleaned_ticker = ticker.strip().upper()

    kwargs = {"interval": interval, "auto_adjust": True, "progress": False}
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = yf.download(cleaned_ticker, **kwargs)

    if df.empty:
        raise ValueError(f"No market data found for symbol '{cleaned_ticker}'.")

    # Handle multi-index columns if returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in required_cols if c in df.columns]
    if "Close" not in available_cols:
        raise ValueError(f"Data for '{cleaned_ticker}' is missing 'Close' prices.")

    df = df[available_cols].copy()
    df.dropna(subset=["Close"], inplace=True)

    # Compute returns
    df["Daily_Return"] = df["Close"].pct_change()

    return df


@cache_data_decorator
def fetch_ticker_info(ticker: str) -> dict[str, Any]:
    """Fetch fundamental and company profile information for a ticker.

    Args:
        ticker: Stock symbol string

    Returns:
        dict: Financial fundamentals and profile data
    """
    cleaned_ticker = ticker.strip().upper()
    yticker = yf.Ticker(cleaned_ticker)

    try:
        info = yticker.info or {}
    except Exception:
        info = {}

    summary = {
        "symbol": cleaned_ticker,
        "name": info.get("shortName") or info.get("longName") or cleaned_ticker,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or 0.0,
        "previous_close": (
            info.get("previousClose") or info.get("regularMarketPreviousClose") or 0.0
        ),
        "open": info.get("open") or info.get("regularMarketOpen") or 0.0,
        "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh") or 0.0,
        "day_low": info.get("dayLow") or info.get("regularMarketDayLow") or 0.0,
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh") or 0.0,
        "fifty_two_week_low": info.get("fiftyTwoWeekLow") or 0.0,
        "market_cap": info.get("marketCap") or 0,
        "volume": info.get("volume") or info.get("regularMarketVolume") or 0,
        "avg_volume": info.get("averageVolume") or 0,
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or None,
        "forward_pe": info.get("forwardPE") or None,
        "eps": info.get("trailingEps") or None,
        "dividend_yield": (
            (info.get("dividendYield") or 0.0) * 100 if info.get("dividendYield") else 0.0
        ),
        "beta": info.get("beta") or None,
        "sector": info.get("sector") or "N/A",
        "industry": info.get("industry") or "N/A",
        "currency": info.get("currency") or "USD",
        "exchange": info.get("exchange") or "N/A",
        "description": info.get("longBusinessSummary") or "No business summary available.",
        "website": info.get("website") or "",
    }

    return summary


def get_curated_tickers(csv_path: str | None = None) -> pd.DataFrame:
    """Load curated tickers list from CSV.

    Args:
        csv_path: Optional custom path to tickers.csv

    Returns:
        pd.DataFrame: Table of curated tickers
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "tickers.csv"

    if Path(csv_path).exists():
        return pd.read_csv(csv_path)

    # Fallback default dataframe
    return pd.DataFrame([
        {
            "Symbol": "AAPL",
            "Name": "Apple Inc.",
            "Sector": "Technology",
            "Category": "Large Cap",
        },
        {
            "Symbol": "MSFT",
            "Name": "Microsoft Corp",
            "Sector": "Technology",
            "Category": "Large Cap",
        },
        {
            "Symbol": "GOOGL",
            "Name": "Alphabet Inc.",
            "Sector": "Communication",
            "Category": "Large Cap",
        },
        {
            "Symbol": "NVDA",
            "Name": "NVIDIA Corp",
            "Sector": "Technology",
            "Category": "Semiconductors",
        },
        {
            "Symbol": "BTC-USD",
            "Name": "Bitcoin (USD)",
            "Sector": "Crypto",
            "Category": "Digital Asset",
        },
    ])


def validate_ticker(ticker: str) -> bool:
    """Check if ticker exists and has valid market data.

    Args:
        ticker: Symbol string

    Returns:
        bool: True if data is available, False otherwise
    """
    try:
        df = fetch_stock_data(ticker, period="5d")
        return not df.empty
    except Exception:
        return False
