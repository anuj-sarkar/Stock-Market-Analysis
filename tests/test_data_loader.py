"""Unit tests for data ingestion loader."""

import pandas as pd

from src.data.loader import get_curated_tickers


def test_get_curated_tickers():
    tickers_df = get_curated_tickers()
    assert isinstance(tickers_df, pd.DataFrame)
    assert not tickers_df.empty
    assert "Symbol" in tickers_df.columns
    assert "Name" in tickers_df.columns
    symbols = tickers_df["Symbol"].tolist()
    assert "AAPL" in symbols
    assert "MSFT" in symbols
