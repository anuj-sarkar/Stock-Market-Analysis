"""Configuration settings for Stock Market Analysis."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppConfig:
    """Application constants and default configurations."""

    APP_NAME: str = "AlphaPulse • Stock Analytics"
    APP_VERSION: str = "2.0.0"
    PAGE_ICON: str = "📈"

    # Default Tickers
    DEFAULT_TICKERS: list[str] = field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "TSLA",
            "META",
            "BRK-B",
            "JPM",
            "SPY",
            "QQQ",
            "BTC-USD",
        ]
    )

    # Timeframe presets
    TIMEFRAMES: dict[str, str] = field(
        default_factory=lambda: {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "YTD": "ytd",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "Max": "max",
        }
    )

    # Default moving average windows
    SMA_SHORT: int = 20
    SMA_MEDIUM: int = 50
    SMA_LONG: int = 200
    EMA_FAST: int = 12
    EMA_SLOW: int = 26
    RSI_PERIOD: int = 14
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0

    # Forecasting parameters
    DEFAULT_FORECAST_DAYS: int = 30
    DEFAULT_LOOKBACK_DAYS: int = 60

    # Visual Theme Palette
    COLORS: dict[str, str] = field(
        default_factory=lambda: {
            "bullish": "#00E676",
            "bearish": "#FF5252",
            "primary": "#2979FF",
            "accent": "#00E5FF",
            "warning": "#FFD600",
            "background": "#0E1117",
            "card_bg": "#1A1F2C",
            "text": "#E0E6ED",
            "grid": "#2D3748",
        }
    )


CONFIG = AppConfig()
