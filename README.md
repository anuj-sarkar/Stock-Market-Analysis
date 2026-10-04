# 📈 AlphaPulse • Stock Market Analysis & Forecasting Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, high-performance financial analytics and multi-horizon time-series forecasting dashboard built with **Streamlit**, **Plotly**, and **Scikit-Learn/Statsmodels**.

---

## 🚀 Key Features

- **🌐 Universal Asset Support:** Real-time data ingestion for US Equities, Global Indices, Cryptocurrencies, and ETFs via Yahoo Finance.
- **📊 Interactive Financial Charts:** TradingView-style candlestick charts, multi-timeframe volume profiles, and synchronized oscillators.
- **🔬 Comprehensive Technical Indicators:**
  - *Trend:* SMA (20/50/200), EMA (12/26), VWAP.
  - *Momentum:* RSI (14), MACD with Signal & Histogram.
  - *Volatility:* Bollinger Bands (20, 2σ), ATR.
  - *Volume:* On-Balance Volume (OBV).
- **🤖 Out-of-Sample AI Forecasting:** Multi-day future projections (7, 14, 30, 90 days) with confidence intervals and statistical metrics (RMSE, MAE, MAPE).
- **📉 Risk & Performance Analytics:** Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.
- **⚡ High Performance Caching:** Zero-latency interactions powered by `@st.cache_data`.

---

## 🏗️ Project Architecture

```
Stock-Market-Analysis/
├── .streamlit/             # Streamlit dark theme & server configurations
├── config/                 # Centralized settings and constants
│   └── settings.py
├── data/                   # Asset tickers and curated datasets
│   └── tickers.csv
├── notebooks/              # Research & exploratory Jupyter notebooks
│   └── LSTM_Model.ipynb
├── src/                    # Core business logic packages
│   ├── analytics/          # Financial risk & return metrics
│   ├── backtesting/        # Trading strategy backtesting engine
│   ├── data/               # Yahoo Finance cached data loader
│   ├── indicators/         # Technical analysis indicators
│   ├── models/             # Time-series forecasting models
│   └── utils/              # UI & formatting helpers
├── ui/                     # UI components & multi-page layouts
│   └── components/         # Reusable chart cards & metric widgets
├── tests/                  # Automated test suite
├── app.py                  # Main application entry point
├── pyproject.toml          # Build configuration & tooling specs
└── requirements.txt        # Pinned project dependencies
```

---

## 📦 Installation & Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/anuj-sarkar/Stock-Market-Analysis.git
cd Stock-Market-Analysis
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

---

## 🧪 Running Tests & Linting

```bash
# Run test suite
pytest

# Run linter
ruff check .
```

---

## 📄 License

This project is licensed under the MIT License.
