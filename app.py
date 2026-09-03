"""AlphaPulse - Modern Financial Analytics and Time-Series Forecasting Dashboard."""

import pandas as pd
import streamlit as st

from config.settings import CONFIG
from src.analytics.metrics import compute_performance_summary
from src.data.loader import fetch_stock_data, fetch_ticker_info
from src.indicators.technical import add_all_indicators
from src.utils.helpers import (
    apply_custom_styles,
    format_currency,
    format_large_number,
    format_percentage,
)
from ui.components.charts import (
    create_candlestick_chart,
    create_cumulative_returns_chart,
    create_drawdown_chart,
    create_macd_chart,
    create_rsi_chart,
)
from ui.components.metrics_cards import (
    render_header_profile,
    render_market_kpi_cards,
    render_risk_kpi_cards,
)
from ui.components.sidebar import render_sidebar

# Set Streamlit Page Configuration
st.set_page_config(
    page_title=CONFIG.APP_NAME,
    page_icon=CONFIG.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Styling
apply_custom_styles()


def main() -> None:
    """Main application execution pipeline."""
    # Render Sidebar and collect configuration
    sidebar_params = render_sidebar()
    ticker = sidebar_params["ticker"]
    period = sidebar_params["period"]
    start_date = sidebar_params["start_date"]
    end_date = sidebar_params["end_date"]
    overlays = sidebar_params["overlays"]

    if not ticker:
        st.warning("⚠️ Please enter or select a valid stock ticker symbol from the sidebar.")
        return

    # Data Ingestion
    with st.spinner(f"Fetching real-time market data for {ticker}..."):
        try:
            raw_df = fetch_stock_data(
                ticker=ticker,
                period=period,
                start=start_date,
                end=end_date,
            )
            ticker_info = fetch_ticker_info(ticker)
        except Exception as e:
            st.error(f"❌ Failed to load data for symbol '{ticker}': {e}")
            st.info(
                "Tip: Double-check the ticker symbol (e.g., 'AAPL', 'MSFT', 'BTC-USD', '^GSPC')."
            )
            return

    if raw_df.empty:
        st.warning(f"No price data available for '{ticker}'.")
        return

    # Enrich dataset with technical indicators & performance metrics
    df = add_all_indicators(raw_df)
    perf_summary = compute_performance_summary(df)

    # 1. Header Profile & Top Market KPI Cards
    render_header_profile(ticker_info)
    render_market_kpi_cards(ticker_info, perf_summary)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # 2. Main Multi-Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price & Technicals",
        "📊 Risk & Performance",
        "📑 Historical Data",
        "🏢 Profile & Fundamentals",
    ])

    # Tab 1: Price & Technicals
    with tab1:
        st.markdown("### Interactive Candlestick & Volume Chart")
        candle_fig = create_candlestick_chart(df, ticker, overlays)
        st.plotly_chart(candle_fig, use_container_width=True)

        col_osc1, col_osc2 = st.columns(2)
        with col_osc1:
            st.markdown("### Relative Strength Index (RSI 14)")
            rsi_fig = create_rsi_chart(df)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with col_osc2:
            st.markdown("### MACD (12, 26, 9)")
            macd_fig = create_macd_chart(df)
            st.plotly_chart(macd_fig, use_container_width=True)

    # Tab 2: Risk & Performance
    with tab2:
        st.markdown("### Key Risk & Return Metrics")
        render_risk_kpi_cards(perf_summary)

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.markdown("### Cumulative Return Trajectory (%)")
            cum_fig = create_cumulative_returns_chart(df, ticker)
            st.plotly_chart(cum_fig, use_container_width=True)

        with col_perf2:
            st.markdown("### Underwater Drawdown Curve (%)")
            dd_fig = create_drawdown_chart(df)
            st.plotly_chart(dd_fig, use_container_width=True)

        st.markdown("### Performance Statistics Summary")
        summary_table_data = [
            {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in perf_summary.items()
        ]
        st.dataframe(pd.DataFrame(summary_table_data), use_container_width=True, hide_index=True)

    # Tab 3: Historical Data
    with tab3:
        st.markdown("### OHLCV & Indicators Historical Table")
        display_cols = [
            "Open", "High", "Low", "Close", "Volume", "Daily_Return",
            "SMA_20", "SMA_50", "SMA_200", "RSI_14", "MACD", "VWAP",
        ]
        available_display_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(
            df[available_display_cols].sort_index(ascending=False),
            use_container_width=True,
            height=400,
        )

        col_d1, col_d2 = st.columns([1, 4])
        with col_d1:
            csv_data = df.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{ticker}_market_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("### Descriptive Statistics")
        st.dataframe(
            df[["Open", "High", "Low", "Close", "Volume"]].describe(),
            use_container_width=True,
        )

    # Tab 4: Profile & Fundamentals
    with tab4:
        col_prof1, col_prof2 = st.columns([2, 1])

        with col_prof1:
            st.markdown("### About the Company")
            desc = ticker_info.get("description", "No detailed business summary available.")
            st.markdown(
                f"""
                <div style="background: #1A1F2C; border: 1px solid #2D3748;
                border-radius: 8px; padding: 18px; color: #CBD5E1; line-height: 1.6;">
                    {desc}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ticker_info.get("website"):
                site = ticker_info["website"]
                st.markdown(f"🌐 **Official Website:** [{site}]({site})")

        with col_prof2:
            st.markdown("### Key Valuation & Fundamentals")
            div_yield = format_percentage(
                ticker_info.get("dividend_yield"),
                include_sign=False,
            )
            fund_data = [
                {
                    "Metric": "Trailing P/E Ratio",
                    "Value": str(ticker_info.get("pe_ratio") or "N/A"),
                },
                {
                    "Metric": "Forward P/E Ratio",
                    "Value": str(ticker_info.get("forward_pe") or "N/A"),
                },
                {
                    "Metric": "Trailing EPS",
                    "Value": format_currency(ticker_info.get("eps")),
                },
                {
                    "Metric": "Dividend Yield",
                    "Value": div_yield,
                },
                {
                    "Metric": "Beta (Volatility vs Market)",
                    "Value": str(ticker_info.get("beta") or "N/A"),
                },
                {
                    "Metric": "Market Cap",
                    "Value": format_large_number(ticker_info.get("market_cap")),
                },
                {
                    "Metric": "52-Week High",
                    "Value": format_currency(ticker_info.get("fifty_two_week_high")),
                },
                {
                    "Metric": "52-Week Low",
                    "Value": format_currency(ticker_info.get("fifty_two_week_low")),
                },
            ]
            st.dataframe(pd.DataFrame(fund_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
