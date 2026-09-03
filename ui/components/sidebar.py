"""Sidebar component for user inputs, ticker search, and indicator toggles."""

from datetime import date, timedelta
from typing import Any

import streamlit as st

from config.settings import CONFIG
from src.data.loader import get_curated_tickers


def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return selected parameter configuration.

    Returns:
        dict: User configuration dictionary
    """
    with st.sidebar:
        st.markdown("## 🔍 Asset Selection")

        tickers_df = get_curated_tickers()
        curated_symbols = tickers_df["Symbol"].tolist()

        # Selection mode: Curated dropdown or Custom ticker input
        selection_mode = st.radio(
            "Selection Mode",
            ["Curated Assets", "Custom Ticker"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if selection_mode == "Curated Assets":
            formatted_options = [
                f"{row['Symbol']} - {row['Name']} ({row['Category']})"
                for _, row in tickers_df.iterrows()
            ]
            selected_idx = st.selectbox(
                "Choose Asset",
                range(len(formatted_options)),
                format_func=lambda i: formatted_options[i],
                index=0,
            )
            selected_ticker = curated_symbols[selected_idx]
        else:
            custom_input = st.text_input(
                "Enter Ticker Symbol (e.g. NVDA, TSLA, BTC-USD, ^GSPC):",
                value="NVDA",
            )
            selected_ticker = custom_input.strip().upper()

        st.markdown("---")
        st.markdown("## ⏱️ Timeframe")

        timeframe_label = st.selectbox(
            "Preset Horizon",
            list(CONFIG.TIMEFRAMES.keys()),
            index=4,  # Default: '1 Year'
        )
        selected_period = CONFIG.TIMEFRAMES[timeframe_label]

        use_custom_date = st.checkbox("Custom Date Range", value=False)
        start_date = None
        end_date = None

        if use_custom_date:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=date.today() - timedelta(days=365),
                ).strftime("%Y-%m-%d")
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=date.today(),
                ).strftime("%Y-%m-%d")

        st.markdown("---")
        st.markdown("## 🔬 Technical Overlays")

        show_sma_20 = st.checkbox("SMA 20 (Short-term)", value=True)
        show_sma_50 = st.checkbox("SMA 50 (Medium-term)", value=False)
        show_sma_200 = st.checkbox("SMA 200 (Long-term)", value=True)
        show_ema_12 = st.checkbox("EMA 12", value=False)
        show_ema_26 = st.checkbox("EMA 26", value=False)
        show_bollinger = st.checkbox("Bollinger Bands (20, 2σ)", value=True)
        show_vwap = st.checkbox("VWAP (Volume Weighted Avg)", value=False)
        show_volume = st.checkbox("Volume Subplot", value=True)

        st.markdown("---")
        if st.button("🔄 Refresh Market Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #64748B; text-align: center; margin-top: 20px;">
                AlphaPulse v2.0.0 • Powered by Streamlit & Plotly
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "ticker": selected_ticker,
        "period": selected_period,
        "start_date": start_date,
        "end_date": end_date,
        "overlays": {
            "sma_20": show_sma_20,
            "sma_50": show_sma_50,
            "sma_200": show_sma_200,
            "ema_12": show_ema_12,
            "ema_26": show_ema_26,
            "bollinger": show_bollinger,
            "vwap": show_vwap,
            "volume": show_volume,
        },
    }
