"""AlphaPulse - Modern Financial Analytics, AI Forecasting & Algorithmic Trading Dashboard."""

import pandas as pd
import streamlit as st

from config.settings import CONFIG
from src.analytics.metrics import compute_performance_summary
from src.analytics.portfolio import (
    calculate_correlation_matrix,
    calculate_normalized_returns,
    fetch_multiple_stocks,
)
from src.backtesting.strategy import (
    backtest_macd_crossover,
    backtest_rsi_mean_reversion,
    backtest_sma_crossover,
)
from src.data.loader import fetch_stock_data, fetch_ticker_info, get_curated_tickers
from src.indicators.technical import add_all_indicators
from src.models import (
    ExponentialSmoothingForecaster,
    MLForecaster,
    MonteCarloGBM,
)
from src.utils.helpers import (
    apply_custom_styles,
    format_currency,
    format_large_number,
    format_percentage,
)
from ui.components.backtest_charts import (
    create_correlation_heatmap,
    create_equity_curve_chart,
    create_multi_stock_comparison_chart,
    create_signals_price_chart,
)
from ui.components.charts import (
    create_candlestick_chart,
    create_cumulative_returns_chart,
    create_drawdown_chart,
    create_macd_chart,
    create_rsi_chart,
)
from ui.components.forecast_charts import (
    create_forecast_chart,
    create_monte_carlo_chart,
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Price & Technicals",
        "🤖 AI Forecasting",
        "⚡ Strategy Backtesting",
        "🌐 Multi-Stock Comparison",
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

    # Tab 2: AI Forecasting
    with tab2:
        st.markdown("### 🤖 Predictive AI & Time-Series Forecasting Engine")
        st.caption(
            "Dynamically trained on historical closing prices with out-of-sample uncertainty cones."
        )

        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            model_choice = st.selectbox(
                "Select Forecasting Algorithm",
                [
                    "Holt-Winters Exponential Smoothing",
                    "Feature-Engineered Ridge ML",
                    "Monte Carlo GBM (1,000 Paths)",
                ],
                index=0,
            )

            forecast_days = st.slider(
                "Forecast Horizon (Business Days)",
                min_value=7,
                max_value=90,
                value=30,
                step=1,
            )

        with col_f2:
            st.markdown("#### Model Architecture Highlights")
            if model_choice == "Holt-Winters Exponential Smoothing":
                st.info(
                    "**Holt-Winters with Damped Trend:** Decomposes level and trend while "
                    "dampening long-term drift. Produces analytical 80% & 95% confidence intervals."
                )
            elif model_choice == "Feature-Engineered Ridge ML":
                st.info(
                    "**Ridge Regression Forecaster:** Uses multi-lag features with recursive "
                    "multi-step projections and strictly isolated training-set standardization."
                )
            else:
                st.info(
                    "**Geometric Brownian Motion (GBM):** Generates 1,000 stochastic trajectories "
                    "derived from historical drift (μ) and annualized volatility (σ)."
                )

        # Data availability check for forecasting
        if len(df["Close"].dropna()) < 20:
            st.warning(
                f"⚠️ **Insufficient Data for Forecasting:** The selected timeframe only contains "
                f"**{len(df)}** trading days. AI forecasting models require at least "
                "**20 data points** to detect trends and volatility patterns.\n\n"
                "👉 **Suggested Action:** Please switch to a longer timeframe (e.g., **3 Months**, "
                "**6 Months**, **1 Year**, or **YTD**) using the sidebar."
            )
        else:
            try:
                # Run forecast
                with st.spinner("Training model and generating projections..."):
                    if model_choice == "Holt-Winters Exponential Smoothing":
                        model = ExponentialSmoothingForecaster()
                    elif model_choice == "Feature-Engineered Ridge ML":
                        model = MLForecaster()
                    else:
                        model = MonteCarloGBM(num_simulations=1000)

                    model.fit(df["Close"], test_size=0.2)
                    forecast_result = model.predict_future(steps=forecast_days)

                # In-sample validation metrics
                st.markdown("#### 🎯 In-Sample Test Evaluation Metrics")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                metrics = forecast_result.metrics
                with m_col1:
                    st.metric("Test RMSE", f"${metrics.get('RMSE', 0.0):.2f}")
                with m_col2:
                    st.metric("Test MAE", f"${metrics.get('MAE', 0.0):.2f}")
                with m_col3:
                    st.metric("Test MAPE", f"{metrics.get('MAPE (%)', 0.0):.2f}%")
                with m_col4:
                    dir_acc = metrics.get("Directional Accuracy (%)", 50.0)
                    st.metric("Directional Accuracy", f"{dir_acc:.1f}%")

                # Interactive Forecast Visualizer
                st.markdown(f"#### 📈 {forecast_result.model_name} Projections")
                fc_fig = create_forecast_chart(forecast_result, df["Close"], ticker)
                st.plotly_chart(fc_fig, use_container_width=True)

                # If Monte Carlo, render sample paths chart
                if isinstance(model, MonteCarloGBM):
                    st.markdown("#### 🎲 Simulated Stochastic Paths (50 Sample Trajectories)")
                    mc_fig = create_monte_carlo_chart(model, df["Close"], forecast_result, ticker)
                    st.plotly_chart(mc_fig, use_container_width=True)

                # Forecast Data Table
                st.markdown("#### 📋 Projected Future Prices Table")
                forecast_df = pd.DataFrame(
                    {
                        "Date": forecast_result.future_dates.strftime("%Y-%m-%d"),
                        "Forecast Price": forecast_result.forecast.values,
                        "80% Lower Bound": forecast_result.lower_bound_80.values,
                        "80% Upper Bound": forecast_result.upper_bound_80.values,
                        "95% Lower Bound": forecast_result.lower_bound_95.values,
                        "95% Upper Bound": forecast_result.upper_bound_95.values,
                    }
                )

                col_tbl1, col_tbl2 = st.columns([3, 1])
                with col_tbl1:
                    st.dataframe(forecast_df, use_container_width=True, height=260)
                with col_tbl2:
                    fc_csv = forecast_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=fc_csv,
                        file_name=f"{ticker}_forecast_{forecast_days}d.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"⚠️ **Forecasting Engine Notice:** {e}")
                st.info("💡 **Tip:** Try selecting a longer timeframe or a different algorithm.")

    # Tab 3: Strategy Backtesting
    with tab3:
        st.markdown("### ⚡ Algorithmic Trading Strategy Backtester")
        st.caption(
            "Simulate rule-based quant trading strategies vs Buy & Hold with zero lookahead bias."
        )

        strat_col1, strat_col2, strat_col3 = st.columns(3)

        with strat_col1:
            strategy_type = st.selectbox(
                "Choose Strategy",
                ["SMA Crossover", "RSI Mean-Reversion", "MACD Divergence"],
                index=0,
            )

        with strat_col2:
            capital = st.number_input(
                "Initial Portfolio Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000,
            )

        with strat_col3:
            if strategy_type == "SMA Crossover":
                fast_ma = st.slider("Fast SMA Window", 5, 50, 20)
                slow_ma = st.slider("Slow SMA Window", 20, 200, 50)
            elif strategy_type == "RSI Mean-Reversion":
                rsi_oversold = st.slider("RSI Oversold (Buy)", 10, 45, 30)
                rsi_overbought = st.slider("RSI Overbought (Sell)", 55, 90, 70)
            else:
                macd_fast = st.slider("MACD Fast Period", 5, 20, 12)
                macd_slow = st.slider("MACD Slow Period", 20, 40, 26)

        # Pre-check for SMA window vs data length
        if strategy_type == "SMA Crossover" and len(df) <= slow_ma:
            st.warning(
                f"⚠️ **Insufficient Timeframe for SMA Strategy:** Slow SMA window (**{slow_ma}**) "
                f"is greater than the available trading days (**{len(df)}**).\n\n"
                "👉 **Suggested Action:** Lower the Slow SMA window or select a longer "
                "timeframe in the sidebar."
            )
        else:
            try:
                # Run Backtest
                if strategy_type == "SMA Crossover":
                    bt_result = backtest_sma_crossover(
                        df, fast_window=fast_ma, slow_window=slow_ma, initial_capital=capital
                    )
                elif strategy_type == "RSI Mean-Reversion":
                    bt_result = backtest_rsi_mean_reversion(
                        df,
                        oversold=rsi_oversold,
                        overbought=rsi_overbought,
                        initial_capital=capital,
                    )
                else:
                    bt_result = backtest_macd_crossover(
                        df, fast=macd_fast, slow=macd_slow, initial_capital=capital
                    )

                # Backtest KPI Cards
                st.markdown("#### 🏆 Strategy vs Benchmark Performance")
                kpi_b1, kpi_b2, kpi_b3, kpi_b4, kpi_b5, kpi_b6 = st.columns(6)
                with kpi_b1:
                    diff_vs_bench = bt_result.total_return_pct - bt_result.benchmark_return_pct
                    st.metric(
                        "Strategy Return",
                        f"{bt_result.total_return_pct:+.2f}%",
                        delta=f"{diff_vs_bench:+.2f}% vs Buy&Hold",
                    )
                with kpi_b2:
                    st.metric("Buy & Hold Return", f"{bt_result.benchmark_return_pct:+.2f}%")
                with kpi_b3:
                    st.metric("Win Rate", f"{bt_result.win_rate_pct:.1f}%")
                with kpi_b4:
                    st.metric("Total Trades", str(bt_result.total_trades))
                with kpi_b5:
                    pf_str = (
                        f"{bt_result.profit_factor:.2f}"
                        if bt_result.profit_factor < 100
                        else "∞"
                    )
                    st.metric("Profit Factor", pf_str)
                with kpi_b6:
                    st.metric("Max Drawdown", f"{bt_result.max_drawdown_pct:.2f}%")

                # Equity Curve Comparison Chart
                st.markdown("#### 📈 Portfolio Equity Curve ($)")
                eq_fig = create_equity_curve_chart(bt_result, ticker)
                st.plotly_chart(eq_fig, use_container_width=True)

                # Buy / Sell Signal Markers on Price Chart
                st.markdown("#### 🎯 Execution Trade Signals on Price")
                sig_fig = create_signals_price_chart(bt_result, ticker)
                st.plotly_chart(sig_fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ **Backtest Execution Notice:** {e}")
                st.info(
                    "💡 **Tip:** Try adjusting strategy parameters or choosing a longer timeframe."
                )

    # Tab 4: Multi-Stock Comparison
    with tab4:
        st.markdown("### 🌐 Multi-Asset Growth & Correlation Matrix")
        st.caption("Compare normalized trajectories and diversification benefits across assets.")

        tickers_data = get_curated_tickers()
        all_curated_symbols = tickers_data["Symbol"].tolist()

        fallbacks = [s for s in ["SPY", "QQQ", "NVDA", "BTC-USD"] if s != ticker][:3]
        default_compare = [ticker] + fallbacks

        selected_compare = st.multiselect(
            "Select Tickers to Compare (2 to 6 assets):",
            options=all_curated_symbols,
            default=default_compare[:4],
        )

        if len(selected_compare) < 2:
            st.info("Please select at least 2 tickers to generate comparison analytics.")
        else:
            with st.spinner("Fetching and aligning multi-asset prices..."):
                multi_df = fetch_multiple_stocks(
                    selected_compare, period=period, start=start_date, end=end_date
                )

            if not multi_df.empty:
                norm_returns = calculate_normalized_returns(multi_df)
                corr_matrix = calculate_correlation_matrix(multi_df)

                col_cmp1, col_cmp2 = st.columns([3, 2])
                with col_cmp1:
                    st.markdown("#### 🚀 Normalized Relative Growth (%)")
                    cmp_fig = create_multi_stock_comparison_chart(norm_returns)
                    st.plotly_chart(cmp_fig, use_container_width=True)

                with col_cmp2:
                    st.markdown("#### 🧬 Cross-Asset Correlation Heatmap")
                    corr_fig = create_correlation_heatmap(corr_matrix)
                    st.plotly_chart(corr_fig, use_container_width=True)

                st.markdown("#### 📊 Comparative Statistics Summary")
                stats_rows = []
                for sym in multi_df.columns:
                    ret_series = multi_df[sym].pct_change().dropna()
                    tot_growth = ((multi_df[sym].iloc[-1] / multi_df[sym].iloc[0]) - 1.0) * 100.0
                    vol_ann = ret_series.std() * (252 ** 0.5) * 100.0
                    sharpe = (
                        (ret_series.mean() / ret_series.std()) * (252 ** 0.5)
                        if ret_series.std() > 0
                        else 0
                    )
                    stats_rows.append(
                        {
                            "Asset": sym,
                            "Total Return (%)": f"{tot_growth:+.2f}%",
                            "Annualized Volatility (%)": f"{vol_ann:.2f}%",
                            "Sharpe Ratio": f"{sharpe:.2f}",
                            "Latest Price": format_currency(multi_df[sym].iloc[-1]),
                        }
                    )
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # Tab 5: Risk & Performance
    with tab5:
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

    # Tab 6: Historical Data
    with tab6:
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

    # Tab 7: Profile & Fundamentals
    with tab7:
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
