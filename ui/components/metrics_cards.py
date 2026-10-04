"""UI component for rendering key financial metrics, badges, and KPI cards."""

from typing import Any

import streamlit as st

from src.utils.helpers import format_currency, format_large_number, format_percentage


def render_header_profile(info: dict[str, Any]) -> None:
    """Render top header with company name, ticker symbol, and metadata tags."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">
                {info.get('name', info.get('symbol', ''))}
            </h1>
            <span style="font-size: 1.2rem; font-weight: 600; color: #94A3B8;">
                ({info.get('symbol', '')})
            </span>
        </div>
        <div style="margin-bottom: 20px;">
            <span class="asset-badge badge-sector">Sector: {info.get('sector', 'N/A')}</span>
            <span class="asset-badge badge-exchange">Exchange: {info.get('exchange', 'N/A')}</span>
            <span class="asset-badge badge-currency">Currency: {info.get('currency', 'USD')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_kpi_cards(info: dict[str, Any], df_summary: dict[str, Any]) -> None:
    """Render top 4-column KPI metric summary row."""
    current_price = info.get("current_price") or df_summary.get("Current Price", 0.0)
    prev_close = info.get("previous_close", 0.0)

    delta_val = current_price - prev_close if prev_close > 0 else 0.0
    delta_pct = (delta_val / prev_close) * 100.0 if prev_close > 0 else 0.0
    delta_str = f"{delta_val:+.2f} ({delta_pct:+.2f}%)" if prev_close > 0 else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Price",
            value=format_currency(current_price),
            delta=delta_str,
        )

    with col2:
        day_low = info.get("day_low", 0.0)
        day_high = info.get("day_high", 0.0)
        range_str = f"${day_low:.2f} - ${day_high:.2f}" if day_high > 0 else "N/A"
        st.metric(
            label="Day Range",
            value=range_str,
        )

    with col3:
        high_52 = info.get("fifty_two_week_high", 0.0)
        low_52 = info.get("fifty_two_week_low", 0.0)
        range_52_str = f"${low_52:.2f} - ${high_52:.2f}" if high_52 > 0 else "N/A"
        st.metric(
            label="52-Week Range",
            value=range_52_str,
        )

    with col4:
        st.metric(
            label="Market Capitalization",
            value=format_large_number(info.get("market_cap")),
        )


def render_risk_kpi_cards(summary: dict[str, Any]) -> None:
    """Render 4-column risk & performance summary cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cagr = summary.get("Annualized Return (CAGR %)", 0.0)
        st.metric(
            label="Annualized Return (CAGR)",
            value=format_percentage(cagr),
            delta="Positive" if cagr >= 0 else "Negative",
        )

    with col2:
        sharpe = summary.get("Sharpe Ratio", 0.0)
        st.metric(
            label="Sharpe Ratio (Rf=4%)",
            value=f"{sharpe:.2f}",
        )

    with col3:
        max_dd = summary.get("Max Drawdown (%)", 0.0)
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd:.2f}%",
            delta_color="inverse",
        )

    with col4:
        var_95 = summary.get("1-Day VaR (95%)", 0.0)
        st.metric(
            label="1-Day VaR (95%)",
            value=f"{var_95:.2f}%",
        )
