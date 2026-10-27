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
    """Render risk & performance KPI cards across two rows.

    Row 1: Five risk-adjusted return metrics (CAGR, Sharpe, Sortino, Calmar, Max Drawdown).
    Row 2: Two VaR estimates (Historical and Parametric/Gaussian) for comparison.
    """
    # --- Row 1: Risk-adjusted return metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)

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
            help="Excess return per unit of total volatility. >1 is good, >2 is excellent.",
        )

    with col3:
        sortino = summary.get("Sortino Ratio", 0.0)
        st.metric(
            label="Sortino Ratio (Rf=4%)",
            value=f"{sortino:.2f}",
            help="Like Sharpe, but penalizes downside volatility only. Preferred for skewed return distributions.",
        )

    with col4:
        calmar = summary.get("Calmar Ratio", 0.0)
        st.metric(
            label="Calmar Ratio",
            value=f"{calmar:.2f}",
            help="CAGR ÷ |Max Drawdown|. Measures return earned per unit of worst historical loss.",
        )

    with col5:
        max_dd = summary.get("Max Drawdown (%)", 0.0)
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd:.2f}%",
            delta_color="inverse",
            help="Largest peak-to-trough decline over the selected period.",
        )

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

    # --- Row 2: Value at Risk comparison ---
    col_v1, col_v2, col_v3 = st.columns([1, 1, 2])

    with col_v1:
        var_hist = summary.get("1-Day VaR (95%)", 0.0)
        st.metric(
            label="1-Day VaR 95% · Historical",
            value=f"{var_hist:.2f}%",
            help="Empirical 5th-percentile daily loss. Non-parametric; no distributional assumption.",
        )

    with col_v2:
        var_param = summary.get("1-Day VaR (95%, Parametric)", 0.0)
        st.metric(
            label="1-Day VaR 95% · Parametric",
            value=f"{var_param:.2f}%",
            help="Gaussian VaR: μ + z₀.₀₅·σ. Industry-standard closed-form estimate under normality.",
        )
