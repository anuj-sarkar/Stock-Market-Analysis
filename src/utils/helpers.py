"""Formatting and UI helper functions."""

import streamlit as st


def format_currency(value: float | None, precision: int = 2) -> str:
    """Format a numeric value as currency ($)."""
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    return f"${value:,.{precision}f}"


def format_large_number(value: float | int | None) -> str:
    """Format large numbers with B (Billion), M (Million), T (Trillion) suffixes."""
    if value is None or not isinstance(value, (int, float)) or value == 0:
        return "N/A"

    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"${value / 1e3:.2f}K"
    return f"${value:.2f}"


def format_percentage(value: float | None, precision: int = 2, include_sign: bool = True) -> str:
    """Format numeric float into a clean percentage string."""
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    sign = "+" if include_sign and value > 0 else ""
    return f"{sign}{value:.{precision}f}%"


def apply_custom_styles() -> None:
    """Inject custom modern dark trading terminal CSS styles into Streamlit."""
    custom_css = """
    <style>
        /* Main background & typography */
        .stApp {
            background-color: #0E1117;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                Roboto, Helvetica, Arial, sans-serif;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1A1F2C 0%, #151922 100%);
            border: 1px solid #2D3748;
            border-radius: 10px;
            padding: 12px 18px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: all 0.2s ease-in-out;
        }

        div[data-testid="stMetric"]:hover {
            border-color: #00E676;
            transform: translateY(-2px);
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
        }

        /* Custom Badges */
        .asset-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-sector {
            background-color: #1E293B;
            color: #38BDF8;
            border: 1px solid #0284C7;
        }
        .badge-exchange {
            background-color: #1E293B;
            color: #A855F7;
            border: 1px solid #7E22CE;
        }
        .badge-currency {
            background-color: #1E293B;
            color: #10B981;
            border: 1px solid #059669;
        }

        /* Tabs styling */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            padding: 10px 20px !important;
            border-radius: 8px 8px 0 0 !important;
        }

        /* Plotly charts container */
        .stPlotlyChart {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #1F2937;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
