"""
5-Player Trading System Dashboard.

Main entry point for the Streamlit dashboard.
Run with: streamlit run coach_system/dashboard/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from coach_system.dashboard.theme import COACH_COLORS

st.set_page_config(
    page_title="5-Player Trading Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for light theme with better visibility
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COACH_COLORS['background']};
        color: {COACH_COLORS['text']};
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #f0f2f6;
    }}
    [data-testid="stSidebar"] .stMarkdown {{
        color: #1a1a2e;
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: #1a1a2e !important;
    }}

    /* Text */
    p, span, label, .stMarkdown {{
        color: #333333 !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: #1a1a2e !important;
        font-weight: bold;
    }}
    [data-testid="stMetricLabel"] {{
        color: #555555 !important;
    }}

    /* Cards/Expanders */
    .stExpander {{
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }}

    /* Info boxes */
    .stAlert {{
        background-color: #e7f3ff;
        color: #0c5460;
        border: 1px solid #b8daff;
    }}

    /* Success boxes */
    .stSuccess {{
        background-color: #d4edda;
        color: #155724;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: #0066cc;
        color: white;
        border: none;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: #0052a3;
    }}

    /* Tables */
    .stDataFrame {{
        background-color: #ffffff;
    }}

    /* Input widgets */
    .stSlider, .stNumberInput, .stSelectbox {{
        color: #333333;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎯 5-Player Coach")
page = st.sidebar.radio(
    "Navigate",
    ["Continuous Backtest", "Paper Trading", "Knowledge Base"],
    index=0,
)

if page == "Continuous Backtest":
    from coach_system.dashboard.pages.continuous_backtest import render_continuous_backtest
    render_continuous_backtest()

elif page == "Paper Trading":
    st.header("📈 Paper Trading Simulation")
    st.info("Paper trading simulation coming soon. Use the Continuous Backtest for now.")

elif page == "Knowledge Base":
    from coach_system.dashboard.pages.knowledge_base import render_knowledge_base
    render_knowledge_base()
