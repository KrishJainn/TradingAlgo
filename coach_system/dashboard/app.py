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

# Custom CSS
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COACH_COLORS['background']};
    }}
    .stMetric {{
        background-color: {COACH_COLORS['card_bg']};
        padding: 10px;
        border-radius: 5px;
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
    st.header("📚 Knowledge Base")
    try:
        from aqtis.knowledge.knowledge_manager import KnowledgeManager
        km = KnowledgeManager()
        stats = km.get_stats()
        st.json(stats)
    except ImportError:
        st.warning("Knowledge base module not available.")
