# app.py
import streamlit as st

st.set_page_config(
    page_title="Market Intelligence Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Market Intelligence & Adaptive Valuation Suite")
st.markdown("""
Welcome to your **Market Intelligence System**.

**Tabs available in the sidebar:**
1. Market Data Engine — for macro, liquidity, and regime analysis  
2. Adaptive Signals & Valuation — for single-stock fair value models  
""")
st.info("Use the sidebar to navigate between modules.")