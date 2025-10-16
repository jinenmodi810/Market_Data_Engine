import streamlit as st
import pandas as pd
import os
import sys

# Ensure backend path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend import engine

st.set_page_config(page_title="Market Data Engine", layout="wide")
st.title("🌐 Market Data Engine — Macro Regime & Liquidity Signals")

st.markdown("""
Use this module to analyze macro indicators and derive the current market regime.
All metrics use the latest **complete data** (not intraday).
""")

start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.date_input("End Date")
run = st.button("🚀 Run Market Direction Model")

if run:
    with st.spinner("Fetching macro data and computing signals..."):
        try:
            results = engine.run_market_direction_dashboard(str(start), str(end))
            st.success("Run complete.")
            st.subheader("🧭 Market Narrative")
            st.write(results["summary"])
        except Exception as e:
            st.error(f"Error: {e}")

    excel_path = os.path.join("outputs", "market_direction_dashboard.xlsx")
    if os.path.exists(excel_path):
        with open(excel_path, "rb") as f:
            st.download_button("Download Excel Dashboard", f, file_name="market_dashboard.xlsx")

st.caption("Note: Prices and macro data reflect last available close (previous trading day).")