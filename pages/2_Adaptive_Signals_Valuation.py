import streamlit as st
import sys, os, io

# Ensure backend import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend import valuation_engine

st.set_page_config(page_title="Adaptive Valuation Engine", layout="wide")
st.title("💹 Adaptive Signals & Multi-Model Valuation")

st.markdown("""
Run the valuation model for one or multiple tickers to generate detailed fair-value reports,
technical signals, and ensemble summaries.
""")

single_ticker = st.text_input("Enter Single Ticker (e.g., AAPL, NVDA, MSFT)", value="AAPL")
multi_input = st.text_area("Enter Multiple Tickers (comma-separated, optional)", placeholder="AAPL, MSFT, JNJ")
run_button = st.button("🧠 Run Analysis")

if run_button:
    tickers = []
    if multi_input.strip():
        tickers = [t.strip().upper() for t in multi_input.split(",") if t.strip()]
    else:
        tickers = [single_ticker.strip().upper()]

    for tk in tickers:
        st.markdown(f"### 📊 Report for `{tk}`")
        buffer = io.StringIO()
        sys.stdout = buffer
        try:
            valuation_engine.run_once(tk)
            sys.stdout = sys.__stdout__
            report_text = buffer.getvalue()

            # Display text report
            st.text_area(f"Results for {tk}", report_text, height=500)

            # Download button for report
            report_path = f"outputs/valuation_report_{tk}.txt"
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button(f"Download {tk} Report", f, file_name=f"valuation_report_{tk}.txt")

            # === New: Display generated charts ===
            fig_dir = os.path.join("outputs", "figs", tk)
            if os.path.exists(fig_dir):
                st.markdown("#### 📈 Charts")
                # Sort for consistent order
                imgs = sorted([img for img in os.listdir(fig_dir) if img.endswith(".png")])
                # Display two per row in a clean grid
                cols = st.columns(2)
                for i, img in enumerate(imgs):
                    path = os.path.join(fig_dir, img)
                    with cols[i % 2]:
                        st.image(path, caption=img, use_container_width=True)
            else:
                st.info(f"No charts found for {tk}.")

        except Exception as e:
            sys.stdout = sys.__stdout__
            st.error(f"Error processing {tk}: {e}")

st.caption("All valuations are based on the previous market close.")