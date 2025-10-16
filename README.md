# 📊 Market Intelligence App

An end-to-end **Streamlit application** that combines macroeconomic intelligence and adaptive valuation analytics.

This project includes:
- **Market Direction Engine** — interprets FRED + Yahoo Finance data to determine current market regime (bull, mid-cycle, slowdown, or bear).  
- **Adaptive Signals & Multi-Model Valuation Engine** — computes fair values, technical indicators, and multi-lens equity valuations (DDM, FCF, Reverse DCF).

---

## 🧠 Core Features

### 1. Market Direction Engine (`backend/engine.py`)
- Fetches free macro data from **FRED** and market indices from **Yahoo Finance**
- Computes key indicators:
  - Yield Curve (10Y–2Y)
  - Money Supply (M2 YoY)
  - Inflation (CPI YoY)
  - PMI, Unemployment, Sentiment, VIX, Breadth
- Produces a **composite market score** and natural-language narrative
- Outputs Excel dashboard (`market_direction_dashboard.xlsx`) and summary text

### 2. Adaptive Valuation Engine (`backend/valuation_engine.py`)
- Pulls price/dividend/FCF data from Yahoo Finance
- Calculates:
  - RSI, MACD, SMA20/50/200, ATR, volume
  - Yield-based valuation bands (Bull/Base/Bear)
  - DDM, FCF, Reverse DCF, and Ensemble fair values
- Saves text report and 5 auto-generated charts per ticker under `outputs/figs/<ticker>/`

---

## 🧩 Directory Structure

market_intelligence_app/
│
├── backend/
│   ├── engine.py                # Market Direction Engine
│   ├── valuation_engine.py      # Adaptive Valuation Engine
│   └── init.py
│
├── pages/
│   ├── 1_Market_Data_Engine.py  # Streamlit frontend for macro dashboard
│   └── 2_Adaptive_Signals_Valuation.py  # Streamlit frontend for valuation module
│
├── outputs/                     # Generated reports and charts
│   └── figs/
│
├── app.py                       # Streamlit entrypoint
├── requirements.txt
├── config.yaml
└── README.md


---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/jinenmodi810/Market_Data_Engine.git
cd Market_Data_Engine

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # (Mac/Linux)
# or venv\Scripts\activate  # (Windows)

# Install dependencies
pip install -r requirements.txt


 Outputs
	•	outputs/market_direction_dashboard.xlsx — Excel file with signal table and summary sheet
	•	outputs/SNAPSHOT.txt — quick text narrative
	•	outputs/figs/<TICKER>/ — PNG charts for each valuation run

⸻


🧑‍💻 Author

Jinen Modi
M.S. Computer Science — Illinois Institute of Technology