# ============================================================
# MARKET DIRECTION INTELLIGENCE ENGINE
# (Free FRED + Yahoo Finance backend for Streamlit frontend)
# ============================================================

import os
import datetime as dt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

# --------------- GLOBAL SETTINGS ----------------
FRED_SERIES = {
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "M2SL": "M2 Money Supply",
    "NAPM": "ISM Manufacturing PMI",
    "UNRATE": "Unemployment Rate",
    "UMCSENT": "Consumer Sentiment",
}
YF_TICKERS = ["^GSPC", "^VIX", "SPY"]
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FETCHING DATA
# ============================================================

def fetch_fred_data(start: str, end: str) -> pd.DataFrame:
    """Pull core macro indicators from FRED (free)."""
    frames = []
    for code, desc in FRED_SERIES.items():
        try:
            df = pdr.DataReader(code, "fred", start, end)
            df.rename(columns={code: desc}, inplace=True)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] FRED fetch failed for {code}: {e}")
    if not frames:
        raise RuntimeError("No FRED data fetched.")
    out = pd.concat(frames, axis=1)
    out.dropna(how="all", inplace=True)
    return out


def fetch_yf_data(start: str, end: str) -> dict:
    """Fetch market indices (SPX, VIX) from Yahoo Finance (modern API)."""
    frames = {}
    for tkr in YF_TICKERS:
        try:
            df = yf.download(tkr, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                frames[tkr] = df
        except Exception as e:
            print(f"[WARN] Yahoo fetch failed for {tkr}: {e}")
    return frames


# ============================================================
# SIGNAL COMPUTATION
# ============================================================

def compute_signals(fred: pd.DataFrame, yfdata: dict) -> dict:
    """Compute derived macro signals used in composite."""
    sig = {}

    # --- Yield curve (10Y - 2Y) ---
    if "10Y Treasury Yield" in fred.columns and "2Y Treasury Yield" in fred.columns:
        yc = fred["10Y Treasury Yield"] - fred["2Y Treasury Yield"]
        sig["YieldCurve"] = yc.iloc[-1]
        sig["YieldCurveTrend"] = np.sign(yc.iloc[-1] - yc.iloc[-20]) if len(yc) > 20 else 0

    # --- M2 YoY growth ---
    if "M2 Money Supply" in fred.columns:
        m2 = fred["M2 Money Supply"].dropna()
        if len(m2) > 12:
            sig["M2YoY"] = (m2.iloc[-1] / m2.iloc[-13]) - 1

    # --- CPI YoY inflation ---
    if "CPI (All Urban Consumers)" in fred.columns:
        cpi = fred["CPI (All Urban Consumers)"].dropna()
        if len(cpi) > 12:
            sig["CPIYoY"] = (cpi.iloc[-1] / cpi.iloc[-13]) - 1

    # --- PMI level and momentum ---
    if "ISM Manufacturing PMI" in fred.columns:
        pmi = fred["ISM Manufacturing PMI"].dropna()
        if len(pmi) > 3:
            sig["PMI"] = pmi.iloc[-1]
            sig["PMItrend"] = np.sign(pmi.iloc[-1] - pmi.iloc[-3])

    # --- Unemployment rate ---
    if "Unemployment Rate" in fred.columns:
        un = fred["Unemployment Rate"].dropna()
        sig["Unemployment"] = un.iloc[-1]

    # --- Sentiment ---
    if "Consumer Sentiment" in fred.columns:
        cs = fred["Consumer Sentiment"].dropna()
        sig["Sentiment"] = cs.iloc[-1]
        sig["SentimentTrend"] = np.sign(cs.iloc[-1] - cs.iloc[-3]) if len(cs) > 3 else 0

    # --- VIX (volatility) ---
    vix_df = yfdata.get("^VIX")
    if vix_df is not None and not vix_df.empty:
        vix = vix_df["Close"]
        sig["VIX"] = vix.iloc[-1]
        sig["VIXmean5y"] = vix.tail(252 * 5).mean()
        sig["VIXtrend"] = np.sign(vix.iloc[-1] - vix.iloc[-20]) if len(vix) > 20 else 0

    # --- Breadth (SPY > 200DMA) ---
    spy_df = yfdata.get("SPY")
    if spy_df is not None and not spy_df.empty:
        spy = spy_df["Close"]
        ma200 = spy.rolling(200).mean()
        sig["Breadth"] = 1.0 if spy.iloc[-1] > ma200.iloc[-1] else 0.0

    return sig


# ============================================================
# COMPOSITE SCORING
# ============================================================

def build_composite(sig: dict) -> tuple[float, str]:
    """Combine signals into 0–1 composite score and classify regime."""
    weights = {
        "YieldCurve": 0.15,
        "M2YoY": 0.15,
        "CPIYoY": 0.10,
        "PMI": 0.20,
        "Unemployment": 0.10,
        "Sentiment": 0.10,
        "VIXtrend": 0.10,
        "Breadth": 0.10,
    }

    norm = {}
    yc = sig.get("YieldCurve", 0)
    norm["YieldCurve"] = np.clip((yc + 1) / 2, 0, 1)
    norm["M2YoY"] = np.clip((sig.get("M2YoY", 0.05) * 10), 0, 1)
    norm["CPIYoY"] = np.clip(1 - abs(sig.get("CPIYoY", 0.02) - 0.02) * 10, 0, 1)
    norm["PMI"] = np.clip((sig.get("PMI", 50) - 45) / 10, 0, 1)
    norm["Unemployment"] = np.clip(1 - (sig.get("Unemployment", 5) / 10), 0, 1)
    norm["Sentiment"] = np.clip(sig.get("Sentiment", 70) / 100, 0, 1)
    norm["VIXtrend"] = 0 if sig.get("VIXtrend", 0) > 0 else 1
    norm["Breadth"] = sig.get("Breadth", 0)

    composite = float(np.average(list(norm.values()), weights=list(weights.values())))

    if composite >= 0.8:
        regime = "Early / Mid Bull"
    elif composite >= 0.6:
        regime = "Mid Cycle Expansion"
    elif composite >= 0.4:
        regime = "Late Cycle / Slowdown"
    else:
        regime = "Recession Risk / Bear"

    return composite, regime


# ============================================================
# INTERPRETATION TEXT
# ============================================================

def generate_narrative(sig: dict, composite: float, regime: str) -> str:
    """Generate natural-language summary based on signals."""
    parts = []
    m2 = sig.get("M2YoY", 0)
    pmi = sig.get("PMI", 50)
    cpi = sig.get("CPIYoY", 0)
    un = sig.get("Unemployment", 4)
    vix = sig.get("VIX", 20)

    if m2 > 0.05:
        parts.append("Liquidity is improving")
    elif m2 < 0:
        parts.append("Liquidity is tightening")
    else:
        parts.append("Liquidity is stable")

    if pmi >= 55:
        parts.append("growth expanding strongly")
    elif pmi >= 50:
        parts.append("growth stabilizing")
    else:
        parts.append("manufacturing softening")

    if cpi > 0.04:
        parts.append("with inflation still elevated")
    elif cpi < 0.01:
        parts.append("and inflation cooling")
    else:
        parts.append("with moderate price pressures")

    if un > 5:
        parts.append("labor market showing strain")
    else:
        parts.append("labor market steady")

    if vix > 25:
        parts.append("volatility high, risk appetite cautious")
    else:
        parts.append("volatility contained")

    summary = ", ".join(parts) + f" → overall regime: **{regime}** (composite={composite:.2f})."
    return summary


# ============================================================
# EXPORT
# ============================================================

def export_outputs(sig: dict, composite: float, regime: str, summary: str):
    """Write Excel dashboard and text snapshot."""
    out_excel = os.path.join(OUTPUT_DIR, "market_direction_dashboard.xlsx")
    out_text = os.path.join(OUTPUT_DIR, "SNAPSHOT.txt")

    df = pd.DataFrame(sig, index=[0]).T
    df.columns = ["Value"]

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Signals")
        pd.DataFrame(
            {"CompositeScore": [composite], "Regime": [regime], "Summary": [summary]}
        ).to_excel(writer, sheet_name="Summary")

    with open(out_text, "w") as f:
        f.write(f"Composite: {composite:.2f}\nRegime: {regime}\n\n{summary}")

    return out_excel, out_text


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def run_market_direction_dashboard(start="2010-01-01", end=None):
    """Top-level routine used by Streamlit frontend."""
    if end is None:
        end = dt.date.today().isoformat()

    fred = fetch_fred_data(start, end)
    yfdata = fetch_yf_data(start, end)
    sig = compute_signals(fred, yfdata)
    composite, regime = build_composite(sig)
    summary = generate_narrative(sig, composite, regime)
    export_outputs(sig, composite, regime, summary)
    print(summary)
    return {"signals": sig, "composite": composite, "regime": regime, "summary": summary}


if __name__ == "__main__":
    run_market_direction_dashboard("2015-01-01")