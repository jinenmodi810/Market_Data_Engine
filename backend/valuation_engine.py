# ============================================================
# JUPYTER NOTEBOOK / PY SCRIPT: Adaptive Signals & Multi-Model Valuation
# (Matplotlib-Optional, Text-First Fallback + Added Tests)
#
# Presentation-Style Output Edition:
# - Logic kept IDENTICAL to your original (no formula changes).
# - Improved, professional console/text formatting.
# - Auto-save reports to outputs/valuation_report_<TICKER>.txt
# - Auto-save charts (when matplotlib available) to outputs/figs/<TICKER>/
# - Streamlit-callable: run_adaptive_signals(ticker) & run_batch([...])
# - CLI prompt supports single or comma-separated multiple tickers.
# ============================================================

import os
import sys
import io
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- Optional imports (graceful fallbacks) ----------
# ---------- Optional imports (force headless backend) ----------
import warnings
try:
    import matplotlib
    matplotlib.use("Agg")  # use headless backend so Streamlit can render plots
    import matplotlib.pyplot as plt
    from IPython.display import display  # type: ignore
    PLOT_ENABLED = True
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    matplotlib.rcParams['figure.max_open_warning'] = 0
except Exception as e:
    print(f"[WARN] Matplotlib unavailable: {e}")
    PLOT_ENABLED = False
    def display(obj):  # minimal shim
        print("[display]", type(obj).__name__)
    class _DummyFig: pass
    class _DummyAx: pass

# -------------------- USER SETTINGS --------------------
TICKER = "OKE"                # default when run directly
TECH_LOOKBACK_DAYS = 540      # technical lookback for SMAs, RSI, MACD, etc.
VAL_LOOKBACK_YEARS = "5y"     # "5y", "10y", or "max"

# Regime probabilities: dynamic by default
USE_DYNAMIC_PROBS = True
MANUAL_PROBS = {"Bull": 0.30, "Base": 0.50, "Bear": 0.20}  # when USE_DYNAMIC_PROBS=False

# Trading risk controls
ATR_MULT = 2.0                 # ATR multiple for stop
VOL_SURGE_MULT = 1.5           # volume surge threshold vs 20D avg
FALLBACK_STOP_PCT = 0.08       # if ATR unavailable, use this % stop
VIX_HI_PERCENTILE = 80.0       # dynamic VIX risk cutoff (percentile of last 5y)

# Manually override context tickers? (Leave None to auto-pick by sector)
MANUAL_CONTEXT = None  # e.g., ["SPY","^VIX","XLK","SMH"]

# Cost of equity fallback (if CAPM unavailable). Used by DDM & reverse DCF.
DEFAULT_COST_OF_EQUITY = 0.09   # 9%

# Price reversion fraction toward fair value over 12 months (for expected return)
PRICE_REVERSION_FRAC = 0.7

# Smoke tests (no plots) — set True to run a quick check for multiple tickers
RUN_SMOKE_TESTS = False
SMOKE_TEST_TICKERS = ["OKE", "AAPL", "KO", "GS", "WMT", "HWM", "SYF", "FI", "KMI", "WAB", "CINF", "IRM", "NRG", "VST", "KKR", "FICO", "NVDA"]

# IO settings
OUTPUT_DIR = "outputs"
FIGS_DIR = os.path.join(OUTPUT_DIR, "figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# -------------------- HELPERS: Indicators --------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = (df['High'] - df['Low']).abs()
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -------------------- HELPERS: Data & Context --------------------
SECTOR_CONTEXT = {
    "Energy":              ["XLE", "CL=F", "NG=F"],
    "Materials":           ["XLB", "HG=F", "GC=F"],
    "Industrials":         ["XLI", "IYT"],
    "Consumer Cyclical":   ["XLY", "XRT"],
    "Consumer Discretionary":["XLY", "XRT"],
    "Consumer Defensive":  ["XLP"],
    "Consumer Staples":    ["XLP"],
    "Health Care":         ["XLV", "IBB"],
    "Financial Services":  ["XLF", "KRE"],
    "Real Estate":         ["XLRE", "VNQ"],
    "Technology":          ["XLK", "SMH"],
    "Communication Services":["XLC"],
    "Utilities":           ["XLU"],
    "Basic Materials":     ["XLB"],
}

SECTOR_DEFAULT_PCTL = {
    # (bull, base, bear)
    "Energy":              (0.20, 0.50, 0.80),
    "Materials":           (0.20, 0.50, 0.80),
    "Industrials":         (0.20, 0.50, 0.80),
    "Consumer Cyclical":   (0.10, 0.50, 0.90),
    "Consumer Discretionary":(0.10, 0.50, 0.90),
    "Consumer Defensive":  (0.20, 0.50, 0.80),
    "Consumer Staples":    (0.20, 0.50, 0.80),
    "Health Care":         (0.10, 0.50, 0.90),
    "Financial Services":  (0.20, 0.50, 0.80),
    "Real Estate":         (0.25, 0.55, 0.80),
    "Technology":          (0.10, 0.50, 0.90),
    "Communication Services":(0.10, 0.50, 0.90),
    "Utilities":           (0.25, 0.55, 0.80),
    "Basic Materials":     (0.20, 0.50, 0.80),
}

def safe_ticker_info(tkr: str):
    try:
        return yf.Ticker(tkr).info or {}
    except Exception:
        return {}

def suggest_context_for(tkr: str):
    base = ["SPY", "^VIX"]
    info = safe_ticker_info(tkr)
    sector = (info.get("sector") or info.get("industry") or "").strip()
    for key in SECTOR_CONTEXT:
        if key.lower() in sector.lower():
            return base + SECTOR_CONTEXT[key]
    return base + ["QQQ", "IWM"]

def fetch_history(tickers, start=None, end=None, period=None):
    frames = {}
    try:
        data = yf.download(
            tickers, start=start, end=end, period=period,
            auto_adjust=True, progress=False, group_by='ticker'
        )
        if isinstance(tickers, list):
            for t in tickers:
                try:
                    df = data[t].dropna().copy()
                    if not df.empty:
                        frames[t] = df
                except Exception:
                    pass
        else:
            df = data.dropna().copy()
            if not df.empty:
                frames[tickers] = df
    except Exception:
        pass
    return frames

# -------------------- Dividend: Proper Historical TTM --------------------

def compute_historical_ttm_dividend_series(tkr: str, history_yrs: str) -> pd.Series:
    tk = yf.Ticker(tkr)
    price = tk.history(period=history_yrs)['Close'].dropna()
    div = tk.dividends
    if div is None or div.empty:
        return pd.Series(0.0, index=price.index, name="div_ttm")
    daily = pd.Series(0.0, index=price.index)
    matching_idx = div.index.intersection(daily.index)
    if len(matching_idx) > 0:
        daily.loc[matching_idx] = div.loc[matching_idx].values
    div_ttm = daily.rolling(window=365, min_periods=1).sum()
    div_ttm.name = "div_ttm"
    return div_ttm

# -------------------- Technical Signal (with relaxed entries) --------------------

def compute_technical_signal(tkr: str, frames: dict,
                             vix_hi_cut: float, atr_mult_local: float,
                             vol_surge_mult_local: float, fallback_stop_pct_local: float):
    f = frames.get(tkr)
    if f is None or f.empty:
        raise ValueError(f"No price data for {tkr}.")
    for col in ['Open','High','Low','Close','Volume']:
        if col not in f.columns:
            raise ValueError(f"Missing column {col} in {tkr} data.")

    f = f.copy()
    f['SMA20']  = f['Close'].rolling(20).mean()
    f['SMA50']  = f['Close'].rolling(50).mean()
    f['SMA200'] = f['Close'].rolling(200).mean()
    f['RSI14']  = rsi(f['Close'], 14)
    f['MACD'], f['MACDsig'], f['MACDhist'] = macd(f['Close'])
    f['ATR14']  = atr(f, 14)
    f['Vol20']  = f['Volume'].rolling(20).mean()
    f['52w_high'] = f['Close'].rolling(252).max()
    f['52w_low']  = f['Close'].rolling(252).min()

    spy = frames.get("SPY")
    vix = frames.get("^VIX")

    last = f.iloc[-1]
    prev = f.iloc[-2] if len(f) > 1 else last

    notes, rationale, risk_filters = [], [], []

    trend_up = bool(last.Close > last.SMA50 > last.SMA200)
    trend_down = bool(last.Close < last.SMA50 < last.SMA200)
    notes.append(f"Trend: {'UP' if trend_up else ('DOWN' if trend_down else 'MIXED')} "
                 f"(Close={last.Close:.2f}, 50D={last.SMA50:.2f}, 200D={last.SMA200:.2f})")

    if pd.notna(last['52w_low']) and last['52w_low'] > 0:
        dist_low = (last.Close - last['52w_low']) / last['52w_low'] * 100
        if dist_low <= 2.0:
            notes.append("Near 52-week LOW zone (<=2% from low): potential rebound setup.")
    if pd.notna(last['52w_high']) and last['52w_high'] > 0:
        dist_high = (last['52w_high'] - last.Close) / last['52w_high'] * 100
        if dist_high <= 2.0:
            notes.append("Near 52-week HIGH zone (<=2% from high): watch for exhaustion/breakout.")

    rsi_upcross = (prev.RSI14 < 30) and (last.RSI14 >= 30)
    macd_up = (prev.MACD - prev.MACDsig) < 0 and (last.MACD - last.MACDsig) > 0
    macd_down = (prev.MACD - prev.MACDsig) > 0 and (last.MACD - last.MACDsig) < 0
    vol_surge = pd.notna(last.Vol20) and (last.Volume > vol_surge_mult_local * last.Vol20)

    spy_ok, vix_ok = True, True
    if spy is not None and not spy.empty:
        spy200 = spy['Close'].rolling(200).mean().iloc[-1]
        spy_ok = spy['Close'].iloc[-1] >= spy200 if pd.notna(spy200) else True
        if not spy_ok: risk_filters.append("SPY < 200-DMA (risk-off)")
    if vix is not None and not vix.empty:
        vix_ok = vix['Close'].iloc[-1] <= vix_hi_cut
        if not vix_ok: risk_filters.append(f"VIX above dynamic threshold ({vix_hi_cut:.1f})")

    # Relaxed entry: 2-of-3 confirms for partial, 3-of-3 for strength
    buy_triggers = sum([
        1 if rsi_upcross else 0,
        1 if macd_up else 0,
        1 if (last.Close > last.SMA20) else 0
    ])

    signal = "HOLD"
    if buy_triggers >= 3:
        signal = "BUY on strength"
        rationale.append("All 3 confirms: RSI upcross + MACD bull cross + close > 20-DMA.")
        if vol_surge:
            rationale.append("Volume surge vs 20D (adds conviction).")
    elif buy_triggers == 2:
        signal = "BUY (partial)"
        rationale.append("2 of 3 confirms; scaling-in allowed.")

    if trend_down and macd_down and (last.Close < last.SMA20):
        signal = "REDUCE / AVOID"
        rationale = ["Downtrend + negative momentum (below 20-DMA, MACD turning down)."]

    # Macro filter: require BOTH to block buys
    risk_hit = (not spy_ok) + (not vix_ok)
    if risk_hit >= 2 and signal.startswith("BUY"):
        signal = "HOLD (risk filter)"
        rationale.append("Both macro risk filters tripped.")

    atr_val = last.ATR14 if pd.notna(last.ATR14) else np.nan
    if pd.notna(atr_val) and atr_val > 0:
        stop = last.Close - atr_mult_local * atr_val
        risk_per_share = last.Close - stop
        notes.append(f"ATR(14)={atr_val:.2f} → ex. stop ≈ {stop:.2f} ({atr_mult_local}×ATR).")
        sizing_hint = f"Risk/Share ≈ ${risk_per_share:.2f}. Size so total $risk stays within your limit."
    else:
        stop = last.Close * (1 - fallback_stop_pct_local)
        risk_per_share = last.Close - stop
        notes.append(f"ATR unavailable → using fallback {int(fallback_stop_pct_local*100)}% stop method.")
        sizing_hint = f"Fallback stop ≈ ${stop:.2f}. Risk/Share ≈ ${risk_per_share:.2f}."

    return {
        "df": f,
        "price": float(last.Close),
        "date": f.index[-1].date(),
        "signal": signal,
        "rationale": rationale,
        "notes": notes,
        "risk_filters": risk_filters,
        "sizing_hint": sizing_hint
    }

# -------------------- Dynamic Probabilities & VIX threshold --------------------

def compute_dynamic_probabilities(frames: dict):
    spy = frames.get("SPY")
    vix = frames.get("^VIX")
    if spy is None or spy.empty or vix is None or vix.empty:
        return {"Bull": 1/3, "Base": 1/3, "Bear": 1/3}, None

    spy200 = spy['Close'].rolling(200).mean()
    spy_up = bool(spy['Close'].iloc[-1] >= spy200.iloc[-1])

    v = vix['Close'].dropna()
    if len(v) > 252*5:
        v = v.tail(252*5)
    vix_pctl = (v.rank(pct=True).iloc[-1]) * 100.0

    if spy_up and vix_pctl <= 40:
        probs = {"Bull": 0.50, "Base": 0.40, "Bear": 0.10}
    elif spy_up and vix_pctl <= 60:
        probs = {"Bull": 0.40, "Base": 0.45, "Bear": 0.15}
    elif (not spy_up) and vix_pctl >= 80:
        probs = {"Bull": 0.10, "Base": 0.30, "Bear": 0.60}
    elif (not spy_up) or vix_pctl >= 60:
        probs = {"Bull": 0.20, "Base": 0.40, "Bear": 0.40}
    else:
        probs = {"Bull": 0.30, "Base": 0.50, "Bear": 0.20}
    meta = {"spy_up": spy_up, "vix_percentile": vix_pctl}
    return probs, meta

def dynamic_vix_threshold(frames: dict, hi_percentile: float = VIX_HI_PERCENTILE) -> float:
    vix = frames.get("^VIX")
    if vix is None or vix.empty:
        return 25.0
    v = vix['Close'].dropna()
    if len(v) > 252*5:
        v = v.tail(252*5)
    return float(np.percentile(v, hi_percentile))

# -------------------- Valuation: Yield Bands (fixed) --------------------

def get_sector_and_defaults(tkr: str):
    info = safe_ticker_info(tkr)
    sector = info.get('sector') or info.get('industry') or 'Unknown'
    sector = str(sector)
    if sector in SECTOR_DEFAULT_PCTL:
        p_bull, p_base, p_bear = SECTOR_DEFAULT_PCTL[sector]
    else:
        p_bull, p_base, p_bear = (0.20, 0.50, 0.80)
    return sector, p_bull, p_base, p_bear

def compute_valuation_yield_bands(tkr: str, history_yrs: str,
                                  probs: dict | None,
                                  pctls: tuple[float,float,float] | None = None):
    tk = yf.Ticker(tkr)
    px = tk.history(period="1d")["Close"].iloc[-1]
    price = float(px)

    div_ttm_series = compute_historical_ttm_dividend_series(tkr, history_yrs)
    annual_div_now = float(div_ttm_series.iloc[-1])

    hist_price = tk.history(period=history_yrs)["Close"].dropna()
    common_idx = hist_price.index.intersection(div_ttm_series.index)
    hist_price = hist_price.loc[common_idx]
    div_ttm_series = div_ttm_series.loc[common_idx]

    yld_series = (div_ttm_series / hist_price).replace([np.inf, -np.inf], np.nan).dropna()

    if annual_div_now <= 0 or len(yld_series) < 30:
        return {
            "price": price,
            "annual_div": float(annual_div_now),
            "bands": None,
            "scenarios": [],
            "expected_total_return": None,
            "current_yield": float(0.0),
            "yield_series": yld_series,
        }

    if pctls is None:
        sector, pb, pm, pr = get_sector_and_defaults(tkr)
    else:
        pb, pm, pr = pctls

    bull_y = float(np.percentile(yld_series, pb*100))
    base_y = float(np.percentile(yld_series, pm*100))
    bear_y = float(np.percentile(yld_series, pr*100))
    bands = {"Bull": bull_y, "Base": base_y, "Bear": bear_y}

    def fair_price(y):
        return annual_div_now / y if y > 0 else np.nan

    rows = []
    for name, y in bands.items():
        fp = fair_price(y)
        price_ret = (fp / price) - 1
        div_yld_now = annual_div_now / price
        total_ret = price_ret + div_yld_now
        rows.append((name, y, fp, price_ret, total_ret))

    if probs is None:
        probs = {"Bull": 1/3, "Base": 1/3, "Bear": 1/3}
    ev = sum(r[4] * probs.get(r[0], 0) for r in rows)

    return {
        "price": price,
        "annual_div": float(annual_div_now),
        "bands": bands,
        "scenarios": rows,
        "expected_total_return": float(ev),
        "current_yield": float(annual_div_now / price),
        "yield_series": yld_series,
    }

# -------------------- Additional Valuation Lenses --------------------

# (1) Dividend Discount Model (Gordon) with data-driven g and fallback k
def dividend_cagr_5y(div_series: pd.Series) -> float:
    if div_series is None or div_series.empty:
        return 0.0
    yearly = div_series.resample('A').sum().dropna()
    if len(yearly) < 6:
        return 0.0
    start = yearly.iloc[-6]
    end = yearly.iloc[-1]
    if start <= 0 or end <= 0:
        return 0.0
    years = 5
    return (end / start) ** (1/years) - 1

def fair_value_ddm(tkr: str, history_yrs: str, k: float | None = None):
    tk = yf.Ticker(tkr)
    _ = float(tk.history(period="1d")["Close"].iloc[-1])  # (price fetched but unused here)
    div_series_daily = compute_historical_ttm_dividend_series(tkr, history_yrs)
    D0 = float(div_series_daily.iloc[-1])
    if D0 <= 0:
        return np.nan, {"reason": "no dividend"}
    raw_cagr = dividend_cagr_5y(tk.dividends)
    g = float(np.clip(raw_cagr, -0.05, 0.06))  # cap between -5% and +6%
    if k is None:
        k = DEFAULT_COST_OF_EQUITY
    k = float(max(k, 0.06))
    if k - g <= 0.005:
        g = k - 0.005
    D1 = D0 * (1 + g)
    fair = D1 / (k - g)
    return float(fair), {"k": k, "g": g, "D1": D1, "note": "Gordon"}

# (2) Shareholder Yield (dividend + net buyback)
def estimate_shares_outstanding(tk: yf.Ticker, hist_years: str) -> pd.Series:
    try:
        sh = tk.get_shares_full(start=None, end=None)
        if sh is not None and not sh.empty:
            sh = sh.dropna()
            return sh
    except Exception:
        pass
    so = tk.info.get('sharesOutstanding', None)
    px = tk.history(period=hist_years)['Close']
    if so is None:
        return pd.Series(index=px.index, dtype=float)
    return pd.Series(float(so), index=px.index)

def current_shareholder_yield(tkr: str, hist_years: str):
    tk = yf.Ticker(tkr)
    price = float(tk.history(period="1d")["Close"].iloc[-1])
    div_ttm = float(compute_historical_ttm_dividend_series(tkr, hist_years).iloc[-1])
    div_yield = div_ttm / price if price > 0 else 0.0
    shares = estimate_shares_outstanding(tk, hist_years)
    if not shares.empty:
        shares_m = shares.resample('M').last().dropna()
        if len(shares_m) >= 13:
            buyback_rate = (shares_m.iloc[-13] - shares_m.iloc[-1]) / shares_m.iloc[-13]
            sh_yield = div_yield + max(buyback_rate, -0.3)
        else:
            sh_yield = div_yield
    else:
        sh_yield = div_yield
    return float(sh_yield), {"div_yield": div_yield}

# (3) FCF-Yield Bands
def get_ttm_fcf_and_history(tkr: str):
    tk = yf.Ticker(tkr)
    try:
        qcf = tk.quarterly_cashflow
        ocf = qcf.loc['Total Cash From Operating Activities'] if 'Total Cash From Operating Activities' in qcf.index else qcf.loc['Operating Cash Flow']
        capex = qcf.loc['Capital Expenditures']
        fcf_ttm = float((ocf.iloc[:4] - capex.iloc[:4]).sum())
    except Exception:
        fcf_ttm = np.nan
    try:
        acf = tk.cashflow
        ocf_a = acf.loc['Total Cash From Operating Activities'] if 'Total Cash From Operating Activities' in acf.index else acf.loc['Operating Cash Flow']
        capex_a = acf.loc['Capital Expenditures']
        fcf_hist = (ocf_a - capex_a).dropna()
    except Exception:
        fcf_hist = pd.Series(dtype=float)
    return fcf_ttm, fcf_hist

def fair_value_fcf_yield_bands(tkr: str, history_yrs: str,
                               pctls: tuple[float,float,float] | None = None):
    tk = yf.Ticker(tkr)
    info = tk.info or {}
    sector = info.get('sector') or info.get('industry') or 'Unknown'
    if pctls is None:
        if sector in ('Utilities','Real Estate','Communication Services','Energy'):
            pctls = (0.25, 0.55, 0.80)
        else:
            pctls = (0.20, 0.50, 0.80)
    pb, pm, pr = pctls
    price = float(tk.history(period="1d")["Close"].iloc[-1])
    market_cap = tk.info.get('marketCap', None)
    shares_out = tk.info.get('sharesOutstanding', None)
    fcf_ttm, fcf_hist = get_ttm_fcf_and_history(tkr)
    if (market_cap is None or market_cap <= 0) and shares_out is not None:
        market_cap = price * float(shares_out)
    if market_cap is None or market_cap <= 0 or np.isnan(fcf_ttm):
        return np.nan, {"reason": "insufficient FCF/market cap data"}
    ylds = []
    if shares_out is None:
        shares_out = (market_cap / price) if price > 0 else np.nan
    for dt_col, fcf_val in fcf_hist.items():
        try:
            dt_col = pd.to_datetime(dt_col)
            px_hist = tk.history(start=str(dt_col.year-1), end=str(dt_col.year+1))['Close']
            if px_hist.empty or np.isnan(shares_out):
                continue
            mktcap_hist = float(shares_out) * float(px_hist.asfreq('M').last().dropna().iloc[-1])
            if mktcap_hist > 0:
                ylds.append(float(fcf_val) / mktcap_hist)
        except Exception:
            continue
    if len(ylds) >= 4:
        ylds = pd.Series(ylds).clip(lower=-1, upper=1).dropna()
        bull_y, base_y, bear_y = np.percentile(ylds, [pb*100, pm*100, pr*100])
        fair_bull = (fcf_ttm / bull_y) if bull_y > 0 else np.nan
        fair_base = (fcf_ttm / base_y) if base_y > 0 else np.nan
        fair_bear = (fcf_ttm / bear_y) if bear_y > 0 else np.nan
        if shares_out and shares_out > 0:
            fair_prices = [fair_bull/shares_out, fair_base/shares_out, fair_bear/shares_out]
            fair = float(np.nanmedian(fair_prices))
        else:
            fair = np.nan
        return fair, {"fcf_ttm": fcf_ttm, "ylds_n": int(len(ylds))}
    else:
        return np.nan, {"reason": "insufficient FCF history"}

# (4) Reverse DCF (implied growth) — simple one-stage over 10y + terminal
def reverse_dcf_implied_g(tkr: str, k: float | None = None):
    tk = yf.Ticker(tkr)
    px = float(tk.history(period="1d")['Close'].iloc[-1])
    shares = tk.info.get('sharesOutstanding', None)
    if shares is None or shares <= 0:
        return np.nan, {"reason": "missing sharesOutstanding"}
    fcf_ttm, _ = get_ttm_fcf_and_history(tkr)
    if np.isnan(fcf_ttm) or fcf_ttm <= 0:
        return np.nan, {"reason": "non-positive FCF"}
    mktcap = px * float(shares)
    if k is None:
        k = DEFAULT_COST_OF_EQUITY

    def pv_given_g(g):
        fcf = fcf_ttm
        pv = 0.0
        for t in range(1, 11):
            fcf *= (1 + g)
            pv += fcf / ((1 + k) ** t)
        g_term = min(0.02, max(-0.02, g))
        tv = fcf * (1 + g_term) / (k - g_term)
        pv += tv / ((1 + k) ** 10)
        return pv

    lo, hi = -0.10, 0.15
    for _ in range(60):
        mid = (lo + hi) / 2
        pv = pv_given_g(mid)
        if pv > mktcap:
            lo = mid
        else:
            hi = mid
    implied_g = (lo + hi) / 2

    g_star = min(max(implied_g, -0.02), 0.08)
    fcf = fcf_ttm
    pv2 = 0.0
    for t in range(1, 11):
        fcf *= (1 + g_star)
        pv2 += fcf / ((1 + k) ** t)
    tv = fcf * (1 + 0.02) / (k - 0.02)
    pv2 += tv / ((1 + k) ** 10)
    fair_mcap = pv2
    fair_price = fair_mcap / shares
    return float(fair_price), {"implied_g": float(implied_g), "g_star": float(g_star), "k": float(k)}

# -------------------- Ensemble & Expected Return --------------------
def winsorize(arr, p=10):
    lo, hi = np.nanpercentile(arr, [p, 100-p])
    return np.clip(arr, lo, hi)

def combine_fair_values(fairs: list[float]) -> float:
    arr = np.array([x for x in fairs if x is not None and not np.isnan(x)])
    if arr.size == 0:
        return np.nan
    arr = winsorize(arr, 10)
    return float(np.nanmean(arr))

# -------------------- Formatting Helpers (Presentation) --------------------
def _line(width=70, char="─"):
    return char * width

def _title(txt, width=70):
    pad = max(0, (width - len(txt)) // 2)
    return f"{' ' * pad}{txt}"

def _section(title):
    width = 70
    return "\n".join([
        "═" * width,
        _title(title.upper(), width),
        "═" * width
    ])

def _subsection(title):
    width = 70
    return "\n".join([
        _line(width, "─"),
        _title(title, width),
        _line(width, "─")
    ])

def _fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:.{digits}f}%"

def _fmt_num(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.{digits}f}"

# -------------------- Console Report (Presentation Style) --------------------
def build_console_report(tkr: str, tech: dict, val_yield: dict,
                         fair_ddm: float, fair_fcf: float, fair_rdcf: float,
                         ensemble_fair: float,
                         final_probs: dict,
                         sh_yield_now: float,
                         yld_sample_n: int | None,
                         yld_current_pctl: float | None) -> str:
    date_str = tech['date']
    px = tech['price']
    # Header
    lines = []
    lines.append(_section("Adaptive Signals & Multi-Model Valuation"))
    lines.append(f"Ticker: {tkr}    Date: {date_str}    Last Price: ${px:,.2f}")
    lines.append(_subsection("Technical Signals"))
    lines.append(f"Signal: {tech['signal']}")
    if tech['rationale']:
        lines.append("Rationale:")
        for r in tech['rationale']:
            lines.append(f"  • {r}")
    if tech['risk_filters']:
        lines.append("Risk Filters:")
        for rf in tech['risk_filters']:
            lines.append(f"  • {rf}")
    lines.append("Notes:")
    for n in tech["notes"]:
        lines.append(f"  • {n}")
    lines.append(f"Position Sizing: {tech['sizing_hint']}")

    # Yield-based valuation
    lines.append(_subsection("Yield-Based Valuation"))
    if val_yield["annual_div"] <= 0 or not val_yield["scenarios"]:
        lines.append("No trailing dividend detected. Yield valuation not applicable.")
    else:
        lines.append(f"Annual Dividend (TTM): ${val_yield['annual_div']:.2f} | "
                     f"Current Yield: {_fmt_pct(val_yield['current_yield'])}")
        order = {"Bull":0,"Base":1,"Bear":2}
        lines.append("Scenarios (Band Yield → Fair, Price Return, Total Return):")
        for name, yld, fp, pr, tr in sorted(val_yield["scenarios"], key=lambda x: order.get(x[0], 99)):
            lines.append(f"  {name:<5} | {yld*100:5.2f}% → ${fp:8.2f}, {pr*100:6.1f}%, {tr*100:6.1f}%")
        if val_yield["expected_total_return"] is not None:
            lines.append(f"Probability-Weighted 12-mo Total Return: {_fmt_pct(val_yield['expected_total_return'])}")
        b = val_yield["bands"]
        if b:
            lines.append(f"Bands (Bull / Base / Bear): {b['Bull']*100:.2f}% / {b['Base']*100:.2f}% / {b['Bear']*100:.2f}%")
    if yld_sample_n is not None and yld_current_pctl is not None:
        lines.append(f"Yield Sample Size: {yld_sample_n} | "
                     f"Current-Yield Percentile: {yld_current_pctl:4.1f}th")

    # Other lenses
    lines.append(_subsection("Other Valuation Lenses"))
    lines.append(f"DDM Fair:     ${_fmt_num(fair_ddm)}")
    lines.append(f"FCF Fair:     ${_fmt_num(fair_fcf)}")
    lines.append(f"RevDCF Fair:  ${_fmt_num(fair_rdcf)}")

    # Ensemble + expected return
    lines.append(_subsection("Ensemble View"))
    lines.append(f"Ensemble Fair (winsorized mean): ${_fmt_num(ensemble_fair)}")
    lines.append(f"Shareholder Yield (approx): {_fmt_pct(sh_yield_now)}")
    if not np.isnan(ensemble_fair):
        price_move = (ensemble_fair/px - 1.0) * PRICE_REVERSION_FRAC
        exp_12m_total = price_move + sh_yield_now
        lines.append(f"Expected 12-month total return (ensemble): {_fmt_pct(exp_12m_total)}")

    # Regime probabilities
    lines.append(_subsection("Regime Probabilities"))
    lines.append(f"Bull / Base / Bear: {final_probs['Bull']:.2f} / {final_probs['Base']:.2f} / {final_probs['Bear']:.2f}")

    # Footer
    lines.append(_line(70, "═"))
    return "\n".join(lines)

# -------------------- Chart helpers (save + show) --------------------
def _ensure_ticker_dir(tkr: str):
    td = os.path.join(FIGS_DIR, tkr.upper())
    os.makedirs(td, exist_ok=True)
    return td

def _save_fig(fig, path):
    if not PLOT_ENABLED:
        return
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception:
        pass

def show_and_close(fig):
    if not PLOT_ENABLED:
        return
    display(fig)
    try:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    except Exception:
        pass

def fig_price_sma(df: pd.DataFrame, tkr: str):
    if not PLOT_ENABLED:
        return _DummyFig()
    fig, ax = plt.subplots()
    df[['Close','SMA20','SMA50','SMA200']].plot(ax=ax)
    ax.set_title(f"{tkr} — Price with SMA20/50/200")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    return fig

def fig_rsi(df: pd.DataFrame, tkr: str):
    if not PLOT_ENABLED:
        return _DummyFig()
    fig, ax = plt.subplots()
    df['RSI14'].plot(ax=ax)
    ax.axhline(30); ax.axhline(70)
    ax.set_title(f"{tkr} — RSI(14)")
    ax.set_xlabel("Date"); ax.set_ylabel("RSI")
    ax.grid(True, alpha=0.3)
    return fig

def fig_macd(df: pd.DataFrame, tkr: str):
    if not PLOT_ENABLED:
        return _DummyFig()
    fig, ax = plt.subplots()
    df['MACD'].plot(ax=ax, label="MACD")
    df['MACDsig'].plot(ax=ax, label="Signal")
    ax.bar(df.index, df['MACDhist'], width=1.0, alpha=0.3)
    ax.set_title(f"{tkr} — MACD")
    ax.set_xlabel("Date"); ax.set_ylabel("MACD")
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig

def fig_volume(df: pd.DataFrame, tkr: str):
    if not PLOT_ENABLED:
        return _DummyFig()
    fig, ax = plt.subplots()
    ax.bar(df.index, df['Volume'])
    df['Vol20'].plot(ax=ax)
    ax.set_title(f"{tkr} — Volume & 20-day Avg")
    ax.set_xlabel("Date"); ax.set_ylabel("Shares")
    ax.grid(True, alpha=0.3)
    return fig

def fig_valuation_curve(val: dict, tkr: str):
    if not PLOT_ENABLED:
        return _DummyFig()
    scenarios = sorted(val["scenarios"], key=lambda x: x[1])
    yields = [s[1]*100 for s in scenarios]
    prices = [s[2] for s in scenarios]
    labels = [s[0] for s in scenarios]
    fig, ax = plt.subplots()
    ax.plot(yields, prices, marker="o")
    for i, label in enumerate(labels):
        ax.text(yields[i] + 0.05, prices[i], label)
    ax.axhline(val["price"])
    ax.set_title(f"{tkr} — Yield-Based Valuation (Fair Price vs Target Yield)")
    ax.set_xlabel("Target Yield (%)"); ax.set_ylabel("Implied Fair Price ($)")
    ax.grid(True, alpha=0.3)
    return fig

# -------------------- Report Saving --------------------
def save_report(ticker: str, text: str) -> str:
    safe_t = ticker.upper().replace("/", "_")
    path = os.path.join(OUTPUT_DIR, f"valuation_report_{safe_t}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

# -------------------- RUN (single ticker) --------------------
def run_once(ticker: str,
             tech_lookback_days: int = TECH_LOOKBACK_DAYS,
             val_lookback_years: str = VAL_LOOKBACK_YEARS,
             use_dynamic_probs: bool = USE_DYNAMIC_PROBS,
             manual_probs: dict | None = MANUAL_PROBS,
             save_charts: bool = True) -> dict:
    # 1) Build dynamic context
    if MANUAL_CONTEXT is not None:
        context = ["SPY", "^VIX"] + list(MANUAL_CONTEXT)
    else:
        context = suggest_context_for(ticker)
    print("Context tickers:", context)

    # 2) Fetch data
    # Adjust end date to ensure we get the last complete close (yesterday if today is selected)
    import datetime as dt
    import yfinance as yf

    today = dt.date.today()
    # one day before ensures we don't include today's partial candle
    end = today - dt.timedelta(days=1)
    start = end - dt.timedelta(days=tech_lookback_days)

    # yfinance end is exclusive, so add +1 day to include the last full bar
    frames = fetch_history(
        [ticker] + context,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat()
    )

# Verify we have at least one data point
    if ticker not in frames or frames[ticker].empty:
        raise SystemExit(f"No valid historical data for {ticker} up to {end}.")
    if ticker not in frames:
        raise SystemExit(f"Could not load price data for {ticker}.")

    # 3) Compute dynamic VIX threshold & technicals
    vix_threshold = dynamic_vix_threshold(frames, hi_percentile=VIX_HI_PERCENTILE)
    tech = compute_technical_signal(
        ticker, frames,
        vix_hi_cut=vix_threshold,
        atr_mult_local=ATR_MULT,
        vol_surge_mult_local=VOL_SURGE_MULT,
        fallback_stop_pct_local=FALLBACK_STOP_PCT
    )

    # 4) Dynamic probabilities (or manual)
    dyn_probs, regime_meta = compute_dynamic_probabilities(frames)
    final_probs = dyn_probs if use_dynamic_probs else (manual_probs or {"Bull":1/3,"Base":1/3,"Bear":1/3})

    # 5) Valuation — Yield Bands (fixed) with sector defaults
    val_yield = compute_valuation_yield_bands(
        ticker,
        history_yrs=val_lookback_years,
        probs=final_probs,
        pctls=None
    )

    # Diagnostics for yield series
    yld_series = val_yield.get('yield_series', pd.Series(dtype=float))
    yld_sample_n = int(len(yld_series)) if yld_series is not None else None
    yld_current_pctl = None
    if yld_sample_n and yld_sample_n > 0 and val_yield['current_yield'] is not None:
        yld_current_pctl = float(((yld_series <= val_yield['current_yield']).mean())*100)

    # 6) Other valuation lenses
    ddm_fair, _meta_ddm = fair_value_ddm(ticker, val_lookback_years, k=DEFAULT_COST_OF_EQUITY)
    fcf_fair, _meta_fcf = fair_value_fcf_yield_bands(ticker, val_lookback_years)
    rdcf_fair, _meta_rdcf = reverse_dcf_implied_g(ticker, k=DEFAULT_COST_OF_EQUITY)

    ensemble_fair = combine_fair_values([
        (val_yield['scenarios'][1][2] if val_yield.get('scenarios') else np.nan),
        ddm_fair,
        fcf_fair,
        rdcf_fair,
    ])

    # 7) Shareholder yield (carry)
    sh_yield_now, _sh_meta = current_shareholder_yield(ticker, val_lookback_years)

    # 8) Build presentation report
    report_text = build_console_report(
        ticker, tech, val_yield,
        fair_ddm=ddm_fair,
        fair_fcf=fcf_fair,
        fair_rdcf=rdcf_fair,
        ensemble_fair=ensemble_fair,
        final_probs=final_probs,
        sh_yield_now=sh_yield_now,
        yld_sample_n=yld_sample_n,
        yld_current_pctl=yld_current_pctl
    )

    # Print to stdout
    print(report_text)

    # Save report
    report_path = save_report(ticker, report_text)

    # 9) Charts: show and save
    fig_paths = []
    if PLOT_ENABLED:
        df_plot = tech["df"].copy()
        tdir = _ensure_ticker_dir(ticker)
        try:
            f1 = fig_price_sma(df_plot, ticker); _save_fig(f1, os.path.join(tdir, "price_sma.png")); show_and_close(f1); fig_paths.append(os.path.join(tdir, "price_sma.png"))
            f2 = fig_rsi(df_plot, ticker); _save_fig(f2, os.path.join(tdir, "rsi.png")); show_and_close(f2); fig_paths.append(os.path.join(tdir, "rsi.png"))
            f3 = fig_macd(df_plot, ticker); _save_fig(f3, os.path.join(tdir, "macd.png")); show_and_close(f3); fig_paths.append(os.path.join(tdir, "macd.png"))
            f4 = fig_volume(df_plot, ticker); _save_fig(f4, os.path.join(tdir, "volume.png")); show_and_close(f4); fig_paths.append(os.path.join(tdir, "volume.png"))
            if val_yield["annual_div"] > 0 and val_yield["scenarios"]:
                f5 = fig_valuation_curve(val_yield, ticker); _save_fig(f5, os.path.join(tdir, "valuation_curve.png")); show_and_close(f5); fig_paths.append(os.path.join(tdir, "valuation_curve.png"))
            else:
                print("No dividend detected → valuation curve not applicable.")
        except Exception as _e:
            print(f"[WARN] Chart save/display issue: {_e}")
    else:
        print("[Info] Plotting disabled (matplotlib not available). Text report generated above.")

    return {
        "ticker": ticker.upper(),
        "date": str(tech["date"]),
        "price": float(tech["price"]),
        "report_text": report_text,
        "report_path": report_path,
        "fig_paths": fig_paths,
        "final_probs": final_probs
    }

# -------------------- Streamlit Helpers --------------------
def run_adaptive_signals(ticker: str) -> dict:
    """
    Streamlit entry: run a single ticker and return report + file paths.
    """
    return run_once(ticker)

def run_batch(tickers: list[str]) -> dict:
    """
    Run multiple tickers; returns dict keyed by ticker with run_once outputs.
    """
    results = {}
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue
        try:
            results[tk] = run_once(tk)
        except Exception as e:
            print(f"[ERROR] {tk}: {e}")
    return results

# -------------------- MAIN & PROMPT --------------------
if __name__ == "__main__":
    if RUN_SMOKE_TESTS:
        print("=== Running smoke tests (no plots) ===")
        for tk in SMOKE_TEST_TICKERS:
            try:
                run_once(tk)
            except SystemExit as e:
                print(f"[WARN] {tk}: {e}")
            except Exception as e:
                print(f"[ERROR] {tk}: {e}")
        print("=== Smoke tests finished ===")
        sys.exit(0)

    # Interactive prompt: single or multiple tickers (comma-separated)
    try:
        user_in = input("Enter ticker(s) (e.g., AAPL or AAPL, MSFT, NVDA): ").strip()
    except EOFError:
        user_in = TICKER
    if not user_in:
        user_in = TICKER

    if "," in user_in:
        tickers = [t.strip().upper() for t in user_in.split(",") if t.strip()]
        res = run_batch(tickers)
        print("\n=== Batch complete ===")
        for tk, out in res.items():
            print(f"{tk}: report → {out.get('report_path')}")
    else:
        out = run_once(user_in.strip().upper())
        print("\nReport saved to:", out.get("report_path"))
        if out.get("fig_paths"):
            print("Charts saved:")
            for p in out["fig_paths"]:
                print(" -", p)