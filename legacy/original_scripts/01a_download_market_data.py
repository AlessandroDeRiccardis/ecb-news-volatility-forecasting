"""
Market Data Download Script
============================
Downloads all financial data needed for the NA-GARCH + H_t sentiment paper.

Data collected:
  1. Euro Stoxx 50 daily prices (2006-2023)         — primary target variable
  2. VSTOXX daily (2006-2023)                        — robustness check
  3. ECB AAA 10Y bond yield (2006-2023)              — macro control variable
  4. Rolling volatility: 10d, 20d (base), 60d        — H_t robustness checks

Run:
    pip install yfinance pandas_datareader pandas requests
    python3 download_market_data.py

Outputs (all in ./market_data/):
    eurostoxx50_prices.csv          — raw OHLCV prices
    eurostoxx50_returns.csv         — daily log returns + squared returns
    rolling_vol_and_ht.csv          — rolling vol (10d/20d/60d) + H_t variants
    vstoxx_daily.csv                — VSTOXX implied vol (robustness)
    ecb_aaa_10y_yield.csv           — Euro area AAA 10Y bond yield (macro control)
    market_data_summary.txt         — data quality report
"""

import os
import re
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

START_DATE     = "2007-03-01"   # Yahoo Finance starts Mar 2007 for ^STOXX50E
END_DATE       = "2023-12-31"
IN_SAMPLE_END  = "2018-12-31"   # long-run vol mean computed over this period only

# H_t threshold multiplier — c in H_t = 1(roll_vol(t-1) > c * long_run_mean)
# 1.5 is the base spec placeholder; c is estimated freely in the model
H_T_MULTIPLIER = 1.5

OUTPUT_DIR = Path("./market_data")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}")


def save(df, filename, description):
    path = OUTPUT_DIR / filename
    df.to_csv(path)
    log(f"  Saved {description}: {len(df)} rows → {filename}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. EURO STOXX 50 PRICES
# ─────────────────────────────────────────────────────────────────────────────

def download_eurostoxx50():
    """
    Download Euro Stoxx 50 daily prices.
    Strategy:
      1. yfinance (^STOXX50E)  — works from ~Mar 2007
      2. Stooq (^SX5E)         — fills 2006 to Mar 2007 gap, data from 1999
    """
    log("── Euro Stoxx 50 prices ──")

    df_yf = pd.DataFrame()
    try:
        import yfinance as yf
        df_yf = yf.Ticker("^STOXX50E").history(
            start=START_DATE, end=END_DATE, auto_adjust=True
        )
        df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
        df_yf = df_yf[["Open", "High", "Low", "Close", "Volume"]]
        df_yf.index.name = "date"
        log(f"  yfinance: {len(df_yf)} rows | {df_yf.index.min().date()} to {df_yf.index.max().date()}")
    except Exception as e:
        log(f"  yfinance failed: {e}")

    # Fill 2006–Mar 2007 gap from Stooq (direct CSV download via requests)
    df_stooq = pd.DataFrame()
    yf_start = df_yf.index.min() if not df_yf.empty else pd.to_datetime("2007-04-01")
    if yf_start > pd.to_datetime("2006-06-01"):
        try:
            from io import StringIO
            stooq_url = (
                f"https://stooq.com/q/d/l/?s=^sx5e"
                f"&d1={START_DATE.replace('-','')}"
                f"&d2={yf_start.strftime('%Y%m%d')}&i=d"
            )
            resp = requests.get(stooq_url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            # Stooq sometimes returns metadata lines before the CSV header
            # Skip lines until we find the "Date" header line
            raw_lines = resp.text.strip().split("\n")
            header_idx = next(
                (i for i, l in enumerate(raw_lines) if "Date" in l and "Close" in l),
                0
            )
            clean_text = "\n".join(raw_lines[header_idx:])
            df_stooq = pd.read_csv(StringIO(clean_text))
            df_stooq.columns = [c.strip() for c in df_stooq.columns]
            df_stooq["Date"] = pd.to_datetime(df_stooq["Date"])
            df_stooq = df_stooq.set_index("Date").sort_index()
            df_stooq.index.name = "date"
            df_stooq.index = df_stooq.index.tz_localize(None)
            # Rename to standard OHLCV columns
            col_map = {c: c.capitalize() for c in df_stooq.columns}
            df_stooq = df_stooq.rename(columns=col_map)
            if "Volume" not in df_stooq.columns:
                df_stooq["Volume"] = 0
            df_stooq = df_stooq[["Open", "High", "Low", "Close", "Volume"]]
            log(f"  Stooq gap-fill: {len(df_stooq)} rows | {df_stooq.index.min().date()} to {df_stooq.index.max().date()}")
        except Exception as e:
            log(f"  Stooq failed: {e}")

    # Combine
    if not df_stooq.empty and not df_yf.empty:
        df = pd.concat([df_stooq[df_stooq.index < yf_start], df_yf]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    elif not df_yf.empty:
        df = df_yf
        if df.index.min() > pd.to_datetime("2006-06-01"):
            log("  WARNING: Data starts after Jan 2006 — GARCH pre-sample will be shorter")
    else:
        log("  ERROR: Both sources failed. See manual download notes in script header.")
        raise RuntimeError("Could not download Euro Stoxx 50 data")

    df = df[
        (df.index >= pd.to_datetime(START_DATE)) &
        (df.index <= pd.to_datetime(END_DATE))
    ]
    log(f"  Final: {len(df)} rows | {df.index.min().date()} to {df.index.max().date()}")
    save(df, "eurostoxx50_prices.csv", "Euro Stoxx 50 prices")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOG RETURNS + SQUARED RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices_df):
    """
    Compute daily log returns and squared returns.

    log_return   — input series for GARCH estimation
    sq_return    — realized volatility proxy for out-of-sample evaluation
                   (used in QLIKE and RMSE loss functions)
    """
    log("── Log returns ──")

    close = prices_df["Close"].copy()
    r = np.log(close / close.shift(1)).dropna()
    r.name = "log_return"

    # Sanity check
    extreme = (r.abs() > 0.10).sum()
    if extreme > 0:
        log(f"  WARNING: {extreme} days with |log_return| > 10% — flagged below")
        log(f"    {r[r.abs() > 0.10].to_dict()}")

    df = pd.DataFrame({
        "log_return": r,
        "sq_return":  r ** 2,       # realized vol proxy for QLIKE/RMSE
        "abs_return": r.abs(),      # alternative proxy (more robust to outliers)
    })
    df.index.name = "date"

    log(f"  {len(df)} observations")
    log(f"  Mean return: {r.mean():.6f} | Std: {r.std():.6f}")
    log(f"  Min: {r.min():.4f} | Max: {r.max():.4f}")

    save(df, "eurostoxx50_returns.csv", "Log returns + realized vol proxies")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLLING VOLATILITY + H_t (THREE WINDOW SIZES)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_vol_and_ht(returns_df):
    """
    Compute rolling volatility and H_t indicator for three window sizes:
      - 10 days  (robustness: more reactive)
      - 20 days  (BASE SPECIFICATION — one trading month)
      - 60 days  (robustness: smoother, slower to react)

    For each window:
      - Long-run mean computed over in-sample period ONLY (2008-2018)
        → treated as fixed constant during out-of-sample forecasting
      - H_t uses roll_vol(t-1) — yesterday's value — for forecast compatibility
        → ensures H_t is fully observable at time t when forecasting t+1

    Justification for 20-day base window:
      - Standard in empirical finance: corresponds to one trading month
      - Responsive enough to detect new stress regimes quickly (e.g. COVID onset)
      - Persistent enough to avoid false triggers from single-day spikes
      - Consistent with GARCH model's short-memory volatility clustering focus
    """
    log("── Rolling volatility + H_t indicator ──")

    r = returns_df["log_return"].copy()
    results = {}

    for window, label in [(10, "10d"), (20, "20d"), (60, "60d")]:
        min_p = max(8, int(window * 0.75))  # require 75% of window at minimum
        roll_vol = r.rolling(window=window, min_periods=min_p).std()

        # Long-run mean: IN-SAMPLE ONLY — never use out-of-sample data here
        in_sample = roll_vol[
            (roll_vol.index >= pd.to_datetime("2008-01-01")) &
            (roll_vol.index <= pd.to_datetime(IN_SAMPLE_END))
        ]
        long_run_mean = in_sample.mean()

        # H_t: lagged by 1 day for forecast compatibility
        H_t = (roll_vol.shift(1) > H_T_MULTIPLIER * long_run_mean).astype(int)

        regime_pct = H_t[H_t.index >= pd.to_datetime("2008-01-01")].mean() * 100
        log(f"  Window {window:2d}d | long-run mean: {long_run_mean:.5f} | "
            f"threshold: {H_T_MULTIPLIER * long_run_mean:.5f} | "
            f"H_t=1: {regime_pct:.1f}% of sample")

        results[f"roll_vol_{label}"]      = roll_vol
        results[f"long_run_mean_{label}"] = long_run_mean  # scalar → broadcast
        results[f"H_t_{label}"]           = H_t

    df = pd.DataFrame(results)
    df.index.name = "date"

    # ── Hockey stick H_t (robustness check) ──────────────────────────────────
    # Formula: H_t_hs = max(0, sigma_roll(t-1) / sigma_bar - c)
    # This equals 0 when rolling vol is below the threshold (regime "off"),
    # and grows continuously above it — a smooth ramp rather than a hard jump.
    # Advantage over binary: response above threshold is proportional to how far
    # vol exceeds the threshold, addressing the "arbitrary hard jump" criticism.
    # Uses the 20-day window and in-sample long-run mean (same as base spec).
    roll_vol_20d   = df["roll_vol_20d"]
    long_run_mean_20d = df["long_run_mean_20d"].iloc[0]  # scalar stored per row
    # Use lagged rolling vol (t-1) for forecast compatibility — same as binary
    ratio_lagged = roll_vol_20d.shift(1) / long_run_mean_20d
    df["H_t_hockey"] = (ratio_lagged - H_T_MULTIPLIER).clip(lower=0)

    hs_nonzero = (df["H_t_hockey"] > 0).mean() * 100
    hs_mean_when_on = df.loc[df["H_t_hockey"] > 0, "H_t_hockey"].mean()
    log(f"  Hockey stick (20d, c={H_T_MULTIPLIER}): active on {hs_nonzero:.1f}% of days | "
        f"mean value when active: {hs_mean_when_on:.4f}")

    # Flag the base spec clearly
    df["H_t_BASE"] = df["H_t_20d"]  # binary, 20d window — BASE SPECIFICATION

    save(df, "rolling_vol_and_ht.csv", "Rolling vol + H_t (all window sizes + hockey stick)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. VSTOXX (robustness check)
# ─────────────────────────────────────────────────────────────────────────────

def download_vstoxx():
    """
    Download VSTOXX daily implied volatility index.

    Source: STOXX official historical data file (free, goes back to 1999)
    URL: https://www.stoxx.com/document/Indices/Current/HistoricalData/h_vstoxx.txt
    Format: tab-separated, date column + VSTOXX columns, header on row 3
    """
    log("── VSTOXX ──")

    # Primary: STOXX official historical data file
    try:
        from io import StringIO
        url = "https://www.stoxx.com/document/Indices/Current/HistoricalData/h_vstoxx.txt"
        log("  Fetching from STOXX official data file...")
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        raw = resp.text

        # File has a few header rows — find the actual data start
        # Look for the line starting with "Date" or a date pattern
        lines = raw.split("\n")
        header_idx = next(
            (i for i, l in enumerate(lines)
             if l.strip().startswith("Date") or
             (len(l.strip()) > 0 and l.strip()[0].isdigit())),
            0
        )
        clean_csv = "\n".join(lines[header_idx:])
        # File format (confirmed live):
        # Row 0: "EURO STOXX 50 Volatility Indices,,,,..."
        # Row 1: " ,VSTOXX,Sub-Index 1M,..."
        # Row 2: "Date,V2TX,V6I1,..."  <-- actual header
        # Data: "04.01.1999,18.2033,..."  comma-separated, dot decimal
        # We want the "Date" and "V2TX" columns
        def parse_vstoxx_file(text):
            """Parse a STOXX VSTOXX txt file into a DataFrame."""
            df_raw = pd.read_csv(
                StringIO(text),
                skiprows=2,
                sep=",",
                decimal=".",
                on_bad_lines="skip"
            )
            df_raw.columns = [c.strip() for c in df_raw.columns]
            return pd.DataFrame({
                "date":   pd.to_datetime(df_raw["Date"], dayfirst=True, errors="coerce"),
                "vstoxx": pd.to_numeric(df_raw["V2TX"], errors="coerce")
            }).dropna()

        df1 = parse_vstoxx_file(resp.text)

        # STOXX splits history into two files — fetch the second file (2016 onwards)
        url2 = "https://www.stoxx.com/download/historical_values/h_vstoxx.txt"
        try:
            resp2 = requests.get(url2, timeout=30,
                                 headers={"User-Agent": "Mozilla/5.0"})
            if resp2.status_code == 200 and "Date" in resp2.text:
                df2 = parse_vstoxx_file(resp2.text)
                # Combine and deduplicate
                df_all = pd.concat([df1, df2]).drop_duplicates(subset="date")
            else:
                df_all = df1
        except Exception:
            df_all = df1

        df = df_all
        df = df.set_index("date").sort_index()
        df.index = df.index.tz_localize(None)

        # Filter to sample period
        df = df[
            (df.index >= pd.to_datetime(START_DATE)) &
            (df.index <= pd.to_datetime(END_DATE))
        ]

        missing = df["vstoxx"].isna().sum()
        if missing > 0:
            df["vstoxx"] = df["vstoxx"].ffill(limit=1)

        log(f"  {len(df)} rows | {df.index.min().date()} to {df.index.max().date()}")
        log(f"  VSTOXX range: {df.vstoxx.min():.1f} to {df.vstoxx.max():.1f}")
        save(df, "vstoxx_daily.csv", "VSTOXX")
        return df

    except Exception as e:
        log(f"  STOXX file failed: {e}")
        log("  Trying Yahoo Finance fallback (V2TX.DE)...")
        try:
            import yfinance as yf
            df = yf.Ticker("V2TX.DE").history(
                start=START_DATE, end=END_DATE, auto_adjust=True
            )
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Close"]].rename(columns={"Close": "vstoxx"})
            df.index.name = "date"
            if len(df) > 0:
                log(f"  Yahoo fallback: {len(df)} rows")
                save(df, "vstoxx_daily.csv", "VSTOXX (Yahoo fallback)")
                return df
        except Exception as e2:
            log(f"  Yahoo fallback also failed: {e2}")

        log("  VSTOXX unavailable — skipping (only needed for robustness checks)")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. ECB AAA 10Y BOND YIELD (macro control variable)
# ─────────────────────────────────────────────────────────────────────────────

def download_ecb_bond_yield():
    """
    Download Euro area AAA-rated government bond 10-year spot rate
    from the ECB Data Portal SDMX API.

    Series: YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y
    - Daily business week frequency
    - Euro area (changing composition)
    - AAA-rated issuers only (Svensson model)
    - 10-year spot rate, % per annum

    Why this over German Bund:
      - Represents the entire euro area AAA sovereign bond market
      - Consistent with Euro Stoxx 50 as a euro area aggregate
      - Published directly by the ECB — methodologically clean
      - Avoids Germany-specific idiosyncratic effects
    """
    log("── ECB AAA 10Y bond yield ──")

    series_key = "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"
    # Correct URL: dataset = YC, key = B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y
    # (drop the "YC." prefix from the series key in the path)
    series_path = series_key.replace("YC.", "", 1)
    url = (
        f"https://data-api.ecb.europa.eu/service/data/YC/{series_path}"
        f"?startPeriod={START_DATE}&endPeriod={END_DATE}"
        f"&format=csvdata"
    )

    try:
        log(f"  Fetching from ECB SDMX API...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(r.text))

        # ECB SDMX CSV format has TIME_PERIOD and OBS_VALUE columns
        if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
            df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
            df.columns = ["date", "yield_10y_aaa"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["yield_10y_aaa"] = pd.to_numeric(df["yield_10y_aaa"], errors="coerce")

            missing = df["yield_10y_aaa"].isna().sum()
            if missing > 0:
                log(f"  {missing} missing values — forward-filling")
                df["yield_10y_aaa"] = df["yield_10y_aaa"].ffill()

            log(f"  {len(df)} rows | {df.index.min().date()} to {df.index.max().date()}")
            log(f"  Yield range: {df['yield_10y_aaa'].min():.3f}% to {df['yield_10y_aaa'].max():.3f}%")
            save(df, "ecb_aaa_10y_yield.csv", "ECB AAA 10Y bond yield")
            return df
        else:
            log(f"  Unexpected CSV format. Columns: {df.columns.tolist()}")
            log(f"  Raw response (first 500 chars): {r.text[:500]}")
            return None

    except Exception as e:
        log(f"  ECB API failed: {e}")
        log("  Trying fallback: ^TNX (US 10Y, for testing only — replace with proper EU data)")
        try:
            import yfinance as yf
            df = yf.Ticker("^TNX").history(
                start=START_DATE, end=END_DATE, auto_adjust=True
            )
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Close"]].rename(columns={"Close": "yield_10y_us_fallback"})
            df.index.name = "date"
            log(f"  FALLBACK (US 10Y): {len(df)} rows — replace with EU data!")
            save(df, "ecb_aaa_10y_yield.csv", "Bond yield (US FALLBACK — replace!)")
            return df
        except Exception as e2:
            log(f"  Fallback also failed: {e2}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. DATA QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def create_summary(prices, returns, vol, vstoxx, bond):
    lines = []
    lines.append("=" * 65)
    lines.append("MARKET DATA SUMMARY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sample: {START_DATE} to {END_DATE}")
    lines.append(f"In-sample: 2008-01-01 to {IN_SAMPLE_END}")
    lines.append(f"Out-of-sample: 2019-01-01 to {END_DATE}")
    lines.append("=" * 65)

    datasets = [
        ("Euro Stoxx 50 Prices",         prices,  "eurostoxx50_prices.csv"),
        ("Log Returns + Realized Vol",    returns, "eurostoxx50_returns.csv"),
        ("Rolling Vol + H_t (all windows)", vol,   "rolling_vol_and_ht.csv"),
        ("VSTOXX (robustness)",           vstoxx,  "vstoxx_daily.csv"),
        ("ECB AAA 10Y Yield (control)",   bond,    "ecb_aaa_10y_yield.csv"),
    ]

    for name, df, fname in datasets:
        if df is None:
            lines.append(f"\n{name}: NOT DOWNLOADED")
            continue
        lines.append(f"\n{name} ({fname}):")
        lines.append(f"  Rows:           {len(df)}")
        lines.append(f"  Date range:     {df.index.min().date()} → {df.index.max().date()}")
        missing = df.isna().sum().sum()
        lines.append(f"  Missing values: {missing}")

    lines.append("\n" + "=" * 65)
    lines.append("H_t INDICATOR SUMMARY (c=1.5 placeholder):")
    if vol is not None:
        for w in ["10d", "20d", "60d"]:
            col = f"H_t_{w}"
            if col in vol.columns:
                in_s = vol[col][vol.index >= pd.to_datetime("2008-01-01")]
                oos  = vol[col][vol.index >= pd.to_datetime("2019-01-01")]
                lines.append(f"  {w} window: H_t=1 on {in_s.mean()*100:.1f}% in-sample, "
                              f"{oos.mean()*100:.1f}% out-of-sample")
    lines.append("  NOTE: c=1.5 is a placeholder — c is estimated freely in model")
    lines.append("  BASE SPEC: 20d window binary (H_t_BASE column = H_t_20d)")
    lines.append("  ROBUSTNESS: H_t_10d, H_t_60d (alternative windows)")
    lines.append("  ROBUSTNESS: H_t_hockey (continuous hockey stick, 20d window)")
    lines.append("    Formula: H_t_hockey = max(0, roll_vol(t-1)/long_run_mean - c)")

    lines.append("\n" + "=" * 65)
    lines.append("HANDOFF TO PERSON 2 (model implementation):")
    lines.append("  Primary inputs:")
    lines.append("    eurostoxx50_returns.csv     → log_return (GARCH input)")
    lines.append("    eurostoxx50_returns.csv     → sq_return (realized vol proxy)")
    lines.append("    rolling_vol_and_ht.csv      → H_t_BASE (base model)")
    lines.append("    sentiment_daily_all_sources.csv → P_t, N_t (from ECB pipeline)")
    lines.append("  Robustness inputs:")
    lines.append("    rolling_vol_and_ht.csv      → H_t_10d, H_t_60d")
    lines.append("    vstoxx_daily.csv            → vstoxx")
    lines.append("    ecb_aaa_10y_yield.csv       → yield_10y_aaa")
    lines.append("=" * 65)

    report = "\n".join(lines)
    (OUTPUT_DIR / "market_data_summary.txt").write_text(report)
    print("\n" + report)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    log("=" * 65)
    log("MARKET DATA DOWNLOAD — START")
    log(f"Period: {START_DATE} to {END_DATE}")
    log("=" * 65 + "\n")

    prices  = download_eurostoxx50()
    returns = compute_returns(prices)
    vol     = compute_rolling_vol_and_ht(returns)
    vstoxx  = download_vstoxx()
    bond    = download_ecb_bond_yield()

    create_summary(prices, returns, vol, vstoxx, bond)

    log(f"\nAll files saved to: {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
