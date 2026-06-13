"""
download_controls.py — fetch EUR/USD and VIX from Yahoo Finance into market_data/.

Run from the ecb_sentiment folder with the venv active:
    python download_controls.py
"""

from pathlib import Path
import yfinance as yf

OUT = Path("market_data")
OUT.mkdir(exist_ok=True)

START = "2007-04-01"
END   = "2024-01-01"


def fetch(ticker, colname, filename):
    print(f"Fetching {ticker} ...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # yfinance can return a multi-index column when given a single ticker,
    # depending on version; flatten it.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": colname})
    out.index.name = "date"
    out.index = out.index.strftime("%Y-%m-%d")
    out.to_csv(OUT / filename)
    print(f"  → {OUT / filename}  ({len(out):,} rows)")


if __name__ == "__main__":
    fetch("EURUSD=X", "eur_usd", "eur_usd_daily.csv")
    fetch("^VIX",     "vix",     "vix_daily.csv")
    fetch("BZ=F",     "brent",   "brent_daily.csv")
    print("Done.")
