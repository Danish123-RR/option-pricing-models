"""Download a snapshot of the S&P 500 index (SPX) option chain from Yahoo Finance.

Run from the repo root:
    python scripts/fetch_spx_chain.py

Saves two files in data/:
    spx_chain_<YYYY-MM-DD>.csv   one row per option (calls and puts, ~10 expiries)
    spx_market_<YYYY-MM-DD>.json spot level, risk-free rate and fetch time

Why SPX and not SPY: SPX options are European (exercise only at expiry),
which is exactly what Black-Scholes and Heston price. SPY options are American.
"""
import datetime as dt
import json
import pathlib

import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Expiries we try to cover, in calendar days from today. We pick the listed
# expiry closest to each target, so the fit sees short, medium and long maturities.
TARGET_DAYS = [14, 30, 45, 60, 90, 120, 180, 270, 365]


def pick_expiries(listed, today):
    """Return the listed expiry dates closest to each target maturity (no duplicates)."""
    dates = [dt.date.fromisoformat(s) for s in listed]
    chosen = []
    for target in TARGET_DAYS:
        best = min(dates, key=lambda d: abs((d - today).days - target))
        if best not in chosen and (best - today).days >= 7:
            chosen.append(best)
    return sorted(chosen)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    today = dt.date.today()

    spx = yf.Ticker("^SPX")
    # dropna: before the US open Yahoo can add an empty row for today.
    closes = spx.history(period="5d")["Close"].dropna()
    spot = float(closes.iloc[-1])
    last_close_date = closes.index[-1].date().isoformat()
    # ^IRX = 13-week US T-bill yield, quoted in percent (e.g. 4.2 means 4.2%).
    rate = float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100.0

    rows = []
    for expiry in pick_expiries(spx.options, today):
        chain = spx.option_chain(expiry.isoformat())
        for kind, frame in (("call", chain.calls), ("put", chain.puts)):
            frame = frame.copy()
            frame["type"] = kind
            frame["expiry"] = expiry.isoformat()
            rows.append(frame)
        print(f"  {expiry}  calls={len(chain.calls)}  puts={len(chain.puts)}")

    df = pd.concat(rows, ignore_index=True)
    keep = ["contractSymbol", "type", "expiry", "strike", "bid", "ask", "lastPrice",
            "lastTradeDate", "volume", "openInterest", "impliedVolatility"]
    df = df[keep]

    stamp = today.isoformat()
    chain_path = DATA_DIR / f"spx_chain_{stamp}.csv"
    df.to_csv(chain_path, index=False)
    market = {
        "date": stamp,
        "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spot": spot,
        "last_close_date": last_close_date,
        "rate_13w_tbill": rate,
    }
    (DATA_DIR / f"spx_market_{stamp}.json").write_text(json.dumps(market, indent=2))
    print(f"Saved {len(df)} options to {chain_path.name}; spot={spot:.2f}, r={rate:.4f}")


if __name__ == "__main__":
    main()
