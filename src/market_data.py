"""Load the saved SPX option chain and turn it into clean quotes for calibration.

Cleaning steps (each one is a decision; see the study guide for alternatives):
  1. Maturity T in years = calendar days to expiry / 365.
  2. Market price = mid of bid and ask. Quotes with no bid, a crossed market
     (ask < bid) or a very wide spread are dropped: their 'price' is unreliable.
     (If the snapshot was taken while the US market was closed and bid/ask are blank,
     we fall back to recent last-traded prices and record that in `price_source`.)
  3. Forward F for each expiry from put-call parity, F = K + e^{rT} (C - P), using
     strikes near the money. This captures dividends without needing a dividend forecast.
  4. Keep out-of-the-money options only (puts below F, calls above). They are the
     liquid ones, and in-the-money prices add no new information (parity links them).
  5. Keep strikes within a moneyness band, then compute each quote's implied vol
     and vega; drop any quote whose implied vol can't be found.
"""
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from src.black_scholes import vega as bs_vega
from src.implied_vol import implied_vol_array

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


def latest_snapshot(data_dir=DATA_DIR):
    """Return (chain DataFrame, market dict) for the most recent saved snapshot."""
    chains = sorted(data_dir.glob("spx_chain_*.csv"))
    if not chains:
        raise FileNotFoundError("No data/spx_chain_*.csv found. Run: python scripts/fetch_spx_chain.py")
    chain_path = chains[-1]
    stamp = chain_path.stem.replace("spx_chain_", "")
    market = json.loads((data_dir / f"spx_market_{stamp}.json").read_text())
    return pd.read_csv(chain_path), market


def implied_forwards(df, rate, n_near=6):
    """Forward price per expiry from put-call parity on the strikes nearest the money."""
    forwards = {}
    for expiry, g in df.groupby("expiry"):
        T = g["T"].iloc[0]
        calls = g[g["type"] == "call"].set_index("strike")["mid"]
        puts = g[g["type"] == "put"].set_index("strike")["mid"]
        both = calls.index.intersection(puts.index)
        if len(both) == 0:
            continue
        diff = (calls[both] - puts[both]).abs()          # smallest |C - P| = closest to the money
        near = diff.nsmallest(n_near).index
        f_est = near + np.exp(rate * T) * (calls[near] - puts[near])
        forwards[expiry] = float(np.median(f_est))
    return forwards


def quote_date_of(market):
    """Trading date the quotes belong to.

    A snapshot fetched before the US open (13:30 UTC) carries the previous session's
    closing quotes, so its quote date is the previous weekday, not the fetch date.
    """
    if market.get("quote_date"):
        return dt.date.fromisoformat(market["quote_date"])
    fetched = dt.datetime.fromisoformat(market["fetched_utc"])
    day = fetched.date()
    if fetched.hour * 60 + fetched.minute < 13 * 60 + 30:
        day -= dt.timedelta(days=1)
        while day.weekday() >= 5:          # skip Saturday/Sunday
            day -= dt.timedelta(days=1)
    return day


def prepare_quotes(df, market, min_days=7, max_days=400, moneyness=(0.80, 1.15),
                   max_rel_spread=0.25, min_mid=0.05):
    """Clean a raw chain into the quotes table the calibrator expects."""
    quote_date = quote_date_of(market) if "fetched_utc" in market else dt.date.fromisoformat(market["date"])
    rate = float(market["rate_13w_tbill"])
    spot = float(market["spot"]) if market.get("spot") is not None else float("nan")

    df = df.copy()
    df["days"] = (pd.to_datetime(df["expiry"]).dt.date.map(lambda d: (d - quote_date).days))
    df = df[(df["days"] >= min_days) & (df["days"] <= max_days)]
    df["T"] = df["days"] / 365.0

    # Step 2: usable two-sided quotes only.
    two_sided = df[(df["bid"] > 0) & (df["ask"] > df["bid"])].copy()
    if len(two_sided) >= 100:
        df = two_sided
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
        df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
        df = df[(df["rel_spread"] <= max_rel_spread) & (df["mid"] >= min_mid)]
        df["price_source"] = "mid"
    else:
        # Outside US market hours Yahoo sometimes blanks bid/ask. Fall back to the last
        # traded price, but only for contracts that actually traded in the last 3 days.
        traded = pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce").dt.date
        recent = traded.map(lambda d: d is not pd.NaT and d is not None and (quote_date - d).days <= 3)
        df = df[recent & (df["lastPrice"] >= min_mid)].copy()
        df["mid"] = df["lastPrice"]
        df["rel_spread"] = np.nan
        df["price_source"] = "last"

    # Step 3: forwards per expiry.
    fwd = implied_forwards(df, rate)
    df = df[df["expiry"].isin(fwd)]
    df["F"] = df["expiry"].map(fwd)
    df["r"] = rate
    if not np.isfinite(spot):
        # Spot missing from the snapshot: estimate it from the nearest forward, F e^{-rT}.
        # Over ~2 weeks dividends move this by only a few index points.
        nearest = df.loc[df["T"].idxmin()]
        spot = float(nearest["F"] * np.exp(-rate * nearest["T"]))
        market["spot_estimated"] = spot
    df["q_implied"] = rate - np.log(df["F"] / spot) / df["T"]   # dividend yield implied by F

    # Step 4 and 5: out-of-the-money quotes within the moneyness band.
    otm = ((df["type"] == "put") & (df["strike"] < df["F"])) | ((df["type"] == "call") & (df["strike"] >= df["F"]))
    df = df[otm]
    ratio = df["strike"] / df["F"]
    df = df[(ratio >= moneyness[0]) & (ratio <= moneyness[1])]

    df = df.rename(columns={"strike": "K"})
    df["is_call"] = df["type"] == "call"
    spot_equiv = df["F"] * np.exp(-df["r"] * df["T"])   # Black-Scholes on the forward, q = 0
    df["iv"] = implied_vol_array(df["mid"], spot_equiv, df["K"], df["T"], df["r"], 0.0,
                                 np.where(df["is_call"], "call", "put"))
    df = df[np.isfinite(df["iv"])]
    df["vega"] = bs_vega(df["F"] * np.exp(-df["r"] * df["T"]), df["K"], df["T"], df["r"], df["iv"])
    df["log_moneyness"] = np.log(df["K"] / df["F"])
    cols = ["expiry", "days", "T", "K", "F", "r", "q_implied", "type", "is_call", "bid", "ask",
            "mid", "price_source", "rel_spread", "iv", "vega", "log_moneyness", "volume", "openInterest"]
    return df[cols].sort_values(["T", "K"]).reset_index(drop=True)
