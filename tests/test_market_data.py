import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.heston import HestonParams, heston_price_chain
from src.market_data import prepare_quotes

TRUE = HestonParams(v0=0.02, kappa=3.0, theta=0.035, xi=1.0, rho=-0.8)
SPOT, RATE, DIV = 6000.0, 0.04, 0.013


def fake_yahoo_chain(params, quote_date=dt.date(2026, 10, 21), spread=0.02, seed=0):
    """A chain in the same column layout yfinance returns, priced with Heston."""
    rng = np.random.default_rng(seed)
    rows = []
    for days in (14, 30, 60, 91, 182, 365):
        T = days / 365
        F = SPOT * np.exp((RATE - DIV) * T)
        strikes = np.arange(round(F * 0.7, -1), F * 1.3, 25.0)
        for kind in ("call", "put"):
            prices = heston_price_chain(np.full(strikes.size, F), strikes, np.full(strikes.size, T),
                                        np.full(strikes.size, RATE), params, np.full(strikes.size, kind == "call"))
            half = np.maximum(spread * prices, 0.05) / 2 * (1 + 0.2 * rng.random(strikes.size))
            for K, p, h in zip(strikes, prices, half):
                rows.append({"contractSymbol": f"SPXW{days}{kind[0]}{K:.0f}", "type": kind,
                             "expiry": (quote_date + dt.timedelta(days=days)).isoformat(), "strike": K,
                             "bid": max(p - h, 0.0), "ask": p + h, "lastPrice": p,
                             "lastTradeDate": "2026-10-21 19:59:00+00:00", "volume": 10, "openInterest": 100,
                             "impliedVolatility": 0.2})
    market = {"date": quote_date.isoformat(), "spot": SPOT, "rate_13w_tbill": RATE}
    return pd.DataFrame(rows), market


def test_forward_and_dividend_recovered():
    df, market = fake_yahoo_chain(TRUE)
    q = prepare_quotes(df, market)
    expected_F = SPOT * np.exp((RATE - DIV) * q["T"])
    assert np.allclose(q["F"], expected_F, rtol=1e-4)
    assert q["q_implied"].median() == pytest.approx(DIV, abs=5e-4)


def test_only_out_of_the_money_quotes_kept():
    df, market = fake_yahoo_chain(TRUE)
    q = prepare_quotes(df, market)
    assert (q.loc[q.is_call, "K"] >= q.loc[q.is_call, "F"]).all()
    assert (q.loc[~q.is_call, "K"] < q.loc[~q.is_call, "F"]).all()
    assert q["iv"].between(0.03, 1.5).all()


def test_falls_back_to_last_price_when_bid_ask_blank():
    df, market = fake_yahoo_chain(TRUE)
    df[["bid", "ask"]] = 0.0
    q = prepare_quotes(df, market)
    assert len(q) > 50 and (q["price_source"] == "last").all()
