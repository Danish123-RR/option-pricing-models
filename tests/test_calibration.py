import numpy as np
import pandas as pd
import pytest

from src.black_scholes import vega as bs_vega
from src.calibration import calibrate_heston
from src.heston import HestonParams, heston_price_chain
from src.implied_vol import implied_vol_array

TRUE = HestonParams(v0=0.03, kappa=2.5, theta=0.05, xi=0.8, rho=-0.75)


def synthetic_quotes(params, spot=100.0, rate=0.04):
    """A fake option chain whose prices come from known Heston parameters."""
    rows = []
    for T in (0.08, 0.25, 0.5, 1.0, 2.0):
        F = spot * np.exp(rate * T)
        for m in np.linspace(0.8, 1.15, 15):
            K = round(F * m, 2)
            rows.append({"T": T, "F": F, "K": K, "r": rate, "is_call": K >= F})
    q = pd.DataFrame(rows)
    q["mid"] = heston_price_chain(q.F, q.K, q["T"], q.r, params, q.is_call)
    spot_equiv = q.F * np.exp(-q.r * q["T"])
    q["iv"] = implied_vol_array(q["mid"], spot_equiv, q.K, q["T"], q.r, 0.0, np.where(q.is_call, "call", "put"))
    q["vega"] = bs_vega(spot_equiv, q.K, q["T"], q.r, q["iv"])
    return q


def test_recovers_known_parameters():
    quotes = synthetic_quotes(TRUE)
    res = calibrate_heston(quotes)
    assert res.rmse_iv < 0.01                       # under 0.01 vol points
    got, want = res.params.as_array(), TRUE.as_array()
    assert got == pytest.approx(want, rel=0.02, abs=1e-3)
