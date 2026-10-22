import math

import numpy as np
import pytest

from src.black_scholes import bs_price
from src.implied_vol import implied_vol, implied_vol_array, price_bounds


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("strike", [60.0, 95.0, 100.0, 110.0, 160.0])
@pytest.mark.parametrize("T", [0.02, 0.5, 3.0])
def test_round_trip(option, sigma, strike, T):
    price = bs_price(100.0, strike, T, 0.03, sigma, 0.01, option)
    time_value = price - price_bounds(100.0, strike, T, 0.03, 0.01, option)[0]
    if time_value < 1e-4:
        # Deep in or out of the money: the price equals its no-arbitrage bound to
        # many decimals, so it carries no information about sigma. Real calibrations
        # avoid these quotes for the same reason.
        pytest.skip("no time value left to invert")
    assert implied_vol(price, 100.0, strike, T, 0.03, 0.01, option) == pytest.approx(sigma, abs=1e-6)


def test_price_outside_bounds_returns_nan():
    assert math.isnan(implied_vol(0.5, 100, 80, 1, 0.0, option="call"))   # below intrinsic value 20
    assert math.isnan(implied_vol(150, 100, 80, 1, 0.0, option="call"))   # above the spot price


def test_array_version():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bs_price(100, strikes, 0.25, 0.02, 0.3)
    assert np.allclose(implied_vol_array(prices, 100, strikes, 0.25, 0.02), 0.3, atol=1e-7)
