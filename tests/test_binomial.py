import pytest

from src.binomial import crr_price
from src.black_scholes import bs_price


@pytest.mark.parametrize("option", ["call", "put"])
def test_converges_to_black_scholes(option):
    exact = bs_price(100, 105, 0.5, 0.04, 0.25, 0.01, option)
    assert crr_price(100, 105, 0.5, 0.04, 0.25, 0.01, option, steps=2000) == pytest.approx(exact, abs=5e-3)


def test_error_shrinks_with_steps():
    exact = bs_price(100, 100, 1, 0.05, 0.2)
    errors = [abs(crr_price(100, 100, 1, 0.05, 0.2, steps=n) - exact) for n in (50, 200, 800)]
    assert errors[0] > errors[1] > errors[2]


def test_american_put_worth_at_least_european():
    eu = crr_price(100, 110, 1, 0.08, 0.2, option="put", steps=500)
    am = crr_price(100, 110, 1, 0.08, 0.2, option="put", steps=500, american=True)
    assert am > eu + 0.1   # early exercise has real value for an in-the-money put


def test_american_call_no_dividend_equals_european():
    # With no dividends it is never optimal to exercise a call early.
    eu = crr_price(100, 95, 1, 0.05, 0.3, option="call", steps=500)
    am = crr_price(100, 95, 1, 0.05, 0.3, option="call", steps=500, american=True)
    assert am == pytest.approx(eu, abs=1e-10)
