import numpy as np
import pytest

from src.black_scholes import bs_price, delta, gamma, vega, theta, rho, put_call_parity_gap

# Textbook example (Hull): S=100, K=100, T=1, r=5%, sigma=20%, no dividends.
S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20


def test_textbook_call_and_put():
    assert bs_price(S, K, T, R, SIG, option="call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(S, K, T, R, SIG, option="put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    strikes = np.linspace(50, 150, 21)
    c = bs_price(S, strikes, T, R, SIG, q, "call")
    p = bs_price(S, strikes, T, R, SIG, q, "put")
    assert np.allclose(put_call_parity_gap(c, p, S, strikes, T, R, q), 0.0, atol=1e-10)


def finite_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_greeks_match_finite_differences(option, strike):
    q = 0.01
    price = lambda s=S, k=strike, t=T, r=R, v=SIG: bs_price(s, k, t, r, v, q, option)
    assert delta(S, strike, T, R, SIG, q, option) == pytest.approx(finite_diff(lambda s: price(s=s), S, 1e-3), rel=1e-6)
    assert gamma(S, strike, T, R, SIG, q) == pytest.approx(
        (price(s=S + 1e-2) - 2 * price() + price(s=S - 1e-2)) / 1e-4, rel=1e-4)
    assert vega(S, strike, T, R, SIG, q) == pytest.approx(finite_diff(lambda v: price(v=v), SIG, 1e-5), rel=1e-6)
    # theta is the change as calendar time passes, i.e. minus the derivative in T.
    assert theta(S, strike, T, R, SIG, q, option) == pytest.approx(-finite_diff(lambda t: price(t=t), T, 1e-5), rel=1e-6)
    assert rho(S, strike, T, R, SIG, q, option) == pytest.approx(finite_diff(lambda r: price(r=r), R, 1e-6), rel=1e-6)


def test_bad_option_type():
    with pytest.raises(ValueError):
        bs_price(S, K, T, R, SIG, option="straddle")
