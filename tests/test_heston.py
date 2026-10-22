import numpy as np
import pytest
from scipy.integrate import quad

from src.black_scholes import bs_price
from src.heston import HestonParams, heston_cf, heston_price, heston_price_chain, bs_price_via_lewis
from src.monte_carlo import mc_heston

P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


def test_characteristic_function_basic_properties():
    assert heston_cf(0.0, 1.0, P) == pytest.approx(1.0)        # total probability is 1
    assert heston_cf(-1j, 1.0, P) == pytest.approx(1.0)        # E[S_T / F_T] = 1: forward is fair


def test_lewis_integral_reproduces_black_scholes():
    strikes = np.array([60.0, 90.0, 100.0, 110.0, 150.0])
    for T in (0.02, 0.5, 3.0):
        assert np.allclose(bs_price_via_lewis(100, strikes, T, 0.05, 0.25, 0.01),
                           bs_price(100, strikes, T, 0.05, 0.25, 0.01), atol=1e-8)


def test_reduces_to_black_scholes_without_vol_of_vol():
    # xi -> 0 and v0 = theta: variance stays at 0.04, i.e. constant 20% volatility.
    flat = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=1e-4, rho=0.0)
    strikes = np.array([80.0, 100.0, 120.0])
    assert np.allclose(heston_price(100, strikes, 1.0, 0.05, flat), bs_price(100, strikes, 1.0, 0.05, 0.2), atol=1e-5)


def adaptive_quad_price(S, K, T, r, p):
    F = S * np.exp(r * T)
    k = np.log(K / F)
    f = lambda u: np.real(np.exp(-1j * u * k) * heston_cf(u - 0.5j, T, p)) / (u * u + 0.25)
    integral = sum(quad(f, a, b, limit=1000, epsabs=1e-13)[0] for a, b in ((0, 50), (50, 500), (500, 5000)))
    return np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integral)


@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_fixed_grid_matches_adaptive_quadrature(T, strike):
    assert heston_price(100, strike, T, 0.03, P) == pytest.approx(adaptive_quad_price(100, strike, T, 0.03, P), abs=1e-6)


def test_put_call_parity():
    strikes = np.array([90.0, 100.0, 110.0])
    c = heston_price(100, strikes, 0.5, 0.03, P, q=0.01, option="call")
    p = heston_price(100, strikes, 0.5, 0.03, P, q=0.01, option="put")
    assert np.allclose(c - p, 100 * np.exp(-0.01 * 0.5) - strikes * np.exp(-0.03 * 0.5), atol=1e-10)


def test_matches_monte_carlo():
    fourier = heston_price(100, 100, 1.0, 0.03, P)
    sim = mc_heston(100, 100, 1.0, 0.03, P, n_paths=200_000, n_steps=200, seed=7)
    assert abs(sim.price - fourier) < 3 * sim.std_error + 0.02   # 0.02 allows for Euler time-step bias


def test_chain_pricer_matches_single_pricer():
    F = np.array([101.0, 101.0, 103.0, 103.0])
    K = np.array([95.0, 105.0, 95.0, 105.0])
    T = np.array([0.25, 0.25, 1.0, 1.0])
    r = np.array([0.04, 0.04, 0.03, 0.03])
    is_call = np.array([False, True, False, True])
    chain = heston_price_chain(F, K, T, r, P, is_call)
    for i in range(4):
        S = F[i] * np.exp(-r[i] * T[i])   # spot implied by the forward with q = 0
        single = heston_price(S, K[i], T[i], r[i], P, option="call" if is_call[i] else "put")
        assert chain[i] == pytest.approx(single, abs=1e-10)
