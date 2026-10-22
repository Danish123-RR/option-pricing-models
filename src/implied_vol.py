"""Implied volatility: the sigma that makes Black-Scholes match a market price.

Method: Newton-Raphson (fast, uses vega as the derivative), with a fall-back to
Brent's bracketing method when Newton is unsafe (vega near zero, or a step that
leaves the search interval). Returns NaN when no volatility can reproduce the price
(e.g. the price is below intrinsic value).
"""
import numpy as np
from scipy.optimize import brentq

from src.black_scholes import bs_price, vega

SIGMA_LOW, SIGMA_HIGH = 1e-4, 5.0   # search interval: 0.01% to 500% volatility


def price_bounds(S, K, T, r, q=0.0, option="call"):
    """No-arbitrage bounds any European option price must sit inside."""
    fwd_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
    if option == "call":
        return max(fwd_s - disc_k, 0.0), fwd_s
    return max(disc_k - fwd_s, 0.0), disc_k


def implied_vol(price, S, K, T, r, q=0.0, option="call", tol=1e-8, max_iter=50):
    lower, upper = price_bounds(S, K, T, r, q, option)
    if not (lower < price < upper):
        return float("nan")

    def error(sig):
        return bs_price(S, K, T, r, sig, q, option) - price

    # Newton-Raphson from a reasonable starting guess.
    sigma = 0.2
    for _ in range(max_iter):
        diff = error(sigma)
        if abs(diff) < tol:
            return float(sigma)
        v = vega(S, K, T, r, sigma, q)
        if v < 1e-10:
            break                       # flat region: Newton would jump wildly
        sigma_next = sigma - diff / v
        if not SIGMA_LOW < sigma_next < SIGMA_HIGH:
            break                       # left the interval: switch to bracketing
        sigma = sigma_next

    # Brent's method: slower but guaranteed when the root is bracketed.
    try:
        return float(brentq(error, SIGMA_LOW, SIGMA_HIGH, xtol=tol, maxiter=200))
    except ValueError:
        return float("nan")


def implied_vol_array(prices, S, K, T, r, q=0.0, option="call"):
    """Implied vols for arrays of prices/strikes/maturities. `option` may also be an array."""
    prices, K, T = np.broadcast_arrays(np.asarray(prices, float), np.asarray(K, float), np.asarray(T, float))
    S, r, q = (np.broadcast_to(np.asarray(x, float), prices.shape) for x in (S, r, q))
    opts = np.broadcast_to(np.asarray(option), prices.shape)
    out = np.empty(prices.shape)
    for i in np.ndindex(prices.shape):
        out[i] = implied_vol(prices[i], S[i], K[i], T[i], r[i], q[i], str(opts[i]))
    return out
