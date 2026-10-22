"""Black-Scholes prices and Greeks for European options.

Every function accepts plain floats or NumPy arrays (they broadcast), so the
same code prices one option or a whole option chain at once.

Inputs used everywhere:
    S      spot price of the underlying
    K      strike price
    T      time to expiry in years (e.g. 30 days -> 30/365)
    r      continuously compounded risk-free rate (0.05 = 5%)
    sigma  volatility per year (0.20 = 20%)
    q      continuous dividend yield (0.0 for a stock with no dividends)
    option "call" or "put"
"""
import numpy as np
from scipy.stats import norm


def _check_option(option):
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def d1_d2(S, K, T, r, sigma, q=0.0):
    """The two standardised distances that appear in every Black-Scholes formula.

    d2 is how many standard deviations the (risk-neutral) log-price must move
    to finish exactly at the strike; d1 = d2 + sigma * sqrt(T).
    """
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option="call"):
    """Black-Scholes(-Merton) price of a European call or put.

    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    """
    _check_option(option)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_s = S * np.exp(-q * T)   # spot, net of dividends paid before expiry
    disc_k = K * np.exp(-r * T)   # strike, in today's money
    if option == "call":
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)


def delta(S, K, T, r, sigma, q=0.0, option="call"):
    """dV/dS: shares of the underlying needed to hedge one option."""
    _check_option(option)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """d2V/dS2: how fast delta changes. Same for calls and puts."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """dV/dsigma per 1.00 change in volatility (divide by 100 for per 1 vol point).

    Same for calls and puts. Also the derivative the implied-vol solver needs.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option="call"):
    """dV/dt per year of calendar time passing (divide by 365 for per day).

    Negative for a long option in most cases: the option loses value as expiry nears.
    """
    _check_option(option)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option == "call":
        return (decay - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1))
    return (decay + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1))


def rho(S, K, T, r, sigma, q=0.0, option="call"):
    """dV/dr per 1.00 change in the interest rate."""
    _check_option(option)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def put_call_parity_gap(call, put, S, K, T, r, q=0.0):
    """C - P - (S e^{-qT} - K e^{-rT}). Zero (up to rounding) when prices are arbitrage-free."""
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))
