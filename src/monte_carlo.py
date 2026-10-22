"""Monte Carlo pricing: simulate the risk-neutral world many times and average.

price = e^{-rT} * mean(payoff), reported with a standard error and a 95%
confidence interval so the estimate's precision is always visible.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self):
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def _summarise(discounted_payoffs, n_paths):
    return MCResult(float(discounted_payoffs.mean()),
                    float(discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs))),
                    n_paths)


def _payoff(ST, K, option):
    return np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)


def mc_gbm(S, K, T, r, sigma, q=0.0, option="call", n_paths=100_000, antithetic=True, seed=0):
    """European option under geometric Brownian motion (the Black-Scholes world).

    GBM can be sampled exactly at expiry, so no time steps are needed:
        S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1)
    Antithetic variates: for every Z also use -Z, and average each pair. The pair
    averages are less noisy than independent draws, so the error shrinks for free.
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(half)
    drift = (r - q - 0.5 * sigma**2) * T
    shock = sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if antithetic:
        pay_plus = _payoff(S * np.exp(drift + shock * Z), K, option)
        pay_minus = _payoff(S * np.exp(drift - shock * Z), K, option)
        return _summarise(disc * 0.5 * (pay_plus + pay_minus), n_paths)
    return _summarise(disc * _payoff(S * np.exp(drift + shock * Z), K, option), n_paths)


def mc_heston(S, K, T, r, params, q=0.0, option="call", n_paths=100_000, n_steps=252, seed=0):
    """European option under the Heston model, simulated with full-truncation Euler.

    Variance can't be sampled exactly in one jump, so we step through time.
    'Full truncation' (Lord et al., 2010): wherever variance would be negative we
    use max(v, 0) in the drift and diffusion, which keeps the scheme stable and
    has a small bias. Used here only to cross-check the Fourier pricer.
    """
    v0, kappa, theta, xi, rho = params.v0, params.kappa, params.theta, params.xi, params.rho
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    log_s = np.full(n_paths, np.log(S))
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal(n_paths)  # correlated shock
        v_pos = np.maximum(v, 0.0)
        log_s += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1
        v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z2
    return _summarise(np.exp(-r * T) * _payoff(np.exp(log_s), K, option), n_paths)
