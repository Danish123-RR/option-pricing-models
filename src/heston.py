"""Heston (1993) stochastic-volatility model.

Risk-neutral dynamics:
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v,     dW^S dW^v = rho dt

Pricing route: Heston has no Black-Scholes-style closed form, but the
characteristic function of the log-price IS known in closed form. We price
calls with the Lewis (2001) single-integral formula and get puts by put-call parity.
"""
from dataclasses import dataclass, astuple

import numpy as np



@dataclass
class HestonParams:
    v0: float      # variance today (sigma_0 = sqrt(v0))
    kappa: float   # speed at which variance is pulled back towards theta
    theta: float   # long-run variance level
    xi: float      # volatility of variance ("vol of vol")
    rho: float     # correlation between stock shocks and variance shocks

    def as_array(self):
        return np.array(astuple(self), dtype=float)

    def feller_ok(self):
        """Feller condition 2 kappa theta > xi^2: variance can never touch zero."""
        return bool(2.0 * self.kappa * self.theta > self.xi**2)


def heston_cf(u, T, p):
    """Characteristic function E[exp(i u X_T)] of X_T = ln(S_T / F_T) under Heston.

    F_T = S e^{(r-q)T} is the forward, so X_T has no r or q in it. Uses the
    'little Heston trap' form (Albrecher et al., 2007): it is algebraically the same as
    Heston's original formula but avoids a branch-cut jump in the complex log that
    makes the original formula wrong for long maturities.
    `u` may be complex (the Lewis formula evaluates it at u - i/2).
    """
    v0, kappa, theta, xi, rho = p.v0, p.kappa, p.theta, p.xi, p.rho
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    exp_dt = np.exp(-d * T)
    C = (kappa * theta / xi**2) * ((b - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
    D = ((b - d) / xi**2) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return np.exp(C + D * v0)


def bs_cf(u, T, sigma):
    """Characteristic function of ln(S_T/F_T) under Black-Scholes (used in tests)."""
    return np.exp(-0.5 * sigma**2 * T * (1j * u + u**2))


# Gauss-Legendre nodes on [-1, 1]; mapped onto [0, u_max] for the Lewis integral.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(1024)


def _u_max(T, p):
    """Upper integration limit for the Lewis integral.

    Two decay regimes matter: for small vol-of-vol the integrand falls like a Gaussian,
    exp(-v T u^2 / 2); for large u it falls only exponentially, roughly
    exp(-(v0 + kappa theta T) sqrt(1 - rho^2) u / xi). We integrate far enough for both
    to be negligible (about e^-35), capped to keep the node spacing fine.
    """
    v_eff = max(min(p.v0, p.theta), 1e-3)
    gaussian = np.sqrt(80.0 / (v_eff * T))
    exponential = 35.0 * p.xi / ((p.v0 + p.kappa * p.theta * T) * np.sqrt(1.0 - p.rho**2))
    return float(np.clip(max(gaussian, exponential), 50.0, 2000.0))


def lewis_call(F, K, T, r, cf):
    """Lewis (2001) formula for a European call, given the characteristic function of ln(S_T/F).

        C = e^{-rT} [ F - sqrt(F K)/pi * int_0^inf Re( e^{-i u k} cf(u - i/2) ) / (u^2 + 1/4) du ],
        k = ln(K / F).
    `cf` is a function of u only; K can be an array (one maturity at a time).
    """
    K = np.atleast_1d(np.asarray(K, float))
    u_max = cf.u_max if hasattr(cf, "u_max") else 200.0
    u = 0.5 * u_max * (_GL_X + 1.0)          # nodes mapped to [0, u_max]
    w = 0.5 * u_max * _GL_W
    phi = cf(u - 0.5j)                        # evaluated once, reused for every strike
    k = np.log(K / F)[:, None]
    integrand = np.real(np.exp(-1j * u[None, :] * k) * phi[None, :]) / (u**2 + 0.25)
    integral = integrand @ w
    return np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integral)


def heston_price(S, K, T, r, params, q=0.0, option="call"):
    """Heston price for one maturity T and one or many strikes K (call or put)."""
    F = S * np.exp((r - q) * T)

    def cf(u):
        return heston_cf(u, T, params)
    cf.u_max = _u_max(T, params)

    call = lewis_call(F, K, T, r, cf)
    if option == "call":
        out = call
    elif option == "put":
        out = call - np.exp(-r * T) * (F - np.asarray(K, float))   # put-call parity
    else:
        raise ValueError("option must be 'call' or 'put'")
    return out if np.ndim(K) else float(out[0])


def heston_price_chain(F, K, T, r, params, is_call):
    """Price a whole option chain (many maturities) given each row's forward F and rate r.

    Rows with the same maturity share one characteristic-function evaluation, which is
    what makes calibration fast enough. All inputs are 1-D arrays of the same length.
    """
    F, K, T, r = (np.asarray(x, float) for x in (F, K, T, r))
    is_call = np.asarray(is_call, bool)
    out = np.empty_like(K)
    for t in np.unique(T):
        m = T == t

        def cf(u, t=t):
            return heston_cf(u, t, params)
        cf.u_max = _u_max(t, params)

        Fm, rm = F[m][0], r[m][0]
        call = lewis_call(Fm, K[m], t, rm, cf)
        put = call - np.exp(-rm * t) * (Fm - K[m])
        out[m] = np.where(is_call[m], call, put)
    return out


def bs_price_via_lewis(S, K, T, r, sigma, q=0.0):
    """Black-Scholes call computed through the Lewis integral (a test of the integrator)."""
    F = S * np.exp((r - q) * T)

    def cf(u):
        return bs_cf(u, T, sigma)
    cf.u_max = float(np.clip(np.sqrt(80.0 / (sigma**2 * T)), 50.0, 2000.0))
    return lewis_call(F, K, T, r, cf)
