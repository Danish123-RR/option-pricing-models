"""Calibrate the Heston model to market option quotes.

Goal: find the 5 parameters (v0, kappa, theta, xi, rho) whose model prices best
match the market. We minimise

    sum_i w_i [ (model_price_i - market_price_i) / vega_i ]^2

where w_i gives every expiry the same total weight.

Dividing a price error by the option's Black-Scholes vega turns it into
(approximately) an implied-volatility error, so every option counts on the same
scale. Without it, a 0.5-point price miss would count the same on a cheap far
out-of-the-money put (where it is a huge volatility error) as on an expensive
at-the-money option (where it is a tiny one).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from src.heston import HestonParams, heston_price_chain
from src.implied_vol import implied_vol_array

# Parameter bounds: v0, kappa, theta, xi, rho
LOWER = np.array([1e-4, 1e-2, 1e-4, 1e-2, -0.999])
UPPER = np.array([1.0, 20.0, 1.0, 5.0, 0.999])


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_iv: float              # root-mean-square implied-vol error, in vol points (1.0 = 1%)
    max_abs_iv_error: float
    n_options: int
    cost: float
    success: bool
    starts: list = field(default_factory=list)   # (start point, final cost) for every start


def _residuals(x, F, K, T, r, is_call, mid, vega_w, expiry_w):
    model = heston_price_chain(F, K, T, r, HestonParams(*x), is_call)
    return expiry_w * (model - mid) / vega_w


def model_implied_vols(params, F, K, T, r, is_call):
    """Price every quote with Heston, then convert each price back to a Black-Scholes vol."""
    prices = heston_price_chain(F, K, T, r, params, is_call)
    spot_equiv = F * np.exp(-r * T)          # with q = 0 this spot has forward F
    opts = np.where(is_call, "call", "put")
    return implied_vol_array(prices, spot_equiv, K, T, r, 0.0, opts)


def calibrate_heston(quotes, starts=None, vega_floor=None):
    """Fit Heston to a quotes table.

    `quotes` needs columns: F (forward), K, T, r, is_call, mid (market price),
    vega (Black-Scholes vega at the market implied vol) and iv (market implied vol).
    Several starting points are tried because the fit has local minima; the best wins.
    """
    F, K, T, r = (quotes[c].to_numpy(float) for c in ("F", "K", "T", "r"))
    is_call = quotes["is_call"].to_numpy(bool)
    mid = quotes["mid"].to_numpy(float)
    vega = quotes["vega"].to_numpy(float)
    floor = vega_floor if vega_floor is not None else 0.05 * np.median(vega)
    vega_w = np.maximum(vega, floor)   # stops near-zero vegas from dominating the fit
    # Equal weight per expiry: monthly expiries list strikes every 5 points and weeklies
    # every 25, so without this the densely quoted expiries would dominate the fit.
    _, inverse, counts = np.unique(T, return_inverse=True, return_counts=True)
    expiry_w = np.sqrt(counts.mean() / counts[inverse])

    atm_var = float(np.median(quotes.loc[(quotes["K"] / quotes["F"] - 1).abs() < 0.03, "iv"]) ** 2) \
        if ((quotes["K"] / quotes["F"] - 1).abs() < 0.03).any() else float(np.median(quotes["iv"]) ** 2)
    if starts is None:
        starts = [
            [atm_var, 2.0, atm_var, 0.5, -0.7],
            [atm_var, 5.0, 1.5 * atm_var, 1.0, -0.8],
            [atm_var, 1.0, atm_var, 0.3, -0.5],
            [0.5 * atm_var, 3.0, 2.0 * atm_var, 1.5, -0.9],
        ]

    best, history = None, []
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), LOWER + 1e-6, UPPER - 1e-6)
        fit = least_squares(_residuals, x0, bounds=(LOWER, UPPER), method="trf",
                            args=(F, K, T, r, is_call, mid, vega_w, expiry_w),
                            x_scale="jac", max_nfev=400)
        history.append((x0.round(4).tolist(), float(fit.cost)))
        if best is None or fit.cost < best.cost:
            best = fit

    params = HestonParams(*best.x)
    iv_model = model_implied_vols(params, F, K, T, r, is_call)
    err = (iv_model - quotes["iv"].to_numpy(float)) * 100.0
    err = err[np.isfinite(err)]
    return CalibrationResult(
        params=params,
        rmse_iv=float(np.sqrt(np.mean(err**2))),
        max_abs_iv_error=float(np.max(np.abs(err))),
        n_options=len(quotes),
        cost=float(best.cost),
        success=bool(best.success),
        starts=history,
    )
