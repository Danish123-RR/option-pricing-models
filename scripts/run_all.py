"""Reproduce every figure and result in the README.

Run from the repo root:
    python scripts/run_all.py

Part 1 (theory checks) needs no data. Part 2 (market calibration) uses the most
recent snapshot in data/ (create one with scripts/fetch_spx_chain.py).
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `import src...` works when run as a script

from src.binomial import crr_price                       # noqa: E402
from src.black_scholes import bs_price, delta, gamma, vega, theta   # noqa: E402
from src.calibration import calibrate_heston, model_implied_vols    # noqa: E402
from src.heston import HestonParams, heston_price          # noqa: E402
from src.implied_vol import implied_vol_array              # noqa: E402
from src.market_data import latest_snapshot, prepare_quotes, quote_date_of   # noqa: E402
from src.monte_carlo import mc_gbm                         # noqa: E402
from src.plotting import plt, save, BLUE, ORANGE, AQUA, INK, MUTED, BLUES   # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20   # the textbook example used throughout


def fig_binomial_convergence():
    exact = bs_price(S, K, T, R, SIG)
    steps = np.arange(6, 301)
    prices = np.array([crr_price(S, K, T, R, SIG, steps=n) for n in steps])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    # Odd and even step counts approach from opposite sides (the strike sits on a node
    # only for even counts), so we draw them separately instead of as one zigzag.
    even = steps % 2 == 0
    ax.plot(steps[even], prices[even], color=BLUE, label="Even number of steps")
    ax.plot(steps[~even], prices[~even], color=ORANGE, label="Odd number of steps")
    ax.legend(loc="upper right")
    ax.axhline(exact, color=MUTED, linestyle="--", linewidth=1)
    ax.text(steps[-1], exact - 0.03, f"Black-Scholes {exact:.4f}", ha="right", va="top", color=MUTED, fontsize=9)
    ax.set(title="Binomial tree price converges to Black-Scholes", xlabel="Number of tree steps",
           ylabel="Call price")
    save(fig, FIG / "binomial_convergence.png")
    return {"bs": exact, "crr_1000": crr_price(S, K, T, R, SIG, steps=1000)}


def fig_mc_convergence():
    exact = bs_price(S, K, T, R, SIG)
    ns = np.unique(np.logspace(3, 6, 13).astype(int))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    out = {}
    for anti, color, label in ((False, ORANGE, "Plain"), (True, BLUE, "Antithetic")):
        res = [mc_gbm(S, K, T, R, SIG, n_paths=n, antithetic=anti, seed=11) for n in ns]
        est = np.array([x.price for x in res])
        se = np.array([x.std_error for x in res])
        ax.fill_between(ns, est - 1.96 * se, est + 1.96 * se, color=color, alpha=0.15, linewidth=0)
        ax.plot(ns, est, color=color, marker="o", markersize=4, label=f"{label} (95% band)")
        out[label.lower()] = {"n": int(ns[-1]), "price": float(est[-1]), "std_error": float(se[-1])}
    ax.axhline(exact, color=MUTED, linestyle="--", linewidth=1)
    ax.text(ns[0], exact + 0.05, f"Black-Scholes {exact:.4f}", color=MUTED, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set(title="Monte Carlo estimate tightens as paths grow", xlabel="Number of simulated paths (log scale)",
           ylabel="Call price")
    ax.legend(loc="upper right")
    save(fig, FIG / "mc_convergence.png")
    return out


def fig_greeks():
    spots = np.linspace(60, 140, 300)
    panels = [("Delta", delta(spots, K, T, R, SIG), "shares per option"),
              ("Gamma", gamma(spots, K, T, R, SIG), "change in delta per 1.00 move"),
              ("Vega", vega(spots, K, T, R, SIG) / 100, "price change per 1 vol point"),
              ("Theta", theta(spots, K, T, R, SIG) / 365, "price change per day")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True)
    for ax, (name, values, unit) in zip(axes.flat, panels):
        ax.plot(spots, values, color=BLUE)
        ax.axvline(K, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(name)
        ax.set_ylabel(unit, fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Spot price (strike = 100)")
    fig.suptitle("Black-Scholes Greeks of a 1-year call, strike 100", x=0.01, ha="left", color=INK, weight="bold")
    fig.tight_layout()
    save(fig, FIG / "greeks.png")


def fig_heston_parameter_effects():
    """How rho and xi shape the smile: rho tilts it, xi curves it."""
    strikes = np.linspace(75, 125, 41)
    base = dict(v0=0.04, kappa=2.0, theta=0.04)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (label, key, values) in zip(axes, (("rho", "rho", (-0.8, 0.0, 0.5)), ("xi", "xi", (0.2, 0.6, 1.2)))):
        for v, color in zip(values, (BLUE, ORANGE, AQUA)):
            kw = {"xi": 0.6, "rho": -0.5, **{key: v}}
            p = HestonParams(**base, **kw)
            prices = heston_price(100, strikes, 0.5, 0.0, p)
            ivs = implied_vol_array(prices, 100, strikes, 0.5, 0.0)
            ax.plot(strikes, ivs * 100, color=color, label=f"{label} = {v}")
        ax.set_title(f"Changing {label}")
        ax.set_xlabel("Strike (spot = 100, 6 months)")
        ax.legend()
    axes[0].set_ylabel("Implied volatility (%)")
    fig.suptitle("Heston: rho tilts the smile, vol-of-vol curves it", x=0.01, ha="left", color=INK, weight="bold")
    fig.tight_layout()
    save(fig, FIG / "heston_parameter_effects.png")


def market_part():
    df, market = latest_snapshot()
    quotes = prepare_quotes(df, market)
    quote_day = quote_date_of(market).isoformat()   # trading day the prices come from
    quotes.to_csv(RES / "spx_quotes_clean.csv", index=False)
    expiries = sorted(quotes["expiry"].unique(), key=lambda e: quotes.loc[quotes.expiry == e, "T"].iloc[0])

    # Market smile, one line per maturity (light = short, dark = long).
    fig, ax = plt.subplots(figsize=(8, 4.4))
    idx = np.linspace(1, len(BLUES) - 1, len(expiries)).round().astype(int)
    colors = [BLUES[i] for i in idx]   # evenly spread from light to dark
    for e, c in zip(expiries, colors):
        g = quotes[quotes.expiry == e]
        ax.plot(g.log_moneyness, g.iv * 100, color=c, linewidth=1.6, label=f"{e} ({int(g.days.iloc[0])}d)")
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set(title=f"SPX implied volatility skew, closing quotes {quote_day}",
           xlabel="Log-moneyness ln(K / F)   (left = out-of-the-money puts)", ylabel="Implied volatility (%)")
    ax.legend(title="Expiry (days to expiry)", fontsize=8, title_fontsize=8, ncol=2, loc="upper right")
    save(fig, FIG / "spx_smile.png")

    t0 = time.time()
    res = calibrate_heston(quotes)
    seconds = time.time() - t0
    quotes["iv_heston"] = model_implied_vols(res.params, quotes.F, quotes.K, quotes["T"], quotes.r, quotes.is_call)
    quotes.to_csv(RES / "spx_quotes_clean.csv", index=False)

    # Fit per maturity: market dots vs Heston line.
    n = len(expiries)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, e in zip(axes, expiries):
        g = quotes[quotes.expiry == e]
        ax.scatter(g.log_moneyness, g.iv * 100, s=16, facecolors="none", edgecolors=BLUE,
                   linewidths=1.0, label="Market", zorder=3)
        ax.plot(g.log_moneyness, g.iv_heston * 100, color=ORANGE, label="Heston fit", zorder=4)
        ax.set_title(f"{int(g.days.iloc[0])} days", fontsize=10)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("Implied vol (%)")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Heston calibrated to SPX options ({res.n_options} quotes): RMSE {res.rmse_iv:.2f} vol points",
                 x=0.01, ha="left", color=INK, weight="bold")
    fig.supxlabel("Log-moneyness ln(K / F)", color=MUTED, fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "heston_fit.png")

    per_expiry = (quotes.assign(err=(quotes.iv_heston - quotes.iv) * 100)
                  .groupby("expiry")
                  .agg(days=("days", "first"), n=("K", "size"), F=("F", "first"),
                       q_implied=("q_implied", "first"),
                       rmse_vol_pts=("err", lambda e: float(np.sqrt(np.mean(e**2)))))
                  .sort_values("days"))
    per_expiry.to_csv(RES / "fit_by_expiry.csv")
    summary = {
        "snapshot": market,
        "quote_date": quote_day,
        "price_source": "bid-ask mids" if (quotes["price_source"] == "mid").all() else "recent last trades",
        "n_quotes": res.n_options,
        "n_expiries": n,
        "params": {k: round(v, 6) for k, v in vars(res.params).items()},
        "feller_condition_met": res.params.feller_ok(),
        "rmse_iv_vol_points": round(res.rmse_iv, 4),
        "max_abs_iv_error_vol_points": round(res.max_abs_iv_error, 4),
        "optimizer_success": res.success,
        "calibration_seconds": round(seconds, 1),
        "starts": res.starts,
    }
    (RES / "calibration.json").write_text(json.dumps(summary, indent=2, default=str).replace("NaN", "null"))
    update_readme(summary, per_expiry)
    return summary


def update_readme(summary, per_expiry):
    """Write the latest calibration results between the RESULTS markers in README.md."""
    p = summary["params"]
    snap = summary["snapshot"]
    spot = snap.get("spot")
    if spot is not None and spot == spot:          # spot == spot is False for NaN
        spot_text = f"{spot:,.2f}"
    else:
        spot_text = f"≈{snap['spot_estimated']:,.0f} (estimated from the nearest forward)"
    lines = [
        f"**Latest calibration** (SPX quotes at the {summary['quote_date']} close, spot {spot_text}, "
        f"{summary['n_quotes']} out-of-the-money quotes across {summary['n_expiries']} expiries, "
        f"prices from {summary['price_source']}):",
        "",
        "| v0 | kappa | theta | xi (vol of vol) | rho | Fit RMSE | Feller 2κθ > ξ² |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| {p['v0']:.4f} | {p['kappa']:.3f} | {p['theta']:.4f} | {p['xi']:.3f} | {p['rho']:.3f} "
        f"| {summary['rmse_iv_vol_points']:.2f} vol pts | {'met' if summary['feller_condition_met'] else 'not met'} |",
        "",
        f"Spot volatility √v0 = {100 * p['v0'] ** 0.5:.1f}%, long-run volatility √θ = {100 * p['theta'] ** 0.5:.1f}%.",
        "",
        "| Days to expiry | Quotes | Forward (from put-call parity) | Fit RMSE (vol pts) |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in per_expiry.iterrows():
        lines.append(f"| {int(row.days)} | {int(row.n)} | {row.F:,.1f} | {row.rmse_vol_pts:.2f} |")
    lines += ["", "![SPX implied volatility skew](figures/spx_smile.png)", "",
              "![Heston fit to SPX](figures/heston_fit.png)"]
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(head + start + "\n" + "\n".join(lines) + "\n" + end + tail, encoding="utf-8")


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    theory = {"binomial": fig_binomial_convergence(), "monte_carlo": fig_mc_convergence()}
    fig_greeks()
    fig_heston_parameter_effects()
    (RES / "theory_checks.json").write_text(json.dumps(theory, indent=2))
    print("Theory figures written to figures/")
    try:
        summary = market_part()
    except FileNotFoundError as err:
        print(err)
        return
    print(json.dumps({k: summary[k] for k in ("n_quotes", "params", "rmse_iv_vol_points", "feller_condition_met")}, indent=2))


if __name__ == "__main__":
    main()
