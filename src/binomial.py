"""Cox-Ross-Rubinstein (CRR) binomial tree.

The stock moves up by u = e^{sigma sqrt(dt)} or down by d = 1/u each step.
The option is priced by working backwards from expiry, at each node taking the
discounted risk-neutral average of the two children:

    V = e^{-r dt} [ p V_up + (1 - p) V_down ],   p = (e^{(r-q) dt} - d) / (u - d)

As the number of steps grows the European price converges to Black-Scholes.
American options are handled by comparing with immediate exercise at every node.
"""
import numpy as np


def crr_price(S, K, T, r, sigma, q=0.0, option="call", steps=500, american=False):
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)      # risk-neutral up probability
    if not 0.0 < p < 1.0:
        raise ValueError("steps too small for these inputs (p outside (0, 1))")
    disc = np.exp(-r * dt)

    # Stock prices at expiry: S u^j d^(steps-j) for j = 0..steps up-moves.
    j = np.arange(steps + 1)
    stock = S * u**j * d**(steps - j)
    payoff = np.maximum(stock - K, 0.0) if option == "call" else np.maximum(K - stock, 0.0)

    values = payoff
    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            stock = S * u**np.arange(n + 1) * d**(n - np.arange(n + 1))
            exercise = np.maximum(stock - K, 0.0) if option == "call" else np.maximum(K - stock, 0.0)
            values = np.maximum(values, exercise)
    return float(values[0])
