"""Black–Scholes pricing and Greeks with dividend yield q.
This module intentionally avoids SciPy and uses closed forms.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _norm_cdf(x: float) -> float:
    # Abramowitz-Stegun: CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _d1(s: float, k: float, r: float, q: float, t: float, sigma: float) -> float:
    if sigma <= 0 or t <= 0 or s <= 0 or k <= 0:
        return float('nan')
    return (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))

def _d2(d1: float, sigma: float, t: float) -> float:
    return d1 - sigma * math.sqrt(t)

def price(option_type: str, s: float, k: float, r: float, q: float, t: float, sigma: float) -> float:
    option_type = option_type.lower()
    d1 = _d1(s, k, r, q, t, sigma)
    d2 = _d2(d1, sigma, t)
    if option_type == "call":
        return s * math.exp(-q*t) * _norm_cdf(d1) - k * math.exp(-r*t) * _norm_cdf(d2)
    elif option_type == "put":
        return k * math.exp(-r*t) * _norm_cdf(-d2) - s * math.exp(-q*t) * _norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

def greeks(option_type: str, s: float, k: float, r: float, q: float, t: float, sigma: float) -> Greeks:
    option_type = option_type.lower()
    d1 = _d1(s, k, r, q, t, sigma)
    d2 = _d2(d1, sigma, t)
    nd1 = _norm_pdf(d1)
    sqrt_t = math.sqrt(t)
    disc_r = math.exp(-r*t)
    disc_q = math.exp(-q*t)

    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (- (s * disc_q * nd1 * sigma) / (2 * sqrt_t)
                 - r * k * disc_r * _norm_cdf(d2)
                 + q * s * disc_q * _norm_cdf(d1))
        rho = k * t * disc_r * _norm_cdf(d2)
    elif option_type == "put":
        delta = -disc_q * _norm_cdf(-d1)
        theta = (- (s * disc_q * nd1 * sigma) / (2 * sqrt_t)
                 + r * k * disc_r * _norm_cdf(-d2)
                 - q * s * disc_q * _norm_cdf(-d1))
        rho = -k * t * disc_r * _norm_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = (disc_q * nd1) / (s * sigma * sqrt_t)
    vega = s * disc_q * nd1 * sqrt_t
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
