from __future__ import annotations
import math
from .bs_pricing import price, greeks

def implied_vol(option_type: str, s: float, k: float, r: float, q: float, t: float, market_price: float,
                initial_sigma: float = 0.2, tol: float = 1e-8, max_iter: int = 100, 
                lo: float = 1e-6, hi: float = 5.0) -> float:
    """Find implied volatility using Newton-Raphson with bisection fallback."""
    sigma = max(initial_sigma, lo)
    for _ in range(max_iter):
        model_price = price(option_type, s, k, r, q, t, sigma)
        diff = model_price - market_price
        if abs(diff) < tol:
            return max(sigma, lo)
        v = greeks(option_type, s, k, r, q, t, sigma).vega
        if v <= 1e-12:
            break
        sigma_next = sigma - diff / v
        if sigma_next <= lo or sigma_next >= hi or math.isnan(sigma_next):
            break
        sigma = sigma_next

    # Bisection
    lo_p = price(option_type, s, k, r, q, t, lo) - market_price
    hi_p = price(option_type, s, k, r, q, t, hi) - market_price
    if lo_p * hi_p > 0:
        # If prices at bounds don't bracket the root, expand hi or return best guess
        return max(min(sigma, hi), lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mp = price(option_type, s, k, r, q, t, mid) - market_price
        if abs(mp) < tol:
            return max(mid, lo)
        if lo_p * mp <= 0:
            hi = mid
            hi_p = mp
        else:
            lo = mid
            lo_p = mp
    return 0.5 * (lo + hi)
