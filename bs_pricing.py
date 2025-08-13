
"""
Vectorized Black–Scholes pricing utilities.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union

#sciPy is used for standard normal CDF for numerical stability
try:
    from scipy.stats import norm
    _have_scipy = True
except Exception:
    _have_scipy = False

ArrayLike = Union[float, np.ndarray]

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    if _have_scipy:
        return norm.cdf(x)
    #fallback: erf-based approximation if SciPy is not installed
    #N(x) = 0.5 * [1 + erf(x / sqrt(2))]
    from math import erf, sqrt
    vec_erf = np.vectorize(erf)
    return 0.5 * (1.0 + vec_erf(x / np.sqrt(2.0)))

def black_scholes_call_put(
    S: ArrayLike, K: ArrayLike, r: ArrayLike, sigma: ArrayLike, T: ArrayLike, q: ArrayLike = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute European call and put prices with Black–Scholes w/ broadcasting over array inputs (S grid and sigma grid).

    Parameters
    ----------
    S: spot price (can be scalar or array)
    K: strike
    r: risk-free rate (annualized, in decimals)
    sigma: volatility (annualized, in decimals)
    T: time to expiry (in years)
    q: dividend yield (annualized, default 0)

    Returns
    -------
    (call_prices, put_prices) as numpy arrays broadcast to a common shape
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    q = np.asarray(q, dtype=float)

    #nroadcast to common shape
    S, K, r, sigma, T, q = np.broadcast_arrays(S, K, r, sigma, T, q)

    #handle T==0 or sigma==0 (intrinsic value limit)
    eps = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log((S + eps) / (K + eps)) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + eps)
        d2 = d1 - sigma * np.sqrt(T)

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    Nmd1 = _norm_cdf(-d1)
    Nmd2 = _norm_cdf(-d2)

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    call = S * disc_q * Nd1 - K * disc_r * Nd2
    put  = K * disc_r * Nmd2 - S * disc_q * Nmd1

    #fix up cases where T==0 explicitly to intrinsic values
    intrinsic_call = np.maximum(S - K, 0.0)
    intrinsic_put  = np.maximum(K - S, 0.0)
    mask_T0 = (T <= eps)
    if np.any(mask_T0):
        call = np.where(mask_T0, intrinsic_call, call)
        put  = np.where(mask_T0, intrinsic_put,  put)

    return call, put

def build_grids(min_S: float, max_S: float, min_vol: float, max_vol: float, nS: int, nV: int):
    S_grid = np.linspace(min_S, max_S, nS)
    V_grid = np.linspace(min_vol, max_vol, nV)
    SS, VV = np.meshgrid(S_grid, V_grid)  #shape (nV, nS)
    return SS, VV, S_grid, V_grid

def pnl_surface(option_price_surface: np.ndarray, purchase_price: float) -> np.ndarray:
    return option_price_surface - float(purchase_price)
