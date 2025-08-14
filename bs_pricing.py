
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Dict
try:
    from scipy.stats import norm
    _have_scipy = True
except Exception:
    _have_scipy = False


"""
Vectorized Black–Scholes pricing utilities.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal
from datetime import datetime

#Import pricing utilities
from db import create_engine_from_url, create_all, insert_calculation

#page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Black-Scholes Option Pricer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


ArrayLike = Union[float, np.ndarray]
"""
Enhanced Black-Scholes pricing utilities with Greeks calculation.
"""

#SciPy is used for standard normal CDF and PDF for numerical stability
try:
    from scipy.stats import norm

    _have_scipy = True
except Exception:
    _have_scipy = False

ArrayLike = Union[float, np.ndarray]

#sciPy is used for standard normal CDF for numerical stability
def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Cumulative distribution function for standard normal."""
    if _have_scipy:
        return norm.cdf(x)
    #Fallback: erf-based approximation if SciPy is not installed
    from math import erf, sqrt
    vec_erf = np.vectorize(erf)
    return 0.5 * (1.0 + vec_erf(x / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Probability density function for standard normal."""
    if _have_scipy:
        return norm.pdf(x)
    #Fallback calculation
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def black_scholes_call_put(
        S: ArrayLike,
        K: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
        T: ArrayLike,
        q: ArrayLike = 0.0,
        compute_greeks: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """
    Compute European call and put prices using Black-Scholes formula.

    Parameters
    ----------
    S : Spot price (scalar or array)
    K : Strike price
    r : Risk-free rate (annualized, in decimals)
    sigma : Volatility (annualized, in decimals)
    T : Time to expiry (in years)
    q : Dividend yield (annualized, default 0)
    compute_greeks : If True, also return Greeks

    Returns
    -------
    If compute_greeks=False: (call_prices, put_prices)
    If compute_greeks=True: (call_prices, put_prices, greeks_dict)
    """
    #Convert to numpy arrays
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    q = np.asarray(q, dtype=float)

    #Broadcast to common shape
    S, K, r, sigma, T, q = np.broadcast_arrays(S, K, r, sigma, T, q)

    #Handle edge cases
    eps = 1e-12
    sqrt_T = np.sqrt(T + eps)

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log((S + eps) / (K + eps)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T + eps)
        d2 = d1 - sigma * sqrt_T

    #Calculate CDFs
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    Nmd1 = _norm_cdf(-d1)
    Nmd2 = _norm_cdf(-d2)

    #Discount factors
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    #Option prices
    call = S * disc_q * Nd1 - K * disc_r * Nd2
    put = K * disc_r * Nmd2 - S * disc_q * Nmd1

    #Handle T=0 case (intrinsic value)
    intrinsic_call = np.maximum(S - K, 0.0)
    intrinsic_put = np.maximum(K - S, 0.0)
    mask_T0 = (T <= eps)

    if np.any(mask_T0):
        call = np.where(mask_T0, intrinsic_call, call)
        put = np.where(mask_T0, intrinsic_put, put)

    if not compute_greeks:
        return call, put

    #Calculate Greeks
    nd1 = _norm_pdf(d1)
    nd2 = _norm_pdf(d2)

    greeks = {
        #Delta
        'call_delta': disc_q * Nd1,
        'put_delta': -disc_q * Nmd1,

        #Gamma (same for calls and puts)
        'gamma': disc_q * nd1 / (S * sigma * sqrt_T + eps),

        #Vega (same for calls and puts, in % terms)
        'vega': S * disc_q * nd1 * sqrt_T / 100,

        #Theta (per day)
        'call_theta': (-S * disc_q * nd1 * sigma / (2 * sqrt_T)
                       - r * K * disc_r * Nd2
                       + q * S * disc_q * Nd1) / 365,
        'put_theta': (-S * disc_q * nd1 * sigma / (2 * sqrt_T)
                      + r * K * disc_r * Nmd2
                      - q * S * disc_q * Nmd1) / 365,

        #Rho (per 1% change)
        'call_rho': K * T * disc_r * Nd2 / 100,
        'put_rho': -K * T * disc_r * Nmd2 / 100
    }

    #Handle edge cases for Greeks
    if np.any(mask_T0):
        for key in greeks:
            greeks[key] = np.where(mask_T0, 0.0, greeks[key])

    return call, put, greeks


def build_grids(
        min_S: float,
        max_S: float,
        min_vol: float,
        max_vol: float,
        nS: int,
        nV: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2D grids for spot price and volatility.

    Returns:
        SS: 2D array of spot prices (shape: nV x nS)
        VV: 2D array of volatilities (shape: nV x nS)
        S_grid: 1D array of spot prices
        V_grid: 1D array of volatilities
    """
    S_grid = np.linspace(min_S, max_S, nS)
    V_grid = np.linspace(min_vol, max_vol, nV)
    SS, VV = np.meshgrid(S_grid, V_grid)  #shape (nV, nS)
    return SS, VV, S_grid, V_grid


def pnl_surface(
        option_price_surface: np.ndarray,
        purchase_price: float
) -> np.ndarray:
    """Calculate P&L surface from option prices and purchase price."""
    return option_price_surface - float(purchase_price)


def implied_volatility(
        price: float,
        S: float,
        K: float,
        r: float,
        T: float,
        option_type: str = 'call',
        q: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 100
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.

    Parameters
    ----------
    price : Market price of the option
    S : Spot price
    K : Strike price
    r : Risk-free rate
    T : Time to expiry
    option_type : 'call' or 'put'
    q : Dividend yield
    tol : Tolerance for convergence
    max_iter : Maximum iterations

    Returns
    -------
    Implied volatility or None if not converged
    """
    if T <= 0:
        return None

    #Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * (price / S)
    sigma = max(0.01, min(sigma, 5.0))  #Bound initial guess

    for _ in range(max_iter):
        call_price, put_price = black_scholes_call_put(S, K, r, sigma, T, q)

        if option_type.lower() == 'call':
            model_price = call_price
        else:
            model_price = put_price

        diff = model_price - price

        if abs(diff) < tol:
            return float(sigma)

        #Calculate vega for Newton-Raphson update
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)

        if abs(vega) < 1e-10:  #Avoid division by very small number
            break

        #Newton-Raphson update
        sigma = sigma - diff / vega
        sigma = max(0.001, min(sigma, 10.0))  #Keep in reasonable bounds

    return None  #Failed to converge
