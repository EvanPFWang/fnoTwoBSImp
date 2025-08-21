from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
from .bs_pricing import price, greeks
from .error_minimization import implied_vol

OptionType = Literal["call", "put"]

@dataclass
class BSInput:
    s: float
    k: float
    r: float
    q: float
    t: float
    option_type: OptionType
    sigma: Optional[float] = None
    market_price: Optional[float] = None

def compute(payload: BSInput) -> Dict[str, Any]:
    if payload.sigma is not None:
        p = price(payload.option_type, payload.s, payload.k, payload.r, payload.q, payload.t, payload.sigma)
        g = greeks(payload.option_type, payload.s, payload.k, payload.r, payload.q, payload.t, payload.sigma)
        return {"price": p, "greeks": g.__dict__, "implied_vol": None}
    elif payload.market_price is not None:
        iv = implied_vol(payload.option_type, payload.s, payload.k, payload.r, payload.q, payload.t, payload.market_price)
        p = price(payload.option_type, payload.s, payload.k, payload.r, payload.q, payload.t, iv)
        g = greeks(payload.option_type, payload.s, payload.k, payload.r, payload.q, payload.t, iv)
        return {"price": p, "greeks": g.__dict__, "implied_vol": iv}
    else:
        raise ValueError("Provide either sigma or market_price")
