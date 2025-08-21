from __future__ import annotations
import os
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field
import numpy as np

from .blackscholes_unified import compute, BSInput, OptionType
from .bs_pricing import price, greeks
from .db import save_calculation, list_calculations

class PriceRequest(BaseModel):
    s: float
    k: float
    r: float
    q: float = 0.0
    t: float = Field(..., description="Time to expiry in years")
    sigma: float
    option_type: Literal["call", "put"]

    @field_validator("option_type")
    @classmethod
    def check_ot(cls, v: str) -> str:
        v = v.lower()
        if v not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return v

class IVRequest(BaseModel):
    s: float
    k: float
    r: float
    q: float = 0.0
    t: float
    market_price: float
    option_type: Literal["call", "put"]

    @field_validator("option_type")
    @classmethod
    def check_ot(cls, v: str) -> str:
        v = v.lower()
        if v not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return v

class CurveRequest(BaseModel):
    k: float
    r: float
    q: float = 0.0
    t: float
    sigma: float
    option_type: Literal["call","put"]
    s_min: float = 10.0
    s_max: float = 200.0
    steps: int = 50

class PriceResponse(BaseModel):
    price: float
    greeks: Dict[str, float]
    implied_vol: Optional[float] = None

class CurveResponse(BaseModel):
    s: List[float]
    price: List[float]

app = FastAPI(title="Black‑Scholes API", version="1.0.0")

# CORS
origins = [o.strip() for o in os.getenv("CORS_ORIGINS","http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/api/price", response_model=PriceResponse)
def api_price(req: PriceRequest):
    data = BSInput(**req.dict())
    out = compute(data)
    # persist
    record = {
        "s": req.s, "k": req.k, "r": req.r, "q": req.q, "t": req.t,
        "sigma": req.sigma, "option_type": req.option_type,
        "price": out["price"], **out["greeks"], "implied_vol": None, "market_price": None
    }
    try:
        save_calculation(record)
    except Exception as e:
        # do not fail the API if DB is down in dev
        pass
    return out

@app.post("/api/implied-vol", response_model=PriceResponse)
def api_iv(req: IVRequest):
    data = BSInput(**req.dict())
    out = compute(data)
    # persist
    record = {
        "s": req.s, "k": req.k, "r": req.r, "q": req.q, "t": req.t,
        "sigma": out["implied_vol"], "option_type": req.option_type,
        "price": out["price"], **out["greeks"], "implied_vol": out["implied_vol"], "market_price": req.market_price
    }
    try:
        save_calculation(record)
    except Exception:
        pass
    return out

@app.post("/api/curve", response_model=CurveResponse)
def api_curve(req: CurveRequest):
    s_vals = np.linspace(req.s_min, req.s_max, req.steps).tolist()
    p_vals = [price(req.option_type, float(s), req.k, req.r, req.q, req.t, req.sigma) for s in s_vals]
    return {"s": s_vals, "price": p_vals}

@app.get("/api/calculations")
def api_list(limit: int = 50):
    try:
        return {"items": list_calculations(limit=limit)}
    except Exception:
        return {"items": []}

class SaveRecord(BaseModel):
    s: float; k: float; r: float; q: float; t: float
    sigma: float | None = None
    option_type: Literal["call","put"]
    price: float
    delta: float; gamma: float; theta: float; vega: float; rho: float
    implied_vol: float | None = None
    market_price: float | None = None

@app.post("/api/calculations/save")
def api_save(rec: SaveRecord):
    try:
        rid = save_calculation(rec.dict())
        return {"id": rid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
