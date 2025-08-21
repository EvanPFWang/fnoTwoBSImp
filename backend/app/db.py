from __future__ import annotations
import os
from typing import Dict, Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

def _db_url() -> str:
    user = os.getenv("MYSQL_USER", "bs_user")
    pwd = os.getenv("MYSQL_PASSWORD", "bs_password123")
    host = os.getenv("MYSQL_HOST", "mysql")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    db = os.getenv("MYSQL_DATABASE", "blackscholes")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"

_ENGINE: Engine | None = None

def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(_db_url(), pool_pre_ping=True, pool_recycle=3600)
    return _ENGINE

def save_calculation(record: Dict[str, Any]) -> int:
    sql = text("""        INSERT INTO calculations
    (s,k,r,q,t,sigma,option_type,price,delta,gamma,theta,vega,rho,implied_vol,market_price)
    VALUES
    (:s,:k,:r,:q,:t,:sigma,:option_type,:price,:delta,:gamma,:theta,:vega,:rho,:implied_vol,:market_price)
    """)
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(sql, record)
        return res.lastrowid if hasattr(res, "lastrowid") else 0

def list_calculations(limit: int = 50) -> List[Dict[str, Any]]:
    sql = text("""SELECT * FROM calculations ORDER BY created_at DESC, id DESC LIMIT :limit""")
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()
        return [dict(r) for r in rows]
