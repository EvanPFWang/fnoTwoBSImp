
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, Tuple
import numpy as np

from sqlalchemy import (
    create_engine, Column, Integer, Numeric, Boolean, String, ForeignKey, Index, MetaData
)
from sqlalchemy.orm import declarative_base, Session, relationship, Mapped, mapped_column
from sqlalchemy.dialects.mysql import TINYINT

#use MySQL naming convention
metadata = MetaData()
Base = declarative_base(metadata=metadata)

NUM = lambda: Numeric(18, 9, asdecimal=True)

class BlackScholesInputs(Base):
    __tablename__ = "BlackScholesInputs"
    CalculationId: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    StockPrice = Column(NUM(), nullable=False)
    StrikePrice = Column(NUM(), nullable=False)
    InterestRate = Column(NUM(), nullable=False)
    Volatility = Column(NUM(), nullable=False)
    TimeToExpiry = Column(NUM(), nullable=False)
    PurchaseCallPrice = Column(NUM(), nullable=True)
    PurchasePutPrice  = Column(NUM(), nullable=True)
    MinSpot = Column(NUM(), nullable=False)
    MaxSpot = Column(NUM(), nullable=False)
    MinVol  = Column(NUM(), nullable=False)
    MaxVol  = Column(NUM(), nullable=False)
    GridNSpot = Column(Integer, nullable=False, default=50)
    GridNVol  = Column(Integer, nullable=False, default=50)

    outputs = relationship("BlackScholesOutputs", back_populates="inputs", cascade="all, delete-orphan")

class BlackScholesOutputs(Base):
    __tablename__ = "BlackScholesOutputs"
    CalculationOutputId: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    VolatilityShock = Column(NUM(), nullable=False)
    StockPriceShock = Column(NUM(), nullable=False)
    OptionPrice     = Column(NUM(), nullable=False)
    IsCall          = Column(TINYINT(1), nullable=False)  #1 = call, 0 = put
    CalculationId   = Column(Integer, ForeignKey("BlackScholesInputs.CalculationId"), nullable=False, index=True)

    inputs = relationship("BlackScholesInputs", back_populates="outputs")

    __table_args__ = (
        Index("FK_BlackScholesInput_BlackScholesOutput_CalculationId_idx", "CalculationId"),
    )

def quant9(x: float) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)

def create_engine_from_url(url: str):
    #expect SQLAlchemy URL, mysql+pymysql://user:pass@host:3306/Evan_dev_db
    return create_engine(url, pool_pre_ping=True, future=True)

def create_all(engine):
    Base.metadata.create_all(engine)

def insert_calculation(
    engine,
    inputs: Dict[str, Any],
    S_grid: np.ndarray,
    V_grid: np.ndarray,
    call_surface: np.ndarray,
    put_surface: np.ndarray,
) -> Tuple[int, int]:
    """
    Insert one calculation row and the flattened output surfaces (both call & put).
    Returns (calculation_id, n_rows_inserted).
    """
    with Session(engine) as sess:
        bs_in = BlackScholesInputs(
            StockPrice=quant9(inputs["StockPrice"]),
            StrikePrice=quant9(inputs["StrikePrice"]),
            InterestRate=quant9(inputs["InterestRate"]),
            Volatility=quant9(inputs["Volatility"]),
            TimeToExpiry=quant9(inputs["TimeToExpiry"]),
            PurchaseCallPrice=quant9(inputs.get("PurchaseCallPrice", 0.0)) if inputs.get("PurchaseCallPrice") is not None else None,
            PurchasePutPrice=quant9(inputs.get("PurchasePutPrice", 0.0)) if inputs.get("PurchasePutPrice") is not None else None,
            MinSpot=quant9(inputs["MinSpot"]), MaxSpot=quant9(inputs["MaxSpot"]),
            MinVol=quant9(inputs["MinVol"]),   MaxVol=quant9(inputs["MaxVol"]),
            GridNSpot=int(inputs["GridNSpot"]), GridNVol=int(inputs["GridNVol"]),
        )
        sess.add(bs_in)
        sess.flush()  #get CalculationId
        calc_id = bs_in.CalculationId

        nV, nS = call_surface.shape
        rows = []
        for i in range(nV):
            for j in range(nS):
                rows.append(BlackScholesOutputs(
                    VolatilityShock=quant9(float(V_grid[i])),
                    StockPriceShock=quant9(float(S_grid[j])),
                    OptionPrice=quant9(float(call_surface[i, j])),
                    IsCall=1,
                    CalculationId=calc_id
                ))
                rows.append(BlackScholesOutputs(
                    VolatilityShock=quant9(float(V_grid[i])),
                    StockPriceShock=quant9(float(S_grid[j])),
                    OptionPrice=quant9(float(put_surface[i, j])),
                    IsCall=0,
                    CalculationId=calc_id
                ))
        sess.add_all(rows)
        sess.commit()
        return calc_id, len(rows)
