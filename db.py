from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime

"""
Enhanced Database Module for Black-Scholes Application
Optimized for ML tasks with efficient data retrieval and storage
"""


from sqlalchemy import (
    create_engine, Column, Integer, Numeric, Boolean, String,
    ForeignKey, Index, MetaData, DateTime, Float, select, and_
)
from sqlalchemy.orm import declarative_base, Session, relationship, Mapped, mapped_column
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.pool import QueuePool

#MySQL naming convention
metadata = MetaData()
Base = declarative_base(metadata=metadata)

#prec numeric type
NUM = lambda: Numeric(18, 9, asdecimal=True)


class BlackScholesInputs(Base):
    """
    Store input parameters for Black-Scholes calculations.
    Optimized with indexes for ML data retrieval.
    """
    __tablename__ = "BlackScholesInputs"

    #Primary key
    CalculationId: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #core Black-Scholes parameters
    StockPrice = Column(NUM(), nullable=False, index=True)
    StrikePrice = Column(NUM(), nullable=False, index=True)
    InterestRate = Column(NUM(), nullable=False)
    Volatility = Column(NUM(), nullable=False, index=True)
    TimeToExpiry = Column(NUM(), nullable=False, index=True)

    #purchase prices for P&L calculation
    PurchaseCallPrice = Column(NUM(), nullable=True)
    PurchasePutPrice = Column(NUM(), nullable=True)

    #grid param
    MinSpot = Column(NUM(), nullable=False)
    MaxSpot = Column(NUM(), nullable=False)
    MinVol = Column(NUM(), nullable=False)
    MaxVol = Column(NUM(), nullable=False)
    GridNSpot = Column(Integer, nullable=False, default=50)
    GridNVol = Column(Integer, nullable=False, default=50)

    #metadata for ML tracking
    CreatedAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    SessionId = Column(String(100), nullable=True, index=True)
    UserId = Column(String(100), nullable=True, index=True)

    #relationships
    outputs = relationship("BlackScholesOutputs", back_populates="inputs", cascade="all, delete-orphan")

    #composite indexes for ML queries
    __table_args__ = (
        Index("idx_bs_inputs_params", "StockPrice", "StrikePrice", "Volatility", "TimeToExpiry"),
        Index("idx_bs_inputs_session", "SessionId", "CreatedAt"),
        Index("idx_bs_inputs_user", "UserId", "CreatedAt"),
    )


class BlackScholesOutputs(Base):
    """
    Store output grid results for Black-Scholes calculations.
    Optimized for bulk retrieval and ML training.
    """
    __tablename__ = "BlackScholesOutputs"

    #Primary key
    CalculationOutputId: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #grid coords
    VolatilityShock = Column(NUM(), nullable=False, index=True)
    StockPriceShock = Column(NUM(), nullable=False, index=True)

    #option price at this grid point
    OptionPrice = Column(NUM(), nullable=False)

    #option type: 1 = call, 0 = put
    IsCall = Column(TINYINT(1), nullable=False, index=True)

    #foreign key to inputs
    CalculationId = Column(Integer, ForeignKey("BlackScholesInputs.CalculationId"), nullable=False)

    #felationships
    inputs = relationship("BlackScholesInputs", back_populates="outputs")

    #indexes for efficient querying
    __table_args__ = (
        Index("FK_BlackScholesInput_BlackScholesOutput_CalculationId_idx", "CalculationId"),
        Index("idx_bs_outputs_grid", "CalculationId", "IsCall", "VolatilityShock", "StockPriceShock"),
        Index("idx_bs_outputs_option_type", "IsCall", "CalculationId"),
    )


class MLDataCache(Base):
    """
    Cache preprocessed data for ML training.
    Stores feature vectors and labels for quick retrieval.
    """
    __tablename__ = "MLDataCache"

    CacheId: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    CalculationId = Column(Integer, ForeignKey("BlackScholesInputs.CalculationId"), nullable=False, index=True)

    #feature vector (JSON or pickle)
    Features = Column(String(5000), nullable=False)

    #labels
    CallPrice = Column(Float, nullable=False)
    PutPrice = Column(Float, nullable=False)

    #greeks (optional)
    CallDelta = Column(Float, nullable=True)
    PutDelta = Column(Float, nullable=True)
    Gamma = Column(Float, nullable=True)
    Vega = Column(Float, nullable=True)
    CallTheta = Column(Float, nullable=True)
    PutTheta = Column(Float, nullable=True)
    CallRho = Column(Float, nullable=True)
    PutRho = Column(Float, nullable=True)

    #Metadata
    CreatedAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_ml_cache_calc", "CalculationId"),
        Index("idx_ml_cache_created", "CreatedAt"),
    )


def quant9(x: float) -> Decimal:
    """Convert float to Decimal with 9 decimal places precision."""
    return Decimal(str(x)).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)


def create_engine_from_url(url: str, **kwargs):
    """
    Create SQLAlchemy engine with connection pooling optimized for ML workloads.

    url: str
        SQLAlchemy database URL
    **kwargs: dict
        Additional engine parameters
    """
    default_params = {
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 3600,
        "poolclass": QueuePool,
        "future": True
    }
    default_params.update(kwargs)
    return create_engine(url, **default_params)


def create_all(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def drop_all(engine):
    """Drop all tables in the database."""
    Base.metadata.drop_all(engine)


def insert_calculation(
        engine,
        inputs: Dict[str, Any],
        S_grid: np.ndarray,
        V_grid: np.ndarray,
        call_surface: np.ndarray,
        put_surface: np.ndarray,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
) -> Tuple[int, int]:
    """
    Insert calculation inputs and outputs into database.

    engine: SQLAlchemy engine
    inputs: dict
        Input parameters
    S_grid: np.ndarray
        Spot price grid
    V_grid: np.ndarray
        Volatility grid
    call_surface: np.ndarray
        Call prices grid
    put_surface: np.ndarray
        Put prices grid
    session_id: str, optional
        Session identifier for tracking
    user_id: str, optional
        User identifier for ML personalization

    Returns:
    calculation_id: int
        ID of inserted calculation
    n_rows: int
        Number of output rows inserted
    """
    with Session(engine) as sess:
        #create input record
        bs_in = BlackScholesInputs(
            StockPrice=quant9(inputs["StockPrice"]),
            StrikePrice=quant9(inputs["StrikePrice"]),
            InterestRate=quant9(inputs["InterestRate"]),
            Volatility=quant9(inputs["Volatility"]),
            TimeToExpiry=quant9(inputs["TimeToExpiry"]),
            PurchaseCallPrice=quant9(inputs.get("PurchaseCallPrice", 0.0)) if inputs.get(
                "PurchaseCallPrice") is not None else None,
            PurchasePutPrice=quant9(inputs.get("PurchasePutPrice", 0.0)) if inputs.get(
                "PurchasePutPrice") is not None else None,
            MinSpot=quant9(inputs["MinSpot"]),
            MaxSpot=quant9(inputs["MaxSpot"]),
            MinVol=quant9(inputs["MinVol"]),
            MaxVol=quant9(inputs["MaxVol"]),
            GridNSpot=int(inputs["GridNSpot"]),
            GridNVol=int(inputs["GridNVol"]),
            SessionId=session_id,
            UserId=user_id,
            CreatedAt=datetime.utcnow()
        )
        sess.add(bs_in)
        sess.flush()  #get CalculationId
        calc_id = bs_in.CalculationId

        #prep bulk insert for outputs
        nV, nS = call_surface.shape
        rows = []

        for i in range(nV):
            for j in range(nS):
                #call option output
                rows.append({
                    "VolatilityShock": quant9(float(V_grid[i])),
                    "StockPriceShock": quant9(float(S_grid[j])),
                    "OptionPrice": quant9(float(call_surface[i, j])),
                    "IsCall": 1,
                    "CalculationId": calc_id
                })
                #Put option output
                rows.append({
                    "VolatilityShock": quant9(float(V_grid[i])),
                    "StockPriceShock": quant9(float(S_grid[j])),
                    "OptionPrice": quant9(float(put_surface[i, j])),
                    "IsCall": 0,
                    "CalculationId": calc_id
                })

        #bulk insert for performance
        sess.bulk_insert_mappings(BlackScholesOutputs, rows)
        sess.commit()

        return calc_id, len(rows)


def retrieve_calculation(engine, calculation_id: int) -> Dict[str, Any]:
    """
    Retrieve complete calculation data for ML training.

    engine: SQLAlchemy engine
    calculation_id: int
        ID of calculation to retrieve

    Returns dict containing inputs and outputs
    """
    with Session(engine) as sess:
        #Get inputs
        inputs = sess.query(BlackScholesInputs).filter_by(CalculationId=calculation_id).first()
        if not inputs:
            raise ValueError(f"Calculation {calculation_id} not found")

        #Get outputs
        outputs = sess.query(BlackScholesOutputs).filter_by(CalculationId=calculation_id).all()

        #Organize data
        result = {
            "inputs": {
                "StockPrice": float(inputs.StockPrice),
                "StrikePrice": float(inputs.StrikePrice),
                "InterestRate": float(inputs.InterestRate),
                "Volatility": float(inputs.Volatility),
                "TimeToExpiry": float(inputs.TimeToExpiry),
                "PurchaseCallPrice": float(inputs.PurchaseCallPrice) if inputs.PurchaseCallPrice else None,
                "PurchasePutPrice": float(inputs.PurchasePutPrice) if inputs.PurchasePutPrice else None,
                "MinSpot": float(inputs.MinSpot),
                "MaxSpot": float(inputs.MaxSpot),
                "MinVol": float(inputs.MinVol),
                "MaxVol": float(inputs.MaxVol),
                "GridNSpot": inputs.GridNSpot,
                "GridNVol": inputs.GridNVol,
                "CreatedAt": inputs.CreatedAt,
                "SessionId": inputs.SessionId,
                "UserId": inputs.UserId
            },
            "outputs": {
                "calls": [],
                "puts": []
            }
        }

        #separate calls and puts
        for output in outputs:
            data = {
                "VolatilityShock": float(output.VolatilityShock),
                "StockPriceShock": float(output.StockPriceShock),
                "OptionPrice": float(output.OptionPrice)
            }
            if output.IsCall:
                result["outputs"]["calls"].append(data)
            else:
                result["outputs"]["puts"].append(data)

        return result


def get_ml_training_data(
        engine,
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        min_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Retrieve data formatted for ML training.

    Parameters
    engine: SQLAlchemy engine
    limit: int, optional
        Maximum number of calculations to retrieve
    user_id: str, optional
        Filter by user ID
    min_date: datetime, optional
        Minimum creation date

    Returns pd.DataFrame with features and labels
    """
    with Session(engine) as sess:
        #Build query
        query = sess.query(BlackScholesInputs)

        if user_id:
            query = query.filter(BlackScholesInputs.UserId == user_id)
        if min_date:
            query = query.filter(BlackScholesInputs.CreatedAt >= min_date)
        if limit:
            query = query.limit(limit)

        #Get inputs
        inputs = query.all()

        #Prepare data
        data = []
        for inp in inputs:
            #Get base prices (at current spot and vol)
            base_outputs = sess.query(BlackScholesOutputs).filter(
                and_(
                    BlackScholesOutputs.CalculationId == inp.CalculationId,
                    BlackScholesOutputs.StockPriceShock.between(
                        inp.StockPrice * Decimal("0.99"),
                        inp.StockPrice * Decimal("1.01")
                    ),
                    BlackScholesOutputs.VolatilityShock.between(
                        inp.Volatility * Decimal("0.99"),
                        inp.Volatility * Decimal("1.01")
                    )
                )
            ).all()

            if base_outputs:
                call_price = next((float(o.OptionPrice) for o in base_outputs if o.IsCall), None)
                put_price = next((float(o.OptionPrice) for o in base_outputs if not o.IsCall), None)

                if call_price and put_price:
                    data.append({
                        "StockPrice": float(inp.StockPrice),
                        "StrikePrice": float(inp.StrikePrice),
                        "InterestRate": float(inp.InterestRate),
                        "Volatility": float(inp.Volatility),
                        "TimeToExpiry": float(inp.TimeToExpiry),
                        "CallPrice": call_price,
                        "PutPrice": put_price,
                        "Moneyness": float(inp.StockPrice / inp.StrikePrice),
                        "CreatedAt": inp.CreatedAt
                    })

        return pd.DataFrame(data)


def cleanup_old_data(engine, days_to_keep: int = 30) -> int:
    """
    Remove old calculation data to maintain database performance.

    engine: SQLAlchemy engine
    days_to_keep: int
        Number of days of data to retain

    Returns int: Number of calculations deleted
    """
    from datetime import timedelta

    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    with Session(engine) as sess:
        #Find old calculations
        old_calcs = sess.query(BlackScholesInputs).filter(
            BlackScholesInputs.CreatedAt < cutoff_date
        ).all()

        count = len(old_calcs)

        #Delete (cascade will handle outputs)
        for calc in old_calcs:
            sess.delete(calc)

        sess.commit()

        return count