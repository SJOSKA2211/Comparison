from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base

class Exchange(str, Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    NSE_KENYA = "NSE_KENYA"
    CRYPTO = "CRYPTO"

class MarketTick(BaseModel):
    timestamp: datetime
    symbol: str
    exchange: Exchange
    price: float
    volume: Optional[float] = None
    source: str

class NumericalExperiment(Base):
    __tablename__ = "numerical_experiments"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    researcher_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    underlying_symbol: Mapped[str] = mapped_column(String)
    spot_price: Mapped[float] = mapped_column(Float)
    strike: Mapped[float] = mapped_column(Float)
    risk_free_rate: Mapped[float] = mapped_column(Float)
    volatility: Mapped[float] = mapped_column(Float)
    time_to_maturity: Mapped[float] = mapped_column(Float)
    option_type: Mapped[str] = mapped_column(String)

    analytical_price: Mapped[float] = mapped_column(Float, nullable=True)

    fdm_price: Mapped[float] = mapped_column(Float, nullable=True)
    fdm_time_us: Mapped[int] = mapped_column(Integer, nullable=True)
    fdm_error_pct: Mapped[float] = mapped_column(Float, nullable=True)

    mc_price: Mapped[float] = mapped_column(Float, nullable=True)
    mc_time_us: Mapped[int] = mapped_column(Integer, nullable=True)
    mc_error_pct: Mapped[float] = mapped_column(Float, nullable=True)

    tree_price: Mapped[float] = mapped_column(Float, nullable=True)
    tree_time_us: Mapped[int] = mapped_column(Integer, nullable=True)
    tree_error_pct: Mapped[float] = mapped_column(Float, nullable=True)
