"""Trading schemas."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.models.trading import OrderSide, OrderStatus, OrderType


class PositionResponse(BaseModel):
    """Position response schema."""
    """Position response schema."""
    id: UUID
    portfolio_id: UUID
    symbol: str
    quantity: float
    average_price: float
    current_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class OrderCreate(BaseModel):
    """Order creation schema."""
    """Order creation schema."""
    portfolio_id: UUID
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(None, gt=0)

class OrderResponse(BaseModel):
    """Order response schema."""
    """Order response schema."""
    id: UUID
    portfolio_id: UUID
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    filled_price: Optional[float]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PortfolioCreate(BaseModel):
    """Portfolio creation schema."""
    """Portfolio creation schema."""
    name: str = "Main Portfolio"
    currency: str = "USD"

class PortfolioResponse(BaseModel):
    """Portfolio response schema."""
    """Portfolio response schema."""
    id: UUID
    user_id: UUID
    name: str
    currency: str
    cash_balance: float
    positions: List[PositionResponse] = []
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class WatchlistCreate(BaseModel):
    """Watchlist creation schema."""
    """Watchlist creation schema."""
    name: str
    symbols: List[str] = []

class WatchlistUpdate(BaseModel):
    """Watchlist update schema."""
    """Watchlist update schema."""
    symbols: List[str]

class WatchlistResponse(BaseModel):
    """Watchlist response schema."""
    """Watchlist response schema."""
    id: UUID
    user_id: UUID
    name: str
    items: List[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
