"""Trading schemas."""
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.models.trading import OrderSide, OrderStatus, OrderType


class PositionResponse(BaseModel):
    """Schema for position response."""
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
    """Schema for order creation."""
    portfolio_id: UUID
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(None, gt=0)

class OrderResponse(BaseModel):
    """Schema for order response."""
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
    """Schema for portfolio creation."""
    name: str = "Main Portfolio"
    currency: str = "USD"

class PortfolioResponse(BaseModel):
    """Schema for portfolio response."""
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
    """Schema for watchlist creation."""
    name: str
    symbols: List[str] = []

class WatchlistUpdate(BaseModel):
    """Schema for watchlist update."""
    symbols: List[str]

class WatchlistResponse(BaseModel):
    """Schema for watchlist response."""
    id: UUID
    user_id: UUID
    name: str
    items: List[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
