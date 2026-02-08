from datetime import datetime
from uuid import UUID
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from src.models.trading import OrderSide, OrderType, OrderStatus

class PositionResponse(BaseModel):
    id: UUID
    portfolio_id: UUID
    symbol: str
    quantity: float
    average_price: float
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class OrderCreate(BaseModel):
    portfolio_id: UUID
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(None, gt=0)

class OrderResponse(BaseModel):
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
    name: str = "Main Portfolio"
    currency: str = "USD"

class PortfolioResponse(BaseModel):
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
    name: str
    symbols: List[str] = []

class WatchlistUpdate(BaseModel):
    symbols: List[str]

class WatchlistResponse(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    items: List[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
