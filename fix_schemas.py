import sys

# src/schemas/auth.py
auth_content = """\"\"\"Authentication schemas.\"\"\"
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserResponse(BaseModel):
    \"\"\"User response schema.\"\"\"
    id: UUID
    email: str
    role: str
    email_verified: bool = False
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None


class LoginRequest(BaseModel):
    \"\"\"Login request schema.\"\"\"
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    \"\"\"Token response schema.\"\"\"
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    user: Optional[UserResponse] = None


class UserCreate(BaseModel):
    \"\"\"User creation schema.\"\"\"
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = Field(default="trader", pattern="^(trader|researcher|admin)$")
"""

with open("src/schemas/auth.py", "w") as f:
    f.write(auth_content)

# src/schemas/trading.py
trading_content = """\"\"\"Trading schemas.\"\"\"
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.models.trading import OrderSide, OrderStatus, OrderType


class PositionResponse(BaseModel):
    \"\"\"Position response schema.\"\"\"
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
    \"\"\"Order creation schema.\"\"\"
    portfolio_id: UUID
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(None, gt=0)


class OrderResponse(BaseModel):
    \"\"\"Order response schema.\"\"\"
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
    \"\"\"Portfolio creation schema.\"\"\"
    name: str = "Main Portfolio"
    currency: str = "USD"


class PortfolioResponse(BaseModel):
    \"\"\"Portfolio response schema.\"\"\"
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
    \"\"\"Watchlist creation schema.\"\"\"
    name: str
    symbols: List[str] = []


class WatchlistUpdate(BaseModel):
    \"\"\"Watchlist update schema.\"\"\"
    symbols: List[str]


class WatchlistResponse(BaseModel):
    \"\"\"Watchlist response schema.\"\"\"
    id: UUID
    user_id: UUID
    name: str
    items: List[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
"""

with open("src/schemas/trading.py", "w") as f:
    f.write(trading_content)
