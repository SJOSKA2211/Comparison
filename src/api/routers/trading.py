from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.database import get_db
from src.models.trading import Portfolio, Position, Order, Watchlist, OrderStatus
from src.schemas.trading import (
    PortfolioCreate, PortfolioResponse, 
    OrderCreate, OrderResponse, 
    WatchlistCreate, WatchlistUpdate, WatchlistResponse,
    PositionResponse
)
from src.api.main import get_current_user

router = APIRouter(prefix="/trading", tags=["Trading"])

# Dependency to get current user ID
async def get_current_user_id(user: dict = Depends(get_current_user)) -> UUID:
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return UUID(str(user["user_id"]))

# =============================================================================
# Portfolios
# =============================================================================

@router.post("/portfolios", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio: PortfolioCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    new_portfolio = Portfolio(
        user_id=user_id,
        name=portfolio.name,
        currency=portfolio.currency
    )
    db.add(new_portfolio)
    await db.commit()
    await db.refresh(new_portfolio)
    return new_portfolio

@router.get("/portfolios", response_model=List[PortfolioResponse])
async def get_portfolios(
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.user_id == user_id)
        .options(selectinload(Portfolio.positions))
    )
    return result.scalars().all()

@router.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
        .options(selectinload(Portfolio.positions))
    )
    portfolio = result.scalars().first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

# =============================================================================
# Orders
# =============================================================================

@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order: OrderCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(Portfolio.id == order.portfolio_id, Portfolio.user_id == user_id)
    )
    portfolio = portfolio_result.scalars().first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # TODO: Validate balance for BUY orders
    # TODO: Validate position for SELL orders

    new_order = Order(
        portfolio_id=order.portfolio_id,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        price=order.price,
        status=OrderStatus.PENDING # Default
    )
    
    # Mock execution for MVP
    # In a real system, this would go to a matching engine or queue
    # For now, immediate fill for MARKET orders
    if new_order.order_type == "market":
        new_order.status = OrderStatus.FILLED
        new_order.filled_quantity = new_order.quantity
        # Fetch current price mock
        # In real implementation use MarketDataRouter
        mock_price = 150.0 
        new_order.filled_price = mock_price
        
        # Update Position
        # Check if position exists
        pos_result = await db.execute(
            select(Position).where(Position.portfolio_id == portfolio.id, Position.symbol == order.symbol)
        )
        position = pos_result.scalars().first()
        
        if new_order.side == "buy":
            cost = new_order.filled_quantity * new_order.filled_price
            if portfolio.cash_balance >= cost:
                portfolio.cash_balance -= cost
                if position:
                    total_cost = (position.quantity * position.average_price) + cost
                    position.quantity += new_order.filled_quantity
                    position.average_price = total_cost / position.quantity
                else:
                    new_position = Position(
                        portfolio_id=portfolio.id,
                        symbol=order.symbol,
                        quantity=new_order.filled_quantity,
                        average_price=new_order.filled_price
                    )
                    db.add(new_position)
            else:
                 new_order.status = OrderStatus.REJECTED # Insufficient funds
        
        elif new_order.side == "sell":
            if position and position.quantity >= new_order.filled_quantity:
                revenue = new_order.filled_quantity * new_order.filled_price
                portfolio.cash_balance += revenue
                position.quantity -= new_order.filled_quantity
                if position.quantity == 0:
                    await db.delete(position)
            else:
                new_order.status = OrderStatus.REJECTED # Insufficient position

    db.add(new_order)
    await db.commit()
    await db.refresh(new_order)
    return new_order

@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    portfolio_id: Optional[UUID] = None,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    query = select(Order).join(Portfolio).where(Portfolio.user_id == user_id)
    if portfolio_id:
        query = query.where(Order.portfolio_id == portfolio_id)
    
    result = await db.execute(query)
    return result.scalars().all()

# =============================================================================
# Watchlists
# =============================================================================

@router.post("/watchlists", response_model=WatchlistResponse)
async def create_watchlist(
    watchlist: WatchlistCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    new_watchlist = Watchlist(
        user_id=user_id,
        name=watchlist.name,
        items=watchlist.symbols
    )
    db.add(new_watchlist)
    await db.commit()
    await db.refresh(new_watchlist)
    return new_watchlist

@router.get("/watchlists", response_model=List[WatchlistResponse])
async def get_watchlists(
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Watchlist).where(Watchlist.user_id == user_id))
    return result.scalars().all()

@router.put("/watchlists/{watchlist_id}", response_model=WatchlistResponse)
async def update_watchlist(
    watchlist_id: UUID,
    update: WatchlistUpdate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Watchlist).where(Watchlist.id == watchlist_id, Watchlist.user_id == user_id)
    )
    watchlist = result.scalars().first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    
    watchlist.items = update.symbols
    await db.commit()
    await db.refresh(watchlist)
    return watchlist
