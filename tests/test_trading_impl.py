# pylint: disable=wrong-import-position, wrong-import-order
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# Patch UUID for SQLite compatibility
import sqlalchemy.dialects.postgresql
from sqlalchemy import Uuid

sqlalchemy.dialects.postgresql.UUID = Uuid

from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.api.routers.trading import get_portfolio  # noqa: E402
from src.database import Base  # noqa: E402
from src.models.market import MarketTick  # noqa: E402
from src.models.trading import Portfolio, Position  # noqa: E402

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture
async def db_session():
    """Create async db session for testing"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Configure SQLite dialect to handle UUID
    from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
    # Add a custom compilation rule for UUID if visit_uuid is missing
    if not hasattr(sqlite_dialect, 'visit_uuid'):
        def visit_uuid(self, type_, **kw):
            return "CHAR(32)"
        sqlite_dialect.visit_UUID = visit_uuid

    # Also patch visit_UUID directly in case it's used instead of visit_uuid lookup
    if not hasattr(sqlite_dialect, 'visit_UUID'):
        sqlite_dialect.visit_UUID = sqlite_dialect.visit_uuid

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_maker() as session:
        yield session

    await engine.dispose()

@pytest.mark.asyncio
async def test_get_portfolio_prices_implementation(db_session): # pylint: disable=redefined-outer-name
    """Test optimized price fetching logic"""
    # Setup data
    user_id = uuid.uuid4()
    portfolio = Portfolio(user_id=user_id, name="Test Portfolio")
    db_session.add(portfolio)
    await db_session.flush()

    # Position 1: AAPL
    pos1 = Position(portfolio_id=portfolio.id, symbol="AAPL", quantity=10, average_price=150.0)
    db_session.add(pos1)

    # Position 2: GOOG (No ticks)
    pos2 = Position(portfolio_id=portfolio.id, symbol="GOOG", quantity=5, average_price=2800.0)
    db_session.add(pos2)

    # Ticks for AAPL
    now = datetime.now(timezone.utc)
    # Old tick
    tick1 = MarketTick(symbol="AAPL", price=155.0, time=now - timedelta(minutes=5))
    # New tick
    tick2 = MarketTick(symbol="AAPL", price=160.0, time=now)
    # Older tick (to verify order)
    tick3 = MarketTick(symbol="AAPL", price=140.0, time=now - timedelta(minutes=10))

    db_session.add_all([tick1, tick2, tick3])
    await db_session.commit()

    # Call implementation
    fetched_portfolio = await get_portfolio(portfolio.id, db=db_session)

    # Verify
    assert len(fetched_portfolio.positions) == 2

    fetched_pos1 = next(p for p in fetched_portfolio.positions if p.symbol == "AAPL")
    fetched_pos2 = next(p for p in fetched_portfolio.positions if p.symbol == "GOOG")

    # AAPL should be 160.0
    assert fetched_pos1.current_price == 160.0, f"Expected 160.0, got {fetched_pos1.current_price}"

    # GOOG should be average_price (2800.0) because no ticks
    assert fetched_pos2.current_price == 2800.0, f"Expected 2800.0, got {fetched_pos2.current_price}"
