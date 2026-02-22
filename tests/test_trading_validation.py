"""Trading Validation Tests"""

# pylint: disable=redefined-outer-name, unused-argument
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.api.routers.trading import get_current_user_id
from src.database import Base, get_db
from src.models.trading import Portfolio, Position

# Setup in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)


async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def db_session():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestingSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
def mock_user_id():
    return uuid.uuid4()


@pytest_asyncio.fixture
async def client(db_session, mock_user_id):
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user_id] = lambda: mock_user_id
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_sell_order_insufficient_position(client, db_session, mock_user_id):
    # Create a portfolio
    portfolio = Portfolio(
        user_id=mock_user_id, name="Test Portfolio", currency="USD", cash_balance=1000.0
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)

    # Try to sell AAPL (no position)
    response = await client.post(
        "/api/trading/orders",
        json={
            "portfolio_id": str(portfolio.id),
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "market",
            "quantity": 10.0,
        },
    )

    assert response.status_code == 400
    assert "Insufficient position" in response.json()["detail"]


@pytest.mark.asyncio
async def test_sell_order_sufficient_position(client, db_session, mock_user_id):
    # Create a portfolio with position
    portfolio = Portfolio(
        user_id=mock_user_id, name="Test Portfolio", currency="USD", cash_balance=1000.0
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)

    position = Position(
        portfolio_id=portfolio.id, symbol="AAPL", quantity=20.0, average_price=150.0
    )
    db_session.add(position)
    await db_session.commit()

    # Try to sell AAPL (valid quantity)
    response = await client.post(
        "/api/trading/orders",
        json={
            "portfolio_id": str(portfolio.id),
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "market",
            "quantity": 10.0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "filled"  # Market order mock execution
    assert data["filled_quantity"] == 10.0


@pytest.mark.asyncio
async def test_sell_order_partial_insufficient_position(client, db_session, mock_user_id):
    # Create a portfolio with position
    portfolio = Portfolio(
        user_id=mock_user_id, name="Test Portfolio", currency="USD", cash_balance=1000.0
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)

    position = Position(portfolio_id=portfolio.id, symbol="AAPL", quantity=5.0, average_price=150.0)
    db_session.add(position)
    await db_session.commit()

    # Try to sell AAPL (quantity > position)
    response = await client.post(
        "/api/trading/orders",
        json={
            "portfolio_id": str(portfolio.id),
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "limit",  # Test with non-market order too
            "quantity": 10.0,
            "price": 160.0,
        },
    )

    assert response.status_code == 400
    assert "Insufficient position" in response.json()["detail"]
