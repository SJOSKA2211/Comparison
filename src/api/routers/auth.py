import secrets
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.api.utils import get_password_hash, get_token_hash, verify_password
from src.database import get_db
from src.models.trading import Portfolio
from src.models.user import OAuthProvider, Session as UserSession, User, UserRole
from src.schemas.auth import LoginRequest, TokenResponse, UserCreate, UserResponse

# pylint: disable=unused-import

router = APIRouter(prefix="/auth", tags=["Auth"])

async def create_user_session(db: AsyncSession, user_id: UUID, request: Request = None) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = get_token_hash(token)
    new_session = UserSession(
        user_id=user_id,
        token_hash=token_hash, # Store hash(token)
        expires_at=datetime.utcnow() + timedelta(hours=24),
        ip_address=request.client.host if request and request.client else None,
        user_agent=request.headers.get("User-Agent") if request else None
    )
    db.add(new_session)
    return token

async def ensure_default_portfolio(db: AsyncSession, user_id: UUID):
    """Ensure user has at least one portfolio"""
    # Check if user already has a portfolio
    result = await db.execute(select(Portfolio).where(Portfolio.user_id == user_id))
    if not result.scalars().first():
        new_portfolio = Portfolio(
            user_id=user_id,
            name="Main Portfolio",
            currency="USD",
            cash_balance=100000.0 # Default starting balance
        )
        db.add(new_portfolio)
        # Commit will happen in the calling endpoint

@router.post("/register", response_model=TokenResponse)
async def register(user_in: UserCreate, request: Request, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = User(
        email=user_in.email,
        password_hash=get_password_hash(user_in.password),
        role=UserRole(user_in.role),
        oauth_provider=OAuthProvider.LOCAL,
        email_verified=False
    )
    db.add(new_user)
    await db.flush() # Get user_id

    # Feature Upgrade: Auto-create portfolio
    await ensure_default_portfolio(db, new_user.id)

    token = await create_user_session(db, new_user.id, request)
    await db.commit()

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": new_user
    }

@router.post("/login", response_model=TokenResponse)
async def login(login_in: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == login_in.email))
    user = result.scalars().first()

    if not user or not user.password_hash or not verify_password(login_in.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    user.last_login = datetime.utcnow()
    token = await create_user_session(db, user.id, request)

    # Ensure portfolio (even for existing users migrating)
    await ensure_default_portfolio(db, user.id)

    await db.commit()

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user
    }


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = await db.execute(select(User).where(User.id == user["user_id"]))
    db_user = result.scalars().first()
    return db_user
