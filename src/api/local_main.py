"""
BS-Opt Local Development API
Simplified version using SQLite for local development
"""

from __future__ import annotations

import os
import sys
import traceback

# Startup diagnostics
try:
    with open("startup.log", "w") as f:
        f.write("Starting local_main...\n")
except:
    pass

try:
    import secrets
    import time
    from contextlib import asynccontextmanager
    from typing import Any

    import structlog
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import ORJSONResponse
    from pydantic import BaseModel, EmailStr, Field
except Exception as e:
    with open("startup_error.log", "w") as f:
        f.write(f"Import Error: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logger = structlog.get_logger()
with open("startup.log", "a") as f:
    f.write("Imports successful\n")

# =============================================================================
# Configuration
# =============================================================================

DATABASE_PATH = os.getenv("DATABASE_PATH", "data/bsopt.db")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# =============================================================================
# Database (SQLite)
# =============================================================================

_db_connection = None


async def get_db():
    """Get SQLite database connection"""
    import aiosqlite

    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


# =============================================================================
# Models
# =============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = Field(default="trader", pattern="^(trader|researcher|admin)$")


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    email_verified: bool = False


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse | None = None


class PricingRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    rate: float = Field(ge=0, le=1)
    volatility: float = Field(gt=0, le=5)
    time_to_maturity: float = Field(gt=0)
    option_type: str = Field(pattern="^(call|put)$")


class PricingResponse(BaseModel):
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    computation_time_us: int
    model: str = "black-scholes"


# =============================================================================
# Auth Helpers
# =============================================================================

async def get_current_user(request: Request, db=Depends(get_db)) -> dict | None:
    """Extract user from Authorization header"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]

    cursor = await db.execute(
        """SELECT u.id, u.email, u.role, u.email_verified
           FROM sessions s JOIN users u ON s.user_id = u.id
           WHERE s.token = ? AND datetime(s.expires_at) > datetime('now')""",
        (token,)
    )
    row = await cursor.fetchone()

    if row:
        return {"user_id": row["id"], "email": row["email"],
                "user_role": row["role"], "email_verified": row["email_verified"]}
    return None


def require_auth(user: dict | None = Depends(get_current_user)) -> dict:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BS-Opt API (local dev)", environment=ENVIRONMENT)
    yield
    logger.info("Shutting down")


# =============================================================================
# Application
# =============================================================================

app = FastAPI(
    title="BS-Opt API (Local Dev)",
    description="Quantitative Finance Research Platform - Local Development",
    version="1.0.0-dev",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "environment": ENVIRONMENT, "version": "1.0.0-dev"}


@app.get("/ready", tags=["Health"])
async def readiness_check(db=Depends(get_db)):
    cursor = await db.execute("SELECT 1")
    await cursor.fetchone()
    return {"status": "ready", "database": "sqlite"}


# =============================================================================
# Auth Endpoints
# =============================================================================

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register_user(user: UserCreate, db=Depends(get_db)):
    """Register with email/password"""
    import hashlib

    # Simple password hash for dev
    password_hash = hashlib.sha256(user.password.encode()).hexdigest()
    user_id = secrets.token_hex(16)

    try:
        await db.execute(
            "INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (user_id, user.email, password_hash, user.role)
        )
        await db.commit()
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(status_code=409, detail="Email already registered")
        raise

    # Create session
    token = secrets.token_urlsafe(32)
    expires_at = "datetime('now', '+24 hours')"
    await db.execute(
        f"INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, {expires_at})",
        (user_id, token)
    )
    await db.commit()

    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user_id, email=user.email, role=user.role)
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(credentials: LoginRequest, db=Depends(get_db)):
    """Login with email/password"""
    import hashlib

    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()

    cursor = await db.execute(
        "SELECT id, email, role, email_verified FROM users WHERE email = ? AND password_hash = ?",
        (credentials.email, password_hash)
    )
    row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create session
    token = secrets.token_urlsafe(32)
    await db.execute(
        "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, datetime('now', '+24 hours'))",
        (row["id"], token)
    )
    await db.commit()

    return TokenResponse(
        access_token=token,
        user=UserResponse(id=row["id"], email=row["email"],
                         role=row["role"], email_verified=bool(row["email_verified"]))
    )


@app.get("/auth/me", response_model=UserResponse, tags=["Auth"])
async def get_me(user: dict = Depends(require_auth)):
    """Get current user"""
    return UserResponse(
        id=user["user_id"],
        email=user["email"],
        role=user["user_role"],
        email_verified=bool(user["email_verified"])
    )


# =============================================================================
# Pricing Endpoints
# =============================================================================

@app.post("/pricing/black-scholes", response_model=PricingResponse, tags=["Pricing"])
async def price_option(request: PricingRequest, user: dict = Depends(require_auth)):
    """Calculate option price with Black-Scholes"""
    start_time = time.perf_counter_ns()

    from src.pricing.numerical_methods import black_scholes_price

    result = black_scholes_price(
        spot=request.spot, strike=request.strike, rate=request.rate,
        sigma=request.volatility, time_to_maturity=request.time_to_maturity,
        option_type=request.option_type,
    )

    return PricingResponse(
        **result,
        computation_time_us=(time.perf_counter_ns() - start_time) // 1000,
    )


@app.post("/pricing/compare", tags=["Pricing", "Research"])
async def compare_methods(request: PricingRequest, user: dict = Depends(require_auth)):
    """Compare all pricing methods"""
    from src.pricing.numerical_methods import NumericalMethodComparator

    comparator = NumericalMethodComparator()
    results = comparator.compare_all(
        spot=request.spot, strike=request.strike, rate=request.rate,
        sigma=request.volatility, time_to_maturity=request.time_to_maturity,
        option_type=request.option_type,
    )

    return results


# =============================================================================
# Demo Data Endpoints
# =============================================================================

@app.get("/demo/price/{symbol}", tags=["Demo"])
async def get_demo_price(symbol: str):
    """Get a demo price for testing (no auth required)"""
    import random

    prices = {
        "AAPL": 185.0,
        "GOOGL": 140.0,
        "MSFT": 400.0,
        "SAFCOM": 25.0,  # Safaricom (NSE Kenya)
        "EQTY": 45.0,    # Equity Bank (NSE Kenya)
    }

    base_price = prices.get(symbol.upper(), 100.0)
    current_price = base_price * (1 + random.uniform(-0.02, 0.02))

    return {
        "symbol": symbol.upper(),
        "price": round(current_price, 2),
        "bid": round(current_price * 0.999, 2),
        "ask": round(current_price * 1.001, 2),
        "source": "demo"
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.local_main:app", host="0.0.0.0", port=8000, reload=True)
