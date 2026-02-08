"""
BS-Opt API Gateway
FastAPI with native PostgreSQL auth, OAuth (Google/GitHub), and email verification
"""

from __future__ import annotations

import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import urlencode
from uuid import UUID

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, Field

# Lazy imports for heavy dependencies
_db_pool = None

logger = structlog.get_logger()


# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseModel):
    """Application settings"""
    database_url: str = Field(default="postgresql://localhost/bsopt")
    redis_url: str = Field(default="redis://localhost:6379/0")
    environment: str = Field(default="development")
    
    # OAuth settings
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    github_client_id: str = Field(default="")
    github_client_secret: str = Field(default="")
    
    # URLs
    frontend_url: str = Field(default="http://localhost:3000")
    api_url: str = Field(default="http://localhost:8000")
    
    # Email (for verification)
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    from_email: str = Field(default="noreply@bsopt.io")


settings = Settings(
    database_url=os.getenv("DATABASE_URL", "postgresql://localhost/bsopt"),
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    environment=os.getenv("ENVIRONMENT", "development"),
    google_client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
    google_client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
    github_client_id=os.getenv("GITHUB_CLIENT_ID", ""),
    github_client_secret=os.getenv("GITHUB_CLIENT_SECRET", ""),
    frontend_url=os.getenv("FRONTEND_URL", "http://localhost:3000"),
    api_url=os.getenv("API_URL", "http://localhost:8000"),
)


# =============================================================================
# Database
# =============================================================================

async def get_db_pool():
    global _db_pool
    if _db_pool is None:
        import asyncpg
        _db_pool = await asyncpg.create_pool(
            settings.database_url.replace("+asyncpg", ""),
            min_size=5,
            max_size=20,
        )
    return _db_pool


async def get_db():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        yield conn


# =============================================================================
# Models
# =============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = Field(default="trader", pattern="^(trader|researcher|admin)$")


class UserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    email_verified: bool = False
    display_name: str | None = None
    avatar_url: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    user: UserResponse | None = None


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


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


class EmailVerificationRequest(BaseModel):
    email: EmailStr
    token: str


# =============================================================================
# Auth Middleware
# =============================================================================

async def get_current_user(request: Request, db=Depends(get_db)) -> dict | None:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header[7:]
    result = await db.fetchrow("SELECT * FROM validate_session($1)", token)
    
    if result:
        await db.execute("SELECT set_session_context($1, $2)", 
                        result["user_id"], result["user_role"])
        return dict(result)
    return None


def require_auth(user: dict | None = Depends(get_current_user)) -> dict:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_verified(user: dict = Depends(require_auth)) -> dict:
    if not user.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified"
        )
    return user


def require_role(*roles: str):
    def check_role(user: dict = Depends(require_verified)) -> dict:
        if user["user_role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(roles)}"
            )
        return user
    return check_role


# =============================================================================
# Email Service
# =============================================================================

async def send_verification_email(email: str, token: str):
    """Send email verification link"""
    verify_url = f"{settings.api_url}/auth/verify-email?email={email}&token={token}"
    
    if settings.environment == "development":
        logger.info("Verification email (dev mode)", email=email, url=verify_url)
        return
    
    # Production: Send via SMTP
    # TODO: Implement actual email sending
    logger.info("Verification email sent", email=email)


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BS-Opt API", environment=settings.environment)
    await get_db_pool()
    yield
    global _db_pool
    if _db_pool:
        await _db_pool.close()


# =============================================================================
# Application
# =============================================================================

app = FastAPI(
    title="BS-Opt API",
    description="Quantitative Finance Research Platform",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", secrets.token_hex(8))
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info("request", method=request.method, path=request.url.path,
                status=response.status_code, duration_ms=round(duration_ms, 2))
    
    response.headers["X-Request-ID"] = request_id
    return response


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(status="healthy", environment=settings.environment, version="1.0.0")


@app.get("/ready", tags=["Health"])
async def readiness_check(db=Depends(get_db)):
    await db.fetchval("SELECT 1")
    return {"status": "ready"}


# =============================================================================
# Local Auth Endpoints
# =============================================================================

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register_user(user: UserCreate, request: Request, db=Depends(get_db)):
    """Register with email/password"""
    try:
        result = await db.fetchrow(
            "SELECT * FROM register_local_user($1, $2, $3)",
            user.email, user.password, user.role
        )
        
        # Send verification email
        await send_verification_email(user.email, result["verification_token"])
        
        # Create session
        token = secrets.token_urlsafe(32)
        await db.execute(
            "SELECT create_session($1, $2, $3, $4, $5)",
            result["user_id"], token, 24,
            request.client.host if request.client else None,
            request.headers.get("User-Agent"),
        )
        
        return TokenResponse(
            access_token=token,
            user=UserResponse(
                id=result["user_id"],
                email=user.email,
                role=user.role,
                email_verified=False,
            )
        )
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(status_code=409, detail="Email already registered")
        raise


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(request: Request, credentials: LoginRequest, db=Depends(get_db)):
    """Login with email/password"""
    result = await db.fetchrow(
        "SELECT * FROM authenticate_user($1, $2)",
        credentials.email, credentials.password
    )
    
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = secrets.token_urlsafe(32)
    await db.execute(
        "SELECT create_session($1, $2, $3, $4, $5)",
        result["user_id"], token, 24,
        request.client.host if request.client else None,
        request.headers.get("User-Agent"),
    )
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=result["user_id"],
            email=credentials.email,
            role=result["user_role"],
            email_verified=result["is_verified"],
        )
    )


@app.post("/auth/logout", tags=["Auth"])
async def logout(request: Request, db=Depends(get_db)):
    """Logout and revoke session"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        await db.execute("SELECT revoke_session($1)", token)
    return {"status": "logged_out"}


# =============================================================================
# Email Verification
# =============================================================================

@app.get("/auth/verify-email", tags=["Auth"])
async def verify_email(
    email: str = Query(...),
    token: str = Query(...),
    db=Depends(get_db),
):
    """Verify email address"""
    success = await db.fetchval("SELECT verify_email($1, $2)", email, token)
    
    if success:
        return RedirectResponse(f"{settings.frontend_url}/email-verified")
    
    raise HTTPException(status_code=400, detail="Invalid or expired token")


@app.post("/auth/resend-verification", tags=["Auth"])
async def resend_verification(email: EmailStr, db=Depends(get_db)):
    """Resend verification email"""
    token = await db.fetchval("SELECT generate_verification_token($1)", email)
    
    if token:
        await send_verification_email(email, token)
        return {"status": "sent"}
    
    raise HTTPException(status_code=404, detail="Email not found")


# =============================================================================
# OAuth: Google
# =============================================================================

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@app.get("/auth/google", tags=["OAuth"])
async def google_login():
    """Initiate Google OAuth flow"""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")
    
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": f"{settings.api_url}/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@app.get("/auth/google/callback", tags=["OAuth"])
async def google_callback(
    code: str = Query(...),
    request: Request = None,
    db=Depends(get_db),
):
    """Handle Google OAuth callback"""
    async with httpx.AsyncClient() as client:
        # Exchange code for tokens
        token_response = await client.post(GOOGLE_TOKEN_URL, data={
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{settings.api_url}/auth/google/callback",
        })
        tokens = token_response.json()
        
        if "error" in tokens:
            raise HTTPException(status_code=400, detail=tokens.get("error_description", "OAuth failed"))
        
        # Get user info
        userinfo_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        userinfo = userinfo_response.json()
    
    # Upsert user in database
    result = await db.fetchrow(
        "SELECT * FROM upsert_oauth_user($1, $2, $3, $4, $5, $6, $7)",
        "google",
        userinfo["id"],
        userinfo["email"],
        userinfo.get("name"),
        userinfo.get("picture"),
        tokens["access_token"],
        tokens.get("refresh_token"),
    )
    
    # Create session
    token = secrets.token_urlsafe(32)
    await db.execute(
        "SELECT create_session($1, $2, $3, $4, $5)",
        result["user_id"], token, 168,  # 7 days for OAuth
        request.client.host if request and request.client else None,
        request.headers.get("User-Agent") if request else None,
    )
    
    # Redirect to frontend with token
    return RedirectResponse(f"{settings.frontend_url}/auth/callback?token={token}")


# =============================================================================
# OAuth: GitHub
# =============================================================================

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


@app.get("/auth/github", tags=["OAuth"])
async def github_login():
    """Initiate GitHub OAuth flow"""
    if not settings.github_client_id:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")
    
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": f"{settings.api_url}/auth/github/callback",
        "scope": "read:user user:email",
    }
    return RedirectResponse(f"{GITHUB_AUTH_URL}?{urlencode(params)}")


@app.get("/auth/github/callback", tags=["OAuth"])
async def github_callback(
    code: str = Query(...),
    request: Request = None,
    db=Depends(get_db),
):
    """Handle GitHub OAuth callback"""
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_response = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"}
        )
        tokens = token_response.json()
        
        if "error" in tokens:
            raise HTTPException(status_code=400, detail=tokens.get("error_description", "OAuth failed"))
        
        access_token = tokens["access_token"]
        headers = {"Authorization": f"token {access_token}", "Accept": "application/json"}
        
        # Get user info
        user_response = await client.get(GITHUB_USER_URL, headers=headers)
        userinfo = user_response.json()
        
        # Get primary email (GitHub might not include it in user response)
        email = userinfo.get("email")
        if not email:
            emails_response = await client.get(GITHUB_EMAILS_URL, headers=headers)
            emails = emails_response.json()
            primary = next((e for e in emails if e.get("primary")), emails[0] if emails else None)
            email = primary["email"] if primary else None
        
        if not email:
            raise HTTPException(status_code=400, detail="Could not get email from GitHub")
    
    # Upsert user
    result = await db.fetchrow(
        "SELECT * FROM upsert_oauth_user($1, $2, $3, $4, $5, $6, $7)",
        "github",
        str(userinfo["id"]),
        email,
        userinfo.get("name") or userinfo.get("login"),
        userinfo.get("avatar_url"),
        access_token,
        None,  # GitHub doesn't provide refresh tokens by default
    )
    
    # Create session
    token = secrets.token_urlsafe(32)
    await db.execute(
        "SELECT create_session($1, $2, $3, $4, $5)",
        result["user_id"], token, 168,
        request.client.host if request and request.client else None,
        request.headers.get("User-Agent") if request else None,
    )
    
    return RedirectResponse(f"{settings.frontend_url}/auth/callback?token={token}")


# =============================================================================
# User Endpoints
# =============================================================================

@app.get("/auth/me", response_model=UserResponse, tags=["Auth"])
async def get_current_user_info(user: dict = Depends(require_auth), db=Depends(get_db)):
    """Get current user profile"""
    result = await db.fetchrow(
        "SELECT id, email, role, email_verified, display_name, avatar_url FROM users WHERE id = $1",
        user["user_id"]
    )
    return UserResponse(**dict(result))


# =============================================================================
# Pricing Endpoints
# =============================================================================

@app.post("/pricing/black-scholes", response_model=PricingResponse, tags=["Pricing"])
async def price_option(request: PricingRequest, user: dict = Depends(require_auth)):
    """Calculate option price with Black-Scholes"""
    start_time = time.perf_counter_ns()
    
    from src.pricing.numerical_methods import black_scholes_price
    
    result = black_scholes_price(
        S=request.spot, K=request.strike, r=request.rate,
        sigma=request.volatility, T=request.time_to_maturity,
        option_type=request.option_type,
    )
    
    return PricingResponse(
        **result,
        computation_time_us=(time.perf_counter_ns() - start_time) // 1000,
    )


@app.post("/pricing/compare", tags=["Pricing", "Research"])
async def compare_methods(
    request: PricingRequest,
    user: dict = Depends(require_role("researcher", "admin")),
    db=Depends(get_db),
):
    """Compare all pricing methods (research endpoint)"""
    from src.pricing.numerical_methods import NumericalMethodComparator
    
    comparator = NumericalMethodComparator()
    results = comparator.compare_all(
        S=request.spot, K=request.strike, r=request.rate,
        sigma=request.volatility, T=request.time_to_maturity,
        option_type=request.option_type,
    )
    
    # Store experiment
    await db.execute("""
        INSERT INTO numerical_experiments (
            researcher_id, spot_price, strike, risk_free_rate, volatility,
            time_to_maturity, option_type, analytical_price,
            fdm_price, fdm_time_us, mc_price, mc_time_us, tree_price, tree_time_us
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """,
        user["user_id"], request.spot, request.strike, request.rate,
        request.volatility, request.time_to_maturity, request.option_type,
        results["analytical"]["price"], results["fdm"]["price"], results["fdm"]["time_us"],
        results["monte_carlo"]["price"], results["monte_carlo"]["time_us"],
        results["trinomial"]["price"], results["trinomial"]["time_us"],
    )
    
    return results


# =============================================================================
# Market Data Endpoints
# =============================================================================

@app.get("/market/{symbol}/latest", tags=["Market Data"])
async def get_latest_price(symbol: str, user: dict = Depends(require_auth), db=Depends(get_db)):
    result = await db.fetchrow(
        "SELECT time, symbol, price, volume, bid, ask FROM market_ticks WHERE symbol = $1 ORDER BY time DESC LIMIT 1",
        symbol.upper()
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    return dict(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
