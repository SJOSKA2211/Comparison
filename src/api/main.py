"""
BS-Opt API Gateway (Refactored)
Modernized with SQLAlchemy 2.0 and Modular Routers
"""
from __future__ import annotations

import os
import secrets
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.api.routers import auth, pricing, trading, user
from src.database import engine

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
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BS-Opt API (SQLAlchemy Mode)", environment=settings.environment)
    yield
    # Close SQLAlchemy engine
    await engine.dispose()

# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="BS-Opt API",
    description="Quantitative Finance Research Platform",
    version="2.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# Middlewares
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
# Routers
# =============================================================================

app.include_router(auth.router, prefix="/api")
app.include_router(pricing.router, prefix="/api")
app.include_router(trading.router, prefix="/api")
app.include_router(user.router, prefix="/api")

# =============================================================================
# Health Check
# =============================================================================

@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "2.0.0",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
