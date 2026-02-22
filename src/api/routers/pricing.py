"""Option Pricing API Router"""

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import require_auth
from src.database import get_db
from src.models.market import NumericalExperiment
from src.pricing.numerical_methods import NumericalMethodComparator, black_scholes_price

router = APIRouter(prefix="/pricing", tags=["Pricing"])
"""Request model for option pricing."""


class PricingRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    rate: float = Field(ge=0, le=1)
    volatility: float = Field(gt=0, le=5)
    option_type: str = Field(pattern="^(call|put)$")


"""Response model for option pricing."""


class PricingResponse(BaseModel):
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    computation_time_us: int
    model: str = "black-scholes"


@router.post("/black-scholes", response_model=PricingResponse)
async def price_option(request: PricingRequest, user: dict = Depends(require_auth)):
    """Calculate option price with Black-Scholes"""
    import time

    start_time = time.perf_counter_ns()

    result = black_scholes_price(
        S=request.spot,
        K=request.strike,
        r=request.rate,
        sigma=request.volatility,
        T=request.time_to_maturity,
        option_type=request.option_type,
    )

    return {**result, "computation_time_us": (time.perf_counter_ns() - start_time) // 1000}


@router.post("/compare")
async def compare_methods(
    request: PricingRequest, user: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)
):
    """Compare all numerical methods and persist experiment"""
    comparator = NumericalMethodComparator()
    results = comparator.compare_all(
        S=request.spot,
        K=request.strike,
        r=request.rate,
        sigma=request.volatility,
        T=request.time_to_maturity,
        option_type=request.option_type,
    )

    # Persist the experiment (Feature Stack Upgrade!)
    experiment = NumericalExperiment(
        researcher_id=user["user_id"],
        underlying_symbol="CUSTOM",
        spot_price=request.spot,
        strike=request.strike,
        risk_free_rate=request.rate,
        volatility=request.volatility,
        time_to_maturity=request.time_to_maturity,
        option_type=request.option_type,
        analytical_price=results["analytical"]["price"],
        fdm_price=results["fdm"]["price"],
        fdm_time_us=results["fdm"]["time_us"],
        fdm_error_pct=results["fdm"]["error_pct"],
        mc_price=results["monte_carlo"]["price"],
        mc_time_us=results["monte_carlo"]["time_us"],
        mc_error_pct=results["monte_carlo"]["error_pct"],
        tree_price=results["trinomial"]["price"],
        tree_time_us=results["trinomial"]["time_us"],
        tree_error_pct=results["trinomial"]["error_pct"],
    )

    db.add(experiment)
    await db.commit()

    return results


@router.get("/experiments", response_model=List[dict])
async def get_experiments(user: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    """Get historical experiments for the current user"""
    result = await db.execute(
        select(NumericalExperiment)
        .where(NumericalExperiment.researcher_id == user["user_id"])
        .order_by(NumericalExperiment.created_at.desc())
    )
    return result.scalars().all()
