"""
BS-Opt Numerical Pricing Methods
Black-Scholes, FDM, Monte Carlo, and Trinomial Trees
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

# =============================================================================
# Analytical Black-Scholes
# =============================================================================

def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """
    Analytical Black-Scholes option pricing with Greeks.

    Parameters:
        spot: Current stock price
        strike: Strike price
        rate: Risk-free interest rate
        volatility: Volatility
        time_to_maturity: Time to maturity (in years)
        option_type: 'call' or 'put'

    Returns:
        Dictionary with price and all Greeks
    """
    if time_to_maturity <= 0:
        # At expiration
        if option_type == "call":
            return {"price": max(spot - strike, 0), "delta": 1 if spot > strike else 0,
                    "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        return {"price": max(strike - spot, 0), "delta": -1 if spot < strike else 0,
                "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    sqrt_time = math.sqrt(time_to_maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt_time)
    d2 = d1 - volatility * sqrt_time

    # Standard normal CDF and PDF
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    # Discount factor
    df = math.exp(-rate * time_to_maturity)

    if option_type == "call":
        price = spot * cdf_d1 - strike * df * cdf_d2
        delta = cdf_d1
        rho = strike * time_to_maturity * df * cdf_d2 / 100
    else:
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)
        price = strike * df * cdf_neg_d2 - spot * cdf_neg_d1
        delta = cdf_d1 - 1
        rho = -strike * time_to_maturity * df * cdf_neg_d2 / 100

    # Greeks (same for call/put except delta and rho)
    gamma = pdf_d1 / (spot * volatility * sqrt_time)
    vega = spot * pdf_d1 * sqrt_time / 100
    theta = (-(spot * pdf_d1 * volatility) / (2 * sqrt_time) - rate * strike * df *
             (cdf_d2 if option_type == "call" else norm.cdf(-d2))) / 365

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


# =============================================================================
# Finite Difference Method (Crank-Nicolson)
# =============================================================================

def crank_nicolson_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
    spot_max: float = None,
    grid_steps: int = 100,  # Asset price steps
    time_steps: int = 100,  # Time steps
) -> dict:
    """
    Crank-Nicolson finite difference method for option pricing.
    Unconditionally stable, second-order accurate in both space and time.
    """
    start_time = time.perf_counter_ns()

    if spot_max is None:
        spot_max = 4 * strike

    dt = time_to_maturity / time_steps
    # dS = spot_max / grid_steps  # Unused

    # Grid
    spot_grid = np.linspace(0, spot_max, grid_steps + 1)

    # Initialize option values at maturity
    if option_type == "call":
        value = np.maximum(spot_grid - strike, 0)
    else:
        value = np.maximum(strike - spot_grid, 0)

    # Coefficients for tridiagonal system
    j = np.arange(1, grid_steps)
    alpha = 0.25 * dt * (volatility**2 * j**2 - rate * j)
    beta = -0.5 * dt * (volatility**2 * j**2 + rate)
    gamma = 0.25 * dt * (volatility**2 * j**2 + rate * j)

    # Build tridiagonal matrices
    # A * V_new = B * V_old (Crank-Nicolson)

    # Implicit part (LHS)
    lhs_diag = 1 - beta
    lhs_lower = -alpha[1:]
    lhs_upper = -gamma[:-1]

    # Explicit part (RHS)
    rhs_diag = 1 + beta
    rhs_lower = alpha[1:]
    rhs_upper = gamma[:-1]

    # Time stepping
    for n in range(time_steps):
        # Build RHS
        rhs = np.zeros(grid_steps - 1)
        rhs[0] = rhs_lower[0] * value[0] + rhs_diag[0] * value[1] + rhs_upper[0] * value[2]
        rhs[1:-1] = rhs_lower[1:-1] * value[1:-2] + rhs_diag[1:-1] * value[2:-1] + rhs_upper[1:-1] * value[3:-1]
        rhs[-1] = rhs_lower[-1] * value[-3] + rhs_diag[-1] * value[-2] + rhs_upper[-1] * value[-1]

        # Apply boundary conditions
        if option_type == "call":
            # V(0, t) = 0, V(S_max, t) = S_max - K * exp(-r*(T-t))
            rhs[-1] += gamma[-1] * (spot_max - strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt)))
        else:
            # V(0, t) = K * exp(-r*(T-t)), V(S_max, t) = 0
            rhs[0] += alpha[0] * strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt))

        # Solve tridiagonal system using Thomas algorithm
        value[1:-1] = solve_tridiagonal(lhs_lower, lhs_diag, lhs_upper, rhs)

        # Update boundary values
        if option_type == "call":
            value[0] = 0
            value[-1] = spot_max - strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt))
        else:
            value[0] = strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt))
            value[-1] = 0

    # Interpolate to find price at spot
    price = np.interp(spot, spot_grid, value)

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "time_us": computation_time,
        "grid_size": grid_steps,
    }


def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal systems"""
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# =============================================================================
# Monte Carlo (Antithetic Variance Reduction)
# =============================================================================

def monte_carlo_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
    num_paths: int = 100000,
    seed: int = None,
) -> dict:
    """
    Monte Carlo option pricing with antithetic variance reduction.
    """
    start_time = time.perf_counter_ns()

    if seed is not None:
        np.random.seed(seed)

    # Generate random numbers
    z_score = np.random.standard_normal(num_paths // 2)

    # Antithetic variates
    z_score = np.concatenate([z_score, -z_score])

    # Simulate terminal stock prices
    drift = (rate - 0.5 * volatility**2) * time_to_maturity
    diffusion = volatility * np.sqrt(time_to_maturity) * z_score
    spot_t = spot * np.exp(drift + diffusion)

    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(spot_t - strike, 0)
    else:
        payoffs = np.maximum(strike - spot_t, 0)

    # Discounted expected payoff
    price = np.exp(-rate * time_to_maturity) * np.mean(payoffs)
    std_error = np.exp(-rate * time_to_maturity) * np.std(payoffs) / np.sqrt(num_paths)

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "std_error": std_error,
        "time_us": computation_time,
        "num_paths": num_paths,
    }


# =============================================================================
# Trinomial Tree with Richardson Extrapolation
# =============================================================================

def trinomial_tree_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
    time_steps: int = 200,
    use_richardson: bool = True,
) -> dict:
    """
    Trinomial tree option pricing with optional Richardson extrapolation.
    """
    start_time = time.perf_counter_ns()

    def _tree_price(steps: int) -> float:
        dt = time_to_maturity / steps

        # Trinomial parameters
        u = np.exp(volatility * np.sqrt(2 * dt))
        # d = 1 / u # Unused
        # m = 1  # middle factor # Unused

        # Risk-neutral probabilities
        sqrt_dt = np.sqrt(dt / 2)
        pu = ((np.exp(rate * dt / 2) - np.exp(-volatility * sqrt_dt)) /
              (np.exp(volatility * sqrt_dt) - np.exp(-volatility * sqrt_dt)))**2
        pd = ((np.exp(volatility * sqrt_dt) - np.exp(rate * dt / 2)) /
              (np.exp(volatility * sqrt_dt) - np.exp(-volatility * sqrt_dt)))**2
        pm = 1 - pu - pd

        # Initialize asset prices at maturity
        num_nodes = 2 * steps + 1
        spot_t = spot * (u ** np.arange(steps, -steps - 1, -1))

        # Option values at maturity
        if option_type == "call":
            value = np.maximum(spot_t - strike, 0)
        else:
            value = np.maximum(strike - spot_t, 0)

        # Backward induction
        df = np.exp(-rate * dt)
        for i in range(steps - 1, -1, -1):
            num_nodes = 2 * i + 1
            value_new = np.zeros(num_nodes)
            for j in range(num_nodes):
                value_new[j] = df * (pu * value[j] + pm * value[j + 1] + pd * value[j + 2])
            value = value_new

        return value[0]

    if use_richardson:
        # Richardson extrapolation: 2 * V(2N) - V(N)
        value_2n = _tree_price(time_steps)
        value_n = _tree_price(time_steps // 2)
        price = 2 * value_2n - value_n
    else:
        price = _tree_price(time_steps)

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "time_us": computation_time,
        "steps": time_steps,
    }


# =============================================================================
# Numerical Method Comparator (Research Tool)
# =============================================================================

@dataclass
class NumericalMethodComparator:
    """Compare all numerical methods for academic research."""

    fdm_grid_size: int = 200
    mc_paths: int = 100000
    tree_steps: int = 200

    def compare_all(
        self,
        spot: float,
        strike: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        option_type: Literal["call", "put"] = "call",
    ) -> dict:
        """Run all methods and return comparative results."""

        # Analytical (benchmark)
        analytical = black_scholes_price(
            spot, strike, rate, volatility, time_to_maturity, option_type
        )

        # FDM
        fdm = crank_nicolson_price(
            spot, strike, rate, volatility, time_to_maturity, option_type,
            grid_steps=self.fdm_grid_size
        )

        # Monte Carlo
        mc = monte_carlo_price(
            spot, strike, rate, volatility, time_to_maturity, option_type,
            num_paths=self.mc_paths
        )

        # Trinomial Tree
        tree = trinomial_tree_price(
            spot, strike, rate, volatility, time_to_maturity, option_type,
            time_steps=self.tree_steps
        )

        # Calculate errors
        analytical_price = analytical["price"]

        return {
            "analytical": analytical,
            "fdm": {
                **fdm,
                "error_pct": abs(fdm["price"] - analytical_price) / analytical_price * 100,
            },
            "monte_carlo": {
                **mc,
                "error_pct": abs(mc["price"] - analytical_price) / analytical_price * 100,
            },
            "trinomial": {
                **tree,
                "error_pct": abs(tree["price"] - analytical_price) / analytical_price * 100,
            },
        }
