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
        else:
            return {"price": max(strike - spot, 0), "delta": -1 if spot < strike else 0,
                    "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    sqrt_time = math.sqrt(time_to_maturity)
    d_one = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt_time)
    d_two = d_one - volatility * sqrt_time

    # Standard normal CDF and PDF
    cdf_d1 = norm.cdf(d_one)
    cdf_d2 = norm.cdf(d_two)
    pdf_d1 = norm.pdf(d_one)

    # Discount factor
    discount_factor = math.exp(-rate * time_to_maturity)

    if option_type == "call":
        price = spot * cdf_d1 - strike * discount_factor * cdf_d2
        delta = cdf_d1
        rho = strike * time_to_maturity * discount_factor * cdf_d2 / 100
    else:
        cdf_neg_d1 = norm.cdf(-d_one)
        cdf_neg_d2 = norm.cdf(-d_two)
        price = strike * discount_factor * cdf_neg_d2 - spot * cdf_neg_d1
        delta = cdf_d1 - 1
        rho = -strike * time_to_maturity * discount_factor * cdf_neg_d2 / 100

    # Greeks (same for call/put except delta and rho)
    gamma = pdf_d1 / (spot * volatility * sqrt_time)
    vega = spot * pdf_d1 * sqrt_time / 100
    theta = (-(spot * pdf_d1 * volatility) / (2 * sqrt_time) - rate * strike * discount_factor *
             (cdf_d2 if option_type == "call" else norm.cdf(-d_two))) / 365

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
    steps: int = 100,  # Time steps
) -> dict:
    """
    Crank-Nicolson finite difference method for option pricing.
    Unconditionally stable, second-order accurate in both space and time.
    """
    start_time = time.perf_counter_ns()

    if spot_max is None:
        spot_max = 4 * strike

    delta_t = time_to_maturity / steps
    delta_s = spot_max / grid_steps

    # Grid
    spot_grid = np.linspace(0, spot_max, grid_steps + 1)

    # Initialize option values at maturity
    if option_type == "call":
        option_values = np.maximum(spot_grid - strike, 0)
    else:
        option_values = np.maximum(strike - spot_grid, 0)

    # Coefficients for tridiagonal system
    j = np.arange(1, grid_steps)
    alpha = 0.25 * delta_t * (volatility**2 * j**2 - rate * j)
    beta = -0.5 * delta_t * (volatility**2 * j**2 + rate)
    gamma = 0.25 * delta_t * (volatility**2 * j**2 + rate * j)

    # Build tridiagonal matrices
    # A * option_values_new = B * V_old (Crank-Nicolson)

    # Implicit part (LHS)
    a_diag = 1 - beta
    a_lower = -alpha[1:]
    a_upper = -gamma[:-1]

    # Explicit part (RHS)
    b_diag = 1 + beta
    b_lower = alpha[1:]
    b_upper = gamma[:-1]

    # Time stepping
    for n in range(steps):
        # Build RHS
        rhs = np.zeros(grid_steps - 1)
        rhs[0] = b_lower[0] * option_values[0] + b_diag[0] * option_values[1] + b_upper[0] * option_values[2]
        rhs[1:-1] = b_lower[1:-1] * option_values[1:-2] + b_diag[1:-1] * option_values[2:-1] + b_upper[1:-1] * option_values[3:-1]
        rhs[-1] = b_lower[-1] * option_values[-3] + b_diag[-1] * option_values[-2] + b_upper[-1] * option_values[-1]

        # Apply boundary conditions
        if option_type == "call":
            # option_values(0, t) = 0, option_values(spot_max, t) = spot_max - strike * exp(-rate*(time_to_maturity-t))
            rhs[-1] += gamma[-1] * (spot_max - strike * np.exp(-rate * (time_to_maturity - (n + 1) * delta_t)))
        else:
            # option_values(0, t) = strike * exp(-rate*(time_to_maturity-t)), option_values(spot_max, t) = 0
            rhs[0] += alpha[0] * strike * np.exp(-rate * (time_to_maturity - (n + 1) * delta_t))

        # Solve tridiagonal system using Thomas algorithm
        option_values[1:-1] = solve_tridiagonal(a_lower, a_diag, a_upper, rhs)

        # Update boundary values
        if option_type == "call":
            option_values[0] = 0
            option_values[-1] = spot_max - strike * np.exp(-rate * (time_to_maturity - (n + 1) * delta_t))
        else:
            option_values[0] = strike * np.exp(-rate * (time_to_maturity - (n + 1) * delta_t))
            option_values[-1] = 0

    # Interpolate to find price at spot
    price = np.interp(spot, spot_grid, option_values)

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
    random_variates = np.random.standard_normal(num_paths // 2)

    # Antithetic variates
    random_variates = np.concatenate([random_variates, -random_variates])

    # Simulate terminal stock prices
    drift = (rate - 0.5 * volatility**2) * time_to_maturity
    diffusion = volatility * np.sqrt(time_to_maturity) * random_variates
    spot_terminal = spot * np.exp(drift + diffusion)

    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(spot_terminal - strike, 0)
    else:
        payoffs = np.maximum(strike - spot_terminal, 0)

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
    steps: int = 200,
    use_richardson: bool = True,
) -> dict:
    """
    Trinomial tree option pricing with optional Richardson extrapolation.
    """
    start_time = time.perf_counter_ns()

    def _tree_price(steps: int) -> float:
        delta_t = time_to_maturity / steps

        # Trinomial parameters
        u = np.exp(volatility * np.sqrt(2 * delta_t))
        # Risk-neutral probabilities
        sqrt_dt = np.sqrt(delta_t / 2)
        pu = ((np.exp(rate * delta_t / 2) - np.exp(-volatility * sqrt_dt)) /
              (np.exp(volatility * sqrt_dt) - np.exp(-volatility * sqrt_dt)))**2
        pd = ((np.exp(volatility * sqrt_dt) - np.exp(rate * delta_t / 2)) /
              (np.exp(volatility * sqrt_dt) - np.exp(-volatility * sqrt_dt)))**2
        pm = 1 - pu - pd

        # Initialize asset prices at maturity
        num_nodes = 2 * steps + 1
        spot_terminal = spot * (u ** np.arange(steps, -steps - 1, -1))

        # Option values at maturity
        if option_type == "call":
            option_values = np.maximum(spot_terminal - strike, 0)
        else:
            option_values = np.maximum(strike - spot_terminal, 0)

        # Backward induction
        discount_factor = np.exp(-rate * delta_t)
        for i in range(steps - 1, -1, -1):
            num_nodes = 2 * i + 1
            option_values_new = np.zeros(num_nodes)
            for j in range(num_nodes):
                option_values_new[j] = discount_factor * (pu * option_values[j] + pm * option_values[j + 1] + pd * option_values[j + 2])
            option_values = option_values_new

        return option_values[0]

    if use_richardson:
        # Richardson extrapolation: 2 * option_values(2N) - option_values(steps)
        price_high_res = _tree_price(steps)
        price_low_res = _tree_price(steps // 2)
        price = 2 * price_high_res - price_low_res
    else:
        price = _tree_price(steps)

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "time_us": computation_time,
        "steps": steps,
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
        analytical = black_scholes_price(spot, strike, rate, volatility, time_to_maturity, option_type)

        # FDM
        fdm = crank_nicolson_price(spot, strike, rate, volatility, time_to_maturity, option_type, grid_steps=self.fdm_grid_size)

        # Monte Carlo
        mc = monte_carlo_price(spot, strike, rate, volatility, time_to_maturity, option_type, num_paths=self.mc_paths)

        # Trinomial Tree
        tree = trinomial_tree_price(spot, strike, rate, volatility, time_to_maturity, option_type, steps=self.tree_steps)

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
