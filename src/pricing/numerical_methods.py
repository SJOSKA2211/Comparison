"""
Numerical methods for option pricing.
Includes Black-Scholes (analytical), Crank-Nicolson (FDM),
Monte Carlo, and Trinomial Trees.
"""

import math
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm

# =============================================================================
# Black-Scholes Analytical Formula
# =============================================================================

def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """
    Calculate the Black-Scholes price and Greeks for a European option.
    """
    if time_to_maturity <= 0:
        # At expiration
        if option_type == "call":
            price = max(spot - strike, 0.0)
        else:
            price = max(strike - spot, 0.0)

        return {
            "price": price,
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    sqrt_time = math.sqrt(time_to_maturity)
    d1_val = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * time_to_maturity) / (sigma * sqrt_time)
    d2_val = d1_val - sigma * sqrt_time

    # Standard normal CDF and PDF
    norm_d1 = norm.cdf(d1_val)
    norm_d2 = norm.cdf(d2_val)
    pdf_d1 = norm.pdf(d1_val)

    # Discount factor
    discount_factor = math.exp(-rate * time_to_maturity)

    if option_type == "call":
        price = spot * norm_d1 - strike * discount_factor * norm_d2
        delta = norm_d1
        rho = strike * time_to_maturity * discount_factor * norm_d2 / 100
    else:
        norm_neg_d1 = norm.cdf(-d1_val)
        norm_neg_d2 = norm.cdf(-d2_val)
        price = strike * discount_factor * norm_neg_d2 - spot * norm_neg_d1
        delta = norm_d1 - 1
        rho = -strike * time_to_maturity * discount_factor * norm_neg_d2 / 100

    # Greeks (same for call/put except delta and rho)
    gamma = pdf_d1 / (spot * sigma * sqrt_time)
    vega = spot * pdf_d1 * sqrt_time / 100
    theta = (-(spot * pdf_d1 * sigma) / (2 * sqrt_time) - rate * strike * discount_factor *
             (norm_d2 if option_type == "call" else norm.cdf(-d2_val))) / 365

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
    sigma: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
    max_spot: Optional[float] = None,
    grid_steps: int = 100,  # Asset price steps (M)
    time_steps: int = 100,  # Time steps (N)
) -> dict:
    """
    Crank-Nicolson finite difference method for option pricing.
    Unconditionally stable, second-order accurate in both space and time.
    """
    start_time = time.perf_counter_ns()

    if max_spot is None:
        max_spot = 4 * strike

    dt = time_to_maturity / time_steps

    # Grid
    spot_grid = np.linspace(0, max_spot, grid_steps + 1)

    # Initialize option values at maturity
    if option_type == "call":
        values = np.maximum(spot_grid - strike, 0)
    else:
        values = np.maximum(strike - spot_grid, 0)

    # Coefficients for tridiagonal system
    j = np.arange(1, grid_steps)
    alpha = 0.25 * dt * (sigma**2 * j**2 - rate * j)
    beta = -0.5 * dt * (sigma**2 * j**2 + rate)
    gamma = 0.25 * dt * (sigma**2 * j**2 + rate * j)

    # Build tridiagonal matrices
    # A * V_new = B * V_old (Crank-Nicolson)

    # Implicit part (LHS)
    lhs_diag = 1 - beta
    lhs_lower = -alpha[1:]
    lhs_upper = -gamma[:-1]

    # Explicit part (RHS) - computed on the fly
    # B = tridiag(alpha, 1+beta, gamma)

    # Time stepping
    for n in range(time_steps):
        # Build RHS vector B * V_old
        # V_old has indices 0..grid_steps
        # Internal nodes 1..grid_steps-1
        rhs = alpha * values[:-2] + (1 + beta) * values[1:-1] + gamma * values[2:]

        # Apply boundary conditions
        if option_type == "call":
            # V(0, t) = 0, V(max_spot, t) = max_spot - K * exp(-r*(T-t))
            rhs[-1] += gamma[-1] * (max_spot - strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt)))
        else:
            # V(0, t) = K * exp(-r*(T-t)), V(max_spot, t) = 0
            rhs[0] += alpha[0] * strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt))

        # Solve tridiagonal system using Thomas algorithm
        values[1:-1] = solve_tridiagonal(lhs_lower, lhs_diag, lhs_upper, rhs)

        # Update boundary values
        if option_type == "call":
            values[0] = 0
            values[-1] = max_spot - strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt))
        else:
            values[0] = strike * np.exp(-rate * (time_to_maturity - (n + 1) * dt))
            values[-1] = 0

    # Interpolate to find price at spot
    price = np.interp(spot, spot_grid, values)

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
    sigma: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
    num_paths: int = 100000,
    seed: Optional[int] = None,
) -> dict:
    """
    Monte Carlo option pricing with antithetic variance reduction.
    """
    start_time = time.perf_counter_ns()

    if seed is not None:
        np.random.seed(seed)

    # Generate random numbers
    z_rand = np.random.standard_normal(num_paths // 2)

    # Antithetic variates
    z_rand = np.concatenate([z_rand, -z_rand])

    # Simulate terminal stock prices
    drift = (rate - 0.5 * sigma**2) * time_to_maturity
    diffusion = sigma * np.sqrt(time_to_maturity) * z_rand
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
    sigma: float,
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
        u = np.exp(sigma * np.sqrt(2 * dt))

        # Risk-neutral probabilities
        sqrt_dt = np.sqrt(dt / 2)
        pu = ((np.exp(rate * dt / 2) - np.exp(-sigma * sqrt_dt)) /
              (np.exp(sigma * sqrt_dt) - np.exp(-sigma * sqrt_dt)))**2
        pd = ((np.exp(sigma * sqrt_dt) - np.exp(rate * dt / 2)) /
              (np.exp(sigma * sqrt_dt) - np.exp(-sigma * sqrt_dt)))**2
        pm = 1 - pu - pd

        # Initialize asset prices at maturity
        spot_terminal = spot * (u ** np.arange(steps, -steps - 1, -1))

        # Option values at maturity
        if option_type == "call":
            values = np.maximum(spot_terminal - strike, 0)
        else:
            values = np.maximum(strike - spot_terminal, 0)

        # Backward induction
        df = np.exp(-rate * dt)
        for i in range(steps - 1, -1, -1):
            num_nodes = 2 * i + 1
            values_new = np.zeros(num_nodes)
            for j in range(num_nodes):
                values_new[j] = df * (pu * values[j] + pm * values[j + 1] + pd * values[j + 2])
            values = values_new

        return values[0]

    if use_richardson:
        # Richardson extrapolation: 2 * V(2N) - V(N)
        val_2n = _tree_price(time_steps)
        val_n = _tree_price(time_steps // 2)
        price = 2 * val_2n - val_n
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
        sigma: float,
        time_to_maturity: float,
        option_type: Literal["call", "put"] = "call",
    ) -> dict:
        """Run all methods and return comparative results."""

        # Analytical (benchmark)
        analytical = black_scholes_price(
            spot, strike, rate, sigma, time_to_maturity, option_type
        )

        # FDM
        fdm = crank_nicolson_price(
            spot, strike, rate, sigma, time_to_maturity, option_type,
            grid_steps=self.fdm_grid_size
        )

        # Monte Carlo
        mc = monte_carlo_price(
            spot, strike, rate, sigma, time_to_maturity, option_type,
            num_paths=self.mc_paths
        )

        # Trinomial Tree
        tree = trinomial_tree_price(
            spot, strike, rate, sigma, time_to_maturity, option_type,
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
