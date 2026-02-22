# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-positional-arguments
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
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """
    Analytical Black-Scholes option pricing with Greeks.

    Parameters:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate
        sigma: Volatility
        T: Time to maturity (in years)
        option_type: 'call' or 'put'

    Returns:
        Dictionary with price and all Greeks
    """
    if T <= 0:
        # At expiration
        if option_type == "call":
            return {"price": max(S - K, 0), "delta": 1 if S > K else 0,
                    "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        else:
            return {"price": max(K - S, 0), "delta": -1 if S < K else 0,
                    "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Standard normal CDF and PDF
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    # Discount factor
    df = math.exp(-r * T)

    if option_type == "call":
        price = S * N_d1 - K * df * N_d2
        delta = N_d1
        rho = K * T * df * N_d2 / 100
    else:
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        price = K * df * N_neg_d2 - S * N_neg_d1
        delta = N_d1 - 1
        rho = -K * T * df * N_neg_d2 / 100

    # Greeks (same for call/put except delta and rho)
    gamma = n_d1 / (S * sigma * sqrt_T)
    vega = S * n_d1 * sqrt_T / 100
    theta = (-(S * n_d1 * sigma) / (2 * sqrt_T) - r * K * df *
             (N_d2 if option_type == "call" else norm.cdf(-d2))) / 365

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
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
    S_max: float = None,
    M: int = 100,  # Asset price steps
    N: int = 100,  # Time steps
) -> dict:
    """
    Crank-Nicolson finite difference method for option pricing.
    Unconditionally stable, second-order accurate in both space and time.
    """
    start_time = time.perf_counter_ns()

    if S_max is None:
        S_max = 4 * K

    dt = T / N
    dS = S_max / M

    # Grid
    S_grid = np.linspace(0, S_max, M + 1)

    # Initialize option values at maturity
    if option_type == "call":
        V = np.maximum(S_grid - K, 0)
    else:
        V = np.maximum(K - S_grid, 0)

    # Coefficients for tridiagonal system
    j = np.arange(1, M)
    alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)
    beta = -0.5 * dt * (sigma**2 * j**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)

    # Build tridiagonal matrices
    # A * V_new = B * V_old (Crank-Nicolson)

    # Implicit part (LHS)
    A_diag = 1 - beta
    A_lower = -alpha[1:]
    A_upper = -gamma[:-1]

    # Explicit part (RHS)
    B_diag = 1 + beta
    B_lower = alpha[1:]
    B_upper = gamma[:-1]

    # Time stepping
    for n in range(N):
        # Build RHS
        rhs = np.zeros(M - 1)
        rhs[0] = B_lower[0] * V[0] + B_diag[0] * V[1] + B_upper[0] * V[2]
        rhs[1:-1] = B_lower[1:-1] * V[1:-2] + B_diag[1:-1] * V[2:-1] + B_upper[1:-1] * V[3:-1]
        rhs[-1] = B_lower[-1] * V[-3] + B_diag[-1] * V[-2] + B_upper[-1] * V[-1]

        # Apply boundary conditions
        if option_type == "call":
            # V(0, t) = 0, V(S_max, t) = S_max - K * exp(-r*(T-t))
            rhs[-1] += gamma[-1] * (S_max - K * np.exp(-r * (T - (n + 1) * dt)))
        else:
            # V(0, t) = K * exp(-r*(T-t)), V(S_max, t) = 0
            rhs[0] += alpha[0] * K * np.exp(-r * (T - (n + 1) * dt))

        # Solve tridiagonal system using Thomas algorithm
        V[1:-1] = solve_tridiagonal(A_lower, A_diag, A_upper, rhs)

        # Update boundary values
        if option_type == "call":
            V[0] = 0
            V[-1] = S_max - K * np.exp(-r * (T - (n + 1) * dt))
        else:
            V[0] = K * np.exp(-r * (T - (n + 1) * dt))
            V[-1] = 0

    # Interpolate to find price at S
    price = np.interp(S, S_grid, V)

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "time_us": computation_time,
        "grid_size": M,
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
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
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
    Z = np.random.standard_normal(num_paths // 2)

    # Antithetic variates
    Z = np.concatenate([Z, -Z])

    # Simulate terminal stock prices
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diffusion)

    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    # Discounted expected payoff
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(num_paths)

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
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
    N: int = 200,
    use_richardson: bool = True,
) -> dict:
    """
    Trinomial tree option pricing with optional Richardson extrapolation.
    """
    start_time = time.perf_counter_ns()

    def _tree_price(steps: int) -> float:
        dt = T / steps

        # Trinomial parameters
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1 / u
        m = 1  # middle factor

        # Risk-neutral probabilities
        sqrt_dt = np.sqrt(dt / 2)
        pu = ((np.exp(r * dt / 2) - np.exp(-sigma * sqrt_dt)) /
              (np.exp(sigma * sqrt_dt) - np.exp(-sigma * sqrt_dt)))**2
        pd = ((np.exp(sigma * sqrt_dt) - np.exp(r * dt / 2)) /
              (np.exp(sigma * sqrt_dt) - np.exp(-sigma * sqrt_dt)))**2
        pm = 1 - pu - pd

        # Initialize asset prices at maturity
        num_nodes = 2 * steps + 1
        S_T = S * (u ** np.arange(steps, -steps - 1, -1))

        # Option values at maturity
        if option_type == "call":
            V = np.maximum(S_T - K, 0)
        else:
            V = np.maximum(K - S_T, 0)

        # Backward induction
        df = np.exp(-r * dt)
        for i in range(steps - 1, -1, -1):
            num_nodes = 2 * i + 1
            V_new = np.zeros(num_nodes)
            for j in range(num_nodes):
                V_new[j] = df * (pu * V[j] + pm * V[j + 1] + pd * V[j + 2])
            V = V_new

        return V[0]

    if use_richardson:
        # Richardson extrapolation: 2 * V(2N) - V(N)
        V_2N = _tree_price(N)
        V_N = _tree_price(N // 2)
        price = 2 * V_2N - V_N
    else:
        price = _tree_price(N)

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "time_us": computation_time,
        "steps": N,
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
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: Literal["call", "put"] = "call",
    ) -> dict:
        """Run all methods and return comparative results."""

        # Analytical (benchmark)
        analytical = black_scholes_price(S, K, r, sigma, T, option_type)

        # FDM
        fdm = crank_nicolson_price(S, K, r, sigma, T, option_type, M=self.fdm_grid_size)

        # Monte Carlo
        mc = monte_carlo_price(S, K, r, sigma, T, option_type, num_paths=self.mc_paths)

        # Trinomial Tree
        tree = trinomial_tree_price(S, K, r, sigma, T, option_type, N=self.tree_steps)

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
