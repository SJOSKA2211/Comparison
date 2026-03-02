import math
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm

# =============================================================================
# Analytical Solution (Black-Scholes)
# =============================================================================

def black_scholes_price(
    spot_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """
    Calculate Black-Scholes price and Greeks.
    """
    start_time = time.perf_counter_ns()

    if time_to_maturity <= 0:
        return {
            "price": max(0, spot_price - strike_price) if option_type == "call" else max(0, strike_price - spot_price),
            "delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0,
            "computation_time_us": 0
        }

    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (math.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    # Standard normal CDF and PDF
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    # Discount factor
    df = math.exp(-risk_free_rate * time_to_maturity)

    if option_type == "call":
        price = spot_price * n_d1 - strike_price * df * n_d2
        delta = n_d1
        rho = strike_price * time_to_maturity * df * n_d2 / 100
    else:
        n_neg_d1 = norm.cdf(-d1)
        n_neg_d2 = norm.cdf(-d2)
        price = strike_price * df * n_neg_d2 - spot_price * n_neg_d1
        delta = n_d1 - 1
        rho = -strike_price * time_to_maturity * df * n_neg_d2 / 100

    # Greeks (same for call/put except delta and rho)
    gamma = pdf_d1 / (spot_price * volatility * sqrt_t)
    vega = spot_price * pdf_d1 * sqrt_t / 100
    theta = (-(spot_price * pdf_d1 * volatility) / (2 * sqrt_t) - risk_free_rate * strike_price * df *
             (n_d2 if option_type == "call" else norm.cdf(-d2))) / 365

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "computation_time_us": computation_time,
    }


# =============================================================================
# Finite Difference Method (Crank-Nicolson)
# =============================================================================

def crank_nicolson_price(
    spot_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: Literal["call", "put"] = "call",
    max_spot_price: Optional[float] = None,
    asset_steps: int = 100,  # Asset price steps (M)
    time_steps: int = 100,  # Time steps (N)
) -> dict:
    """
    Crank-Nicolson finite difference method for option pricing.
    Unconditionally stable, second-order accurate in both space and time.
    """
    start_time = time.perf_counter_ns()

    if max_spot_price is None:
        max_spot_price = 4 * strike_price

    dt = time_to_maturity / time_steps

    # Grid
    spot_grid = np.linspace(0, max_spot_price, asset_steps + 1)

    # Initialize option values at maturity
    if option_type == "call":
        values = np.maximum(spot_grid - strike_price, 0)
    else:
        values = np.maximum(strike_price - spot_grid, 0)

    # Coefficients for tridiagonal system
    j = np.arange(1, asset_steps)
    alpha = 0.25 * dt * (volatility**2 * j**2 - risk_free_rate * j)
    beta = -0.5 * dt * (volatility**2 * j**2 + risk_free_rate)
    gamma = 0.25 * dt * (volatility**2 * j**2 + risk_free_rate * j)

    # Build tridiagonal matrices
    # A * V_new = B * V_old (Crank-Nicolson)

    # Implicit part (LHS)
    a_diag = 1 - beta
    a_lower = -alpha[1:]
    a_upper = -gamma[:-1]

    # Time stepping
    for n in range(time_steps):
        # Build RHS (vectorized)
        # rhs[k] corresponds to j=k+1
        # alpha, beta, gamma are size M-1
        # values size M+1
        # values[1:-1] size M-1
        # values[:-2] size M-1 (from 0 to M-2)
        # values[2:] size M-1 (from 2 to M)
        rhs = alpha * values[:-2] + (1 + beta) * values[1:-1] + gamma * values[2:]

        # Apply boundary conditions
        if option_type == "call":
            # V(0, t) = 0, V(S_max, t) = S_max - K * exp(-r*(T-t))
            rhs[-1] += gamma[-1] * (max_spot_price - strike_price * np.exp(-risk_free_rate * (time_to_maturity - (n + 1) * dt)))
        else:
            # V(0, t) = K * exp(-r*(T-t)), V(S_max, t) = 0
            rhs[0] += alpha[0] * strike_price * np.exp(-risk_free_rate * (time_to_maturity - (n + 1) * dt))

        # Solve tridiagonal system using Thomas algorithm
        values[1:-1] = solve_tridiagonal(a_lower, a_diag, a_upper, rhs)

        # Update boundary values
        if option_type == "call":
            values[0] = 0
            values[-1] = max_spot_price - strike_price * np.exp(-risk_free_rate * (time_to_maturity - (n + 1) * dt))
        else:
            values[0] = strike_price * np.exp(-risk_free_rate * (time_to_maturity - (n + 1) * dt))
            values[-1] = 0

    # Interpolate to find price at spot_price
    price = np.interp(spot_price, spot_grid, values)

    computation_time = (time.perf_counter_ns() - start_time) // 1000

    return {
        "price": price,
        "time_us": computation_time,
        "grid_size": asset_steps,
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
    spot_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
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
    z_score = np.random.standard_normal(num_paths // 2)

    # Antithetic variates
    z_score = np.concatenate([z_score, -z_score])

    # Simulate terminal stock prices
    drift = (risk_free_rate - 0.5 * volatility**2) * time_to_maturity
    diffusion = volatility * np.sqrt(time_to_maturity) * z_score
    spot_terminal = spot_price * np.exp(drift + diffusion)

    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(spot_terminal - strike_price, 0)
    else:
        payoffs = np.maximum(strike_price - spot_terminal, 0)

    # Discounted expected payoff
    price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
    std_error = np.exp(-risk_free_rate * time_to_maturity) * np.std(payoffs) / np.sqrt(num_paths)

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
    spot_price: float,
    strike_price: float,
    risk_free_rate: float,
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

    def _tree_price(num_steps: int) -> float:
        dt = time_to_maturity / num_steps

        # Trinomial parameters
        u = np.exp(volatility * np.sqrt(2 * dt))
        # d = 1 / u # Unused
        # m = 1  # middle factor # Unused

        # Risk-neutral probabilities
        sqrt_dt = np.sqrt(dt / 2)
        pu = ((np.exp(risk_free_rate * dt / 2) - np.exp(-volatility * sqrt_dt)) /
              (np.exp(volatility * sqrt_dt) - np.exp(-volatility * sqrt_dt)))**2
        pd = ((np.exp(volatility * sqrt_dt) - np.exp(risk_free_rate * dt / 2)) /
              (np.exp(volatility * sqrt_dt) - np.exp(-volatility * sqrt_dt)))**2
        pm = 1 - pu - pd

        # Initialize asset prices at maturity
        # num_nodes = 2 * num_steps + 1 # unused except for range
        spot_terminal = spot_price * (u ** np.arange(num_steps, -num_steps - 1, -1))

        # Option values at maturity
        if option_type == "call":
            values = np.maximum(spot_terminal - strike_price, 0)
        else:
            values = np.maximum(strike_price - spot_terminal, 0)

        # Backward induction
        df = np.exp(-risk_free_rate * dt)
        for i in range(num_steps - 1, -1, -1):
            num_nodes_i = 2 * i + 1
            values_new = np.zeros(num_nodes_i)
            for j in range(num_nodes_i):
                values_new[j] = df * (pu * values[j] + pm * values[j + 1] + pd * values[j + 2])
            values = values_new

        return values[0]

    if use_richardson:
        # Richardson extrapolation: 2 * V(2N) - V(N)
        price_2n = _tree_price(steps)
        price_n = _tree_price(steps // 2)
        price = 2 * price_2n - price_n
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
        spot_price: float,
        strike_price: float,
        risk_free_rate: float,
        volatility: float,
        time_to_maturity: float,
        option_type: Literal["call", "put"] = "call",
    ) -> dict:
        """Run all methods and return comparative results."""

        # Analytical (benchmark)
        analytical = black_scholes_price(
            spot_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_type
        )

        # FDM
        fdm = crank_nicolson_price(
            spot_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_type,
            asset_steps=self.fdm_grid_size
        )

        # Monte Carlo
        mc = monte_carlo_price(
            spot_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_type,
            num_paths=self.mc_paths
        )

        # Trinomial Tree
        tree = trinomial_tree_price(
            spot_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_type,
            steps=self.tree_steps
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
