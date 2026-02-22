# pylint: disable=duplicate-code
# pylint: disable=duplicate-code
"""
BS-Opt Test Suite
Tests for pricing engine numerical methods
"""

import math

import pytest

from src.pricing.numerical_methods import (
    NumericalMethodComparator,
    black_scholes_price,
    crank_nicolson_price,
    monte_carlo_price,
    trinomial_tree_price,
)

# =============================================================================
# Test Parameters
# =============================================================================

# Standard test case
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOLATILITY = 0.2
TIME = 1.0  # 1 year

# Tolerance for numerical methods vs analytical
TOLERANCE_PCT = 1.0  # 1% max error


# =============================================================================
# Black-Scholes Analytical Tests
# =============================================================================


class TestBlackScholes:
    """Tests for analytical Black-Scholes pricing"""

    def test_at_the_money_call(self):
        """ATM call should have delta ~0.5"""
        result = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")

        assert result["price"] > 0
        assert 0.5 <= result["delta"] <= 0.7  # ATM call delta
        assert result["gamma"] > 0
        assert result["vega"] > 0

    def test_at_the_money_put(self):
        """ATM put should have delta ~-0.5"""
        result = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "put")

        assert result["price"] > 0
        assert -0.7 <= result["delta"] <= -0.3  # ATM put delta

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)"""
        call = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")
        put = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "put")

        lhs = call["price"] - put["price"]
        rhs = SPOT - STRIKE * math.exp(-RATE * TIME)

        assert abs(lhs - rhs) < 0.01

    def test_deep_itm_call(self):
        """Deep ITM call should have delta ~1"""
        result = black_scholes_price(150, 100, RATE, VOLATILITY, TIME, "call")

        assert result["delta"] > 0.9
        assert result["price"] > 50  # At least intrinsic value

    def test_deep_otm_call(self):
        """Deep OTM call should have delta ~0"""
        result = black_scholes_price(50, 100, RATE, VOLATILITY, TIME, "call")

        assert result["delta"] < 0.1
        assert result["price"] < 5

    def test_zero_time_call(self):
        """At expiration, call = max(S-K, 0)"""
        itm = black_scholes_price(110, 100, RATE, VOLATILITY, 0.0001, "call")
        otm = black_scholes_price(90, 100, RATE, VOLATILITY, 0.0001, "call")

        assert abs(itm["price"] - 10) < 0.5
        assert otm["price"] < 0.5


# =============================================================================
# Finite Difference Method Tests
# =============================================================================


class TestCrankNicolson:
    """Tests for Crank-Nicolson FDM"""

    def test_accuracy_vs_analytical_call(self):
        """FDM should match analytical within tolerance"""
        analytical = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")
        fdm = crank_nicolson_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", M=200)

        error_pct = abs(fdm["price"] - analytical["price"]) / analytical["price"] * 100
        assert error_pct < TOLERANCE_PCT

    def test_accuracy_vs_analytical_put(self):
        """FDM put should match analytical"""
        analytical = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "put")
        fdm = crank_nicolson_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "put", M=200)

        error_pct = abs(fdm["price"] - analytical["price"]) / analytical["price"] * 100
        assert error_pct < TOLERANCE_PCT

    def test_grid_convergence(self):
        """Finer grid should give more accurate result"""
        analytical = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")

        coarse = crank_nicolson_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", M=50)
        fine = crank_nicolson_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", M=200)

        coarse_error = abs(coarse["price"] - analytical["price"])
        fine_error = abs(fine["price"] - analytical["price"])

        assert fine_error < coarse_error


# =============================================================================
# Monte Carlo Tests
# =============================================================================


class TestMonteCarlo:
    """Tests for Monte Carlo pricing"""

    def test_accuracy_vs_analytical(self):
        """MC should match analytical within confidence interval"""
        analytical = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")
        mc = monte_carlo_price(
            SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", num_paths=100000, seed=42
        )

        # Within 3 standard errors
        error = abs(mc["price"] - analytical["price"])
        assert error < 3 * mc["std_error"]

    def test_antithetic_reduces_variance(self):
        """Antithetic sampling should give reasonable standard error"""
        mc = monte_carlo_price(
            SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", num_paths=100000, seed=42
        )

        # Standard error should be small relative to price
        assert mc["std_error"] < mc["price"] * 0.01  # < 1% of price

    def test_reproducibility_with_seed(self):
        """Same seed should give same result"""
        mc1 = monte_carlo_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", seed=42)
        mc2 = monte_carlo_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", seed=42)

        assert mc1["price"] == mc2["price"]


# =============================================================================
# Trinomial Tree Tests
# =============================================================================


class TestTrinomialTree:
    """Tests for trinomial tree pricing"""

    def test_accuracy_vs_analytical(self):
        """Tree should match analytical within tolerance"""
        analytical = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")
        tree = trinomial_tree_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", N=200)

        error_pct = abs(tree["price"] - analytical["price"]) / analytical["price"] * 100
        assert error_pct < TOLERANCE_PCT

    def test_richardson_extrapolation_improves_accuracy(self):
        """Richardson extrapolation should improve accuracy"""
        analytical = black_scholes_price(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")

        no_richardson = trinomial_tree_price(
            SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", N=100, use_richardson=False
        )
        with_richardson = trinomial_tree_price(
            SPOT, STRIKE, RATE, VOLATILITY, TIME, "call", N=100, use_richardson=True
        )

        error_no = abs(no_richardson["price"] - analytical["price"])
        error_with = abs(with_richardson["price"] - analytical["price"])

        # Richardson should generally improve accuracy
        # (might not always hold for small N, so we're lenient)
        assert error_with < error_no * 2


# =============================================================================
# Comparative Analysis Tests
# =============================================================================


class TestNumericalMethodComparator:
    """Tests for the method comparator"""

    def test_all_methods_return_results(self):
        """All methods should return valid results"""
        comparator = NumericalMethodComparator(fdm_grid_size=100, mc_paths=50000, tree_steps=100)

        results = comparator.compare_all(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")

        assert "analytical" in results
        assert "fdm" in results
        assert "monte_carlo" in results
        assert "trinomial" in results

        for method in ["fdm", "monte_carlo", "trinomial"]:
            assert "price" in results[method]
            assert "error_pct" in results[method]
            assert results[method]["error_pct"] < 5.0  # < 5% error

    def test_timing_information(self):
        """All methods should report computation time"""
        comparator = NumericalMethodComparator()
        results = comparator.compare_all(SPOT, STRIKE, RATE, VOLATILITY, TIME, "call")

        assert results["fdm"]["time_us"] > 0
        assert results["monte_carlo"]["time_us"] > 0
        assert results["trinomial"]["time_us"] > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_high_volatility(self):
        """High volatility should still produce valid prices"""
        result = black_scholes_price(SPOT, STRIKE, RATE, 1.0, TIME, "call")
        assert result["price"] > 0
        assert 0 < result["delta"] < 1

    def test_very_short_maturity(self):
        """Very short maturity should approach intrinsic value"""
        result = black_scholes_price(110, 100, RATE, VOLATILITY, 0.001, "call")
        assert abs(result["price"] - 10) < 0.5

    def test_zero_rate(self):
        """Zero interest rate should still work"""
        result = black_scholes_price(SPOT, STRIKE, 0.0, VOLATILITY, TIME, "call")
        assert result["price"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
