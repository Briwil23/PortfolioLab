import numpy as np
import pandas as pd

from src.optimization.mean_variance import (
    compute_equal_weight_portfolio,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    efficient_frontier,
    portfolio_expected_return,
    portfolio_volatility,
    portfolio_sharpe_ratio,
)


def test_equal_weight_portfolio():
    weights = compute_equal_weight_portfolio(4)
    assert weights.shape == (4,)
    np.testing.assert_allclose(weights.sum(), 1.0)
    np.testing.assert_allclose(weights, np.full(4, 0.25))


def test_minimum_variance_portfolio():
    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.03, 0.01],
            [0.00, 0.01, 0.05],
        ]
    )
    result = minimum_variance_portfolio(mu, sigma, max_weight=0.60)
    assert result["success"]
    np.testing.assert_allclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= -1e-8)
    assert np.all(result["weights"] <= 0.60 + 1e-8)


def test_maximum_sharpe_portfolio():
    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.03, 0.01],
            [0.00, 0.01, 0.05],
        ]
    )
    result = maximum_sharpe_portfolio(mu, sigma, risk_free_rate=0.02, max_weight=0.60)
    assert result["success"]
    np.testing.assert_allclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= -1e-8)
    assert np.all(result["weights"] <= 0.60 + 1e-8)


def test_portfolio_weight_constraint_long_only():
    """Verify that long-only constraint is enforced."""
    mu = np.array([0.08, 0.10, 0.12, 0.06])
    sigma = np.array([
        [0.04, 0.01, 0.00, 0.01],
        [0.01, 0.03, 0.01, 0.00],
        [0.00, 0.01, 0.05, 0.01],
        [0.01, 0.00, 0.01, 0.02],
    ])
    result = minimum_variance_portfolio(mu, sigma, max_weight=0.40)
    assert np.all(result["weights"] >= -1e-10), "Negative weights found in long-only portfolio"


def test_portfolio_weight_sum():
    """Verify that portfolio weights sum to exactly 1.0."""
    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])
    result_mv = minimum_variance_portfolio(mu, sigma, max_weight=0.50)
    result_ms = maximum_sharpe_portfolio(mu, sigma, max_weight=0.50)
    
    np.testing.assert_allclose(result_mv["weights"].sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(result_ms["weights"].sum(), 1.0, atol=1e-10)


def test_portfolio_return_and_volatility():
    """Verify portfolio return and volatility calculations match manual computation."""
    weights = np.array([0.30, 0.50, 0.20])
    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])
    
    # Manual calculation
    exp_ret_manual = weights @ mu
    variance_manual = weights @ sigma @ weights
    vol_manual = np.sqrt(variance_manual)
    
    # Function calculation
    exp_ret_func = portfolio_expected_return(weights, mu)
    vol_func = portfolio_volatility(weights, sigma)
    
    np.testing.assert_allclose(exp_ret_manual, exp_ret_func, rtol=1e-10)
    np.testing.assert_allclose(vol_manual, vol_func, rtol=1e-10)


def test_portfolio_sharpe_ratio():
    """Verify Sharpe ratio calculation consistency."""
    weights = np.array([0.30, 0.50, 0.20])
    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])
    rf = 0.02
    
    exp_ret = portfolio_expected_return(weights, mu)
    vol = portfolio_volatility(weights, sigma)
    sharpe_manual = (exp_ret - rf) / vol
    sharpe_func = portfolio_sharpe_ratio(weights, mu, sigma, rf)
    
    np.testing.assert_allclose(sharpe_manual, sharpe_func, rtol=1e-10)


def test_minimum_variance_less_than_equal_weight():
    """Verify that minimum variance solution has lower or equal variance than equal weight."""
    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])
    
    ew_weights = compute_equal_weight_portfolio(len(mu))
    ew_variance = portfolio_volatility(ew_weights, sigma) ** 2
    
    mv_result = minimum_variance_portfolio(mu, sigma, max_weight=0.60)
    mv_variance = portfolio_volatility(mv_result["weights"], sigma) ** 2
    
    assert mv_variance <= ew_variance + 1e-8, "Minimum variance should be <= equal weight variance"


def test_efficient_frontier_feasibility():
    """Verify that efficient frontier points satisfy constraints."""
    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])
    max_w = 0.50
    
    frontier = efficient_frontier(mu, sigma, max_weight=max_w, num_points=10)
    
    assert len(frontier) > 0, "Efficient frontier should have at least one point"
    
    for point in frontier:
        weights = point["weights"]
        
        # Check constraints
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), "Weights don't sum to 1"
        assert np.all(weights >= -1e-8), "Negative weights"
        assert np.all(weights <= max_w + 1e-8), "Weight exceeds max_weight"
        
        # Check that return matches target
        exp_ret = portfolio_expected_return(weights, mu)
        assert np.isclose(exp_ret, point["target_return"], atol=1e-8), "Return doesn't match target"
        
        # Check volatility is positive
        assert point["volatility"] >= 0, "Volatility should be non-negative"


def test_expected_return_shrinkage_lambda_zero_and_one():
    """Shrinkage should recover the original mu at lambda=0 and target at lambda=1."""
    from src.optimization.robust import shrink_expected_returns

    mu = np.array([0.10, 0.12, 0.08])
    target = np.array([0.05, 0.05, 0.05])

    shrunk_zero = shrink_expected_returns(mu, target, lambda_value=0.0)
    shrunk_one = shrink_expected_returns(mu, target, lambda_value=1.0)

    np.testing.assert_allclose(shrunk_zero, mu)
    np.testing.assert_allclose(shrunk_one, target)


def test_expected_return_shrinkage_midpoint_formula():
    """Intermediate lambda should interpolate linearly between sample and target mean."""
    from src.optimization.robust import shrink_expected_returns

    mu = np.array([0.10, 0.12, 0.08])
    target = np.array([0.02, 0.02, 0.02])

    shrunk = shrink_expected_returns(mu, target, lambda_value=0.5)
    expected = 0.5 * mu + 0.5 * target

    np.testing.assert_allclose(shrunk, expected)


def test_turnover_metric_matches_definition():
    """Turnover should be half the L1 distance between weights."""
    from src.optimization.robust import compute_turnover

    prev = pd.Series({"A": 0.50, "B": 0.50})
    curr = pd.Series({"A": 0.80, "B": 0.20})

    turnover = compute_turnover(prev, curr)
    assert np.isclose(turnover, 0.30)


def test_turnover_zero_for_identical_portfolios():
    """Identical portfolios should have zero turnover."""
    from src.optimization.robust import compute_turnover

    prev = pd.Series({"A": 0.20, "B": 0.80})
    curr = pd.Series({"A": 0.20, "B": 0.80})

    assert np.isclose(compute_turnover(prev, curr), 0.0)


def test_transaction_costs_are_zero_when_turnover_is_zero():
    """Zero turnover must imply zero cost."""
    from src.optimization.robust import estimate_transaction_costs

    costs = estimate_transaction_costs(turnover=0.0, cost_rate_bps=10.0)
    assert np.isclose(costs, 0.0)


def test_robust_max_sharpe_constraints():
    """Robust portfolio optimizer should satisfy long-only and sum-to-one constraints."""
    from src.optimization.robust import robust_maximum_sharpe_portfolio

    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])

    result = robust_maximum_sharpe_portfolio(mu, sigma, risk_free_rate=0.02, max_weight=0.60, lambda_value=0.5)

    assert result["success"]
    np.testing.assert_allclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= -1e-8)
    assert np.all(result["weights"] <= 0.60 + 1e-8)


def test_turnover_aware_max_sharpe_constraints():
    """Turnover-aware portfolio optimizer should satisfy long-only and sum-to-one constraints."""
    from src.optimization.robust import turnover_aware_maximum_sharpe_portfolio

    mu = np.array([0.08, 0.10, 0.12])
    sigma = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])
    prev = np.array([0.2, 0.3, 0.5])

    result = turnover_aware_maximum_sharpe_portfolio(
        mu,
        sigma,
        previous_weights=prev,
        gamma=0.25,
        risk_free_rate=0.02,
        max_weight=0.60,
    )

    assert result["success"]
    np.testing.assert_allclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= -1e-8)
    assert np.all(result["weights"] <= 0.60 + 1e-8)
