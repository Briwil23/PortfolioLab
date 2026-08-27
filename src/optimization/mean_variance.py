"""Mean-variance optimization for portfolio construction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def compute_equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """Construct a uniform-weight portfolio."""
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")
    return np.full(n_assets, 1.0 / n_assets, dtype=float)


def portfolio_expected_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Compute the expected portfolio return."""
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)
    if weights.shape != expected_returns.shape:
        raise ValueError("weights and expected_returns must have the same shape.")
    return float(weights @ expected_returns)


def portfolio_volatility(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    """Compute the portfolio volatility from a covariance matrix."""
    weights = np.asarray(weights, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)
    variance = float(weights @ covariance_matrix @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def portfolio_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Compute the Sharpe ratio for a portfolio."""
    exp_ret = portfolio_expected_return(weights, expected_returns)
    vol = portfolio_volatility(weights, covariance_matrix)
    if vol <= 0:
        return 0.0
    return float((exp_ret - risk_free_rate) / vol)


def minimum_variance_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    max_weight: float = 0.30,
) -> dict[str, Any]:
    """Solve the long-only minimum-variance portfolio problem."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    n_assets = expected_returns.shape[0]
    initial_weights = compute_equal_weight_portfolio(n_assets)
    bounds = [(0.0, max_weight)] * n_assets
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w: np.ndarray) -> float:
        return float(w @ covariance_matrix @ w)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = result.x
    else:
        weights = initial_weights.copy()
        logger.warning("Minimum variance optimization failed. Falling back to equal weights.")

    weights = np.clip(weights, 0.0, max_weight)
    weights = weights / weights.sum() if weights.sum() > 0 else initial_weights

    exp_ret = portfolio_expected_return(weights, expected_returns)
    volatility = portfolio_volatility(weights, covariance_matrix)
    sharpe = portfolio_sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate=0.0)

    return {
        "weights": weights,
        "expected_return": exp_ret,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
    }


def maximum_sharpe_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.30,
) -> dict[str, Any]:
    """Numerically maximize the Sharpe ratio under long-only constraints."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    n_assets = expected_returns.shape[0]
    initial_weights = compute_equal_weight_portfolio(n_assets)
    bounds = [(0.0, max_weight)] * n_assets
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w: np.ndarray) -> float:
        exp_ret = portfolio_expected_return(w, expected_returns)
        vol = portfolio_volatility(w, covariance_matrix)
        if vol <= 1e-12:
            return -1e12
        sharp = (exp_ret - risk_free_rate) / vol
        return -float(sharp)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = result.x
    else:
        weights = initial_weights.copy()
        logger.warning("Maximum Sharpe optimization failed. Falling back to equal weights.")

    weights = np.clip(weights, 0.0, max_weight)
    weights = weights / weights.sum() if weights.sum() > 0 else initial_weights

    exp_ret = portfolio_expected_return(weights, expected_returns)
    volatility = portfolio_volatility(weights, covariance_matrix)
    sharpe = portfolio_sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate)

    return {
        "weights": weights,
        "expected_return": exp_ret,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
    }


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    target_returns: np.ndarray | None = None,
    max_weight: float = 0.30,
    num_points: int = 25,
) -> list[dict[str, Any]]:
    """Generate a constrained efficient frontier using target returns."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    if target_returns is None:
        lower_bound = float(np.min(expected_returns))
        upper_bound = float(np.max(expected_returns))
        target_returns = np.linspace(lower_bound, upper_bound, num=num_points)

    frontier = []
    bounds = [(0.0, max_weight)] * expected_returns.shape[0]

    for target_return in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target_return: float(w @ expected_returns) - target},
        )

        initial_weights = compute_equal_weight_portfolio(expected_returns.shape[0])
        result = minimize(
            lambda w: float(w @ covariance_matrix @ w),
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            continue

        weights = np.clip(result.x, 0.0, max_weight)
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()

        portfolio_return = portfolio_expected_return(weights, expected_returns)
        vol = portfolio_volatility(weights, covariance_matrix)
        sharpe = portfolio_sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate=0.0)

        frontier.append(
            {
                "target_return": float(portfolio_return),
                "volatility": float(vol),
                "sharpe_ratio": float(sharpe),
                "weights": weights,
                "success": True,
            }
        )

    return frontier
