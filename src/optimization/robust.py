"""Robust portfolio construction for Milestone 3.

This module isolates the empirical failure modes identified in the diagnostic
study:
- noisy expected-return estimates
- excessive turnover / churn
- cap-bound concentration

The implementation deliberately keeps the classical Maximum-Sharpe optimizer as
an unchanged baseline and adds transparent robustness layers instead of
replacing it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.optimization.mean_variance import (
    compute_equal_weight_portfolio,
    portfolio_expected_return,
    portfolio_volatility,
    portfolio_sharpe_ratio,
)

logger = logging.getLogger(__name__)


def shrink_expected_returns(
    expected_returns: np.ndarray,
    target: np.ndarray | None = None,
    lambda_value: float = 0.0,
    target_type: str = "grand_mean",
) -> np.ndarray:
    """Shrink sample expected returns toward a conservative target.

    mu_robust = (1 - lambda) * mu_sample + lambda * mu_target

    Supported targets:
    - "grand_mean": every asset shrinks toward the cross-sectional average
    - "zero": shrink to zero expected excess return
    - "custom": use a user-provided target vector
    """
    mu = np.asarray(expected_returns, dtype=float)
    if mu.ndim != 1:
        raise ValueError("expected_returns must be a one-dimensional array.")

    if not 0.0 <= float(lambda_value) <= 1.0:
        raise ValueError("lambda_value must be in [0, 1].")

    if target is None:
        if target_type == "grand_mean":
            target = np.full_like(mu, float(mu.mean()))
        elif target_type == "zero":
            target = np.zeros_like(mu)
        elif target_type == "custom":
            raise ValueError("A custom target vector is required when target_type='custom'.")
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")
    else:
        target = np.asarray(target, dtype=float)
        if target.shape != mu.shape:
            raise ValueError("target must have the same shape as expected_returns.")

    return (1.0 - float(lambda_value)) * mu + float(lambda_value) * target


def compute_turnover(
    previous_weights: np.ndarray | list[float] | dict[str, float] | Any,
    current_weights: np.ndarray | list[float] | dict[str, float] | Any,
) -> float:
    """Compute one-step portfolio turnover as 0.5 * sum(abs(w_t - w_{t-1}))."""
    if hasattr(previous_weights, "to_dict"):
        previous_weights = previous_weights.to_dict()
    if hasattr(current_weights, "to_dict"):
        current_weights = current_weights.to_dict()

    if isinstance(previous_weights, dict):
        prev = np.asarray([previous_weights[k] for k in sorted(previous_weights)], dtype=float)
    elif isinstance(previous_weights, (list, tuple, np.ndarray)):
        prev = np.asarray(previous_weights, dtype=float)
    else:
        raise TypeError("Unsupported previous_weights type.")

    if isinstance(current_weights, dict):
        curr = np.asarray([current_weights[k] for k in sorted(current_weights)], dtype=float)
    elif isinstance(current_weights, (list, tuple, np.ndarray)):
        curr = np.asarray(current_weights, dtype=float)
    else:
        raise TypeError("Unsupported current_weights type.")

    if prev.shape != curr.shape:
        raise ValueError("previous_weights and current_weights must have the same shape.")

    return float(0.5 * np.sum(np.abs(curr - prev)))


def estimate_transaction_costs(turnover: float, cost_rate_bps: float = 0.0) -> float:
    """Convert turnover into a proportional trading-cost estimate.

    cost_t = turnover * transaction_cost_rate
    with a stylized cost rate expressed in basis points.
    """
    if turnover < 0:
        raise ValueError("turnover must be non-negative.")
    rate = float(cost_rate_bps) / 10_000.0
    return float(turnover * rate)


def _solve_maximum_sharpe(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.30,
) -> dict[str, Any]:
    """Internal optimizer for Sharpe maximization with a long-only full-investment constraint."""
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
        return -float((exp_ret - risk_free_rate) / vol)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = np.clip(result.x, 0.0, max_weight)
        weights = weights / weights.sum() if weights.sum() > 0 else initial_weights.copy()
        return {
            "weights": weights,
            "success": True,
            "status": int(result.status),
            "message": str(result.message),
        }

    fallback = initial_weights.copy()
    fallback = np.clip(fallback, 0.0, max_weight)
    fallback = fallback / fallback.sum() if fallback.sum() > 0 else initial_weights.copy()
    logger.warning("Maximum-Sharpe optimization failed. Falling back to equal weights. Message: %s", result.message)
    return {
        "weights": fallback,
        "success": False,
        "status": int(result.status),
        "message": str(result.message),
    }


def robust_maximum_sharpe_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.30,
    lambda_value: float = 0.0,
    target_type: str = "grand_mean",
    target: np.ndarray | None = None,
) -> dict[str, Any]:
    """Maximise Sharpe ratio after shrinking expected returns toward a target."""
    mu_sample = np.asarray(expected_returns, dtype=float)
    mu_robust = shrink_expected_returns(
        mu_sample,
        target=target,
        lambda_value=lambda_value,
        target_type=target_type,
    )

    result = _solve_maximum_sharpe(mu_robust, covariance_matrix, risk_free_rate=risk_free_rate, max_weight=max_weight)
    weights = result["weights"]
    exp_ret = portfolio_expected_return(weights, mu_sample)
    vol = portfolio_volatility(weights, covariance_matrix)
    sharpe = portfolio_sharpe_ratio(weights, mu_sample, covariance_matrix, risk_free_rate)

    return {
        "weights": weights,
        "expected_return": float(exp_ret),
        "volatility": float(vol),
        "sharpe_ratio": float(sharpe),
        "mu_sample": mu_sample,
        "mu_robust": mu_robust,
        "lambda_value": float(lambda_value),
        "target_type": target_type,
        "success": bool(result["success"]),
        "status": int(result["status"]),
        "message": str(result["message"]),
    }


def turnover_aware_maximum_sharpe_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    previous_weights: np.ndarray | list[float] | None = None,
    gamma: float = 0.10,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.30,
) -> dict[str, Any]:
    """Maximize Sharpe ratio minus a turnover penalty relative to previous portfolio weights.

    The turnover penalty is implemented explicitly as:
        gamma * 0.5 * sum_i |w_i - w_prev_i|

    This is represented in the SLSQP problem by auxiliary variables s_i with:
        s_i >= w_i - w_prev_i
        s_i >= -(w_i - w_prev_i)
    so the optimization solves a mathematically consistent absolute-value penalty.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    if previous_weights is None:
        previous_weights = compute_equal_weight_portfolio(expected_returns.shape[0])
    previous_weights = np.asarray(previous_weights, dtype=float)
    if previous_weights.shape != expected_returns.shape:
        raise ValueError("previous_weights must match the dimension of expected_returns.")

    n_assets = expected_returns.shape[0]
    initial_weights = compute_equal_weight_portfolio(n_assets)
    initial_slacks = np.zeros(n_assets, dtype=float)
    initial = np.concatenate([initial_weights, initial_slacks])
    bounds = [(0.0, max_weight)] * n_assets + [(0.0, None)] * n_assets
    constraints: list[dict[str, Any]] = [{"type": "eq", "fun": lambda v: np.sum(v[:n_assets]) - 1.0}]

    for idx in range(n_assets):
        def slack_pos(v: np.ndarray, i: int = idx) -> float:
            return float(v[n_assets + i] - (v[i] - previous_weights[i]))

        def slack_neg(v: np.ndarray, i: int = idx) -> float:
            return float(v[n_assets + i] - (previous_weights[i] - v[i]))

        constraints.append({"type": "ineq", "fun": slack_pos})
        constraints.append({"type": "ineq", "fun": slack_neg})

    def objective(v: np.ndarray) -> float:
        w = v[:n_assets]
        s = v[n_assets:]
        exp_ret = portfolio_expected_return(w, expected_returns)
        vol = portfolio_volatility(w, covariance_matrix)
        if vol <= 1e-12:
            return 1e12
        sharpe = (exp_ret - risk_free_rate) / vol
        turnover_penalty = 0.5 * np.sum(s)
        return float(-sharpe + gamma * turnover_penalty)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if result.success:
        weights = np.clip(result.x[:n_assets], 0.0, max_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        turnover = compute_turnover(previous_weights, weights)
        exp_ret = portfolio_expected_return(weights, expected_returns)
        vol = portfolio_volatility(weights, covariance_matrix)
        sharpe = portfolio_sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate)
        return {
            "weights": weights,
            "expected_return": float(exp_ret),
            "volatility": float(vol),
            "sharpe_ratio": float(sharpe),
            "turnover": float(turnover),
            "gamma": float(gamma),
            "previous_weights": previous_weights.copy(),
            "success": True,
            "status": int(result.status),
            "message": str(result.message),
        }

    fallback = initial_weights.copy()
    fallback = np.clip(fallback, 0.0, max_weight)
    if fallback.sum() > 0:
        fallback = fallback / fallback.sum()
    turnover = compute_turnover(previous_weights, fallback)
    exp_ret = portfolio_expected_return(fallback, expected_returns)
    vol = portfolio_volatility(fallback, covariance_matrix)
    sharpe = portfolio_sharpe_ratio(fallback, expected_returns, covariance_matrix, risk_free_rate)
    logger.warning("Turnover-aware optimization failed. Falling back to equal weights. Message: %s", result.message)
    return {
        "weights": fallback,
        "expected_return": float(exp_ret),
        "volatility": float(vol),
        "sharpe_ratio": float(sharpe),
        "turnover": float(turnover),
        "gamma": float(gamma),
        "previous_weights": previous_weights.copy(),
        "success": False,
        "status": int(result.status),
        "message": str(result.message),
    }


def combined_robust_turnover_aware_maximum_sharpe_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    previous_weights: np.ndarray | list[float] | None = None,
    gamma: float = 0.10,
    lambda_value: float = 0.50,
    target_type: str = "grand_mean",
    target: np.ndarray | None = None,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.30,
) -> dict[str, Any]:
    """Combine expected-return shrinkage and a turnover penalty in one objective."""
    mu_sample = np.asarray(expected_returns, dtype=float)
    mu_robust = shrink_expected_returns(
        mu_sample,
        target=target,
        lambda_value=lambda_value,
        target_type=target_type,
    )
    base = turnover_aware_maximum_sharpe_portfolio(
        mu_robust,
        covariance_matrix,
        previous_weights=previous_weights,
        gamma=gamma,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )
    base["mu_sample"] = mu_sample
    base["mu_robust"] = mu_robust
    base["lambda_value"] = float(lambda_value)
    base["target_type"] = target_type
    return base
