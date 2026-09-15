"""Minimum CVaR 95% optimizer mechanics.

This module implements the Rockafellar-Uryasev LP for a pure tail-risk objective
without expected returns, turnover penalties, or empirical backtests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linprog

from src.risk.tail_metrics import empirical_cvar


@dataclass(frozen=True)
class CVaROptimizerResult:
    """Structured, deterministic optimizer output."""

    weights: np.ndarray
    objective_cvar: float
    objective_value: float
    zeta: float
    u: np.ndarray
    success: bool
    status: int
    message: str
    alpha: float
    max_weight: float
    scenario_count: int
    asset_labels: list[str]
    post_validation: bool
    max_constraint_violation: float
    objective_gap: float


def _validate_asset_labels(asset_labels: Iterable[str]) -> list[str]:
    labels = [str(label) for label in asset_labels]
    if not labels:
        raise ValueError("asset_labels must not be empty.")
    if len(labels) != len(set(labels)):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(f"Duplicate asset labels are not allowed: {duplicates}")
    return labels


def _coerce_return_matrix(
    returns: np.ndarray | list[list[float]],
    *,
    asset_labels: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, N).")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("returns must contain at least one scenario row and one asset column.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains NaN or inf values.")
    labels = _validate_asset_labels(asset_labels)
    if len(labels) != arr.shape[1]:
        raise ValueError("asset_labels length must match the number of return columns.")
    return arr, labels


def _canonical_losses_from_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return -returns @ weights


def _build_ru_lp(
    returns: np.ndarray,
    alpha: float,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]], tuple[np.ndarray, np.ndarray]]:
    n_scenarios, n_assets = returns.shape
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1.")
    if n_assets * max_weight < 1.0:
        raise ValueError("max_weight is infeasible because the fully invested portfolio cannot sum to 1.")

    n_vars = n_assets + 1 + n_scenarios
    c = np.zeros(n_vars, dtype=float)
    c[n_assets] = 1.0
    u_coeff = 1.0 / ((1.0 - alpha) * n_scenarios)
    c[n_assets + 1 :] = u_coeff

    A_ub = np.zeros((n_scenarios, n_vars), dtype=float)
    b_ub = np.zeros(n_scenarios, dtype=float)
    for t in range(n_scenarios):
        A_ub[t, :n_assets] = -returns[t]
        A_ub[t, n_assets] = -1.0
        A_ub[t, n_assets + 1 + t] = -1.0
        b_ub[t] = 0.0

    A_eq = np.zeros((1, n_vars), dtype=float)
    A_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, max_weight)] * n_assets
    bounds.append((None, None))
    bounds.extend([(0.0, None)] * n_scenarios)
    return c, A_ub, b_ub, bounds, (A_eq, b_eq)


def minimum_cvar_95_portfolio(
    returns: np.ndarray | list[list[float]],
    *,
    asset_labels: Iterable[str],
    alpha: float = 0.95,
    max_weight: float = 0.30,
) -> dict[str, Any]:
    """Solve the minimum CVaR 95% LP under a long-only sum-to-one constraint.

    The canonical loss is L_t(w) = -r_t' w. The optimizer minimizes the RU
    objective zeta + 1/((1-alpha)T) * sum_t u_t subject to u_t >= L_t(w) - zeta,
    u_t >= 0, sum_i w_i = 1, and 0 <= w_i <= max_weight.
    """
    arr, labels = _coerce_return_matrix(returns, asset_labels=asset_labels)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("returns must contain at least one scenario row and one asset column.")
    if arr.shape[1] * max_weight < 1.0:
        raise ValueError("max_weight is infeasible because the fully invested portfolio cannot sum to 1.")

    c, A_ub, b_ub, bounds, (A_eq, b_eq) = _build_ru_lp(arr, alpha, max_weight)
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )

    if not result.success:
        raise ValueError(f"CVaR optimizer failed: {result.message}")
    if not hasattr(result, "x") or result.x is None:
        raise ValueError("CVaR optimizer returned no solution vector.")
    x = np.asarray(result.x, dtype=float)
    if x.size < arr.shape[1] + 1 + arr.shape[0]:
        raise ValueError("CVaR optimizer returned an invalid variable vector.")

    w = np.asarray(x[: arr.shape[1]], dtype=float)
    zeta = float(x[arr.shape[1]])
    u = np.asarray(x[arr.shape[1] + 1 :], dtype=float)

    losses = _canonical_losses_from_returns(arr, w)
    objective_value = float(zeta + np.sum(u) / ((1.0 - alpha) * arr.shape[0]))
    objective_cvar = float(empirical_cvar(losses, alpha=alpha))

    max_constraint_violation = 0.0
    for t in range(arr.shape[0]):
        lhs = u[t]
        rhs = losses[t] - zeta
        max_constraint_violation = max(max_constraint_violation, max(0.0, rhs - lhs))

    post_validation = (
        np.all(np.isfinite(w))
        and np.isfinite(zeta)
        and np.all(np.isfinite(u))
        and np.isclose(np.sum(w), 1.0, atol=1e-10, rtol=0.0)
        and np.all(w >= -1e-10)
        and np.all(w <= max_weight + 1e-10)
        and np.all(u >= -1e-10)
        and max_constraint_violation <= 1e-8
        and np.isfinite(objective_value)
        and np.isclose(objective_cvar, objective_value, atol=1e-10, rtol=0.0)
    )

    if not post_validation:
        raise ValueError("Post-solution validation failed for Minimum CVaR optimizer.")

    weights_by_ticker = {label: float(weight) for label, weight in zip(labels, w)}
    return {
        "weights": w,
        "weights_by_ticker": weights_by_ticker,
        "objective_cvar": objective_cvar,
        "objective_value": objective_value,
        "zeta": zeta,
        "u": u,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "solver": "linprog",
        "method": "highs",
        "alpha": float(alpha),
        "max_weight": float(max_weight),
        "scenario_count": int(arr.shape[0]),
        "asset_labels": labels,
        "post_validation": bool(post_validation),
        "max_constraint_violation": float(max_constraint_violation),
        "objective_gap": float(abs(objective_cvar - objective_value)),
    }


__all__ = ["minimum_cvar_95_portfolio", "CVaROptimizerResult"]
