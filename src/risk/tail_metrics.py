"""Standalone tail-risk metrics for empirical VaR and RU CVaR."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _as_1d_float_array(values: Iterable[float] | np.ndarray, *, name: str = "values") -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return arr


def _validate_alpha(alpha: float) -> float:
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")
    return float(alpha)


def empirical_var(losses: Iterable[float] | np.ndarray, alpha: float) -> float:
    """Deterministic right-continuous empirical VaR for a loss vector.

    Canonical loss definition is L_t = -r_t. Gains therefore appear as negative
    losses, and no clipping or zero-flooring is applied.
    """
    arr = _as_1d_float_array(losses, name="losses")
    alpha = _validate_alpha(alpha)
    q = int(math.ceil(alpha * arr.size))
    loss_order = np.sort(arr)
    return float(loss_order[q - 1])


def empirical_cvar(losses: Iterable[float] | np.ndarray, alpha: float) -> float:
    """Empirical CVaR using the Rockafellar-Uryasev finite-sample definition.

    Canonical loss definition: L_t = -r_t.
    The RU excess-loss term is computed as max(L_t - zeta, 0), while the original
    loss sample itself remains unclipped.
    """
    arr = _as_1d_float_array(losses, name="losses")
    alpha = _validate_alpha(alpha)
    if arr.size == 0:
        raise ValueError("losses must not be empty.")

    sorted_losses = np.sort(arr)
    candidates = sorted_losses.copy()
    if candidates.size == 1:
        return float(candidates[0])

    def objective(zeta: float) -> float:
        return float(zeta + np.sum(np.maximum(arr - zeta, 0.0)) / ((1.0 - alpha) * arr.size))

    best_value = math.inf
    for zeta in candidates:
        val = objective(zeta)
        if val < best_value:
            best_value = val
    return float(best_value)


def downside_deviation(returns: Iterable[float] | np.ndarray) -> float:
    """Root-mean-square downside deviation relative to zero.

    This is the same negative-only convention used in the repository's Sortino ratio,
    where only returns below zero contribute.
    """
    arr = _as_1d_float_array(returns, name="returns")
    negative = arr[arr < 0.0]
    if negative.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(negative))))


def worst_daily_return(returns: Iterable[float] | np.ndarray) -> float:
    """Return the minimum daily return as a signed return value."""
    arr = _as_1d_float_array(returns, name="returns")
    return float(np.min(arr))


def worst_compounded_n_day_return(returns: Iterable[float] | np.ndarray, window: int) -> float:
    """Return the minimum overlapping compounded return over a fixed horizon."""
    arr = _as_1d_float_array(returns, name="returns")
    if window <= 0:
        raise ValueError("window must be positive.")
    if window > arr.size:
        raise ValueError("window cannot exceed the number of observations.")

    minima = []
    for start in range(0, arr.size - window + 1):
        rolling = arr[start : start + window]
        comp = np.prod(1.0 + rolling) - 1.0
        minima.append(float(comp))
    return float(np.min(minima))
