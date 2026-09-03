"""Risk-only portfolio construction routines for Milestone 4."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LabeledArray:
    values: np.ndarray
    labels: list[str]


def normalize_asset_labels(asset_labels: Iterable[str]) -> list[str]:
    """Validate labels and return a deterministic list."""
    labels = [str(label) for label in asset_labels]
    if not labels:
        raise ValueError("asset_labels must contain at least one asset.")
    if len(labels) != len(set(labels)):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(f"Duplicate asset labels are not allowed: {duplicates}")
    return labels


def _coerce_labeled_vector(
    values: pd.Series | np.ndarray | list[float] | tuple[float, ...],
    asset_labels: Iterable[str] | None = None,
    *,
    name: str,
) -> _LabeledArray:
    if isinstance(values, pd.Series):
        labels = [str(label) for label in values.index.tolist()]
        if asset_labels is not None:
            requested = normalize_asset_labels(asset_labels)
            if set(labels) != set(requested):
                missing = sorted(set(requested) - set(labels))
                extra = sorted(set(labels) - set(requested))
                raise ValueError(f"{name} label mismatch. Missing assets: {missing}. Extra assets: {extra}.")
            values = values.reindex(requested)
            labels = requested
        else:
            labels = normalize_asset_labels(labels)
        return _LabeledArray(values.to_numpy(dtype=float), labels)

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if asset_labels is None:
        raise ValueError(f"{name} requires asset_labels when not provided as a pandas Series.")
    labels = normalize_asset_labels(asset_labels)
    if len(labels) != array.shape[0]:
        raise ValueError(f"{name} length does not match asset_labels.")
    return _LabeledArray(array, labels)


def _coerce_labeled_matrix(
    values: pd.DataFrame | np.ndarray | list[list[float]],
    asset_labels: Iterable[str] | None = None,
    *,
    name: str,
) -> _LabeledArray:
    if isinstance(values, pd.DataFrame):
        labels = [str(label) for label in values.columns.tolist()]
        if len(labels) != len(set(labels)):
            duplicates = sorted({label for label in labels if labels.count(label) > 1})
            raise ValueError(f"Duplicate asset labels are not allowed: {duplicates}")
        if asset_labels is not None:
            requested = normalize_asset_labels(asset_labels)
            if set(labels) != set(requested):
                missing = sorted(set(requested) - set(labels))
                extra = sorted(set(labels) - set(requested))
                raise ValueError(f"{name} label mismatch. Missing assets: {missing}. Extra assets: {extra}.")
            values = values.reindex(index=requested, columns=requested)
            labels = requested
        else:
            labels = normalize_asset_labels(labels)
            values = values.reindex(index=labels, columns=labels)
        matrix = values.to_numpy(dtype=float)
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix.")
        if asset_labels is None:
            raise ValueError(f"{name} requires asset_labels when not provided as a pandas DataFrame.")
        labels = normalize_asset_labels(asset_labels)
        if len(labels) != matrix.shape[0]:
            raise ValueError(f"{name} dimension does not match asset_labels.")

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric.")
    return _LabeledArray(matrix, labels)


def covariance_to_correlation(
    covariance_matrix: pd.DataFrame | np.ndarray | list[list[float]],
    asset_labels: Iterable[str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Convert covariance into labeled volatility and correlation estimates."""
    labeled = _coerce_labeled_matrix(covariance_matrix, asset_labels, name="covariance_matrix")
    covariance = labeled.values
    labels = labeled.labels
    volatility = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if np.any(~np.isfinite(volatility)):
        raise ValueError("covariance_matrix produced non-finite volatilities.")
    if np.any(volatility < 0):
        raise ValueError("covariance_matrix produced negative volatilities.")
    denom = np.outer(volatility, volatility)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0)
    np.fill_diagonal(correlation, 1.0)
    return pd.Series(volatility, index=labels, name="annualized_volatility"), pd.DataFrame(correlation, index=labels, columns=labels)


def reconstruct_covariance_from_correlation(
    volatility: pd.Series | np.ndarray | list[float],
    correlation: pd.DataFrame | np.ndarray | list[list[float]],
    asset_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Reconstruct covariance from volatilities and correlation."""
    vol_labeled = _coerce_labeled_vector(volatility, asset_labels, name="volatility")
    corr_labeled = _coerce_labeled_matrix(correlation, vol_labeled.labels, name="correlation")

    if np.any(~np.isfinite(vol_labeled.values)):
        raise ValueError("volatility contains non-finite values.")
    if np.any(vol_labeled.values < 0):
        raise ValueError("volatility must be non-negative.")

    reconstructed = np.diag(vol_labeled.values) @ corr_labeled.values @ np.diag(vol_labeled.values)
    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("Reconstructed covariance contains non-finite values.")
    if not np.allclose(reconstructed, reconstructed.T, atol=1e-12, rtol=0.0):
        raise ValueError("Reconstructed covariance must be symmetric.")
    if np.any(np.diag(reconstructed) < -1e-12):
        raise ValueError("Reconstructed covariance diagonal must be non-negative.")
    return pd.DataFrame(reconstructed, index=vol_labeled.labels, columns=vol_labeled.labels)


def perturb_single_asset_volatility(
    covariance_matrix: pd.DataFrame | np.ndarray | list[list[float]],
    asset: str,
    scale: float,
    asset_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Perturb one asset volatility while preserving correlation."""
    if scale <= 0:
        raise ValueError("scale must be positive.")
    volatility, correlation = covariance_to_correlation(covariance_matrix, asset_labels=asset_labels)
    if asset not in volatility.index:
        raise ValueError(f"Unknown asset: {asset}")
    perturbed = volatility.copy()
    perturbed.loc[asset] = float(perturbed.loc[asset]) * float(scale)
    return reconstruct_covariance_from_correlation(perturbed, correlation, asset_labels=volatility.index.tolist())


def risk_contributions(
    weights: pd.Series | np.ndarray | list[float],
    covariance_matrix: pd.DataFrame | np.ndarray | list[list[float]],
    asset_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return labeled risk contributions and normalized risk contributions."""
    w = _coerce_labeled_vector(weights, asset_labels, name="weights")
    cov = _coerce_labeled_matrix(covariance_matrix, w.labels, name="covariance_matrix")
    portfolio_volatility = float(np.sqrt(max(w.values @ cov.values @ w.values, 0.0)))

    if portfolio_volatility <= 1e-12:
        zeros = np.zeros_like(w.values)
        return pd.DataFrame(
            {
                "asset": w.labels,
                "weight": w.values,
                "marginal_risk_contribution": zeros,
                "absolute_risk_contribution": zeros,
                "normalized_risk_contribution": zeros,
            }
        )

    marginal = cov.values @ w.values / portfolio_volatility
    absolute = w.values * marginal
    normalized = absolute / portfolio_volatility
    return pd.DataFrame(
        {
            "asset": w.labels,
            "weight": w.values,
            "marginal_risk_contribution": marginal,
            "absolute_risk_contribution": absolute,
            "normalized_risk_contribution": normalized,
        }
    )


def risk_contribution_report(
    weights: pd.Series | np.ndarray | list[float],
    covariance_matrix: pd.DataFrame | np.ndarray | list[list[float]],
    asset_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Summarize risk contribution identities and dispersion."""
    rc = risk_contributions(weights, covariance_matrix, asset_labels=asset_labels)
    normalized = rc["normalized_risk_contribution"].to_numpy(dtype=float)
    absolute = rc["absolute_risk_contribution"].to_numpy(dtype=float)
    portfolio_volatility = float(np.sum(absolute))
    target = 1.0 / max(len(normalized), 1)
    return {
        "assets": rc["asset"].tolist(),
        "portfolio_volatility": portfolio_volatility,
        "absolute_risk_contribution_sum": float(np.sum(absolute)),
        "absolute_risk_contribution_abs_sum": float(np.sum(np.abs(absolute))),
        "normalized_risk_contribution_sum": float(np.sum(normalized)),
        "target_normalized_risk_contribution": float(target),
        "dispersion": float(np.sum((normalized - target) ** 2)),
        "max_normalized_risk_contribution": float(np.max(normalized)) if len(normalized) else 0.0,
        "top3_normalized_risk_contribution": float(np.sum(np.sort(normalized)[-3:])) if len(normalized) else 0.0,
        "table": rc,
    }


def inverse_volatility_portfolio(
    annualized_volatility: pd.Series | np.ndarray | list[float],
    max_weight: float = 0.30,
    asset_labels: Iterable[str] | None = None,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Construct an inverse-volatility portfolio with deterministic cap redistribution."""
    vol = _coerce_labeled_vector(annualized_volatility, asset_labels, name="annualized_volatility")
    values = vol.values
    labels = vol.labels

    if np.any(~np.isfinite(values)):
        raise ValueError("annualized_volatility contains non-finite values.")
    if np.any(values <= 0):
        raise ValueError("annualized_volatility must be strictly positive.")
    if not np.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("max_weight must be positive and finite.")
    if len(values) * float(max_weight) < 1.0 - tolerance:
        raise ValueError("Inverse-volatility cap is infeasible because N * max_weight < 1.")

    scores = 1.0 / values
    weights = np.zeros_like(scores)
    remaining = np.ones_like(scores, dtype=bool)
    remaining_capital = 1.0
    iterations = 0

    while remaining.any():
        iterations += 1
        active_scores = scores[remaining]
        active_total = float(active_scores.sum())
        if active_total <= 0 or not np.isfinite(active_total):
            raise ValueError("Inverse-volatility scores are invalid.")

        provisional = remaining_capital * active_scores / active_total
        over = provisional > (max_weight + tolerance)
        active_indices = np.where(remaining)[0]

        if not np.any(over):
            weights[active_indices] = provisional
            break

        capped_indices = active_indices[over]
        weights[capped_indices] = max_weight
        remaining[capped_indices] = False
        remaining_capital = 1.0 - float(weights[~remaining].sum())
        if remaining_capital < -tolerance:
            raise ValueError("Inverse-volatility cap redistribution became infeasible.")

    weights = np.clip(weights, 0.0, max_weight)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Inverse-volatility optimization failed to produce valid weights.")
    weights = weights / total

    report = risk_contribution_report(weights, np.diag(values ** 2), asset_labels=labels)
    return {
        "weights": weights,
        "weights_by_asset": {label: float(weight) for label, weight in zip(labels, weights)},
        "scores": scores,
        "success": True,
        "status": 0,
        "message": "Inverse-volatility portfolio constructed successfully.",
        "iterations": iterations,
        "risk_contribution_report": report,
    }


def equal_risk_contribution_portfolio(
    covariance_matrix: pd.DataFrame | np.ndarray | list[list[float]],
    max_weight: float = 0.30,
    asset_labels: Iterable[str] | None = None,
    tolerance: float = 1e-10,
    objective_tolerance: float = 1e-12,
    maxiter: int = 2000,
) -> dict[str, Any]:
    """Solve a capped equal-risk-contribution portfolio using covariance only."""
    cov = _coerce_labeled_matrix(covariance_matrix, asset_labels, name="covariance_matrix")
    labels = cov.labels
    matrix = cov.values

    if len(labels) * float(max_weight) < 1.0 - tolerance:
        raise ValueError("ERC cap is infeasible because N * max_weight < 1.")
    if np.any(np.diag(matrix) < -tolerance):
        raise ValueError("covariance_matrix has a negative diagonal entry.")

    n_assets = len(labels)
    initial_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    if np.any(initial_weights > max_weight + tolerance):
        raise ValueError("Equal-weight initialization violates max_weight.")

    def objective(weights: np.ndarray) -> float:
        sigma_p = float(np.sqrt(max(weights @ matrix @ weights, 0.0)))
        if sigma_p <= 1e-12:
            return 1e6
        normalized_rc = (weights * (matrix @ weights) / sigma_p) / sigma_p
        target = 1.0 / n_assets
        return float(np.sum((normalized_rc - target) ** 2))

    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, max_weight)] * n_assets

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": float(objective_tolerance)},
    )

    if result.success:
        weights = np.clip(result.x, 0.0, max_weight)
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("ERC optimization returned invalid weights.")
        weights = weights / total
        success = True
        fallback_used = False
        message = str(result.message)
    else:
        weights = initial_weights.copy()
        success = False
        fallback_used = True
        message = f"ERC optimization failed; falling back to equal weights. {result.message}"
        logger.warning(message)

    report = risk_contribution_report(weights, matrix, asset_labels=labels)
    return {
        "weights": weights,
        "weights_by_asset": {label: float(weight) for label, weight in zip(labels, weights)},
        "success": bool(success),
        "solver_success": bool(success),
        "fallback_used": bool(fallback_used),
        "status": int(result.status),
        "message": message,
        "solver_message": message,
        "objective_value": float(objective(weights)),
        "risk_contribution_report": report,
        "max_normalized_risk_contribution": report["max_normalized_risk_contribution"],
        "top3_normalized_risk_contribution": report["top3_normalized_risk_contribution"],
        "risk_contribution_dispersion": report["dispersion"],
    }