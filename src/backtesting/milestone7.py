"""Milestone 7 minimum-CVaR walk-forward mechanics integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.backtesting.walk_forward import (
    RebalanceInfo,
    extract_holding_data,
    generate_rebalance_dates,
    normalize_asset_order,
)
from src.optimization.cvar import minimum_cvar_95_portfolio

M7_STRATEGY_LABEL = "Minimum CVaR"
M7_ALPHA = 0.95
M7_TRAINING_OBSERVATIONS = 252
M7_COST_BPS_OPTIONS = (0.0, 5.0, 10.0, 25.0)


@dataclass(frozen=True)
class M7RebalanceCandidate:
    """Single rebalance candidate with training and holding boundaries."""

    rebalance_date: pd.Timestamp
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    holding_start: pd.Timestamp
    holding_end: pd.Timestamp | None
    training_returns: pd.DataFrame
    holding_returns: pd.DataFrame
    eligible: bool
    training_observations: int


def _validate_return_frame(frame: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Return frame must not be empty.")
    if any(frame.columns.duplicated()):
        raise ValueError("Duplicate asset columns are not allowed in return data.")
    if not isinstance(frame.columns, pd.Index):
        raise TypeError("Return frame columns must be an index-like object.")
    if not set(expected_columns).issubset(frame.columns):
        missing = sorted(set(expected_columns) - set(frame.columns))
        raise ValueError(f"Missing configured asset columns: {missing}")
    if set(frame.columns) - set(expected_columns):
        extra = sorted(set(frame.columns) - set(expected_columns))
        raise ValueError(f"Extra asset columns detected: {extra}")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Return frame contains NaN or inf values.")
    if not all(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns):
        raise ValueError("Return frame contains nonnumeric values.")
    return normalize_asset_order(frame, list(expected_columns))


def _strip_training_returns(returns_df: pd.DataFrame, rebalance_info: RebalanceInfo, tickers: list[str]) -> pd.DataFrame:
    window = returns_df[
        (returns_df.index >= rebalance_info.training_start)
        & (returns_df.index < rebalance_info.holding_start)
    ].copy()
    if len(window) < M7_TRAINING_OBSERVATIONS:
        return window
    window = _validate_return_frame(window, tickers)
    return window.tail(M7_TRAINING_OBSERVATIONS).copy()


def list_m7_rebalance_candidates(
    returns_df: pd.DataFrame,
    tickers: list[str],
    *,
    holdout_start_date: str | None = None,
    rebalance_frequency: str = "monthly",
    lookback_years: int = 3,
) -> list[M7RebalanceCandidate]:
    """List rebalance candidates for the canonical monthly schedule."""
    prices_df = pd.DataFrame(index=returns_df.index, columns=tickers, dtype=float)
    prices_df.iloc[:, :] = 1.0
    rebalance_infos = generate_rebalance_dates(
        prices_df,
        lookback_years=lookback_years,
        rebalance_frequency=rebalance_frequency,
        holdout_start_date=holdout_start_date,
    )
    candidates: list[M7RebalanceCandidate] = []
    for rebalance_info in rebalance_infos:
        training_window = _strip_training_returns(returns_df, rebalance_info, tickers)
        if len(training_window) < M7_TRAINING_OBSERVATIONS:
            candidates.append(
                M7RebalanceCandidate(
                    rebalance_date=pd.Timestamp(rebalance_info.rebalance_date),
                    training_start=pd.Timestamp(rebalance_info.training_start),
                    training_end=pd.Timestamp(rebalance_info.training_end),
                    holding_start=pd.Timestamp(rebalance_info.holding_start),
                    holding_end=pd.Timestamp(rebalance_info.holding_end) if rebalance_info.holding_end is not None else None,
                    training_returns=training_window,
                    holding_returns=pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype=float),
                    eligible=False,
                    training_observations=len(training_window),
                )
            )
            continue

        training_window = _validate_return_frame(training_window, tickers)
        training_window = training_window.tail(M7_TRAINING_OBSERVATIONS).copy()
        if len(training_window) != M7_TRAINING_OBSERVATIONS:
            raise ValueError("Training window length must be exactly 252 observations.")
        if training_window.index.max() >= rebalance_info.holding_start:
            raise ValueError("Training window must end strictly before the holding period starts.")
        holding_window = extract_holding_data(returns_df, rebalance_info)
        holding_window = _validate_return_frame(holding_window, tickers)
        candidates.append(
            M7RebalanceCandidate(
                rebalance_date=pd.Timestamp(rebalance_info.rebalance_date),
                training_start=pd.Timestamp(rebalance_info.training_start),
                training_end=pd.Timestamp(training_window.index.max()),
                holding_start=pd.Timestamp(rebalance_info.holding_start),
                holding_end=pd.Timestamp(rebalance_info.holding_end) if rebalance_info.holding_end is not None else None,
                training_returns=training_window,
                holding_returns=holding_window,
                eligible=True,
                training_observations=len(training_window),
            )
        )
    return candidates


def _rebalance_keyweight_map(weights: dict[str, float], tickers: list[str]) -> dict[str, float]:
    return {ticker: float(weights.get(ticker, 0.0)) for ticker in tickers}


def _turnover_from_weights(previous_weights: dict[str, float] | None, current_weights: dict[str, float], tickers: list[str]) -> float:
    if previous_weights is None:
        return 0.0
    prev = np.asarray([float(previous_weights.get(ticker, 0.0)) for ticker in tickers], dtype=float)
    curr = np.asarray([float(current_weights.get(ticker, 0.0)) for ticker in tickers], dtype=float)
    return float(np.sum(np.abs(curr - prev)) / 2.0)


def run_milestone7_rebalance(
    rebalance_info: RebalanceInfo,
    returns_df: pd.DataFrame,
    *,
    tickers: list[str],
    max_weight: float = 0.30,
    alpha: float = M7_ALPHA,
    previous_weights: dict[str, float] | None = None,
    cost_bps: float = 0.0,
) -> dict[str, Any]:
    """Compute the minimized-CVaR weight set for a single eligible rebalance."""
    training_window = _strip_training_returns(returns_df, rebalance_info, tickers)
    if len(training_window) < M7_TRAINING_OBSERVATIONS:
        raise ValueError(
            f"Insufficient training history for rebalance {rebalance_info.rebalance_date}. "
            f"Found {len(training_window)} observations; require {M7_TRAINING_OBSERVATIONS}."
        )
    training_window = _validate_return_frame(training_window, tickers)
    training_window = training_window.tail(M7_TRAINING_OBSERVATIONS).copy()
    if len(training_window) != M7_TRAINING_OBSERVATIONS:
        raise ValueError("Training window length must be exactly 252 observations.")
    if training_window.index.max() >= rebalance_info.holding_start:
        raise ValueError("Anti-lookahead violation: training window ends on or after the holding start.")

    training_matrix = training_window.loc[:, tickers].to_numpy(dtype=float)
    result = minimum_cvar_95_portfolio(
        training_matrix,
        asset_labels=tickers,
        alpha=alpha,
        max_weight=max_weight,
    )
    weights = result["weights_by_ticker"]
    weights = _rebalance_keyweight_map(weights, tickers)
    holding_returns = extract_holding_data(returns_df, rebalance_info)
    holding_returns = _validate_return_frame(holding_returns, tickers)

    gross_returns = pd.Series(
        np.asarray(holding_returns.loc[:, tickers].to_numpy(dtype=float) @ np.asarray([weights[ticker] for ticker in tickers], dtype=float)),
        index=holding_returns.index,
        name="gross_return",
    )

    turnover = _turnover_from_weights(previous_weights, weights, tickers)
    cost = turnover * (float(cost_bps) / 10_000.0)
    net_returns = gross_returns.copy()
    if len(net_returns) > 0 and previous_weights is not None:
        net_returns.iloc[0] = net_returns.iloc[0] - cost

    # Independent identity check for the portfolio return construction.
    manual = pd.Series(
        [
            float(np.sum([weights[ticker] * float(holding_returns.loc[idx, ticker]) for ticker in tickers]))
            for idx in holding_returns.index
        ],
        index=holding_returns.index,
        name="manual_return",
    )
    identity_error = float(np.max(np.abs(manual.to_numpy() - gross_returns.to_numpy()))) if len(manual) else 0.0

    return {
        "rebalance_date": pd.Timestamp(rebalance_info.rebalance_date),
        "training_start": pd.Timestamp(training_window.index.min()),
        "training_end": pd.Timestamp(training_window.index.max()),
        "holding_start": pd.Timestamp(rebalance_info.holding_start),
        "holding_end": pd.Timestamp(rebalance_info.holding_end) if rebalance_info.holding_end is not None else None,
        "strategy": M7_STRATEGY_LABEL,
        "alpha": float(alpha),
        "training_observations": int(len(training_window)),
        "scenario_count": int(len(training_window)),
        "weights": weights,
        "weight_sum": float(np.sum(list(weights.values()))),
        "objective_cvar": float(result["objective_cvar"]),
        "zeta": float(result["zeta"]),
        "turnover": float(turnover),
        "transaction_cost_bps": float(cost_bps),
        "transaction_cost": float(cost),
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "holding_return_identity_max_error": float(identity_error),
        "max_constraint_violation": float(result.get("max_constraint_violation", np.nan)),
        "solver_success": bool(result.get("success", True)),
        "solver_status": int(result.get("status", 0)),
        "solver_message": str(result.get("message", "ok")),
    }


def run_milestone7_walk_forward(
    returns_df: pd.DataFrame,
    *,
    tickers: list[str],
    holdout_start_date: str | None = None,
    rebalance_frequency: str = "monthly",
    lookback_years: int = 3,
    max_weight: float = 0.30,
    alpha: float = M7_ALPHA,
    cost_bps: float | Iterable[float] = 0.0,
    previous_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Execute the certified Minimum-CVaR walk-forward mechanics for all eligible monthly rebalances."""
    if isinstance(cost_bps, (int, float)):
        cost_grid = [float(cost_bps)]
    else:
        cost_grid = [float(rate) for rate in list(cost_bps)]

    candidates = list_m7_rebalance_candidates(
        returns_df,
        tickers,
        holdout_start_date=holdout_start_date,
        rebalance_frequency=rebalance_frequency,
        lookback_years=lookback_years,
    )
    eligible = [candidate for candidate in candidates if candidate.eligible]
    rebalance_results = []
    last_weights = previous_weights
    for candidate in eligible:
        rebalance_info = RebalanceInfo(
            rebalance_date=candidate.rebalance_date.to_pydatetime(),
            training_start=candidate.training_start.to_pydatetime(),
            training_end=candidate.training_end.to_pydatetime(),
            holding_start=candidate.holding_start.to_pydatetime(),
            holding_end=candidate.holding_end.to_pydatetime() if candidate.holding_end is not None else None,
        )
        rebalance = run_milestone7_rebalance(
            rebalance_info,
            returns_df,
            tickers=tickers,
            max_weight=max_weight,
            alpha=alpha,
            previous_weights=last_weights,
            cost_bps=float(cost_grid[0]),
        )
        rebalance_results.append(rebalance)
        last_weights = rebalance["weights"]

    summary = {
        "strategy_label": M7_STRATEGY_LABEL,
        "alpha": float(alpha),
        "candidate_rebalance_count": int(len(candidates)),
        "eligible_rebalance_count": int(len(eligible)),
        "ineligible_rebalance_count": int(len(candidates) - len(eligible)),
        "first_candidate_rebalance": candidates[0].rebalance_date if candidates else None,
        "first_eligible_rebalance": eligible[0].rebalance_date if eligible else None,
        "last_eligible_rebalance": eligible[-1].rebalance_date if eligible else None,
        "training_observations": M7_TRAINING_OBSERVATIONS,
        "scenario_count": M7_TRAINING_OBSERVATIONS,
        "rebalance_results": rebalance_results,
    }
    return summary


def _structural_schedule_summary(returns_df: pd.DataFrame, tickers: list[str]) -> dict[str, Any]:
    prices_df = pd.DataFrame(index=returns_df.index, columns=tickers, dtype=float)
    prices_df.iloc[:, :] = 1.0
    candidates = list_m7_rebalance_candidates(
        returns_df,
        tickers,
        holdout_start_date="2018-01-01",
        rebalance_frequency="monthly",
        lookback_years=3,
    )
    eligible = [candidate for candidate in candidates if candidate.eligible]
    return {
        "first_candidate_rebalance": candidates[0].rebalance_date if candidates else None,
        "first_eligible_rebalance": eligible[0].rebalance_date if eligible else None,
        "last_eligible_rebalance": eligible[-1].rebalance_date if eligible else None,
        "candidate_rebalance_count": int(len(candidates)),
        "eligible_rebalance_count": int(len(eligible)),
        "ineligible_rebalance_count": int(len(candidates) - len(eligible)),
        "comparison_start": None,
        "comparison_end": None,
        "comparison_observation_count": 0,
    }


def write_milestone7_validation_artifact(
    artifact_path: str | Path,
    *,
    strategy_label: str = M7_STRATEGY_LABEL,
    alpha: float = M7_ALPHA,
    training_observations: int = M7_TRAINING_OBSERVATIONS,
    scenario_count: int = M7_TRAINING_OBSERVATIONS,
    first_candidate_rebalance: str | None = None,
    first_eligible_rebalance: str | None = None,
    last_eligible_rebalance: str | None = None,
    candidate_rebalance_count: int = 0,
    eligible_rebalance_count: int = 0,
    ineligible_rebalance_count: int = 0,
    anti_lookahead_checks: int = 0,
    anti_lookahead_violations: int = 0,
    future_mutation_weight_diff: float | None = None,
    future_mutation_objective_diff: float | None = None,
    holding_return_identity_max_error: float | None = None,
    permutation_weight_max_diff: float | None = None,
    permutation_return_max_diff: float | None = None,
    permutation_objective_diff: float | None = None,
    determinism_weight_max_diff: float | None = None,
    determinism_return_max_diff: float | None = None,
    determinism_objective_diff: float | None = None,
    zero_cost_max_diff: float | None = None,
    cost_identity_max_error: float | None = None,
    canonical_live_data_calls: int = 0,
    optimizer_failures_in_mechanics_tests: int = 0,
    test_counts: dict[str, int] | None = None,
) -> Path:
    """Write the M7 validation artifact without reporting performance."""
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_label": strategy_label,
        "alpha": float(alpha),
        "training_observations": int(training_observations),
        "scenario_count": int(scenario_count),
        "first_candidate_rebalance": first_candidate_rebalance,
        "first_eligible_rebalance": first_eligible_rebalance,
        "last_eligible_rebalance": last_eligible_rebalance,
        "candidate_rebalance_count": int(candidate_rebalance_count),
        "eligible_rebalance_count": int(eligible_rebalance_count),
        "ineligible_rebalance_count": int(ineligible_rebalance_count),
        "anti_lookahead_checks": int(anti_lookahead_checks),
        "anti_lookahead_violations": int(anti_lookahead_violations),
        "future_mutation_weight_diff": None if future_mutation_weight_diff is None else float(future_mutation_weight_diff),
        "future_mutation_objective_diff": None if future_mutation_objective_diff is None else float(future_mutation_objective_diff),
        "holding_return_identity_max_error": None if holding_return_identity_max_error is None else float(holding_return_identity_max_error),
        "permutation_weight_max_diff": None if permutation_weight_max_diff is None else float(permutation_weight_max_diff),
        "permutation_return_max_diff": None if permutation_return_max_diff is None else float(permutation_return_max_diff),
        "permutation_objective_diff": None if permutation_objective_diff is None else float(permutation_objective_diff),
        "determinism_weight_max_diff": None if determinism_weight_max_diff is None else float(determinism_weight_max_diff),
        "determinism_return_max_diff": None if determinism_return_max_diff is None else float(determinism_return_max_diff),
        "determinism_objective_diff": None if determinism_objective_diff is None else float(determinism_objective_diff),
        "zero_cost_max_diff": None if zero_cost_max_diff is None else float(zero_cost_max_diff),
        "cost_identity_max_error": None if cost_identity_max_error is None else float(cost_identity_max_error),
        "canonical_live_data_calls": int(canonical_live_data_calls),
        "optimizer_failures_in_mechanics_tests": int(optimizer_failures_in_mechanics_tests),
        "test_counts": test_counts or {},
    }
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact


__all__ = [
    "M7_STRATEGY_LABEL",
    "M7_ALPHA",
    "M7_TRAINING_OBSERVATIONS",
    "M7RebalanceCandidate",
    "list_m7_rebalance_candidates",
    "run_milestone7_rebalance",
    "run_milestone7_walk_forward",
    "write_milestone7_validation_artifact",
    "_structural_schedule_summary",
]
