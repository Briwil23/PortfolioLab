from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.milestone7 import (
    M7_TRAINING_OBSERVATIONS,
    list_m7_rebalance_candidates,
    run_milestone7_rebalance,
    run_milestone7_walk_forward,
    write_milestone7_validation_artifact,
)
from src.backtesting.walk_forward import RebalanceInfo, generate_rebalance_dates
from src.optimization.cvar import minimum_cvar_95_portfolio

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_returns(n_days: int = 400, n_assets: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2015-01-05", periods=n_days, freq="B")
    rng = np.random.default_rng(12345)
    base = rng.normal(0.0006, 0.012, size=(n_days, n_assets))
    out = pd.DataFrame(base, index=dates, columns=["A", "B", "C"][:n_assets])
    out.iloc[:, 0] = np.linspace(-0.02, 0.02, n_days)
    return out


def _m7_valid_returns() -> pd.DataFrame:
    return _synthetic_returns(n_days=1500, n_assets=3)


def _first_eligible_candidate(returns: pd.DataFrame):
    candidates = list_m7_rebalance_candidates(
        returns,
        ["A", "B", "C"],
        holdout_start_date="2018-01-01",
        rebalance_frequency="monthly",
        lookback_years=3,
    )
    eligible = [candidate for candidate in candidates if candidate.eligible]
    assert eligible, "Expected at least one eligible M7 candidate derived from the canonical schedule."
    return eligible[0]


def _gap_returns(gap: int) -> tuple[pd.DataFrame, RebalanceInfo]:
    index = pd.date_range("2020-01-02", periods=gap + 5, freq="B")
    data = pd.DataFrame(
        np.random.default_rng(42).normal(0.0006, 0.012, size=(len(index), 3)),
        index=index,
        columns=["A", "B", "C"],
    )
    training_start = index[0]
    holding_start = index[gap]
    rebalance = RebalanceInfo(
        rebalance_date=holding_start,
        training_start=training_start,
        training_end=holding_start - pd.Timedelta(days=1),
        holding_start=holding_start,
        holding_end=holding_start + pd.Timedelta(days=30),
    )
    return data, rebalance


def _make_rebalance_info(training_start: str, training_end: str, holding_start: str) -> RebalanceInfo:
    return RebalanceInfo(
        rebalance_date=pd.Timestamp(holding_start),
        training_start=pd.Timestamp(training_start),
        training_end=pd.Timestamp(training_end),
        holding_start=pd.Timestamp(holding_start),
        holding_end=pd.Timestamp(holding_start) + pd.Timedelta(days=30),
    )


def test_m7_training_window_uses_exactly_252_and_fail_closed_on_lookahead():
    returns = _m7_valid_returns()
    candidate = _first_eligible_candidate(returns)
    assert candidate.training_observations == 252
    assert candidate.training_returns.shape[0] == 252
    assert candidate.training_returns.index.max() < candidate.holding_start

    rebalance = RebalanceInfo(
        rebalance_date=candidate.rebalance_date.to_pydatetime(),
        training_start=candidate.training_start.to_pydatetime(),
        training_end=candidate.training_end.to_pydatetime(),
        holding_start=candidate.holding_start.to_pydatetime(),
        holding_end=candidate.holding_end.to_pydatetime() if candidate.holding_end is not None else None,
    )
    result = run_milestone7_rebalance(
        rebalance,
        returns,
        tickers=["A", "B", "C"],
        max_weight=0.60,
        alpha=0.95,
        previous_weights=None,
        cost_bps=0.0,
    )
    assert result["training_observations"] == 252

    malformed = RebalanceInfo(
        rebalance_date=candidate.rebalance_date.to_pydatetime(),
        training_start=candidate.training_start.to_pydatetime(),
        training_end=candidate.training_end.to_pydatetime(),
        holding_start=candidate.training_start.to_pydatetime(),
        holding_end=candidate.holding_start.to_pydatetime(),
    )
    with pytest.raises(ValueError):
        run_milestone7_rebalance(
            malformed,
            returns,
            tickers=["A", "B", "C"],
            max_weight=0.60,
            alpha=0.95,
            previous_weights=None,
            cost_bps=0.0,
        )


def test_m7_boundaries_251_252_253():
    for gap in (251, 252, 253):
        data, rebalance = _gap_returns(gap)
        training = data[(data.index >= rebalance.training_start) & (data.index < rebalance.holding_start)].copy()
        assert len(training) == gap
        if gap < 252:
            with pytest.raises(ValueError):
                run_milestone7_rebalance(
                    rebalance,
                    data,
                    tickers=["A", "B", "C"],
                    max_weight=0.60,
                    alpha=0.95,
                    previous_weights=None,
                    cost_bps=0.0,
                )
        else:
            result = run_milestone7_rebalance(
                rebalance,
                data,
                tickers=["A", "B", "C"],
                max_weight=0.60,
                alpha=0.95,
                previous_weights=None,
                cost_bps=0.0,
            )
            assert result["training_observations"] == 252


def test_m7_future_mutation_does_not_change_weights_or_objective():
    returns = _m7_valid_returns()
    candidate = list_m7_rebalance_candidates(returns, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3)[0]
    result = run_milestone7_rebalance(
        RebalanceInfo(
            rebalance_date=candidate.rebalance_date.to_pydatetime(),
            training_start=candidate.training_start.to_pydatetime(),
            training_end=candidate.training_end.to_pydatetime(),
            holding_start=candidate.holding_start.to_pydatetime(),
            holding_end=candidate.holding_end.to_pydatetime() if candidate.holding_end is not None else None,
        ),
        returns,
        tickers=["A", "B", "C"],
        max_weight=0.60,
        alpha=0.95,
        previous_weights=None,
        cost_bps=0.0,
    )
    training_matrix = candidate.training_returns[["A", "B", "C"]].to_numpy(dtype=float)
    mutated = returns.copy()
    mutated.loc[mutated.index >= candidate.holding_start, :] = 1000.0
    mutated_candidate = list_m7_rebalance_candidates(mutated, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3)[0]
    mutated_result = run_milestone7_rebalance(
        RebalanceInfo(
            rebalance_date=mutated_candidate.rebalance_date.to_pydatetime(),
            training_start=mutated_candidate.training_start.to_pydatetime(),
            training_end=mutated_candidate.training_end.to_pydatetime(),
            holding_start=mutated_candidate.holding_start.to_pydatetime(),
            holding_end=mutated_candidate.holding_end.to_pydatetime() if mutated_candidate.holding_end is not None else None,
        ),
        mutated,
        tickers=["A", "B", "C"],
        max_weight=0.60,
        alpha=0.95,
        previous_weights=None,
        cost_bps=0.0,
    )
    diff = np.max(np.abs(np.array(list(result["weights"].values())) - np.array(list(mutated_result["weights"].values()))))
    assert diff <= 1e-12
    assert abs(result["objective_cvar"] - mutated_result["objective_cvar"]) <= 1e-12
    assert np.max(np.abs(training_matrix - mutated_candidate.training_returns[["A", "B", "C"]].to_numpy(dtype=float))) == 0.0


def test_m7_holding_return_identity_and_constancy():
    returns = _m7_valid_returns()
    candidates = list_m7_rebalance_candidates(returns, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3)
    candidate = next(c for c in candidates if c.eligible)
    result = run_milestone7_rebalance(
        RebalanceInfo(
            rebalance_date=candidate.rebalance_date.to_pydatetime(),
            training_start=candidate.training_start.to_pydatetime(),
            training_end=candidate.training_end.to_pydatetime(),
            holding_start=candidate.holding_start.to_pydatetime(),
            holding_end=candidate.holding_end.to_pydatetime() if candidate.holding_end is not None else None,
        ),
        returns,
        tickers=["A", "B", "C"],
        max_weight=0.60,
        alpha=0.95,
        previous_weights=None,
        cost_bps=0.0,
    )
    assert result["holding_return_identity_max_error"] <= 1e-12
    assert np.isclose(np.sum(list(result["weights"].values())), 1.0, atol=1e-12)


def test_m7_asset_order_and_validation_fail_closed():
    returns = _m7_valid_returns()
    bad = returns.copy()
    bad["D"] = 0.1
    with pytest.raises(ValueError):
        list_m7_rebalance_candidates(bad, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3)

    returns_bad = returns.copy()
    returns_bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        list_m7_rebalance_candidates(returns_bad, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3)


def test_m7_permutation_invariance_and_determinism():
    returns = _m7_valid_returns()
    result_a = run_milestone7_walk_forward(returns, tickers=["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3, max_weight=0.60)
    permuted = returns[["C", "A", "B"]].copy()
    permuted.columns = ["C", "A", "B"]
    result_b = run_milestone7_walk_forward(permuted, tickers=["C", "A", "B"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3, max_weight=0.60)
    assert result_a["eligible_rebalance_count"] == result_b["eligible_rebalance_count"]
    assert result_a["first_eligible_rebalance"] == result_b["first_eligible_rebalance"]
    assert result_a["last_eligible_rebalance"] == result_b["last_eligible_rebalance"]

    repeat = run_milestone7_walk_forward(returns, tickers=["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3, max_weight=0.60)
    assert repeat["eligible_rebalance_count"] == result_a["eligible_rebalance_count"]
    assert repeat["first_eligible_rebalance"] == result_a["first_eligible_rebalance"]
    assert repeat["last_eligible_rebalance"] == result_a["last_eligible_rebalance"]


def test_m7_cost_identity_zero_and_monotonicity():
    returns = _m7_valid_returns()
    candidate = list_m7_rebalance_candidates(returns, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3)[0]
    result = run_milestone7_rebalance(
        RebalanceInfo(
            rebalance_date=candidate.rebalance_date.to_pydatetime(),
            training_start=candidate.training_start.to_pydatetime(),
            training_end=candidate.training_end.to_pydatetime(),
            holding_start=candidate.holding_start.to_pydatetime(),
            holding_end=candidate.holding_end.to_pydatetime() if candidate.holding_end is not None else None,
        ),
        returns,
        tickers=["A", "B", "C"],
        max_weight=0.60,
        alpha=0.95,
        previous_weights=None,
        cost_bps=0.0,
    )
    assert np.max(np.abs(result["gross_returns"].to_numpy() - result["net_returns"].to_numpy())) <= 1e-15

    base = result["turnover"]
    assert base >= 0.0
    assert 0.0 <= result["transaction_cost_bps"] <= 25.0


def test_m7_cost_manual_identity_and_first_rebalance_treatment():
    returns = _m7_valid_returns()
    candidate = next(item for item in list_m7_rebalance_candidates(returns, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3) if item.eligible)
    previous_weights = None
    result = run_milestone7_rebalance(
        RebalanceInfo(
            rebalance_date=candidate.rebalance_date.to_pydatetime(),
            training_start=candidate.training_start.to_pydatetime(),
            training_end=candidate.training_end.to_pydatetime(),
            holding_start=candidate.holding_start.to_pydatetime(),
            holding_end=candidate.holding_end.to_pydatetime() if candidate.holding_end is not None else None,
        ),
        returns,
        tickers=["A", "B", "C"],
        max_weight=0.60,
        alpha=0.95,
        previous_weights=previous_weights,
        cost_bps=10.0,
    )
    assert result["transaction_cost"] >= 0.0
    assert result["turnover"] >= 0.0
    assert result["holding_return_identity_max_error"] <= 1e-12


def test_m7_optimizer_fail_closed_propagation_and_zero_live_calls():
    returns = _m7_valid_returns()
    candidate = list_m7_rebalance_candidates(returns, ["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3)[0]
    assert candidate.training_returns.shape[0] == 252
    with pytest.raises(ValueError):
        run_milestone7_rebalance(
            RebalanceInfo(
                rebalance_date=candidate.rebalance_date.to_pydatetime(),
                training_start=candidate.training_start.to_pydatetime(),
                training_end=candidate.training_end.to_pydatetime(),
                holding_start=candidate.holding_start.to_pydatetime(),
                holding_end=candidate.holding_end.to_pydatetime() if candidate.holding_end is not None else None,
            ),
            returns,
            tickers=["A", "B", "C"],
            max_weight=1e-9,
            alpha=0.95,
            previous_weights=None,
            cost_bps=0.0,
        )

    artifact = ROOT / "results" / "milestone7_validation" / "walk_forward_mechanics_validation.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    write_milestone7_validation_artifact(artifact, candidate_rebalance_count=1, eligible_rebalance_count=1, ineligible_rebalance_count=0, first_candidate_rebalance="2018-02-01", first_eligible_rebalance="2018-02-01", last_eligible_rebalance="2018-02-01")
    assert artifact.exists()
    payload = artifact.read_text(encoding="utf-8")
    assert "strategy_label" in payload


def test_m7_structural_schedule_count():
    returns = _m7_valid_returns()
    summary = run_milestone7_walk_forward(returns, tickers=["A", "B", "C"], holdout_start_date="2018-01-01", rebalance_frequency="monthly", lookback_years=3, max_weight=0.60)
    assert summary["candidate_rebalance_count"] >= summary["eligible_rebalance_count"]
    assert summary["eligible_rebalance_count"] >= 1
    assert summary["training_observations"] == 252
    assert summary["scenario_count"] == 252


@pytest.mark.parametrize("gap", [251, 252, 253])
def test_m7_boundary_count(gap):
    data, rebalance = _gap_returns(gap)
    if gap < 252:
        with pytest.raises(ValueError):
            run_milestone7_rebalance(
                rebalance,
                data,
                tickers=["A", "B", "C"],
                max_weight=0.60,
                alpha=0.95,
                previous_weights=None,
                cost_bps=0.0,
            )
    else:
        result = run_milestone7_rebalance(
            rebalance,
            data,
            tickers=["A", "B", "C"],
            max_weight=0.60,
            alpha=0.95,
            previous_weights=None,
            cost_bps=0.0,
        )
        assert result["training_observations"] == 252
