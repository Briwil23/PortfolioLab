from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.milestone7_empirical import run_milestone7_canonical_empirical_replicates
from src.risk.tail_metrics import empirical_cvar, empirical_var

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "milestone7_canonical"
RUN2_DIR = ROOT / "results" / "milestone7_validation" / "canonical_empirical_run2"

REQUIRED_OUTPUTS = [
    "walk_forward_returns.csv",
    "walk_forward_weights.csv",
    "rebalance_history.csv",
    "optimizer_diagnostics.csv",
    "portfolio_metrics_gross.csv",
    "portfolio_metrics_net_10bps.csv",
    "tail_risk_metrics.csv",
    "turnover_analysis.csv",
    "concentration_analysis.csv",
    "stress_analysis.csv",
    "transaction_cost_analysis.csv",
    "weight_stability.csv",
    "turnover_top_changes.csv",
    "milestone7_empirical_verification.json",
    "figures/01_cumulative_growth_gross.png",
    "figures/02_cumulative_growth_net_10bps.png",
    "figures/03_drawdown_comparison.png",
    "figures/04_tail_loss_distribution.png",
    "figures/05_realized_var_cvar_comparison.png",
    "figures/06_minimum_cvar_weights.png",
    "figures/07_turnover_comparison.png",
    "figures/08_concentration_comparison.png",
    "figures/09_covid_stress.png",
    "figures/010_2022_stress.png",
    "figures/11_risk_return_tail_map.png",
]


@lru_cache(maxsize=1)
def _run_replicates() -> dict:
    return run_milestone7_canonical_empirical_replicates(OUT_DIR, RUN2_DIR)


def test_m7_empirical_artifacts_and_reproducibility():
    bundle = _run_replicates()
    verification = bundle["verification"]

    assert verification["empirical_certification_status"] == "validated"
    assert verification["canonical_data_hash"]
    assert verification["comparison_start"] == "2018-02-01"
    assert verification["comparison_end"] == "2026-09-02"
    assert verification["comparison_observation_count"] == 2158
    assert verification["candidate_rebalances"] == 67
    assert verification["eligible_rebalances"] == 67
    assert verification["optimizer_calls"] == 67
    assert verification["optimizer_failures"] == 0
    assert verification["primary_net_cost_bps"] == 10.0
    assert verification["run1_vs_run2_hashes_equal"] is True
    assert all(diff <= 1e-12 for diff in verification["run1_vs_run2_max_diffs"].values())
    assert verification["artifact_rows"]
    assert all(flag is True for flag in verification["figure_validation"].values())

    for name in REQUIRED_OUTPUTS:
        assert (OUT_DIR / name).exists(), name
        if name.endswith(".png"):
            assert (OUT_DIR / name).stat().st_size > 0, name

    payload = json.loads((OUT_DIR / "milestone7_empirical_verification.json").read_text(encoding="utf-8"))
    required_keys = {
        "canonical_data_hash",
        "comparison_start",
        "comparison_end",
        "comparison_observation_count",
        "candidate_rebalances",
        "eligible_rebalances",
        "optimizer_calls",
        "optimizer_failures",
        "alpha",
        "training_observations",
        "max_weight",
        "strategy_label",
        "cost_grid_bps",
        "primary_net_cost_bps",
        "stress_periods",
        "optimizer_constraint_errors",
        "metric_objective_max_difference",
        "live_data_calls",
        "artifact_rows",
        "artifact_hashes",
        "figure_validation",
        "closed_artifact_preservation",
        "full_test_count",
        "empirical_certification_status",
    }
    assert required_keys.issubset(payload)
    assert payload["live_data_calls"] == 0
    assert payload["full_test_count"] == 218
    assert payload["stress_periods"]["COVID"] == {"start": "2020-02-20", "end": "2020-04-30"}
    assert payload["stress_periods"]["2022"] == {"start": "2022-01-03", "end": "2022-10-31"}


def test_m7_return_and_cost_identity_contracts():
    _run_replicates()

    returns = pd.read_csv(OUT_DIR / "walk_forward_returns.csv")
    weights = pd.read_csv(OUT_DIR / "walk_forward_weights.csv")
    rebalance = pd.read_csv(OUT_DIR / "rebalance_history.csv")
    optimizer = pd.read_csv(OUT_DIR / "optimizer_diagnostics.csv")
    transaction = pd.read_csv(OUT_DIR / "transaction_cost_analysis.csv")
    tail = pd.read_csv(OUT_DIR / "tail_risk_metrics.csv")
    stress = pd.read_csv(OUT_DIR / "stress_analysis.csv")

    assert returns["date"].duplicated().sum() > 0
    assert returns[["gross_return", "net_return"]].replace([np.inf, -np.inf], np.nan).notna().all().all()
    assert weights["weight"].between(0.0, 0.30 + 1e-12).all()
    assert np.isclose(weights.groupby("date")["weight"].sum().to_numpy(), 1.0, atol=1e-12).all()
    assert len(rebalance) == 67
    assert len(optimizer) == 67
    assert optimizer["optimizer_success"].all()
    assert optimizer["weight_sum_error"].max() <= 1e-12
    assert optimizer["lower_bound_violation"].max() <= 1e-12
    assert optimizer["upper_bound_violation"].max() <= 1e-12
    assert optimizer["ru_scenario_constraint_violation"].max() <= 1e-12
    assert optimizer["metric_objective_max_difference"].max() <= 1e-12

    zero_cost = returns[returns["cost_bps"] == 0.0].copy()
    assert np.max(np.abs(zero_cost["gross_return"] - zero_cost["net_return"])) <= 1e-15

    ten_bps = returns[returns["cost_bps"] == 10.0].copy()
    gross = ten_bps["gross_return"].to_numpy(dtype=float)
    net = ten_bps["net_return"].to_numpy(dtype=float)
    first_diff = float((gross - net).max())
    assert first_diff >= 0.0
    assert transaction["cost_bps"].isin([0.0, 5.0, 10.0, 25.0]).all()
    assert transaction["terminal_wealth"].notna().all()
    assert transaction["mean_turnover"].notna().all()
    assert tail["confidence"].isin([0.95, 0.99]).all()
    assert stress["stress_period"].isin(["COVID", "2022"]).all()

    losses = -ten_bps["net_return"].to_numpy(dtype=float)
    assert np.isfinite(empirical_var(losses, 0.95))
    assert np.isfinite(empirical_cvar(losses, 0.95))


def test_m7_empirical_date_contract_and_no_leaks():
    _run_replicates()

    gross = pd.read_csv(OUT_DIR / "portfolio_metrics_gross.csv")
    net = pd.read_csv(OUT_DIR / "portfolio_metrics_net_10bps.csv")
    assert set(gross["strategy"]) == set(net["strategy"])
    assert gross["sample_start"].nunique() == 1
    assert gross["sample_end"].nunique() == 1
    assert gross["sample_start"].iloc[0] == "2018-02-01"
    assert gross["sample_end"].iloc[0] == "2026-09-02"
    assert gross["n_obs"].nunique() == 1
    assert gross["n_obs"].iloc[0] == 2158

    stress = pd.read_csv(OUT_DIR / "stress_analysis.csv")
    assert set(stress["stress_period"]) == {"COVID", "2022"}
    assert stress.groupby("stress_period")["start"].nunique().to_dict() == {"COVID": 1, "2022": 1}
    assert stress.groupby("stress_period")["end"].nunique().to_dict() == {"COVID": 1, "2022": 1}
