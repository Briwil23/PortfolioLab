from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "milestone3_canonical"
FIGDIR = OUTDIR / "figures"


def test_canonical_output_files_exist():
    required = {
        "walk_forward_returns_gross.csv",
        "walk_forward_returns_net.csv",
        "walk_forward_weights.csv",
        "walk_forward_metrics_gross.csv",
        "walk_forward_metrics_net.csv",
        "rebalance_history.csv",
        "turnover_analysis.csv",
        "concentration_analysis.csv",
        "cap_binding_analysis.csv",
        "expected_return_sensitivity.csv",
        "risk_contribution_analysis.csv",
        "stress_period_analysis.csv",
        "drawdown_analysis.csv",
        "rolling_performance.csv",
        "transaction_cost_analysis.csv",
        "evaluation_verification.json",
    }
    missing = [name for name in required if not (OUTDIR / name).exists()]
    assert not missing, f"Missing canonical output files: {missing}"


def test_canonical_figures_exist():
    expected = [
        "01_cumulative_wealth_primary.png",
        "02_drawdown_comparison.png",
        "03_rolling_12m_sharpe.png",
        "04_turnover_comparison.png",
        "05_hhi_effective_holdings.png",
        "06_transaction_cost_impact.png",
        "07_weight_evolution_max_sharpe.png",
        "08_weight_evolution_combined_robust.png",
        "09_expected_return_sensitivity.png",
        "10_risk_contribution_comparison.png",
    ]
    missing = [name for name in expected if not (FIGDIR / name).exists()]
    assert not missing, f"Missing canonical figure files: {missing}"


def test_verification_contract():
    with (OUTDIR / "evaluation_verification.json").open("r", encoding="utf-8") as f:
        verification = json.load(f)

    assert verification["canonical_hashes_validated"] is True
    assert verification["live_data_calls"] == 0
    assert verification["optimizer_failures"] == 0
    assert verification["oos_observations"] > 0
    assert verification["rebalance_count"] > 0


def test_core_schema_columns():
    gross = pd.read_csv(OUTDIR / "walk_forward_returns_gross.csv")
    net = pd.read_csv(OUTDIR / "walk_forward_returns_net.csv")
    weights = pd.read_csv(OUTDIR / "walk_forward_weights.csv")

    assert "Date" in gross.columns
    assert {"date", "strategy", "cost_bps", "net_return"}.issubset(net.columns)
    assert {"date", "strategy", "asset", "weight"}.issubset(weights.columns)


def test_primary_strategies_present():
    gross = pd.read_csv(OUTDIR / "walk_forward_returns_gross.csv")
    required_strategies = {
        "SPY",
        "Maximum Sharpe",
        "Shrunk Max Sharpe λ=0.50",
        "Turnover-Aware Max Sharpe γ=0.10",
        "Combined Robust Max Sharpe λ=0.50 γ=0.10",
    }
    assert required_strategies.issubset(set(gross.columns))
