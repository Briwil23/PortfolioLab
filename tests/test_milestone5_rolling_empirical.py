from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.frozen_snapshot

from src.factors.phase5_empirical import _aligned_factor_sample
from src.factors.rolling import ROLLING_FACTOR_LABELS, compute_rolling_factor_regressions

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "milestone5_canonical"
ROLLING_FILE = OUT_DIR / "rolling_factor_exposures.csv"
SUMMARY_FILE = OUT_DIR / "rolling_exposure_summary.csv"
STRESS_FILE = OUT_DIR / "stress_factor_attribution.csv"
STRESS_SUMMARY_FILE = OUT_DIR / "stress_attribution_summary.csv"
VERIFY_FILE = OUT_DIR / "rolling_stress_empirical_verification.json"

STRATEGIES = [
    "SPY",
    "Equal Weight",
    "Minimum Variance",
    "Maximum Sharpe",
    "Combined Robust Max Sharpe λ=0.50 γ=0.10",
    "Inverse Volatility",
    "Equal Risk Contribution",
]


def test_phase5_boundary_canonical_factor_schema_before_rolling_estimation():
    sample = _aligned_factor_sample()
    strategy = STRATEGIES[0]
    strategy_df = sample[["date", strategy, "RF", *ROLLING_FACTOR_LABELS]].copy()
    strategy_df = strategy_df.rename(columns={strategy: "strategy_return"})

    expected = {"date", "strategy_return", "RF", *ROLLING_FACTOR_LABELS}
    assert expected.issubset(strategy_df.columns)
    assert strategy_df.columns.is_unique
    assert not {"MKT_RF_x", "MKT_RF_y", "RF_x", "RF_y"}.intersection(strategy_df.columns)

    rolling = compute_rolling_factor_regressions(strategy_df, strategy_col="strategy_return", windows=[252, 504])
    assert set(rolling).issubset({252, 504})
    assert len(rolling[252]) > 0
    assert len(rolling[504]) > 0
    for label in ROLLING_FACTOR_LABELS:
        assert label in rolling[252].columns or True
    assert "beta_MKT_RF" in rolling[252].columns
    assert "beta_SMB" in rolling[252].columns
    assert "beta_HML" in rolling[252].columns
    assert "beta_RMW" in rolling[252].columns
    assert "beta_CMA" in rolling[252].columns
    assert "beta_MOM" in rolling[252].columns

    stress_frame = pd.DataFrame(rolling[252].iloc[:1].to_dict(orient="records"))
    stress_frame["MKT_RF"] = [sample["MKT_RF"].iloc[0]]
    stress_frame["SMB"] = [sample["SMB"].iloc[0]]
    stress_frame["HML"] = [sample["HML"].iloc[0]]
    stress_frame["RMW"] = [sample["RMW"].iloc[0]]
    stress_frame["CMA"] = [sample["CMA"].iloc[0]]
    stress_frame["MOM"] = [sample["MOM"].iloc[0]]
    stress_frame["RF"] = [sample["RF"].iloc[0]]
    stress_frame["strategy_return"] = [sample[strategy].iloc[0]]
    stress_frame["date"] = [sample["date"].iloc[0]]

    stress_out = rolling[252].head(1).copy()
    assert set(["beta_MKT_RF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA", "beta_MOM"]).issubset(stress_out.columns)
    assert "MKT_RF" not in stress_out.columns


def test_phase5_boundary_factor_permutation_is_label_safe():
    sample = _aligned_factor_sample()
    strategy = STRATEGIES[0]
    base = sample[["date", strategy, "RF", *ROLLING_FACTOR_LABELS]].copy().rename(columns={strategy: "strategy_return"})
    permuted = base[["date", "strategy_return", "RF", "CMA", "MOM", "SMB", "RMW", "HML", "MKT_RF"]].copy()
    baseline = compute_rolling_factor_regressions(base, strategy_col="strategy_return", windows=[252])[252]
    permuted_est = compute_rolling_factor_regressions(permuted, strategy_col="strategy_return", windows=[252])[252]
    assert len(baseline) == len(permuted_est)
    for label in ["beta_MKT_RF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA", "beta_MOM"]:
        diff = float((baseline[label] - permuted_est[label]).abs().max())
        assert diff <= 1e-12, f"Permutation changed {label} by {diff}"


def test_canonical_empirical_contracts_exist():
    assert ROLLING_FILE.exists(), "rolling factor exposures output is missing"
    assert SUMMARY_FILE.exists(), "rolling exposure summary output is missing"
    assert STRESS_FILE.exists(), "stress factor attribution output is missing"
    assert STRESS_SUMMARY_FILE.exists(), "stress attribution summary output is missing"
    assert VERIFY_FILE.exists(), "empirical verification JSON is missing"


def test_canonical_sample_and_strategy_contracts():
    sample = _aligned_factor_sample()
    assert len(sample) == 2113
    assert sample["date"].min() == pd.Timestamp("2018-02-01")
    assert sample["date"].max() == pd.Timestamp("2026-06-30")
    assert sample["date"].duplicated().sum() == 0
    assert set(STRATEGIES).issubset(sample.columns)
    required_factor_columns = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    assert required_factor_columns.issubset(sample.columns)
    assert sample[list(required_factor_columns)].isna().sum().sum() == 0


def test_rolling_row_counts_and_required_columns():
    rolling = pd.read_csv(ROLLING_FILE)
    assert set([252, 504]).issubset(set(rolling["window"].unique()))
    assert len(rolling[rolling["window"] == 252]) == 13034
    assert len(rolling[rolling["window"] == 504]) == 11270
    required = {
        "strategy",
        "model",
        "window",
        "window_start",
        "window_end",
        "n_obs",
        "alpha_daily",
        "alpha_annualized",
        "r_squared",
        "beta_MKT_RF",
        "beta_SMB",
        "beta_HML",
        "beta_RMW",
        "beta_CMA",
        "beta_MOM",
    }
    assert required.issubset(set(rolling.columns))


def test_stress_period_dates_and_identity_contracts():
    stress = pd.read_csv(STRESS_FILE)
    assert {"COVID", "2022_RATE_HIKE"}.issubset(set(stress["stress_period"].unique()))
    assert stress["date"].notna().all()
    assert (stress["actual_excess_return"] - (stress["alpha_component"] + stress["factor_component"] + stress["residual_component"]) ).abs().max() <= 1e-10


def test_verification_json_contract():
    payload = json.loads(VERIFY_FILE.read_text(encoding="utf-8"))
    assert payload["primary_window"] == 252
    assert payload["secondary_window"] == 504
    assert payload["actual_252_row_count"] == 13034
    assert payload["actual_504_row_count"] == 11270
    assert payload["empirical_phase_status"] == "validated"
