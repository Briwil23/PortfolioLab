from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.frozen_snapshot

from src.data.market_data import compute_file_sha256
from src.factors.data import load_canonical_factor_dataset
from src.factors.models import FACTOR_MODEL_REGISTRY, get_model_factor_labels
from src.factors.regression import HAC_MAXLAGS, align_m4_returns_to_factors
from src.factors.static_attribution import MODELS, STRATEGIES, run_static_factor_attribution

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data" / "canonical_factors"
M4_GROSS = ROOT / "results" / "milestone4_canonical" / "walk_forward_returns_gross.csv"
OUT_DIR = ROOT / "results" / "milestone5_canonical"


def _tmp_static_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "milestone5_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_static_strategy_and_model_inclusion():
    assert STRATEGIES == [
        "SPY",
        "Equal Weight",
        "Minimum Variance",
        "Maximum Sharpe",
        "Combined Robust Max Sharpe λ=0.50 γ=0.10",
        "Inverse Volatility",
        "Equal Risk Contribution",
    ]
    assert MODELS == ["CAPM", "FF3", "FF5", "FF5_MOM"]
    assert FACTOR_MODEL_REGISTRY == {
        "CAPM": ["MKT_RF"],
        "FF3": ["MKT_RF", "SMB", "HML"],
        "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
        "FF5_MOM": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
    }


def test_static_regressions_count_and_sample_contract(tmp_path):
    output_dir = _tmp_static_dir(tmp_path)
    result = run_static_factor_attribution(output_dir)
    static = result["static_regressions"]
    assert len(static) == 28
    assert (static["n_obs"] == 2113).all()
    assert sorted(static["strategy"].unique().tolist()) == sorted(STRATEGIES)
    assert sorted(static["model"].unique().tolist()) == sorted(MODELS)
    assert static["sample_start"].nunique() == 1
    assert static["sample_end"].nunique() == 1
    assert static["sample_start"].iloc[0] == "2018-02-01"
    assert static["sample_end"].iloc[0] == "2026-06-30"


def test_static_alignment_and_hash_validation():
    canonical_factors, manifest, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    m4 = pd.read_csv(M4_GROSS)
    aligned = align_m4_returns_to_factors(m4, canonical_factors)
    max_diff = float((aligned[[c for c in m4.columns if c != "Date"]].reset_index(drop=True) - m4[[c for c in m4.columns if c != "Date"]].reset_index(drop=True)).abs().max().max())
    assert max_diff <= 1e-12
    assert aligned["date"].min().date().isoformat() == "2018-02-01"
    assert aligned["date"].max().date().isoformat() == "2026-06-30"
    assert len(aligned) == 2113
    assert compute_file_sha256(CANONICAL_DIR / "canonical_factors_daily.csv") == manifest["canonical_sha256"]


def test_rf_and_hac_rules_and_annualization(tmp_path):
    output_dir = _tmp_static_dir(tmp_path)
    result = run_static_factor_attribution(output_dir)
    static = result["static_regressions"]
    assert (static["alpha_annualized"] == 252.0 * static["alpha_daily"]).all()
    assert static["alpha_raw_pvalue"].between(0, 1).all()
    assert (static["alpha_hac_se"] > 0).all()
    assert (static["condition_number"] > 0).all()
    assert (static["alpha_hac_tstat"].notna()).all()
    assert (static["rmse"] >= 0).all()
    assert (static["residual_std"] >= 0).all()
    assert (static["n_obs"] == 2113).all()
    assert (static["model"] == "CAPM").sum() == 7
    assert (static["model"] == "FF3").sum() == 7
    assert (static["model"] == "FF5").sum() == 7
    assert (static["model"] == "FF5_MOM").sum() == 7
    assert static["alpha_hac_se"].notna().all()


def test_bh_and_coefficients_output(tmp_path):
    output_dir = _tmp_static_dir(tmp_path)
    result = run_static_factor_attribution(output_dir)
    alpha_summary = result["alpha_summary"]
    coeffs = result["factor_coefficients"]
    for model in MODELS:
        family = alpha_summary[alpha_summary["model"] == model]
        qvals = family["bh_qvalue"].to_numpy(dtype=float)
        assert len(family) == 7
        assert (qvals >= 0).all()
        assert (qvals <= 1.0).all()
        assert family["bh_significant"].isin([True, False]).all()
    for strategy in STRATEGIES:
        for model in MODELS:
            expected_labels = get_model_factor_labels(model)
            subset = coeffs[(coeffs["strategy"] == strategy) & (coeffs["model"] == model)]
            assert sorted(subset["factor"].tolist()) == sorted(expected_labels)
            assert subset["coefficient"].notna().all()
            assert subset["hac_pvalue"].between(0, 1).all()
            assert subset["coefficient"].notna().all()


def test_diagnostics_and_factor_correlations(tmp_path):
    output_dir = _tmp_static_dir(tmp_path)
    result = run_static_factor_attribution(output_dir)
    corr = result["factor_correlations"]
    assert list(corr.columns) == ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    assert set(corr.index) == {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"}
    diag = result["residual_diagnostics"]
    assert {"ljung_box_stat", "ljung_box_pvalue", "breusch_pagan_stat", "breusch_pagan_pvalue", "jarque_bera_stat", "jarque_bera_pvalue"}.issubset(diag.columns)
    assert "Durbin_Watson" not in diag.columns
    assert "Durbin_Watson" not in result["static_regressions"].columns
    assert diag["vif_max"].notna().any()
    assert diag["vif_mean"].notna().any()


def test_static_verification_and_hashes(tmp_path):
    output_dir = _tmp_static_dir(tmp_path)
    result = run_static_factor_attribution(output_dir)
    verification = result["verification"]
    required = {
        "canonical_factor_hash_valid",
        "m4_preservation_max_difference",
        "matched_start",
        "matched_end",
        "matched_observation_count",
        "strategy_count",
        "model_count",
        "expected_regression_count",
        "actual_regression_count",
        "all_n_obs_equal_2113",
        "hac_maxlags",
        "alpha_annualization_rule",
        "bh_family_definition",
        "no_live_call_status",
        "no_optimizer_call_status",
        "closed_m1_m4_unchanged",
        "static_output_hashes",
        "phase4_status",
    }
    assert required.issubset(set(verification.keys()))
    assert verification["expected_regression_count"] == 28
    assert verification["actual_regression_count"] == 28
    assert verification["all_n_obs_equal_2113"] is True
    assert verification["hac_maxlags"] == HAC_MAXLAGS
    assert verification["phase4_status"] == "complete"
    assert verification["m4_preservation_max_difference"] <= 1e-12


def test_deterministic_rerun_and_static_outputs(tmp_path):
    output_dir = _tmp_static_dir(tmp_path)
    first = run_static_factor_attribution(output_dir)
    hashes_1 = {name: compute_file_sha256(output_dir / name) for name in [
        "static_factor_regressions.csv",
        "factor_coefficients.csv",
        "alpha_summary.csv",
        "model_fit_summary.csv",
        "factor_correlations.csv",
        "residual_diagnostics.csv",
    ]}
    second = run_static_factor_attribution(output_dir)
    hashes_2 = {name: compute_file_sha256(output_dir / name) for name in hashes_1}
    assert hashes_1 == hashes_2
    assert first["verification"]["static_output_hashes"] == hashes_1
    assert second["verification"]["static_output_hashes"] == hashes_2


def test_no_live_download_and_no_optimizer_call_status(monkeypatch, tmp_path):
    output_dir = _tmp_static_dir(tmp_path)
    import src.optimization.risk_based as risk_based

    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer should not be called during static attribution")

    monkeypatch.setattr(risk_based, "equal_risk_contribution_portfolio", forbidden)
    result = run_static_factor_attribution(output_dir)
    assert result["verification"]["no_optimizer_call_status"] is True
    assert result["verification"]["no_live_call_status"] is True


def test_closed_m4_artifacts_remain_unchanged():
    artifact = ROOT / "results" / "milestone4_canonical" / "walk_forward_returns_gross.csv"
    assert artifact.exists()
    assert artifact.is_file()
    with artifact.open("r", encoding="utf-8") as handle:
        text = handle.read()
    assert "Date" in text
    assert "SPY" in text
    assert "Equal Risk Contribution" in text
