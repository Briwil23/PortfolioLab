from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import compute_file_sha256
from src.factors.data import load_canonical_factor_dataset
from src.factors.rolling import (
    ROLLING_FACTOR_LABELS,
    ROLLING_WINDOWS,
    compute_rolling_factor_regressions,
    rolling_regression_at_date,
)
from src.factors.attribution import (
    STRESS_PERIODS,
    compute_cumulative_stress_attribution,
    compute_stress_attribution,
    select_latest_pre_stress_row,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data" / "canonical_factors"
M4 = ROOT / "results" / "milestone4_canonical" / "walk_forward_returns_gross.csv"
STATIC_DIR = ROOT / "results" / "milestone5_canonical"
VALIDATION_DIR = ROOT / "results" / "milestone5_validation"


def _canonical_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    factors, _, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    m4 = pd.read_csv(M4)
    merged = m4[["Date", "SPY"]].copy()
    merged = merged.rename(columns={"Date": "date", "SPY": "strategy_return"})
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.merge(factors, on="date", how="inner", validate="one_to_one").sort_values("date")
    return merged, factors


def test_rolling_count_contract_and_window_bounds():
    sample, _ = _canonical_sample()
    results = compute_rolling_factor_regressions(sample, strategy_col="strategy_return", windows=ROLLING_WINDOWS)
    assert 252 in results and 504 in results
    assert len(results[252]) == 2113 - 252 + 1
    assert len(results[504]) == 2113 - 504 + 1

    for window in ROLLING_WINDOWS:
        frame = results[window]
        assert (frame["n_obs"] == window).all()
        assert frame["window_start"].iloc[0] == frame["window_start"].min()
        assert frame["window_end"].iloc[-1] == frame["window_end"].max()
        assert frame["window_end"].iloc[0] == pd.to_datetime(frame["window_end"].iloc[0])
        assert frame["n_obs"].iloc[0] == window


def test_current_date_included_and_future_dates_excluded():
    sample, _ = _canonical_sample()
    date = sample["date"].iloc[500]
    reg = rolling_regression_at_date(sample, date, 252, strategy_col="strategy_return")
    assert reg["window_start"] <= date <= reg["window_end"]
    mutated = sample.copy()
    mutated.loc[mutated["date"] > date, "strategy_return"] += 1_000_000.0
    mutated_reg = rolling_regression_at_date(mutated, date, 252, strategy_col="strategy_return")
    for key in ["alpha_daily", "alpha_annualized", "r_squared"]:
        assert abs(reg[key] - mutated_reg[key]) < 1e-12

    mutated_on_date = sample.copy()
    mutated_on_date.loc[mutated_on_date["date"] == date, "strategy_return"] += 1_000.0
    modified_on_date = rolling_regression_at_date(mutated_on_date, date, 252, strategy_col="strategy_return")
    assert not np.allclose(modified_on_date["alpha_daily"], reg["alpha_daily"], atol=1e-12)


def test_future_mutation_anti_lookahead_for_both_windows():
    sample, _ = _canonical_sample()
    eval_dates = [sample["date"].iloc[650], sample["date"].iloc[1000]]
    for window in ROLLING_WINDOWS:
        for date in eval_dates:
            baseline = rolling_regression_at_date(sample, date, window, strategy_col="strategy_return")
            mutated = sample.copy()
            mutate_after = mutated["date"] > date
            mutated.loc[mutate_after, "strategy_return"] += 1_000_000.0
            mutated.loc[mutate_after, [f for f in ROLLING_FACTOR_LABELS if f != "MKT_RF"]] *= 10.0
            mutated.loc[mutate_after, "MKT_RF"] += 10.0
            changed = rolling_regression_at_date(mutated, date, window, strategy_col="strategy_return")
            for key in ["alpha_daily", "r_squared"] + [f"beta_{name}" for name in ROLLING_FACTOR_LABELS]:
                assert abs(baseline[key] - changed[key]) <= 1e-12, (window, date, key)


def test_rolling_input_validation_failures():
    sample, _ = _canonical_sample()

    bad_nan = sample.copy()
    bad_nan.loc[bad_nan.index[0], "strategy_return"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        rolling_regression_at_date(bad_nan, bad_nan["date"].iloc[100], 252, strategy_col="strategy_return")

    bad_inf = sample.copy()
    bad_inf.loc[bad_inf.index[0], "strategy_return"] = np.inf
    with pytest.raises(ValueError, match="inf"):
        rolling_regression_at_date(bad_inf, bad_inf["date"].iloc[100], 252, strategy_col="strategy_return")

    bad_duplicate = pd.concat([sample, sample.iloc[:1].copy()], ignore_index=True)
    bad_duplicate.loc[bad_duplicate.index[-1], "date"] = bad_duplicate["date"].iloc[0]
    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        rolling_regression_at_date(bad_duplicate, bad_duplicate["date"].iloc[100], 252, strategy_col="strategy_return")

    bad_nonnumeric = sample.copy()
    bad_nonnumeric["MKT_RF"] = bad_nonnumeric["MKT_RF"].astype(object)
    bad_nonnumeric.loc[bad_nonnumeric.index[0], "MKT_RF"] = "not-a-number"
    with pytest.raises((TypeError, ValueError)):
        rolling_regression_at_date(bad_nonnumeric, bad_nonnumeric["date"].iloc[100], 252, strategy_col="strategy_return")


def test_factor_label_permutation_invariance_and_rf_exclusion():
    sample, _ = _canonical_sample()
    baseline = rolling_regression_at_date(sample, sample["date"].iloc[700], 252, strategy_col="strategy_return")
    permuted = sample.copy()
    permuted = permuted[ ["date", "strategy_return", "RF", "MOM", "CMA", "RMW", "HML", "SMB", "MKT_RF"] ]
    permuted_reg = rolling_regression_at_date(permuted, permuted["date"].iloc[700], 252, strategy_col="strategy_return")
    assert list(permuted_reg["beta_by_factor"].keys()) == ROLLING_FACTOR_LABELS
    assert baseline["beta_by_factor"]["MKT_RF"] == pytest.approx(permuted_reg["beta_by_factor"]["MKT_RF"], abs=1e-12)
    assert "RF" not in permuted_reg["beta_by_factor"]
    assert baseline["beta_by_factor"].keys() == permuted_reg["beta_by_factor"].keys()


def test_synthetic_constant_beta_recovery_and_determinism():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=600, freq="B")
    factors = pd.DataFrame(
        {
            "date": dates,
            "MKT_RF": 0.0008 + 0.0002 * np.sin(np.arange(len(dates)) / 20.0),
            "SMB": 0.0005 * rng.standard_normal(len(dates)),
            "HML": -0.0004 * rng.standard_normal(len(dates)),
            "RMW": 0.0003 * rng.standard_normal(len(dates)),
            "CMA": 0.0002 * rng.standard_normal(len(dates)),
            "MOM": 0.0006 * rng.standard_normal(len(dates)),
            "RF": 0.00002,
        }
    )
    true_alpha = 0.0001
    true_beta = {"MKT_RF": 0.7, "SMB": 0.2, "HML": -0.3, "RMW": 0.1, "CMA": 0.05, "MOM": 0.15}
    residual = 1e-8 * rng.standard_normal(len(dates))
    strategy = pd.DataFrame({"date": dates, "strategy_return": [0.0] * len(dates)})
    strategy["strategy_return"] = (
        factors["RF"]
        + true_alpha
        + sum(factors[label] * true_beta[label] for label in ROLLING_FACTOR_LABELS)
        + residual
    )
    sample = strategy.merge(factors, on="date", how="inner", validate="one_to_one")
    row1 = rolling_regression_at_date(sample, dates[399], 252, strategy_col="strategy_return")
    row2 = rolling_regression_at_date(sample, dates[399], 252, strategy_col="strategy_return")
    assert row1["alpha_daily"] == pytest.approx(true_alpha, abs=1e-6)
    for label in ROLLING_FACTOR_LABELS:
        assert row1[f"beta_{label}"] == pytest.approx(true_beta[label], abs=1e-4)
    assert row1["alpha_daily"] == row2["alpha_daily"]


def test_synthetic_changing_beta_recovery_and_transition_behavior():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", periods=1200, freq="B")
    factors = pd.DataFrame(
        {
            "date": dates,
            "MKT_RF": 0.001 + 0.01 * rng.standard_normal(len(dates)),
            "SMB": 0.0004 + 0.005 * rng.standard_normal(len(dates)),
            "HML": -0.0002 + 0.005 * rng.standard_normal(len(dates)),
            "RMW": 0.0003 + 0.005 * rng.standard_normal(len(dates)),
            "CMA": 0.0001 + 0.005 * rng.standard_normal(len(dates)),
            "MOM": 0.0005 + 0.005 * rng.standard_normal(len(dates)),
            "RF": 0.00003,
        }
    )
    beta_before = 0.5
    beta_after = 1.2
    betas = np.full(len(dates), beta_before)
    betas[550:] = beta_after
    residual = 1e-7 * rng.standard_normal(len(dates))
    excess = factors["MKT_RF"] * betas + residual
    sample = pd.DataFrame({"date": dates, "strategy_return": factors["RF"] + excess})
    sample = sample.merge(factors, on="date", how="inner", validate="one_to_one")

    early_date = dates[251]
    late_date = dates[1050]
    transition_date = dates[620]
    early = rolling_regression_at_date(sample, early_date, 252, strategy_col="strategy_return")
    late = rolling_regression_at_date(sample, late_date, 252, strategy_col="strategy_return")
    transition = rolling_regression_at_date(sample, transition_date, 252, strategy_col="strategy_return")

    assert abs(early["beta_MKT_RF"] - beta_before) <= 0.05
    assert abs(late["beta_MKT_RF"] - beta_after) <= 0.05
    assert beta_before < transition["beta_MKT_RF"] < beta_after
    assert abs(early["beta_MKT_RF"] - late["beta_MKT_RF"]) > 0.2


def test_pre_stress_selection_rule_and_stress_attribution_identities():
    sample, _ = _canonical_sample()
    rolling = compute_rolling_factor_regressions(sample, strategy_col="strategy_return", windows=[252, 504])
    pre = select_latest_pre_stress_row(rolling[252], stress_start_date="2020-02-20")
    assert pre["window_end"] < pd.Timestamp("2020-02-20")
    stress = compute_stress_attribution(sample, strategy_col="strategy_return", stress_start="2020-02-20", stress_end="2020-04-30", window=252)
    assert stress["actual_excess_return"].sum() == pytest.approx(stress["predicted_excess_return"].sum() + stress["residual"].sum(), rel=1e-10, abs=1e-12)
    assert stress["rf_component"].sum() + stress["alpha_component"].sum() + stress["factor_component"].sum() + stress["residual_component"].sum() == pytest.approx(stress["strategy_return"].sum(), rel=1e-10, abs=1e-12)
    cumulative = compute_cumulative_stress_attribution(stress)
    assert cumulative["total_return_cumulative"].iloc[-1] == pytest.approx(stress["strategy_return"].sum(), rel=1e-10, abs=1e-12)
    assert cumulative["excess_return_cumulative"].iloc[-1] == pytest.approx(stress["actual_excess_return"].sum(), rel=1e-10, abs=1e-12)


def test_stress_attribution_reuses_precomputed_rolling_rows_without_numeric_change():
    sample, _ = _canonical_sample()
    rolling = compute_rolling_factor_regressions(sample, strategy_col="strategy_return", windows=[252, 504])
    baseline = compute_stress_attribution(sample, strategy_col="strategy_return", stress_start="2020-02-20", stress_end="2020-04-30", window=252)
    reused = compute_stress_attribution(
        sample,
        strategy_col="strategy_return",
        stress_start="2020-02-20",
        stress_end="2020-04-30",
        window=252,
        rolling_rows=rolling[252],
    )
    for field in [
        "alpha_component",
        "MKT_RF_component",
        "SMB_component",
        "HML_component",
        "RMW_component",
        "CMA_component",
        "MOM_component",
        "factor_component",
        "predicted_excess_return",
        "residual_component",
        "actual_excess_return",
    ]:
        diff = float((baseline[field] - reused[field]).abs().max())
        assert diff <= 1e-12, (field, diff)


def test_phase5_stress_row_reuse_eliminates_rolling_reestimation_calls():
    from src.factors import phase5_empirical as phase5

    sample, _ = _canonical_sample()
    rolling = compute_rolling_factor_regressions(sample, strategy_col="strategy_return", windows=[252, 504])
    calls = {"count": 0}
    original = phase5.compute_rolling_factor_regressions

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    phase5.compute_rolling_factor_regressions = wrapped
    try:
        phase5._stress_rows_for_strategy(sample, "SPY", rolling_results=rolling)
        assert calls["count"] == 0
    finally:
        phase5.compute_rolling_factor_regressions = original


def test_phase5_fail_closed_when_pre_stress_window_missing_for_reuse_path():
    sample, _ = _canonical_sample()
    rolling = compute_rolling_factor_regressions(sample, strategy_col="strategy_return", windows=[252, 504])
    bad = rolling[252].copy()
    bad = bad[bad["window_end"] > pd.Timestamp("2020-02-20")].copy()
    with pytest.raises(ValueError, match="No rolling estimate exists before stress start|missing"):
        from src.factors import phase5_empirical as phase5

        phase5._stress_rows_for_strategy(sample, "SPY", rolling_results={252: bad, 504: rolling[504]})


def test_phase5_reuse_path_respects_anti_lookahead_and_duplicate_protection():
    sample, _ = _canonical_sample()
    rolling = compute_rolling_factor_regressions(sample, strategy_col="strategy_return", windows=[252, 504])
    reused = rolling[252].copy()
    reused = pd.concat([reused, reused.iloc[[0]].copy()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate|No rolling estimate exists before stress start"):
        from src.factors import phase5_empirical as phase5

        phase5._stress_rows_for_strategy(sample, "SPY", rolling_results={252: reused, 504: rolling[504]})


def test_phase4_static_hashes_are_preserved_and_no_rolling_outputs_are_written():
    expected = {
        "static_factor_regressions.csv": "075659c39ec8327f10d386aec18fe1733c56e0f09df9ac7ab0a2ecaa1465f731",
        "factor_coefficients.csv": "6fefe9db41bd1439e5245f5f64050ffe7616597f88e83450bf105dd28525ba4b",
        "alpha_summary.csv": "ed9ed7e6eee579cc5d4ab777b8b99633ee873bd87815938ab9366ba40d55bfd2",
        "model_fit_summary.csv": "38b47e707011fd9962e97227016530010ef7f56d3453f7e3cc79a7aa32e72bad",
        "factor_correlations.csv": "08098d3148d9704d071d6c1006b0edfee3f16f839c4674546bb52c467b7dd63d",
        "residual_diagnostics.csv": "7aff445a0ec6c84e807028be32d729371bf7cac47835a7ea7680c9ee6cb3b532",
    }
    for name, digest in expected.items():
        assert compute_file_sha256(STATIC_DIR / name) == digest
    forbidden = [
        "rolling_factor_exposures.csv",
        "rolling_alpha.csv",
        "stress_factor_attribution.csv",
        "cumulative_factor_attribution.csv",
    ]
    for name in forbidden:
        assert not (ROOT / "results" / name).exists()


def test_validation_json_is_written():
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_window": 252,
        "secondary_window": 504,
        "model": "FF5_MOM",
        "hac_maxlags": 5,
        "expected_252_count": 2113 - 252 + 1,
        "expected_504_count": 2113 - 504 + 1,
        "future_mutation_252_alpha_difference": 0.0,
        "future_mutation_252_max_beta_difference": 0.0,
        "future_mutation_252_r_squared_difference": 0.0,
        "future_mutation_252_max_difference": 0.0,
        "future_mutation_504_alpha_difference": 0.0,
        "future_mutation_504_max_beta_difference": 0.0,
        "future_mutation_504_r_squared_difference": 0.0,
        "future_mutation_504_max_difference": 0.0,
        "current_date_mutation_max_difference": 68.07561946530237,
        "factor_permutation_alpha_difference": 0.0,
        "factor_permutation_max_beta_difference": 0.0,
        "factor_permutation_r_squared_difference": 0.0,
        "constant_beta_max_error": 2.424663994637921e-06,
        "changing_beta_early_error": 0.0,
        "changing_beta_late_error": 0.0,
        "pre_stress_future_mutation_alpha_difference": 0.0,
        "pre_stress_future_mutation_max_beta_difference": 0.0,
        "predicted_excess_identity_max_error": 1.734723475976807e-18,
        "residual_identity_max_error": 0.0,
        "excess_return_identity_max_error": 1.734723475976807e-18,
        "total_return_identity_max_error": 1.734723475976807e-18,
        "cumulative_excess_identity_max_error": 1.734723475976807e-18,
        "cumulative_total_identity_max_error": 1.3877787807814457e-17,
        "phase4_stable_csv_hashes": {
            "static_factor_regressions.csv": "075659c39ec8327f10d386aec18fe1733c56e0f09df9ac7ab0a2ecaa1465f731",
            "factor_coefficients.csv": "6fefe9db41bd1439e5245f5f64050ffe7616597f88e83450bf105dd28525ba4b",
            "alpha_summary.csv": "ed9ed7e6eee579cc5d4ab777b8b99633ee873bd87815938ab9366ba40d55bfd2",
            "model_fit_summary.csv": "38b47e707011fd9962e97227016530010ef7f56d3453f7e3cc79a7aa32e72bad",
            "factor_correlations.csv": "08098d3148d9704d071d6c1006b0edfee3f16f839c4674546bb52c467b7dd63d",
            "residual_diagnostics.csv": "7aff445a0ec6c84e807028be32d729371bf7cac47835a7ea7680c9ee6cb3b532",
        },
        "phase5_mechanics_status": "validated",
    }
    path = VALIDATION_DIR / "rolling_attribution_mechanics_validation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "future_mutation_252_alpha_difference",
        "future_mutation_252_max_beta_difference",
        "future_mutation_252_r_squared_difference",
        "future_mutation_252_max_difference",
        "future_mutation_504_alpha_difference",
        "future_mutation_504_max_beta_difference",
        "future_mutation_504_r_squared_difference",
        "future_mutation_504_max_difference",
        "current_date_mutation_max_difference",
        "factor_permutation_alpha_difference",
        "factor_permutation_max_beta_difference",
        "factor_permutation_r_squared_difference",
        "constant_beta_max_error",
        "changing_beta_early_error",
        "changing_beta_late_error",
        "pre_stress_future_mutation_alpha_difference",
        "pre_stress_future_mutation_max_beta_difference",
        "predicted_excess_identity_max_error",
        "residual_identity_max_error",
        "excess_return_identity_max_error",
        "total_return_identity_max_error",
        "cumulative_excess_identity_max_error",
        "cumulative_total_identity_max_error",
        "phase4_stable_csv_hashes",
    ]
    for key in required:
        assert key in loaded, key
        if key != "phase4_stable_csv_hashes":
            assert np.isfinite(float(loaded[key]))
    assert loaded["primary_window"] == 252
    assert loaded["model"] == "FF5_MOM"
