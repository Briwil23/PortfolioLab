from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.factors.data import load_canonical_factor_dataset
from src.factors.models import FACTOR_MODEL_REGISTRY, get_model_factor_labels, validate_model_spec
from src.factors.regression import (
    FactorRegressionResult,
    align_m4_returns_to_factors,
    build_factor_design_matrix,
    fit_factor_model,
    minimum_static_observations,
    portfolio_excess_return,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data" / "canonical_factors"
M4_GROSS = ROOT / "results" / "milestone4_canonical" / "walk_forward_returns_gross.csv"


def _synthetic_factor_frame(n: int = 120, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=n, freq="B"),
            "MKT_RF": rng.normal(0.0005, 0.01, size=n),
            "SMB": rng.normal(0.0003, 0.008, size=n),
            "HML": rng.normal(-0.0002, 0.009, size=n),
            "RMW": rng.normal(0.0001, 0.007, size=n),
            "CMA": rng.normal(0.0002, 0.006, size=n),
            "MOM": rng.normal(0.0004, 0.01, size=n),
            "RF": rng.normal(0.0001, 0.001, size=n),
        }
    )
    return frame


def _synthetic_portfolio_case(model_name: str, *, alpha: float = 0.0005, beta_map: dict[str, float] | None = None) -> pd.DataFrame:
    factors = _synthetic_factor_frame(n=220, seed=11)
    if beta_map is None:
        beta_map = {
            "MKT_RF": 1.2,
            "SMB": 0.35,
            "HML": -0.55,
            "RMW": 0.25,
            "CMA": 0.18,
            "MOM": 0.30,
        }
    labels = get_model_factor_labels(model_name)
    linear = alpha + sum(beta_map.get(label, 0.0) * factors[label] for label in labels)
    noise = np.linspace(-0.00015, 0.00015, len(factors))
    factors["portfolio_return"] = linear + noise
    factors["portfolio_excess_return"] = portfolio_excess_return(factors["portfolio_return"], factors["RF"])
    return factors


def test_model_registry_exact():
    assert FACTOR_MODEL_REGISTRY == {
        "CAPM": ["MKT_RF"],
        "FF3": ["MKT_RF", "SMB", "HML"],
        "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
        "FF5_MOM": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
    }
    assert get_model_factor_labels("FF5_MOM") == ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_factor_labels("UNKNOWN")


def test_excess_return_construction_and_no_double_rf_subtraction():
    portfolio = pd.Series([0.01, 0.02, 0.03], index=pd.Index(["2020-01-01", "2020-01-02", "2020-01-03"], name="date"))
    rf = pd.Series([0.002, 0.0025, 0.003], index=portfolio.index)
    excess = portfolio_excess_return(portfolio, rf)
    expected = pd.Series([0.0080, 0.0175, 0.0270], index=portfolio.index)
    pd.testing.assert_series_equal(excess, expected)

    factors = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "MKT_RF": [0.01, 0.02, 0.03], "RF": [0.002, 0.0025, 0.003]})
    x = build_factor_design_matrix(factors, "CAPM")
    assert list(x.columns) == ["MKT_RF"]
    assert "RF" not in x.columns


def test_capm_synthetic_recovery():
    data = _synthetic_portfolio_case("CAPM", alpha=0.0007, beta_map={"MKT_RF": 1.35})
    result = fit_factor_model(data, model_name="CAPM", target_col="portfolio_excess_return")
    assert isinstance(result, FactorRegressionResult)
    assert result.model == "CAPM"
    assert result.alpha_daily == pytest.approx(0.0007, abs=5e-4)
    assert result.factor_coefficients["MKT_RF"]["beta"] == pytest.approx(1.35, abs=5e-3)
    assert result.n_obs >= minimum_static_observations("CAPM")
    assert result.condition_number > 0
    assert result.r_squared >= 0.0


def test_ff3_synthetic_recovery():
    data = _synthetic_portfolio_case("FF3", alpha=0.0002, beta_map={"MKT_RF": 0.9, "SMB": 0.35, "HML": -0.45})
    result = fit_factor_model(data, model_name="FF3", target_col="portfolio_excess_return")
    assert result.model == "FF3"
    assert result.alpha_daily == pytest.approx(0.0002, abs=5e-4)
    assert result.factor_coefficients["MKT_RF"]["beta"] == pytest.approx(0.9, abs=5e-2)
    assert result.factor_coefficients["SMB"]["beta"] == pytest.approx(0.35, abs=1.5e-2)
    assert result.factor_coefficients["HML"]["beta"] == pytest.approx(-0.45, abs=1.5e-2)


def test_ff5_momentum_synthetic_recovery():
    data = _synthetic_portfolio_case("FF5_MOM", alpha=0.0005, beta_map={"MKT_RF": 1.0, "SMB": 0.3, "HML": -0.4, "RMW": 0.2, "CMA": 0.1, "MOM": 0.25})
    result = fit_factor_model(data, model_name="FF5_MOM", target_col="portfolio_excess_return")
    assert result.model == "FF5_MOM"
    assert set(result.factor_coefficients) == {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"}
    assert result.alpha_daily == pytest.approx(0.0005, abs=5e-4)
    assert result.factor_coefficients["MOM"]["beta"] == pytest.approx(0.25, abs=5e-3)


def test_ff5_synthetic_recovery_and_intercept_present():
    data = _synthetic_portfolio_case("FF5", alpha=0.0004, beta_map={"MKT_RF": 1.1, "SMB": 0.4, "HML": -0.7, "RMW": 0.2, "CMA": 0.15})
    result = fit_factor_model(data, model_name="FF5", target_col="portfolio_excess_return")
    assert result.alpha_daily == pytest.approx(0.0004, abs=5e-4)
    assert set(result.factor_coefficients) == {"MKT_RF", "SMB", "HML", "RMW", "CMA"}
    assert result.intercept_name == "const"
    assert result.ols_fit is not None


def test_hac_matches_direct_statsmodels_reference():
    data = _synthetic_portfolio_case("FF3", alpha=0.0003, beta_map={"MKT_RF": 0.9, "SMB": 0.3, "HML": -0.4})
    result = fit_factor_model(data, model_name="FF3", target_col="portfolio_excess_return")

    exog = sm.add_constant(build_factor_design_matrix(data, "FF3"), has_constant="add")
    endog = data["portfolio_excess_return"]
    direct = sm.OLS(endog, exog).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    for label in ["MKT_RF", "SMB", "HML"]:
        idx = result.factor_coefficients[label]
        assert idx["beta"] == pytest.approx(direct.params[label], rel=1e-9, abs=1e-10)
        assert idx["HAC_SE"] == pytest.approx(direct.bse[label], rel=1e-9, abs=1e-10)
        assert idx["HAC_tstat"] == pytest.approx(direct.tvalues[label], rel=1e-9, abs=1e-10)
        assert idx["HAC_pvalue"] == pytest.approx(direct.pvalues[label], rel=1e-9, abs=1e-10)
        ci = direct.conf_int(alpha=0.05).loc[label]
        assert idx["HAC_CI_low"] == pytest.approx(ci.iloc[0], rel=1e-9, abs=1e-10)
        assert idx["HAC_CI_high"] == pytest.approx(ci.iloc[1], rel=1e-9, abs=1e-10)

    assert result.alpha_hac["HAC_SE"] == pytest.approx(direct.bse["const"], rel=1e-9, abs=1e-10)


def test_ols_vs_hac_coefficient_equality():
    data = _synthetic_portfolio_case("CAPM", alpha=0.0002, beta_map={"MKT_RF": 1.1})
    result = fit_factor_model(data, model_name="CAPM", target_col="portfolio_excess_return")
    exog = sm.add_constant(build_factor_design_matrix(data, "CAPM"), has_constant="add")
    direct_ols = sm.OLS(data["portfolio_excess_return"], exog).fit()
    assert result.factor_coefficients["MKT_RF"]["beta"] == pytest.approx(direct_ols.params["MKT_RF"], rel=1e-9, abs=1e-10)
    assert result.alpha_daily == pytest.approx(direct_ols.params["const"], rel=1e-9, abs=1e-10)


def test_model_rejection_and_unknown_model_labels():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_factor_labels("UNKNOWN")
    with pytest.raises(ValueError, match="Unknown model"):
        validate_model_spec("UNKNOWN", ["MKT_RF"])
    with pytest.raises(ValueError, match="Missing required factors"):
        validate_model_spec("FF3", ["MKT_RF", "SMB"])
    with pytest.raises(ValueError, match="Duplicate factor labels"):
        validate_model_spec("FF5", ["MKT_RF", "SMB", "HML", "RMW", "CMA", "CMA"])


def test_rf_excluded_from_explanatory_matrix_and_model_registry_exact_specs():
    data = _synthetic_factor_frame(n=80, seed=4)
    x = build_factor_design_matrix(data, "FF5_MOM")
    assert list(x.columns) == ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    assert "RF" not in x.columns
    with pytest.raises(ValueError, match="Unknown model"):
        validate_model_spec("UNKNOWN", ["MKT_RF"])
    with pytest.raises(ValueError, match="Missing required factors"):
        validate_model_spec("FF3", ["MKT_RF", "SMB"])
    with pytest.raises(ValueError, match="Duplicate factor labels"):
        validate_model_spec("FF5", ["MKT_RF", "SMB", "HML", "RMW", "CMA", "CMA"])


def test_factor_permutation_invariance_by_label():
    data = _synthetic_portfolio_case("FF5_MOM", alpha=0.0006, beta_map={"MKT_RF": 1.1, "SMB": 0.3, "HML": -0.5, "RMW": 0.2, "CMA": 0.1, "MOM": 0.25})
    reordered = data[["date", "RF", "portfolio_return", "portfolio_excess_return", "MOM", "HML", "MKT_RF", "CMA", "SMB", "RMW"]].copy()
    result = fit_factor_model(reordered, model_name="FF5_MOM", target_col="portfolio_excess_return")
    assert set(result.factor_coefficients) == {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"}
    assert result.factor_coefficients["MKT_RF"]["beta"] == pytest.approx(1.1, abs=5e-2)
    assert result.factor_coefficients["SMB"]["beta"] == pytest.approx(0.3, abs=2e-2)
    assert result.factor_coefficients["HML"]["beta"] == pytest.approx(-0.5, abs=2e-2)
    assert result.factor_coefficients["MOM"]["beta"] == pytest.approx(0.25, abs=2e-2)


def test_deterministic_repeated_regression_and_stats_outputs():
    data = _synthetic_portfolio_case("FF5", alpha=0.0004, beta_map={"MKT_RF": 1.1, "SMB": 0.4, "HML": -0.7, "RMW": 0.2, "CMA": 0.15})
    first = fit_factor_model(data, model_name="FF5", target_col="portfolio_excess_return")
    second = fit_factor_model(data, model_name="FF5", target_col="portfolio_excess_return")
    assert first.alpha_daily == pytest.approx(second.alpha_daily, rel=1e-12)
    assert first.adjusted_r_squared == pytest.approx(second.adjusted_r_squared, rel=1e-12)
    assert first.residual_std == pytest.approx(second.residual_std, rel=1e-12)
    assert first.condition_number == pytest.approx(second.condition_number, rel=1e-12)
    assert first.r_squared == pytest.approx(second.r_squared, rel=1e-12)


def test_r2_adjusted_r2_residual_std_and_condition_number_validation():
    data = _synthetic_portfolio_case("FF3", alpha=0.0003, beta_map={"MKT_RF": 0.9, "SMB": 0.3, "HML": -0.4})
    result = fit_factor_model(data, model_name="FF3", target_col="portfolio_excess_return")
    exog = sm.add_constant(build_factor_design_matrix(data, "FF3"), has_constant="add")
    direct = sm.OLS(data["portfolio_excess_return"], exog).fit()
    assert result.r_squared == pytest.approx(direct.rsquared, rel=1e-10, abs=1e-12)
    assert result.adjusted_r_squared == pytest.approx(direct.rsquared_adj, rel=1e-10, abs=1e-12)
    assert result.residual_std == pytest.approx(direct.resid.std(ddof=int(direct.df_resid)), rel=1e-10, abs=1e-12)
    assert result.condition_number == pytest.approx(np.linalg.cond(exog.to_numpy(dtype=float)), rel=1e-10, abs=1e-12)


def test_alpha_annualization_and_fitted_value_identity():
    data = _synthetic_portfolio_case("FF5")
    result = fit_factor_model(data, model_name="FF5", target_col="portfolio_excess_return")
    assert result.alpha_annualized == pytest.approx(252.0 * result.alpha_daily, rel=1e-12)
    expected = result.predicted_excess_return + result.residuals
    pd.testing.assert_series_equal(expected, result.actual_excess_return, check_names=False, check_exact=False, rtol=1e-10, atol=1e-12)


def test_minimum_observation_rule_and_zero_residual_dof():
    valid = _synthetic_factor_frame(n=30, seed=10)
    valid["portfolio_excess_return"] = 0.001 + 0.5 * valid["MKT_RF"]
    assert minimum_static_observations("CAPM") == 30
    assert minimum_static_observations("FF3") == 30
    assert minimum_static_observations("FF5") == 30
    assert minimum_static_observations("FF5_MOM") == 30
    result = fit_factor_model(valid, model_name="CAPM", target_col="portfolio_excess_return")
    assert result.n_obs >= minimum_static_observations("CAPM")

    insufficient = _synthetic_factor_frame(n=29, seed=10)
    insufficient["portfolio_excess_return"] = 0.001 + 0.5 * insufficient["MKT_RF"]
    with pytest.raises(ValueError, match="Too few observations for HAC inference"):
        fit_factor_model(insufficient, model_name="CAPM", target_col="portfolio_excess_return")

    degenerate = _synthetic_factor_frame(n=30, seed=11)
    degenerate["portfolio_excess_return"] = 0.001 + 0.5 * degenerate["MKT_RF"]
    degenerate["MKT_RF"] = 0.0
    with pytest.raises(ValueError, match="rank deficient|constant|identification"):
        fit_factor_model(degenerate, model_name="CAPM", target_col="portfolio_excess_return")


def test_rank_deficiency_and_constant_factor_rejection():
    data = _synthetic_factor_frame(n=60, seed=5)
    data["portfolio_excess_return"] = 0.001 + 0.5 * data["MKT_RF"]
    data["MKT_RF"] = 0.0
    with pytest.raises(ValueError, match="rank|collinear|constant"):
        fit_factor_model(data, model_name="CAPM", target_col="portfolio_excess_return")


def test_nan_inf_and_missing_factor_failures():
    data = _synthetic_factor_frame(n=50, seed=9)
    data["portfolio_excess_return"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_factor_model(data, model_name="CAPM", target_col="portfolio_excess_return")

    data = _synthetic_factor_frame(n=50, seed=9)
    data["portfolio_excess_return"] = np.inf
    with pytest.raises(ValueError, match="inf"):
        fit_factor_model(data, model_name="CAPM", target_col="portfolio_excess_return")

    data = _synthetic_factor_frame(n=50, seed=9).drop(columns=["MKT_RF"])
    data["portfolio_excess_return"] = 0.001 + 0.1 * data["SMB"]
    with pytest.raises(ValueError, match="Missing required factors"):
        fit_factor_model(data, model_name="CAPM", target_col="portfolio_excess_return")

    bad_types = _synthetic_factor_frame(n=30, seed=8)
    bad_types["MKT_RF"] = ["bad-value" for _ in range(len(bad_types))]
    bad_types["portfolio_excess_return"] = 0.001 + 0.5 * pd.to_numeric(np.full(len(bad_types), 0.1), errors="raise")
    with pytest.raises((TypeError, ValueError), match="Unable to parse string|could not convert|invalid|float"):
        fit_factor_model(bad_types, model_name="CAPM", target_col="portfolio_excess_return")


def test_duplicate_date_and_insufficient_obs_failures():
    data = _synthetic_factor_frame(n=12, seed=12)
    data["portfolio_excess_return"] = 0.001 + data["MKT_RF"]
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate dates"):
        fit_factor_model(data, model_name="CAPM", target_col="portfolio_excess_return")

    small = _synthetic_factor_frame(n=3, seed=12)
    small["portfolio_excess_return"] = 0.001 + small["MKT_RF"]
    with pytest.raises(ValueError, match="Too few observations|n_obs"):
        fit_factor_model(small, model_name="CAPM", target_col="portfolio_excess_return")


def test_m4_preservation_aligned_to_canonical_factor_sample():
    canonical, _, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    m4 = pd.read_csv(M4_GROSS)
    aligned = align_m4_returns_to_factors(m4, canonical)
    assert aligned["date"].min().date().isoformat() == "2018-02-01"
    assert aligned["date"].max().date().isoformat() == "2026-06-30"
    assert len(aligned) == 2113
    for strategy in [c for c in m4.columns if c != "Date"]:
        assert aligned[strategy].notna().all()
    original = m4[[c for c in m4.columns if c != "Date"]].reset_index(drop=True)
    preserved = aligned[[c for c in m4.columns if c != "Date"]].reset_index(drop=True)
    max_diff = float((preserved - original).abs().max().max())
    assert max_diff == pytest.approx(0.0, abs=1e-12)


def test_no_live_download_and_no_optimizer_call_in_regression_path(monkeypatch):
    import src.optimization.risk_based as risk_based

    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer should not be called during regression mechanics")

    monkeypatch.setattr(risk_based, "equal_risk_contribution_portfolio", forbidden)
    data = _synthetic_factor_frame(n=80, seed=12)
    data["portfolio_return"] = 0.0005 + 0.9 * data["MKT_RF"] + 0.3 * data["SMB"]
    data["portfolio_excess_return"] = portfolio_excess_return(data["portfolio_return"], data["RF"])
    result = fit_factor_model(data, model_name="FF3", target_col="portfolio_excess_return")
    assert result.model == "FF3"


def test_hac_maxlags_enforced_to_five():
    data = _synthetic_portfolio_case("FF3")
    result = fit_factor_model(data, model_name="FF3", target_col="portfolio_excess_return")
    assert result.hac_maxlags == 5
    assert result.cov_type == "HAC"
