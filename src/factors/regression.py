"""Regression mechanics for Milestone 5 canonical factor data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.factors.models import FACTOR_MODEL_REGISTRY, get_model_factor_labels, validate_model_spec

MIN_OBS_RULE = "max(30, number_of_parameters + HAC_MAXLAGS + 2) with number_of_parameters including the intercept"
HAC_MAXLAGS = 5


def minimum_static_observations(model_name: str) -> int:
    """Return the conservative production minimum for static HAC regression inference."""
    if model_name not in FACTOR_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Expected one of {sorted(FACTOR_MODEL_REGISTRY)}.")
    parameter_count = len(get_model_factor_labels(model_name)) + 1
    return max(30, parameter_count + HAC_MAXLAGS + 2)


@dataclass
class FactorRegressionResult:
    model: str
    n_obs: int
    n_params: int
    sample_start: str
    sample_end: str
    alpha_daily: float
    alpha_annualized: float
    r_squared: float
    adjusted_r_squared: float
    residual_std: float
    condition_number: float
    factor_coefficients: dict[str, dict[str, float]]
    alpha_hac: dict[str, float]
    beta_table: dict[str, dict[str, float]]
    ols_fit: object
    hac_maxlags: int = HAC_MAXLAGS
    cov_type: str = "HAC"
    intercept_name: str = "const"
    actual_excess_return: pd.Series | None = None
    predicted_excess_return: pd.Series | None = None
    residuals: pd.Series | None = None


def _as_numeric_series(name: str, values: pd.Series) -> pd.Series:
    series = pd.Series(values, copy=True)
    if series.isna().any():
        raise ValueError(f"{name} contains NaN values.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains inf values.")
    if series.index.has_duplicates:
        raise ValueError(f"{name} has duplicate keys.")
    return series.astype(float)


def portfolio_excess_return(portfolio_daily_return: pd.Series | np.ndarray | list[float], rf: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Return portfolio excess return = portfolio_return - RF.

    The caller is expected to supply decimal returns; the function performs a single
    subtraction and rejects NaN/inf, duplicate keys, and alignment issues.
    """
    if isinstance(portfolio_daily_return, pd.Series):
        portfolio = portfolio_daily_return.copy()
    else:
        portfolio = pd.Series(portfolio_daily_return, copy=True)
    if isinstance(rf, pd.Series):
        rf_series = rf.copy()
    else:
        rf_series = pd.Series(rf, copy=True)

    if len(portfolio) != len(rf_series):
        raise ValueError("portfolio_daily_return and rf must have the same length.")
    if portfolio.index.has_duplicates or rf_series.index.has_duplicates:
        raise ValueError("Duplicate keys are not allowed when constructing excess returns.")

    if not portfolio.index.equals(rf_series.index):
        if not portfolio.index.equals(pd.Index(rf_series.index, name=portfolio.index.name)):
            raise ValueError("portfolio_daily_return and rf are misaligned on dates/keys.")

    portfolio = _as_numeric_series("portfolio_daily_return", portfolio)
    rf_series = _as_numeric_series("rf", rf_series)
    excess = portfolio - rf_series
    if not np.isfinite(excess.to_numpy(dtype=float)).all():
        raise ValueError("Excess-return construction produced non-finite values.")
    return excess


def build_factor_design_matrix(factors: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Build a factor design matrix using the locked model registry and label-based ordering.

    Extra columns like `date`, `RF`, and portfolio targets are ignored for the design
    matrix, but the required model labels must be present and unique.
    """
    if not isinstance(factors, pd.DataFrame):
        raise ValueError("Factor data must be a pandas DataFrame.")
    if factors.empty:
        raise ValueError("Factor data is empty.")
    if factors.columns.has_duplicates:
        raise ValueError("Duplicate factor labels are not allowed in the design matrix.")

    labels = validate_model_spec(model_name, list(factors.columns))
    subset = factors.loc[:, labels].copy()
    for label in subset.columns:
        if not pd.api.types.is_numeric_dtype(subset[label]):
            subset[label] = pd.to_numeric(subset[label], errors="raise")
    if subset.isna().any().any():
        raise ValueError(f"Factor matrix for model {model_name!r} contains NaN values.")
    if not np.isfinite(subset.to_numpy(dtype=float)).all():
        raise ValueError(f"Factor matrix for model {model_name!r} contains inf values.")
    return subset


def align_m4_returns_to_factors(m4_returns: pd.DataFrame, canonical_factors: pd.DataFrame) -> pd.DataFrame:
    """Align the closed M4 returns to the exact canonical-factor date sample without transformation."""
    if m4_returns.empty or canonical_factors.empty:
        raise ValueError("M4 returns and canonical factors must both be non-empty.")
    left = m4_returns.copy()
    right = canonical_factors.copy()

    if "Date" in left.columns:
        left["date"] = pd.to_datetime(left["Date"])
    elif "date" in left.columns:
        left["date"] = pd.to_datetime(left["date"])
    else:
        raise ValueError("M4 returns must contain a Date/date column.")

    if "date" not in right.columns:
        raise ValueError("Canonical factors must contain a date column.")
    right["date"] = pd.to_datetime(right["date"])

    if left["date"].duplicated().any():
        raise ValueError("Duplicate dates in M4 returns.")
    if right["date"].duplicated().any():
        raise ValueError("Duplicate dates in canonical factors.")

    aligned = left.merge(right[["date"]], on="date", how="inner").sort_values("date", kind="mergesort").reset_index(drop=True)
    return aligned


def _prepare_target_and_exog(data: pd.DataFrame, model_name: str, target_col: str, rf_col: str = "RF") -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Regression input must be a pandas DataFrame.")
    if "date" in data.columns:
        if data["date"].duplicated().any():
            raise ValueError("Duplicate dates are not allowed in the regression sample.")
        data = data.sort_values("date", kind="mergesort").reset_index(drop=True)

    if target_col not in data.columns:
        if "portfolio_return" in data.columns and rf_col in data.columns:
            target_col = "portfolio_excess_return"
            data = data.copy()
            data[target_col] = portfolio_excess_return(data["portfolio_return"], data[rf_col])
        else:
            raise ValueError(f"Missing regression target column: {target_col!r}.")

    if target_col == "portfolio_excess_return" and target_col not in data.columns:
        raise ValueError("Target column must be present or derivable from portfolio_return and RF.")

    target = data[target_col].copy()
    if target.isna().any():
        raise ValueError("Target values contain NaN.")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target values contain inf values.")

    exog = build_factor_design_matrix(data, model_name)
    if exog.isna().any().any():
        raise ValueError("Factor design matrix contains NaN values.")
    if not np.isfinite(exog.to_numpy(dtype=float)).all():
        raise ValueError("Factor design matrix contains inf values.")

    if len(target) != len(exog):
        raise ValueError("Target and factor design matrix have mismatched row counts.")
    minimum_n_obs = minimum_static_observations(model_name)
    if len(target) < minimum_n_obs:
        raise ValueError(
            f"Too few observations for HAC inference: n_obs={len(target)} < minimum_n_obs={minimum_n_obs} "
            f"for model {model_name!r}."
        )

    exog_rank = np.linalg.matrix_rank(exog.to_numpy(dtype=float))
    if exog_rank < exog.shape[1]:
        raise ValueError("Regression design is rank deficient or constant in a way that prevents identification.")

    return target, exog, target


def fit_factor_model(data: pd.DataFrame, model_name: str, target_col: str = "portfolio_excess_return", rf_col: str = "RF") -> FactorRegressionResult:
    """Estimate an OLS factor model with explicit intercept and HAC/Newey-West inference."""
    if model_name not in FACTOR_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Expected one of {sorted(FACTOR_MODEL_REGISTRY)}.")

    target, exog, _ = _prepare_target_and_exog(data, model_name, target_col=target_col, rf_col=rf_col)
    exog_full = sm.add_constant(exog, has_constant="add")
    if exog_full.isnull().any().any():
        raise ValueError("Regression design contains NaN values.")

    if exog_full.shape[0] <= exog_full.shape[1]:
        raise ValueError("Zero residual degrees of freedom or insufficient observations.")
    minimum_n_obs = minimum_static_observations(model_name)
    if exog_full.shape[0] < minimum_n_obs:
        raise ValueError(
            f"Too few observations for HAC inference: n_obs={exog_full.shape[0]} < minimum_n_obs={minimum_n_obs} "
            f"for model {model_name!r}."
        )

    ols_fit = sm.OLS(target, exog_full, missing="raise").fit()
    hac_fit = sm.OLS(target, exog_full, missing="raise").fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})

    alpha_hac = {
        "beta": float(hac_fit.params["const"]),
        "HAC_SE": float(hac_fit.bse["const"]),
        "HAC_tstat": float(hac_fit.tvalues["const"]),
        "HAC_pvalue": float(hac_fit.pvalues["const"]),
        "HAC_CI_low": float(hac_fit.conf_int(alpha=0.05).loc["const", 0]),
        "HAC_CI_high": float(hac_fit.conf_int(alpha=0.05).loc["const", 1]),
    }

    factor_coefficients: dict[str, dict[str, float]] = {}
    ci = hac_fit.conf_int(alpha=0.05)
    for label in exog.columns:
        factor_coefficients[label] = {
            "beta": float(hac_fit.params[label]),
            "HAC_SE": float(hac_fit.bse[label]),
            "HAC_tstat": float(hac_fit.tvalues[label]),
            "HAC_pvalue": float(hac_fit.pvalues[label]),
            "HAC_CI_low": float(ci.loc[label, 0]),
            "HAC_CI_high": float(ci.loc[label, 1]),
        }

    sample_start = pd.to_datetime(data["date"]).min().strftime("%Y-%m-%d") if "date" in data.columns else target.index[0].strftime("%Y-%m-%d")
    sample_end = pd.to_datetime(data["date"]).max().strftime("%Y-%m-%d") if "date" in data.columns else target.index[-1].strftime("%Y-%m-%d")

    alpha_daily = float(hac_fit.params["const"])
    result = FactorRegressionResult(
        model=model_name,
        n_obs=int(len(target)),
        n_params=int(exog_full.shape[1]),
        sample_start=sample_start,
        sample_end=sample_end,
        alpha_daily=alpha_daily,
        alpha_annualized=float(252.0 * alpha_daily),
        r_squared=float(ols_fit.rsquared),
        adjusted_r_squared=float(ols_fit.rsquared_adj),
        residual_std=float(ols_fit.resid.std(ddof=int(ols_fit.df_resid))),
        condition_number=float(np.linalg.cond(exog_full.to_numpy(dtype=float))),
        factor_coefficients=factor_coefficients,
        alpha_hac=alpha_hac,
        beta_table={**factor_coefficients},
        ols_fit=ols_fit,
        hac_maxlags=HAC_MAXLAGS,
        cov_type="HAC",
        intercept_name="const",
        actual_excess_return=target.copy(),
        predicted_excess_return=pd.Series(ols_fit.fittedvalues.to_numpy(), index=target.index, name="predicted_excess_return"),
        residuals=pd.Series(ols_fit.resid.to_numpy(), index=target.index, name="residual"),
    )
    return result


__all__ = [
    "HAC_MAXLAGS",
    "MIN_OBS_RULE",
    "FactorRegressionResult",
    "align_m4_returns_to_factors",
    "build_factor_design_matrix",
    "fit_factor_model",
    "minimum_static_observations",
    "portfolio_excess_return",
]
