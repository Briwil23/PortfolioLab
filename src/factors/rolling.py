from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.factors.models import FACTOR_MODEL_REGISTRY, validate_model_spec
from src.factors.regression import HAC_MAXLAGS, fit_factor_model

ROLLING_WINDOWS: tuple[int, ...] = (252, 504)
ROLLING_FACTOR_LABELS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
ROLLING_MODEL = "FF5_MOM"


def _coerce_sample(sample: pd.DataFrame, strategy_col: str) -> pd.DataFrame:
    if not isinstance(sample, pd.DataFrame):
        raise TypeError("Rolling sample must be a pandas DataFrame.")
    if sample.empty:
        raise ValueError("Rolling sample is empty.")
    if "date" not in sample.columns:
        raise ValueError("Rolling sample must contain a 'date' column.")
    if strategy_col not in sample.columns:
        raise ValueError(f"Missing strategy column: {strategy_col!r}.")
    if "RF" not in sample.columns:
        raise ValueError("Rolling sample must include an 'RF' column.")
    required = set(ROLLING_FACTOR_LABELS)
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"Rolling sample is missing required factor columns: {missing!r}.")
    df = sample.copy().sort_values("date", kind="mergesort").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("Rolling sample contains duplicate dates.")
    for column in [strategy_col, *ROLLING_FACTOR_LABELS, "RF"]:
        if df[column].isna().any():
            raise ValueError(f"Column {column!r} contains NaN values.")
        arr = df[column].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"Column {column!r} contains inf values.")
    return df


def _fit_window(window_df: pd.DataFrame, strategy_col: str) -> dict:
    window_df = _coerce_sample(window_df, strategy_col)
    labels = validate_model_spec(ROLLING_MODEL, list(window_df.columns))
    data = window_df[["date", *labels, "RF", strategy_col]].copy()
    data["portfolio_return"] = data[strategy_col]
    data["portfolio_excess_return"] = data[strategy_col] - data["RF"]
    result = fit_factor_model(data, model_name=ROLLING_MODEL, target_col="portfolio_excess_return", rf_col="RF")

    beta_by_factor = {label: float(result.factor_coefficients[label]["beta"]) for label in labels}
    entry = {
        "strategy": strategy_col,
        "model": ROLLING_MODEL,
        "window": int(len(window_df)),
        "window_start": window_df["date"].min(),
        "window_end": window_df["date"].max(),
        "n_obs": int(len(window_df)),
        "alpha_daily": float(result.alpha_daily),
        "alpha_annualized": float(result.alpha_annualized),
        "alpha_hac_se": float(result.alpha_hac["HAC_SE"]),
        "alpha_hac_tstat": float(result.alpha_hac["HAC_tstat"]),
        "alpha_hac_pvalue": float(result.alpha_hac["HAC_pvalue"]),
        "r_squared": float(result.r_squared),
        "adjusted_r_squared": float(result.adjusted_r_squared),
        "residual_std": float(result.residual_std),
        "condition_number": float(result.condition_number),
        "beta_MKT_RF": float(beta_by_factor["MKT_RF"]),
        "beta_SMB": float(beta_by_factor["SMB"]),
        "beta_HML": float(beta_by_factor["HML"]),
        "beta_RMW": float(beta_by_factor["RMW"]),
        "beta_CMA": float(beta_by_factor["CMA"]),
        "beta_MOM": float(beta_by_factor["MOM"]),
        "beta_by_factor": beta_by_factor,
    }
    return entry


def rolling_regression_at_date(
    sample: pd.DataFrame,
    date_value: str | pd.Timestamp,
    window: int,
    strategy_col: str = "strategy_return",
) -> dict:
    """Compute the rolling FF5_MOM regression using the latest window observations ending at date_value."""
    sample_df = _coerce_sample(sample, strategy_col)
    target_date = pd.Timestamp(date_value)
    if target_date not in set(sample_df["date"]):
        raise ValueError(f"Date {target_date!r} is not present in the canonical sample.")
    subset = sample_df[sample_df["date"] <= target_date].copy()
    if len(subset) < window:
        raise ValueError(f"Insufficient history for {window}-day window ending at {target_date}: only {len(subset)} rows available.")
    tail = subset.iloc[-window:].copy()
    return _fit_window(tail, strategy_col)


def compute_rolling_factor_regressions(
    sample: pd.DataFrame,
    strategy_col: str = "strategy_return",
    windows: Iterable[int] = ROLLING_WINDOWS,
) -> dict[int, pd.DataFrame]:
    """Compute rolling FF5_MOM regressions for each requested window length."""
    sample_df = _coerce_sample(sample, strategy_col)
    results: dict[int, pd.DataFrame] = {}
    for window in windows:
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"Rolling window must be a positive integer, got {window!r}.")
        if len(sample_df) < window:
            raise ValueError(f"Sample length {len(sample_df)} is shorter than requested window {window}.")
        rows = []
        for end_idx in range(window - 1, len(sample_df)):
            window_df = sample_df.iloc[end_idx - window + 1 : end_idx + 1].copy()
            rows.append(_fit_window(window_df, strategy_col))
        results[window] = pd.DataFrame(rows)
        if results[window].empty:
            raise ValueError(f"No rolling estimates generated for window {window}.")
    return results


__all__ = [
    "ROLLING_WINDOWS",
    "ROLLING_MODEL",
    "ROLLING_FACTOR_LABELS",
    "compute_rolling_factor_regressions",
    "rolling_regression_at_date",
]
