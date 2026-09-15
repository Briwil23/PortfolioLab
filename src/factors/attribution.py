from __future__ import annotations

import pandas as pd
import numpy as np

from src.factors.rolling import ROLLING_FACTOR_LABELS, ROLLING_MODEL, compute_rolling_factor_regressions, rolling_regression_at_date

STRESS_PERIODS = {
    "COVID": ("2020-02-20", "2020-04-30"),
    "2022_RATE_HIKE": ("2022-01-03", "2022-10-31"),
}


def _ensure_dataframe(sample: pd.DataFrame, strategy_col: str) -> pd.DataFrame:
    if not isinstance(sample, pd.DataFrame):
        raise TypeError("Stress sample must be a pandas DataFrame.")
    if "date" not in sample.columns:
        raise ValueError("Stress sample must contain a 'date' column.")
    if strategy_col not in sample.columns:
        raise ValueError(f"Missing strategy column: {strategy_col!r}.")
    if "RF" not in sample.columns:
        raise ValueError("Stress sample must include an 'RF' column.")
    for label in ROLLING_FACTOR_LABELS:
        if label not in sample.columns:
            raise ValueError(f"Stress sample is missing required factor column: {label!r}.")
    df = sample.copy().sort_values("date", kind="mergesort").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def select_latest_pre_stress_row(rolling_rows: pd.DataFrame, stress_start_date: str | pd.Timestamp) -> pd.Series:
    """Return the most recent rolling estimate whose window_end is strictly before stress start."""
    if rolling_rows.empty:
        raise ValueError("Rolling rows are empty.")
    stress_start = pd.Timestamp(stress_start_date)
    eligible = rolling_rows[rolling_rows["window_end"] < stress_start].copy()
    if eligible.empty:
        raise ValueError(f"No rolling estimate exists before stress start {stress_start.date().isoformat()}.")
    return eligible.sort_values("window_end", kind="mergesort").iloc[-1]


def _coerce_rolling_rows_for_stress(rolling_rows: pd.DataFrame | dict[int, pd.DataFrame] | None, window: int) -> pd.DataFrame:
    if rolling_rows is None:
        raise ValueError("Stress attribution reuse path requires an authoritative rolling table; no regression fallback is allowed.")
    if isinstance(rolling_rows, dict):
        if window not in rolling_rows:
            raise ValueError(f"Rolling reuse data is missing window {window}.")
        rolling_frame = rolling_rows[window].copy()
    elif isinstance(rolling_rows, pd.DataFrame):
        rolling_frame = rolling_rows.copy()
        if "window" in rolling_frame.columns:
            rolling_frame = rolling_frame[rolling_frame["window"] == window].copy()
    else:
        raise TypeError("rolling_rows must be a pandas DataFrame or a dict keyed by window.")

    if rolling_frame.empty:
        raise ValueError(f"Rolling reuse data is empty for window {window}.")

    required = [
        "window_start",
        "window_end",
        "alpha_daily",
        "beta_MKT_RF",
        "beta_SMB",
        "beta_HML",
        "beta_RMW",
        "beta_CMA",
        "beta_MOM",
        "r_squared",
        "adjusted_r_squared",
    ]
    missing = [name for name in required if name not in rolling_frame.columns]
    if missing:
        raise ValueError(f"Rolling reuse data is missing required coefficient columns for window {window}: {missing!r}.")

    numeric_cols = [col for col in required if col in rolling_frame.columns]
    numeric = rolling_frame[numeric_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Rolling reuse data contains nonfinite coefficients for window {window}.")
    if rolling_frame.duplicated(subset=["window_start", "window_end"]).any():
        raise ValueError(f"Rolling reuse data contains duplicate coefficient rows for window {window}.")
    return rolling_frame


def _assert_canonical_factor_contract(frame: pd.DataFrame, *, context: str) -> None:
    required = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]
    missing = [label for label in required if label not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required canonical factor columns: {missing!r}.")
    suffixed = [
        label for label in frame.columns
        if any(label.startswith(prefix) for prefix in ("MKT_RF_", "SMB_", "HML_", "RMW_", "CMA_", "MOM_", "RF_"))
    ]
    if suffixed:
        raise ValueError(f"{context} contains suffixed/duplicate factor columns: {suffixed!r}. This indicates a duplicate merge or wrong factor ownership contract.")


def compute_stress_attribution(
    sample: pd.DataFrame,
    strategy_col: str = "strategy_return",
    stress_start: str = "2020-02-20",
    stress_end: str = "2020-04-30",
    window: int = 252,
    rolling_rows: pd.DataFrame | dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute daily stress attribution using a fixed pre-stress coefficient vector."""
    sample_df = _ensure_dataframe(sample, strategy_col)
    _assert_canonical_factor_contract(sample_df, context="Stress sample")
    stress_start_dt = pd.Timestamp(stress_start)
    stress_end_dt = pd.Timestamp(stress_end)
    stress_window = sample_df[(sample_df["date"] >= stress_start_dt) & (sample_df["date"] <= stress_end_dt)].copy()
    if stress_window.empty:
        raise ValueError(f"No aligned stress dates within requested range [{stress_start}, {stress_end}].")
    if rolling_rows is not None:
        rolling = _coerce_rolling_rows_for_stress(rolling_rows, window)
        selected = select_latest_pre_stress_row(rolling, stress_start_dt)
    else:
        rolling = compute_rolling_factor_regressions(sample_df, strategy_col=strategy_col, windows=[window])[window]
        selected = select_latest_pre_stress_row(rolling, stress_start_dt)
    selected_coeffs = {
        "alpha_pre": float(selected["alpha_daily"]),
        "beta_MKT_RF": float(selected["beta_MKT_RF"]),
        "beta_SMB": float(selected["beta_SMB"]),
        "beta_HML": float(selected["beta_HML"]),
        "beta_RMW": float(selected["beta_RMW"]),
        "beta_CMA": float(selected["beta_CMA"]),
        "beta_MOM": float(selected["beta_MOM"]),
    }

    rows = []
    for _, row in stress_window.iterrows():
        rf = float(row["RF"])
        actual_excess = float(row[strategy_col] - row["RF"])
        predicted = (
            selected_coeffs["alpha_pre"]
            + selected_coeffs["beta_MKT_RF"] * float(row["MKT_RF"])
            + selected_coeffs["beta_SMB"] * float(row["SMB"])
            + selected_coeffs["beta_HML"] * float(row["HML"])
            + selected_coeffs["beta_RMW"] * float(row["RMW"])
            + selected_coeffs["beta_CMA"] * float(row["CMA"])
            + selected_coeffs["beta_MOM"] * float(row["MOM"])
        )
        residual = actual_excess - predicted
        alpha_component = selected_coeffs["alpha_pre"]
        factor_component = (
            selected_coeffs["beta_MKT_RF"] * float(row["MKT_RF"]) +
            selected_coeffs["beta_SMB"] * float(row["SMB"]) +
            selected_coeffs["beta_HML"] * float(row["HML"]) +
            selected_coeffs["beta_RMW"] * float(row["RMW"]) +
            selected_coeffs["beta_CMA"] * float(row["CMA"]) +
            selected_coeffs["beta_MOM"] * float(row["MOM"])
        )
        rows.append(
            {
                "date": row["date"],
                "MKT_RF": float(row["MKT_RF"]),
                "SMB": float(row["SMB"]),
                "HML": float(row["HML"]),
                "RMW": float(row["RMW"]),
                "CMA": float(row["CMA"]),
                "MOM": float(row["MOM"]),
                "RF": rf,
                "strategy_return": float(row[strategy_col]),
                "rf": rf,
                "actual_excess_return": actual_excess,
                "predicted_excess_return": predicted,
                "residual": residual,
                "rf_component": rf,
                "alpha_component": alpha_component,
                "MKT_RF_component": selected_coeffs["beta_MKT_RF"] * float(row["MKT_RF"]),
                "SMB_component": selected_coeffs["beta_SMB"] * float(row["SMB"]),
                "HML_component": selected_coeffs["beta_HML"] * float(row["HML"]),
                "RMW_component": selected_coeffs["beta_RMW"] * float(row["RMW"]),
                "CMA_component": selected_coeffs["beta_CMA"] * float(row["CMA"]),
                "MOM_component": selected_coeffs["beta_MOM"] * float(row["MOM"]),
                "factor_component": factor_component,
                "residual_component": residual,
                "total_return_identity_error": float(row[strategy_col] - (rf + alpha_component + factor_component + residual)),
                "excess_return_identity_error": float(actual_excess - (alpha_component + factor_component + residual)),
            }
        )
    return pd.DataFrame(rows)


def compute_cumulative_stress_attribution(stress_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative arithmetic attribution for the stress sample."""
    if stress_frame.empty:
        raise ValueError("Stress attribution frame is empty.")
    frame = stress_frame.copy()
    for component in [
        "rf_component",
        "alpha_component",
        "MKT_RF_component",
        "SMB_component",
        "HML_component",
        "RMW_component",
        "CMA_component",
        "MOM_component",
        "factor_component",
        "residual_component",
    ]:
        frame[f"{component}_cumulative"] = frame[component].cumsum()
    frame["excess_return_cumulative"] = frame["actual_excess_return"].cumsum()
    frame["total_return_cumulative"] = frame["strategy_return"].cumsum()
    return frame


__all__ = [
    "STRESS_PERIODS",
    "select_latest_pre_stress_row",
    "compute_stress_attribution",
    "compute_cumulative_stress_attribution",
]
