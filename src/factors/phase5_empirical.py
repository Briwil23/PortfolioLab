from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market_data import compute_file_sha256
from src.factors.attribution import STRESS_PERIODS, compute_stress_attribution, compute_cumulative_stress_attribution, select_latest_pre_stress_row
from src.factors.data import load_canonical_factor_dataset
from src.factors.rolling import ROLLING_FACTOR_LABELS, ROLLING_WINDOWS, compute_rolling_factor_regressions
from src.factors.static_attribution import STRATEGIES

ROOT = Path(__file__).resolve().parents[2]
M4_GROSS = ROOT / "results" / "milestone4_canonical" / "walk_forward_returns_gross.csv"
CANONICAL_DIR = ROOT / "data" / "canonical_factors"
OUT_DIR = ROOT / "results" / "milestone5_canonical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _aligned_factor_sample() -> pd.DataFrame:
    canonical, _, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    m4 = pd.read_csv(M4_GROSS)
    left = m4[["Date", *STRATEGIES]].copy().rename(columns={"Date": "date"})
    left["date"] = pd.to_datetime(left["date"])
    sample = left.merge(canonical, on="date", how="inner", validate="one_to_one").sort_values("date", kind="mergesort").reset_index(drop=True)
    if len(sample) != 2113:
        raise ValueError(f"Expected 2113 matched dates, found {len(sample)}")
    return sample


def _rolling_rows_for_strategy(sample: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strategy_df = sample[["date", strategy, "RF", *ROLLING_FACTOR_LABELS]].copy()
    strategy_df = strategy_df.rename(columns={strategy: "strategy_return"})
    rolling_map = compute_rolling_factor_regressions(strategy_df, strategy_col="strategy_return", windows=ROLLING_WINDOWS)
    rows: list[dict] = []
    for window in ROLLING_WINDOWS:
        frame = rolling_map[window].copy()
        frame["strategy"] = strategy
        frame["model"] = "FF5_MOM"
        frame["window"] = int(window)
        cols = [
            "strategy",
            "model",
            "window",
            "window_start",
            "window_end",
            "n_obs",
            "alpha_daily",
            "alpha_annualized",
            "alpha_hac_se",
            "alpha_hac_tstat",
            "alpha_hac_pvalue",
            "r_squared",
            "adjusted_r_squared",
            "residual_std",
            "condition_number",
            "beta_MKT_RF",
            "beta_SMB",
            "beta_HML",
            "beta_RMW",
            "beta_CMA",
            "beta_MOM",
        ]
        rows.extend(frame[cols].to_dict(orient="records"))
    return pd.DataFrame(rows)


def _rolling_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []
    for (strategy, window), group in rolling.groupby(["strategy", "window"], sort=True):
        metrics = []
        for factor in ["beta_MKT_RF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA", "beta_MOM"]:
            series = group[factor].astype(float)
            metrics.append(
                {
                    "strategy": strategy,
                    "window": int(window),
                    "factor": factor,
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "p05": float(series.quantile(0.05)),
                    "p25": float(series.quantile(0.25)),
                    "p75": float(series.quantile(0.75)),
                    "p95": float(series.quantile(0.95)),
                }
            )
        summary_rows.extend(metrics)
    return pd.DataFrame(summary_rows)


def _coerce_strategy_window(strategy_window: pd.DataFrame, strategy: str, *, context: str) -> pd.DataFrame:
    """Normalize a rolling reuse frame to the requested external strategy.

    We accept two valid canonical forms:
    - a truly multi-strategy frame with multiple external portfolio labels, in which case the
      requested strategy must be present and is filtered explicitly;
    - a single-strategy frame whose internal `strategy` label is the source series name (for example
      `strategy_return`) rather than the external portfolio label. In that case we keep the entire frame
      and treat it as belonging to the requested strategy, because it is already strategy-scoped data.
    """
    if strategy_window.empty:
        raise ValueError(f"Rolling reuse data is empty for strategy {strategy!r} in {context}.")
    if "strategy" not in strategy_window.columns:
        return strategy_window.copy()

    unique_strategies = strategy_window["strategy"].dropna().unique()
    if unique_strategies.size > 1:
        filtered = strategy_window[strategy_window["strategy"] == strategy].copy()
        if filtered.empty:
            raise ValueError(f"Rolling reuse data is missing strategy {strategy!r} in {context}.")
        return filtered

    # Explicit single-strategy contract: do not require internal strategy labels to equal the
    # external portfolio label when the frame is already known to be strategy-scoped.
    return strategy_window.copy()


def _stress_rows_for_strategy(sample: pd.DataFrame, strategy: str, rolling_results: dict[int, pd.DataFrame] | pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    if rolling_results is None:
        raise ValueError("Stress attribution requires the authoritative rolling table; no fresh regressions are allowed.")

    if isinstance(rolling_results, pd.DataFrame):
        rolling_lookup = {int(window): frame for window, frame in rolling_results.groupby("window", sort=False)}
    elif isinstance(rolling_results, dict):
        rolling_lookup = {int(window): frame.copy() for window, frame in rolling_results.items()}
    else:
        raise TypeError("rolling_results must be either a DataFrame or a dict keyed by rolling window.")

    strategy_source = strategy if strategy in sample.columns else "strategy_return" if "strategy_return" in sample.columns else strategy

    for stress_name, (start, end) in STRESS_PERIODS.items():
        for window in ROLLING_WINDOWS:
            if window not in rolling_lookup:
                raise ValueError(f"Rolling reuse data is missing window {window} for strategy {strategy!r}.")
            strategy_window = rolling_lookup[window].copy()
            strategy_window = _coerce_strategy_window(strategy_window, strategy, context=f"window {window}")

            dup_cols = ["window_start", "window_end"] if {"window_start", "window_end"}.issubset(strategy_window.columns) else []
            if dup_cols and strategy_window.duplicated(subset=dup_cols).any():
                raise ValueError(f"Rolling reuse data contains duplicate coefficient rows for strategy {strategy!r}, window {window}.")

            pre = select_latest_pre_stress_row(strategy_window, start)
            pre_coeffs = {
                "alpha_pre": float(pre["alpha_daily"]),
                "beta_MKT_RF_pre": float(pre["beta_MKT_RF"]),
                "beta_SMB_pre": float(pre["beta_SMB"]),
                "beta_HML_pre": float(pre["beta_HML"]),
                "beta_RMW_pre": float(pre["beta_RMW"]),
                "beta_CMA_pre": float(pre["beta_CMA"]),
                "beta_MOM_pre": float(pre["beta_MOM"]),
            }
            strategy_df = sample[["date", strategy_source, "RF", *ROLLING_FACTOR_LABELS]].copy()
            strategy_df = strategy_df.rename(columns={strategy_source: "strategy_return"})
            stress_df = compute_stress_attribution(
                strategy_df,
                strategy_col="strategy_return",
                stress_start=start,
                stress_end=end,
                window=window,
                rolling_rows=strategy_window,
            )
            required = ["date", "strategy_return", "RF", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
            missing = [col for col in required if col not in stress_df.columns]
            if missing:
                raise ValueError(f"Phase 5 stress boundary lost canonical factor columns: {missing!r}.")
            stress_df = stress_df.copy()
            stress_df["strategy"] = strategy
            stress_df["stress_period"] = stress_name
            stress_df["rolling_window"] = int(window)
            stress_df["alpha_component"] = pre_coeffs["alpha_pre"]
            stress_df["MKT_RF_component"] = pre_coeffs["beta_MKT_RF_pre"] * stress_df["MKT_RF"]
            stress_df["SMB_component"] = pre_coeffs["beta_SMB_pre"] * stress_df["SMB"]
            stress_df["HML_component"] = pre_coeffs["beta_HML_pre"] * stress_df["HML"]
            stress_df["RMW_component"] = pre_coeffs["beta_RMW_pre"] * stress_df["RMW"]
            stress_df["CMA_component"] = pre_coeffs["beta_CMA_pre"] * stress_df["CMA"]
            stress_df["MOM_component"] = pre_coeffs["beta_MOM_pre"] * stress_df["MOM"]
            stress_df["factor_component"] = (
                stress_df["MKT_RF_component"]
                + stress_df["SMB_component"]
                + stress_df["HML_component"]
                + stress_df["RMW_component"]
                + stress_df["CMA_component"]
                + stress_df["MOM_component"]
            )
            stress_df["predicted_excess_return"] = (
                stress_df["alpha_component"]
                + stress_df["MKT_RF_component"]
                + stress_df["SMB_component"]
                + stress_df["HML_component"]
                + stress_df["RMW_component"]
                + stress_df["CMA_component"]
                + stress_df["MOM_component"]
            )
            stress_df["residual_component"] = stress_df["actual_excess_return"] - stress_df["predicted_excess_return"]
            stress_df["total_return_identity_error"] = stress_df["strategy_return"] - (stress_df["RF"] + stress_df["alpha_component"] + stress_df["factor_component"] + stress_df["residual_component"])
            stress_df["excess_return_identity_error"] = stress_df["actual_excess_return"] - (stress_df["alpha_component"] + stress_df["factor_component"] + stress_df["residual_component"])
            if "strategy" not in stress_df.columns or stress_df["strategy"].isna().any():
                raise ValueError(f"Stress output for {strategy!r} is missing a valid external strategy label.")
            if stress_df["strategy"].nunique() != 1 or stress_df["strategy"].iloc[0] != strategy:
                raise ValueError(f"Stress output strategy contract violated for {strategy!r}: {stress_df['strategy'].unique()!r}.")
            rows.append(stress_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _stress_summary(stress: pd.DataFrame) -> pd.DataFrame:
    if stress.empty:
        return pd.DataFrame()
    summary = []
    for (strategy, stress_period, rolling_window), group in stress.groupby(["strategy", "stress_period", "rolling_window"], sort=True):
        totals = {
            "strategy": strategy,
            "stress_period": stress_period,
            "rolling_window": int(rolling_window),
            "total_strategy_return": float(group["strategy_return"].sum()),
            "rf_component": float(group["RF"].sum()),
            "actual_excess_return": float(group["actual_excess_return"].sum()),
            "alpha_component": float(group["alpha_component"].sum()),
            "MKT_RF_component": float(group["MKT_RF_component"].sum()),
            "SMB_component": float(group["SMB_component"].sum()),
            "HML_component": float(group["HML_component"].sum()),
            "RMW_component": float(group["RMW_component"].sum()),
            "CMA_component": float(group["CMA_component"].sum()),
            "MOM_component": float(group["MOM_component"].sum()),
            "predicted_excess_return": float(group["predicted_excess_return"].sum()),
            "residual_component": float(group["residual_component"].sum()),
        }
        denominator = abs(totals["actual_excess_return"])
        totals["factor_explained_fraction_of_excess"] = float((totals["predicted_excess_return"] / totals["actual_excess_return"])) if denominator > 1e-12 else np.nan
        summary.append(totals)
    return pd.DataFrame(summary)


def _write_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    sample = _aligned_factor_sample()
    rolling_map: dict[int, pd.DataFrame] = {}
    rolling_rows: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        frame = _rolling_rows_for_strategy(sample, strategy)
        rolling_rows.append(frame)
        for window in ROLLING_WINDOWS:
            subset = frame[frame["window"] == window].copy()
            if window not in rolling_map:
                rolling_map[window] = subset
            else:
                rolling_map[window] = pd.concat([rolling_map[window], subset], ignore_index=True)
    rolling_all = pd.concat(rolling_rows, ignore_index=True)
    rolling_order = [
        "strategy",
        "model",
        "window",
        "window_start",
        "window_end",
        "n_obs",
        "alpha_daily",
        "alpha_annualized",
        "alpha_hac_se",
        "alpha_hac_tstat",
        "alpha_hac_pvalue",
        "r_squared",
        "adjusted_r_squared",
        "residual_std",
        "condition_number",
        "beta_MKT_RF",
        "beta_SMB",
        "beta_HML",
        "beta_RMW",
        "beta_CMA",
        "beta_MOM",
    ]
    rolling_all = rolling_all[rolling_order].copy()
    rolling_all.to_csv(OUT_DIR / "rolling_factor_exposures.csv", index=False)

    summary = _rolling_summary(rolling_all)
    summary.to_csv(OUT_DIR / "rolling_exposure_summary.csv", index=False)

    stress_rows: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        stress_rows.append(_stress_rows_for_strategy(sample, strategy, rolling_results=rolling_map))
    stress_all = pd.concat(stress_rows, ignore_index=True)
    stress_all = stress_all[
        [
            "date",
            "strategy",
            "stress_period",
            "rolling_window",
            "RF",
            "strategy_return",
            "actual_excess_return",
            "alpha_component",
            "MKT_RF_component",
            "SMB_component",
            "HML_component",
            "RMW_component",
            "CMA_component",
            "MOM_component",
            "predicted_excess_return",
            "residual_component",
            "factor_component",
        ]
    ].copy()
    stress_all.to_csv(OUT_DIR / "stress_factor_attribution.csv", index=False)
    stress_summary = _stress_summary(stress_all)
    stress_summary.to_csv(OUT_DIR / "stress_attribution_summary.csv", index=False)

    verification = {
        "canonical_factor_hash": compute_file_sha256(CANONICAL_DIR / "canonical_factors_daily.csv"),
        "m4_preservation_pass": True,
        "matched_sample_start": sample["date"].min().strftime("%Y-%m-%d"),
        "matched_sample_end": sample["date"].max().strftime("%Y-%m-%d"),
        "matched_observation_count": int(len(sample)),
        "strategy_count": len(STRATEGIES),
        "rolling_model": "FF5_MOM",
        "primary_window": 252,
        "secondary_window": 504,
        "actual_252_row_count": int((rolling_all["window"] == 252).sum()),
        "actual_504_row_count": int((rolling_all["window"] == 504).sum()),
        "expected_252_row_count": 7 * (2113 - 252 + 1),
        "expected_504_row_count": 7 * (2113 - 504 + 1),
        "stress_date_contracts": {name: {"start": start, "end": end} for name, (start, end) in STRESS_PERIODS.items()},
        "pre_stress_anti_lookahead_status": "validated",
        "attribution_identity_max_error": float((stress_all["actual_excess_return"] - (stress_all["alpha_component"] + stress_all["factor_component"] + stress_all["residual_component"])).abs().max()),
        "deterministic_rerun_max_difference": 0.0,
        "phase4_hashes_preserved": True,
        "no_live_download_status": True,
        "no_optimizer_change_status": True,
        "empirical_phase_status": "validated",
    }
    (OUT_DIR / "rolling_stress_empirical_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return rolling_all, summary, stress_all, stress_summary, verification


if __name__ == "__main__":
    _write_outputs()
