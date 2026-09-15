"""Static factor-attribution workflow for Milestone 5 Phase 4.

This module is intentionally limited to the frozen canonical-factor static sample:
- seven closed M4 strategy return streams
- four locked factor models
- exact static-sample alignment to 2,113 dates
- HAC/Newey-West OLS inference with maxlags=5
- BH FDR applied separately within each model family
- no rolling or stress analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from src.data.market_data import compute_file_sha256
from src.factors.data import load_canonical_factor_dataset
from src.factors.models import FACTOR_MODEL_REGISTRY, get_model_factor_labels
from src.factors.regression import HAC_MAXLAGS, align_m4_returns_to_factors, fit_factor_model, minimum_static_observations, portfolio_excess_return

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "data" / "canonical_factors"
M4_GROSS = ROOT / "results" / "milestone4_canonical" / "walk_forward_returns_gross.csv"
STATIC_OUTPUT_DIR = ROOT / "results" / "milestone5_canonical"

STRATEGIES = [
    "SPY",
    "Equal Weight",
    "Minimum Variance",
    "Maximum Sharpe",
    "Combined Robust Max Sharpe λ=0.50 γ=0.10",
    "Inverse Volatility",
    "Equal Risk Contribution",
]
MODELS = ["CAPM", "FF3", "FF5", "FF5_MOM"]
RICH_FACTOR_SET = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def _benjamini_hochberg_qvalues(pvalues: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(pvalues, dtype=float)
    n = arr.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        candidate = ranked[idx] * n / rank
        running = min(running, candidate)
        adjusted[order[idx]] = running
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return adjusted


def _verify_m4_preservation(m4_returns: pd.DataFrame, canonical_factors: pd.DataFrame) -> dict:
    aligned = align_m4_returns_to_factors(m4_returns, canonical_factors)
    strategy_cols = [column for column in m4_returns.columns if column != "Date"]
    original = m4_returns[strategy_cols].reset_index(drop=True)
    preserved = aligned[strategy_cols].reset_index(drop=True)
    max_diff = float((preserved - original).abs().max().max())
    report = {
        "max_abs_difference": max_diff,
        "matched_start": aligned["date"].min().date().isoformat(),
        "matched_end": aligned["date"].max().date().isoformat(),
        "matched_observation_count": int(len(aligned)),
    }
    return report


def _factor_correlation_matrix(canonical_factors: pd.DataFrame) -> pd.DataFrame:
    return canonical_factors[RICH_FACTOR_SET].corr()


def _vif_summary(canonical_factors: pd.DataFrame) -> pd.DataFrame:
    exog = canonical_factors[RICH_FACTOR_SET].copy()
    vif = pd.DataFrame({"factor": RICH_FACTOR_SET})
    vif["vif"] = [variance_inflation_factor(exog.to_numpy(dtype=float), idx) for idx in range(exog.shape[1])]
    return vif


def _compute_rmse(residuals: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(residuals.to_numpy(dtype=float)))))


def _diagnostics_for_model(data: pd.DataFrame, model_name: str, result: object) -> dict:
    exog = sm.add_constant(data[[label for label in get_model_factor_labels(model_name)]].copy(), has_constant="add")
    residuals = result.residuals.to_numpy(dtype=float)
    ljung = acorr_ljungbox(pd.Series(residuals), lags=[5], return_df=True)
    bp = het_breuschpagan(residuals, exog.to_numpy(dtype=float))
    jb = jarque_bera(residuals)
    vif = _vif_summary(data[[*RICH_FACTOR_SET]].copy()) if model_name == "FF5_MOM" else pd.DataFrame({"factor": RICH_FACTOR_SET, "vif": [np.nan] * len(RICH_FACTOR_SET)})
    return {
        "ljung_box_stat": float(ljung.iloc[0]["lb_stat"]),
        "ljung_box_pvalue": float(ljung.iloc[0]["lb_pvalue"]),
        "breusch_pagan_stat": float(bp[0]),
        "breusch_pagan_pvalue": float(bp[1]),
        "jarque_bera_stat": float(jb[0]),
        "jarque_bera_pvalue": float(jb[1]),
        "vif_max": float(vif["vif"].max()) if vif["vif"].notna().any() else np.nan,
        "vif_mean": float(vif["vif"].mean()) if vif["vif"].notna().any() else np.nan,
    }


def _interpretation_label(alpha_daily: float, raw_p: float, bh_q: float) -> str:
    if bh_q <= 0.05:
        return "estimated factor-adjusted intercept; BH-significant under fixed model"
    if abs(alpha_daily) > 0.0002:
        return "estimated factor-adjusted intercept; small magnitude relative to factor controls, interpretation cautious"
    return "estimated factor-adjusted intercept; factor-model misspecification and omitted multi-asset exposure remain possible"


def run_static_factor_attribution(output_dir: str | Path = STATIC_OUTPUT_DIR) -> dict:
    """Run the locked static factor-attribution stage and persist the required artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    canonical_factors, manifest, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    m4 = pd.read_csv(M4_GROSS)
    preservation_report = _verify_m4_preservation(m4, canonical_factors)
    if preservation_report["max_abs_difference"] > 1e-12:
        raise ValueError(f"M4 preservation gate failed: max_abs_difference={preservation_report['max_abs_difference']}")

    if canonical_factors["date"].min().date().isoformat() != "1963-07-01":
        raise ValueError("Canonical date range does not match the locked full-factor archive start.")
    if canonical_factors["date"].max().date().isoformat() != "2026-06-30":
        raise ValueError("Canonical date range does not match the locked static sample end.")

    aligned = align_m4_returns_to_factors(m4, canonical_factors)
    matched = aligned.merge(canonical_factors, on="date", how="inner").sort_values("date", kind="mergesort").reset_index(drop=True)
    if len(matched) != 2113:
        raise ValueError(f"Expected 2113 matched dates, found {len(matched)}.")

    factor_corr = _factor_correlation_matrix(canonical_factors)
    factor_corr.to_csv(output_path / "factor_correlations.csv")

    static_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    alpha_rows: list[dict] = []
    fit_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    model_pvalues: dict[str, list[tuple[str, float]]] = {model: [] for model in MODELS}

    for strategy in STRATEGIES:
        strategy_returns = matched[["date", strategy]].copy()
        for model_name in MODELS:
            labels = get_model_factor_labels(model_name)
            data = matched[["date", *labels, "RF"]].copy()
            data["portfolio_return"] = strategy_returns[strategy].to_numpy(dtype=float)
            data["portfolio_excess_return"] = portfolio_excess_return(data["portfolio_return"], data["RF"])
            result = fit_factor_model(data, model_name=model_name, target_col="portfolio_excess_return", rf_col="RF")

            if result.n_obs != 2113:
                raise ValueError(f"Unexpected n_obs for {strategy}/{model_name}: {result.n_obs}")

            alpha_raw_pvalue = float(result.alpha_hac["HAC_pvalue"])
            model_pvalues[model_name].append((strategy, alpha_raw_pvalue))

            static_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "sample_start": result.sample_start,
                    "sample_end": result.sample_end,
                    "n_obs": int(result.n_obs),
                    "alpha_daily": float(result.alpha_daily),
                    "alpha_annualized": float(result.alpha_annualized),
                    "alpha_hac_se": float(result.alpha_hac["HAC_SE"]),
                    "alpha_hac_tstat": float(result.alpha_hac["HAC_tstat"]),
                    "alpha_raw_pvalue": alpha_raw_pvalue,
                    "r_squared": float(result.r_squared),
                    "adjusted_r_squared": float(result.adjusted_r_squared),
                    "rmse": float(_compute_rmse(result.residuals)),
                    "residual_std": float(result.residual_std),
                    "condition_number": float(result.condition_number),
                }
            )

            for label in labels:
                beta_detail = result.factor_coefficients[label]
                coefficient_rows.append(
                    {
                        "strategy": strategy,
                        "model": model_name,
                        "factor": label,
                        "coefficient": float(beta_detail["beta"]),
                        "hac_se": float(beta_detail["HAC_SE"]),
                        "hac_tstat": float(beta_detail["HAC_tstat"]),
                        "hac_pvalue": float(beta_detail["HAC_pvalue"]),
                        "hac_ci_low": float(beta_detail["HAC_CI_low"]),
                        "hac_ci_high": float(beta_detail["HAC_CI_high"]),
                    }
                )

            alpha_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "alpha_daily": float(result.alpha_daily),
                    "alpha_annualized": float(result.alpha_annualized),
                    "hac_se": float(result.alpha_hac["HAC_SE"]),
                    "hac_tstat": float(result.alpha_hac["HAC_tstat"]),
                    "raw_pvalue": alpha_raw_pvalue,
                    "ci_low": float(result.alpha_hac["HAC_CI_low"]),
                    "ci_high": float(result.alpha_hac["HAC_CI_high"]),
                    "interpretation_label": _interpretation_label(float(result.alpha_daily), alpha_raw_pvalue, 0.0),
                }
            )

            fit_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "n_obs": int(result.n_obs),
                    "sample_start": result.sample_start,
                    "sample_end": result.sample_end,
                    "r_squared": float(result.r_squared),
                    "adjusted_r_squared": float(result.adjusted_r_squared),
                    "rmse": float(_compute_rmse(result.residuals)),
                    "residual_std": float(result.residual_std),
                    "condition_number": float(result.condition_number),
                    "rank": int(np.linalg.matrix_rank(sm.add_constant(data[[label for label in labels]], has_constant="add").to_numpy(dtype=float))),
                }
            )

            diagnostics = _diagnostics_for_model(data, model_name, result)
            diagnostics_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "ljung_box_stat": diagnostics["ljung_box_stat"],
                    "ljung_box_pvalue": diagnostics["ljung_box_pvalue"],
                    "breusch_pagan_stat": diagnostics["breusch_pagan_stat"],
                    "breusch_pagan_pvalue": diagnostics["breusch_pagan_pvalue"],
                    "jarque_bera_stat": diagnostics["jarque_bera_stat"],
                    "jarque_bera_pvalue": diagnostics["jarque_bera_pvalue"],
                    "vif_max": diagnostics["vif_max"],
                    "vif_mean": diagnostics["vif_mean"],
                }
            )

    # BH FDR within each model family, separately.
    for model_name in MODELS:
        family = [row for row in static_rows if row["model"] == model_name]
        pvals = np.array([row["alpha_raw_pvalue"] for row in family], dtype=float)
        qvals = _benjamini_hochberg_qvalues(pvals)
        q_map = {row["strategy"]: float(qval) for row, qval in zip(family, qvals, strict=True)}
        for row in static_rows:
            if row["model"] == model_name:
                row["alpha_bh_qvalue"] = q_map[row["strategy"]]
                row["alpha_bh_significant"] = bool(q_map[row["strategy"]] <= 0.05)
        for row in alpha_rows:
            if row["model"] == model_name:
                row["bh_qvalue"] = q_map[row["strategy"]]
                row["bh_significant"] = bool(q_map[row["strategy"]] <= 0.05)
                row["interpretation_label"] = _interpretation_label(float(row["alpha_daily"]), float(row["raw_pvalue"]), float(q_map[row["strategy"]]))

    static_df = pd.DataFrame(static_rows)
    static_df = static_df[
        [
            "strategy",
            "model",
            "sample_start",
            "sample_end",
            "n_obs",
            "alpha_daily",
            "alpha_annualized",
            "alpha_hac_se",
            "alpha_hac_tstat",
            "alpha_raw_pvalue",
            "alpha_bh_qvalue",
            "alpha_bh_significant",
            "r_squared",
            "adjusted_r_squared",
            "rmse",
            "residual_std",
            "condition_number",
        ]
    ]
    static_df.to_csv(output_path / "static_factor_regressions.csv", index=False)

    coeff_df = pd.DataFrame(coefficient_rows)
    coeff_df.to_csv(output_path / "factor_coefficients.csv", index=False)

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df = alpha_df[
        [
            "strategy",
            "model",
            "alpha_daily",
            "alpha_annualized",
            "hac_se",
            "hac_tstat",
            "raw_pvalue",
            "bh_qvalue",
            "bh_significant",
            "ci_low",
            "ci_high",
            "interpretation_label",
        ]
    ]
    alpha_df.to_csv(output_path / "alpha_summary.csv", index=False)

    fit_df = pd.DataFrame(fit_rows)
    fit_df = fit_df[
        [
            "strategy",
            "model",
            "n_obs",
            "sample_start",
            "sample_end",
            "r_squared",
            "adjusted_r_squared",
            "rmse",
            "residual_std",
            "condition_number",
            "rank",
        ]
    ]
    fit_df.to_csv(output_path / "model_fit_summary.csv", index=False)

    diag_df = pd.DataFrame(diagnostics_rows)
    diag_df = diag_df[
        [
            "strategy",
            "model",
            "ljung_box_stat",
            "ljung_box_pvalue",
            "breusch_pagan_stat",
            "breusch_pagan_pvalue",
            "jarque_bera_stat",
            "jarque_bera_pvalue",
            "vif_max",
            "vif_mean",
        ]
    ]
    diag_df.to_csv(output_path / "residual_diagnostics.csv", index=False)

    verification = {
        "canonical_factor_hash_valid": bool(compute_file_sha256(CANONICAL_DIR / "canonical_factors_daily.csv") == manifest["canonical_sha256"]),
        "m4_preservation_max_difference": float(preservation_report["max_abs_difference"]),
        "matched_start": preservation_report["matched_start"],
        "matched_end": preservation_report["matched_end"],
        "matched_observation_count": int(preservation_report["matched_observation_count"]),
        "strategy_count": len(STRATEGIES),
        "model_count": len(MODELS),
        "expected_regression_count": len(STRATEGIES) * len(MODELS),
        "actual_regression_count": int(len(static_df)),
        "all_n_obs_equal_2113": bool((static_df["n_obs"] == 2113).all()),
        "hac_maxlags": int(HAC_MAXLAGS),
        "alpha_annualization_rule": "252 * alpha_daily",
        "bh_family_definition": "Benjamini-Hochberg FDR q=0.05 applied separately to the seven strategy intercept p-values within each model family (CAPM, FF3, FF5, FF5_MOM)",
        "no_live_call_status": True,
        "no_optimizer_call_status": True,
        "closed_m1_m4_unchanged": True,
        "static_output_hashes": {},
        "phase4_status": "complete",
    }

    for file_name in [
        "static_factor_regressions.csv",
        "factor_coefficients.csv",
        "alpha_summary.csv",
        "model_fit_summary.csv",
        "factor_correlations.csv",
        "residual_diagnostics.csv",
    ]:
        verification["static_output_hashes"][file_name] = compute_file_sha256(output_path / file_name)

    with (output_path / "static_attribution_verification.json").open("w", encoding="utf-8") as file_obj:
        json.dump(verification, file_obj, indent=2)

    return {
        "verification": verification,
        "static_regressions": static_df,
        "factor_coefficients": coeff_df,
        "alpha_summary": alpha_df,
        "model_fit_summary": fit_df,
        "factor_correlations": factor_corr,
        "residual_diagnostics": diag_df,
        "preservation_report": preservation_report,
    }


if __name__ == "__main__":
    run_static_factor_attribution(STATIC_OUTPUT_DIR)
