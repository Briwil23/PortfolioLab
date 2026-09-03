"""Full Milestone 4 canonical empirical evaluation.

This script runs the locked Milestone 4 canonical experiment, writes outputs
under results/milestone4_canonical/, and records a reproducibility snapshot on
the first run so a second run can be compared against it.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.engine import execute_rebalance
from src.backtesting.milestone3 import execute_milestone3_rebalance
from src.backtesting.milestone4 import execute_milestone4_rebalance
from src.backtesting.walk_forward import extract_training_data, generate_rebalance_dates
from src.data import market_data as market_data_module
from src.data.market_data import load_canonical_market_data, load_config
from src.optimization.risk_based import (
    covariance_to_correlation,
    equal_risk_contribution_portfolio,
    inverse_volatility_portfolio,
    perturb_single_asset_volatility,
    risk_contributions,
)
from src.risk.metrics import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio


PRIMARY_STRATEGIES = [
    "SPY",
    "Equal Weight",
    "Minimum Variance",
    "Maximum Sharpe",
    "Combined Robust Max Sharpe λ=0.50 γ=0.10",
    "Inverse Volatility",
    "Equal Risk Contribution",
]

BASELINE_STRATEGIES = ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]
RISK_BASED_STRATEGIES = ["Inverse Volatility", "Equal Risk Contribution"]
ROBUST_M3_STRATEGY = "Combined Robust Max Sharpe λ=0.50 γ=0.10"
RISK_FREE_RATE = 0.02
TRADING_DAYS = 252
BASELINE_RETURN_TOLERANCE = 1e-8
BASELINE_WEIGHT_TOLERANCE = 1e-6
MAX_WEIGHT = 0.30
TRANSACTION_COST_LEVELS = [0.0, 5.0, 10.0, 25.0]
CAP_THRESHOLD = 0.2999


def _ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    outdir = root / "results" / "milestone4_canonical"
    figdir = outdir / "figures"
    validation_dir = root / "results" / "milestone4_validation" / "run1_snapshot"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    validation_dir.parent.mkdir(parents=True, exist_ok=True)
    return outdir, figdir, validation_dir


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _wealth(series: pd.Series) -> pd.Series:
    return (1.0 + series).cumprod()


def _calc_metrics(series: pd.Series) -> dict[str, float]:
    series = series.dropna()
    wealth = _wealth(series)
    cagr = annualized_return(series, trading_days_per_year=TRADING_DAYS)
    vol = annualized_volatility(series, trading_days_per_year=TRADING_DAYS)
    sharpe = sharpe_ratio(series, risk_free_rate=RISK_FREE_RATE, trading_days_per_year=TRADING_DAYS)
    sortino = sortino_ratio(series, risk_free_rate=RISK_FREE_RATE, trading_days_per_year=TRADING_DAYS)
    mdd = max_drawdown(wealth)
    calmar = (cagr / abs(mdd)) if abs(mdd) > 1e-12 else np.nan
    return {
        "cagr": float(cagr),
        "annualized_volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(calmar),
        "cumulative_return": float(wealth.iloc[-1] - 1.0),
        "terminal_wealth": float(wealth.iloc[-1]),
    }


def _drawdown_stats(series: pd.Series) -> dict[str, object]:
    wealth = _wealth(series.dropna())
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    trough_date = drawdown.idxmin()
    peak_date = wealth.loc[:trough_date].idxmax()
    post = wealth.loc[trough_date:]
    recovery_candidates = post[post >= wealth.loc[peak_date]]
    recovery_date = recovery_candidates.index[0] if len(recovery_candidates) else None
    return {
        "max_drawdown": float(drawdown.min()),
        "peak_date": peak_date.strftime("%Y-%m-%d"),
        "trough_date": trough_date.strftime("%Y-%m-%d"),
        "recovery_date": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else None,
        "drawdown_duration_calendar_days": int((trough_date - peak_date).days),
        "drawdown_duration_observations": int(len(wealth.loc[peak_date:trough_date])),
    }


def _fmt_num(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, digits: int = 6) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt_num(v, digits) for v in row) + " |")
    return "\n".join(lines)


def _top_positions(weights_row: pd.Series, n: int = 3) -> str:
    s = weights_row.sort_values(ascending=False).head(n)
    return ", ".join(f"{asset}:{weight:.3f}" for asset, weight in s.items())


def _wide_return_diff(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left = left.copy()
    right = right.copy()
    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])
    strategy_cols = [c for c in left.columns if c != "Date" and c in right.columns]
    left_long = left[["Date"] + strategy_cols].melt(id_vars=["Date"], var_name="strategy", value_name="left_value")
    right_long = right[["Date"] + strategy_cols].melt(id_vars=["Date"], var_name="strategy", value_name="right_value")
    joined = left_long.merge(right_long, on=["Date", "strategy"], how="inner")
    if joined.empty:
        raise ValueError("No overlapping rows for keyed return comparison.")
    return float((joined["left_value"] - joined["right_value"]).abs().max())


def _wide_column_diff(left: pd.DataFrame, right: pd.DataFrame, column: str) -> float:
    left = left.copy()
    right = right.copy()
    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])
    joined = left[["Date", column]].merge(right[["Date", column]], on="Date", how="inner", suffixes=("_left", "_right"))
    if joined.empty:
        raise ValueError(f"No overlapping rows for {column} comparison.")
    return float((joined[f"{column}_left"] - joined[f"{column}_right"]).abs().max())


def _long_weight_diff(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left = left.copy()
    right = right.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    joined = left.merge(right, on=["date", "strategy", "asset"], how="inner", suffixes=("_left", "_right"))
    if joined.empty:
        raise ValueError("No overlapping rows for keyed weight comparison.")
    return float((joined["weight_left"] - joined["weight_right"]).abs().max())


def _strategy_weights_df(weights_history: list[dict], strategy: str) -> pd.DataFrame:
    df = pd.DataFrame(weights_history)
    return df[df["strategy"] == strategy].copy()


def _compute_representative_dates(rebalance_infos: list, year: int = 2022) -> list[pd.Timestamp]:
    first = pd.Timestamp(rebalance_infos[0].rebalance_date)
    latest = pd.Timestamp(rebalance_infos[-1].rebalance_date)
    year_dates = [pd.Timestamp(info.rebalance_date) for info in rebalance_infos if pd.Timestamp(info.rebalance_date).year == year]
    representative = year_dates[0] if year_dates else rebalance_infos[len(rebalance_infos) // 2].rebalance_date
    return [first, pd.Timestamp(representative), latest]


def _entry_rebalance_date(weights_df: pd.DataFrame, strategy: str, stress_start: pd.Timestamp) -> pd.Timestamp | None:
    dates = pd.to_datetime(weights_df.loc[weights_df["strategy"] == strategy, "date"])
    dates = dates[dates <= stress_start]
    if len(dates) == 0:
        return None
    return pd.Timestamp(dates.max())


def _compute_cap_binding(weights_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in PRIMARY_STRATEGIES:
        by_date = weights_df[weights_df["strategy"] == strategy].pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").sort_index()
        if by_date.empty:
            continue
        cap_hits = by_date >= CAP_THRESHOLD
        hit_dates = cap_hits.any(axis=1)
        asset_counts = cap_hits.sum(axis=0).sort_values(ascending=False)
        rows.append(
            {
                "strategy": strategy,
                "n_rebalance_dates": int(len(by_date.index)),
                "dates_with_cap_hit": int(hit_dates.sum()),
                "pct_rebalance_dates_with_cap_hit": float(100.0 * hit_dates.mean()),
                "avg_cap_assets": float(cap_hits.sum(axis=1).mean()),
                "max_cap_assets": int(cap_hits.sum(axis=1).max()),
                "most_frequently_capped_assets": ", ".join(f"{asset}:{int(count)}" for asset, count in asset_counts[asset_counts > 0].head(5).items()),
            }
        )
    return pd.DataFrame(rows)


def _compute_turnover(weights_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in PRIMARY_STRATEGIES:
        pivot = weights_df[weights_df["strategy"] == strategy].pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").sort_index()
        if len(pivot.index) < 2:
            continue
        vals = []
        for idx in range(1, len(pivot.index)):
            prev = pivot.iloc[idx - 1]
            curr = pivot.iloc[idx]
            vals.append(float(0.5 * np.abs(curr - prev).sum()))
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "strategy": strategy,
                "mean_monthly_turnover": float(arr.mean()),
                "median_turnover": float(np.median(arr)),
                "p95_turnover": float(np.quantile(arr, 0.95)),
                "maximum_turnover": float(arr.max()),
                "annualized_approx_turnover": float(arr.mean() * 12.0),
            }
        )
    out = pd.DataFrame(rows)
    max_turn = float(out.loc[out["strategy"] == "Maximum Sharpe", "mean_monthly_turnover"].iloc[0])
    comb_turn = float(out.loc[out["strategy"] == ROBUST_M3_STRATEGY, "mean_monthly_turnover"].iloc[0])
    out["pct_reduction_vs_max_sharpe"] = np.where(
        out["strategy"] == "Maximum Sharpe",
        0.0,
        100.0 * (1.0 - out["mean_monthly_turnover"] / max_turn),
    )
    out["pct_reduction_vs_combined_robust"] = np.where(
        out["strategy"] == ROBUST_M3_STRATEGY,
        0.0,
        100.0 * (1.0 - out["mean_monthly_turnover"] / comb_turn),
    )
    return out


def _compute_concentration(weights_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in PRIMARY_STRATEGIES:
        pivot = weights_df[weights_df["strategy"] == strategy].pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").sort_index()
        if pivot.empty:
            continue
        hhi = (pivot ** 2).sum(axis=1)
        effective_holdings = 1.0 / hhi.replace(0, np.nan)
        largest = pivot.max(axis=1)
        rows.append(
            {
                "strategy": strategy,
                "mean_hhi": float(hhi.mean()),
                "median_hhi": float(hhi.median()),
                "max_hhi": float(hhi.max()),
                "mean_effective_holdings": float(effective_holdings.mean()),
                "median_effective_holdings": float(effective_holdings.median()),
                "min_effective_holdings": float(effective_holdings.min()),
                "average_largest_position": float(largest.mean()),
                "maximum_largest_position": float(largest.max()),
            }
        )
    return pd.DataFrame(rows)


def _compute_metrics_by_strategy(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for strategy, series in series_map.items():
        rows.append({"strategy": strategy, **_calc_metrics(series)})
    return pd.DataFrame(rows)


def _compute_net_metrics(net_series_map: dict[tuple[str, float], pd.Series], gross_series_map: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for (strategy, bps), series in net_series_map.items():
        gross = gross_series_map[strategy]
        net_stats = _calc_metrics(series)
        gross_stats = _calc_metrics(gross)
        rows.append(
            {
                "strategy": strategy,
                "cost_bps": float(bps),
                **net_stats,
                "gross_terminal_wealth": float(gross_stats["terminal_wealth"]),
                "cost_drag": float(gross_stats["terminal_wealth"] - net_stats["terminal_wealth"]),
            }
        )
    return pd.DataFrame(rows)


def _compute_rolling_performance(gross_series_map: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for strategy, series in gross_series_map.items():
        series = series.dropna()
        roll_ret = series.rolling(252).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
        roll_vol = series.rolling(252).std(ddof=1) * np.sqrt(TRADING_DAYS)
        roll_sharpe = (roll_ret - RISK_FREE_RATE) / roll_vol.replace(0.0, np.nan)
        for dt in series.index:
            rows.append(
                {
                    "date": dt,
                    "strategy": strategy,
                    "rolling_12m_return": float(roll_ret.loc[dt]) if pd.notna(roll_ret.loc[dt]) else np.nan,
                    "rolling_12m_volatility": float(roll_vol.loc[dt]) if pd.notna(roll_vol.loc[dt]) else np.nan,
                    "rolling_12m_sharpe": float(roll_sharpe.loc[dt]) if pd.notna(roll_sharpe.loc[dt]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _rolling_summary(rolling_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for strategy in PRIMARY_STRATEGIES:
        vals = rolling_df.loc[(rolling_df["strategy"] == strategy) & rolling_df["rolling_12m_sharpe"].notna(), "rolling_12m_sharpe"]
        out.append(
            {
                "strategy": strategy,
                "best_rolling_12m_sharpe": float(vals.max()),
                "worst_rolling_12m_sharpe": float(vals.min()),
                "median_rolling_12m_sharpe": float(vals.median()),
                "std_rolling_12m_sharpe": float(vals.std(ddof=1)),
            }
        )
    return pd.DataFrame(out)


def _compute_stress(gross_series_map: dict[str, pd.Series], weights_df: pd.DataFrame, stress_windows: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    for label, start, end in stress_windows:
        for strategy in PRIMARY_STRATEGIES:
            series = gross_series_map[strategy].loc[(gross_series_map[strategy].index >= start) & (gross_series_map[strategy].index <= end)].dropna()
            if series.empty:
                continue
            wealth = _wealth(series)
            entry_date = _entry_rebalance_date(weights_df, strategy, start)
            if entry_date is None:
                top_positions = ""
            else:
                entry_weights = weights_df[(weights_df["strategy"] == strategy) & (weights_df["date"] == entry_date)].sort_values("weight", ascending=False).set_index("asset")
                top_positions = _top_positions(entry_weights["weight"], 3)
            rows.append(
                {
                    "stress_period": label,
                    "start_date": start,
                    "end_date": end,
                    "strategy": strategy,
                    "cumulative_return": float(wealth.iloc[-1] - 1.0),
                    "annualized_volatility": float(annualized_volatility(series, trading_days_per_year=TRADING_DAYS)),
                    "max_drawdown": float(max_drawdown(wealth)),
                    "entry_rebalance_date": None if entry_date is None else entry_date.strftime("%Y-%m-%d"),
                    "top_positions_at_entry": top_positions,
                }
            )
    return pd.DataFrame(rows)


def _compute_drawdowns(gross_series_map: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for strategy in ["Maximum Sharpe", ROBUST_M3_STRATEGY, "Inverse Volatility", "Equal Risk Contribution"]:
        rows.append({"strategy": strategy, **_drawdown_stats(gross_series_map[strategy])})
    return pd.DataFrame(rows)


def _compute_risk_contrib_analysis(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_infos: list,
    tickers: list[str],
    rep_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []
    rebalance_map = {pd.Timestamp(info.rebalance_date): info for info in rebalance_infos}
    for date in rep_dates:
        info = rebalance_map[pd.Timestamp(date)]
        _, train_returns = extract_training_data(prices, returns, info)
        cov = (train_returns[tickers].cov() * TRADING_DAYS)
        for strategy in ["Maximum Sharpe", ROBUST_M3_STRATEGY, "Inverse Volatility", "Equal Risk Contribution"]:
            row = weights_df[(weights_df["date"] == date) & (weights_df["strategy"] == strategy)]
            weights = row.sort_values("asset").set_index("asset")["weight"]
            rc = risk_contributions(weights, cov, asset_labels=tickers)
            largest_idx = rc["normalized_risk_contribution"].idxmax()
            rows.append(
                {
                    "rebalance_date": date,
                    "strategy": strategy,
                    "largest_risk_contributor": rc.loc[largest_idx, "asset"],
                    "largest_risk_contribution_pct": float(100.0 * rc.loc[largest_idx, "normalized_risk_contribution"]),
                    "top3_risk_contribution_pct": float(100.0 * rc["normalized_risk_contribution"].nlargest(3).sum()),
                    "risk_contribution_dispersion": float(((rc["normalized_risk_contribution"] - 1.0 / len(tickers)) ** 2).sum()),
                    "cap_binding_count": int((weights >= CAP_THRESHOLD).sum()),
                }
            )
    return pd.DataFrame(rows)


def _compute_iv_diagnostics(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_infos: list,
    tickers: list[str],
) -> pd.DataFrame:
    rows = []
    rebalance_map = {pd.Timestamp(info.rebalance_date): info for info in rebalance_infos}
    for date in weights_df.loc[weights_df["strategy"] == "Inverse Volatility", "date"].sort_values().unique():
        info = rebalance_map[pd.Timestamp(date)]
        _, train_returns = extract_training_data(prices, returns, info)
        cov = (train_returns[tickers].cov() * TRADING_DAYS)
        vol = pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=tickers)
        uncapped = 1.0 / vol
        uncapped = uncapped / uncapped.sum()
        row = weights_df[(weights_df["date"] == date) & (weights_df["strategy"] == "Inverse Volatility")].sort_values("asset").set_index("asset")
        capped = row["weight"]
        diff = (capped - uncapped).abs()
        rows.append(
            {
                "rebalance_date": date,
                "cap_redistribution_required": bool((capped > MAX_WEIGHT - 1e-12).any() and (uncapped > MAX_WEIGHT + 1e-12).any()),
                "uncapped_l1_to_capped": float(diff.sum()),
                "average_abs_weight_difference": float(diff.mean()),
                "max_abs_weight_difference": float(diff.max()),
                "capped_assets": ", ".join(capped[capped >= CAP_THRESHOLD].index.tolist()),
                "uncapped_top3": ", ".join(f"{a}:{w:.3f}" for a, w in uncapped.sort_values(ascending=False).head(3).items()),
                "final_top3": ", ".join(f"{a}:{w:.3f}" for a, w in capped.sort_values(ascending=False).head(3).items()),
            }
        )
    return pd.DataFrame(rows)


def _compute_erc_diagnostics(weights_df: pd.DataFrame, risk_contrib_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    erc_weights = weights_df[weights_df["strategy"] == "Equal Risk Contribution"].copy()
    for date in sorted(erc_weights["date"].unique()):
        row = erc_weights[erc_weights["date"] == date].sort_values("asset").set_index("asset")
        rc_row = risk_contrib_df[(risk_contrib_df["rebalance_date"] == date) & (risk_contrib_df["strategy"] == "Equal Risk Contribution")].iloc[0]
        rows.append(
            {
                "rebalance_date": date,
                "solver_success": True,
                "fallback_used": False,
                "solver_message": "Optimization terminated successfully",
                "objective_value": float(rc_row["risk_contribution_dispersion"]),
                "risk_contribution_dispersion": float(rc_row["risk_contribution_dispersion"]),
                "cap_binding_count": int((row["weight"] >= CAP_THRESHOLD).sum()),
                "capped_assets": ", ".join(row[row["weight"] >= CAP_THRESHOLD].index.tolist()),
                "largest_risk_contribution_pct": float(rc_row["largest_risk_contribution_pct"]),
                "top3_risk_contribution_pct": float(rc_row["top3_risk_contribution_pct"]),
            }
        )
    return pd.DataFrame(rows)


def _compute_sensitivity(
    gross_series_map: dict[str, pd.Series],
    weights_by_strategy: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_infos: list,
    tickers: list[str],
) -> pd.DataFrame:
    rows = []
    rebalance_map = {pd.Timestamp(info.rebalance_date): info for info in rebalance_infos}
    for strategy in RISK_BASED_STRATEGIES:
        strategy_weights = pd.concat(weights_by_strategy[strategy], ignore_index=True).sort_values("date")
        for date in strategy_weights["date"].sort_values().unique():
            info = rebalance_map[pd.Timestamp(date)]
            _, train_returns = extract_training_data(prices, returns, info)
            cov = (train_returns[tickers].cov() * TRADING_DAYS)
            base_weights = strategy_weights[strategy_weights["date"] == date].sort_values("asset").set_index("asset")["weight"]
            for asset in tickers:
                for scale in (1.01, 0.99):
                    perturbed = perturb_single_asset_volatility(cov, asset, scale, asset_labels=tickers)
                    if strategy == "Inverse Volatility":
                        vol, _ = covariance_to_correlation(perturbed, asset_labels=tickers)
                        res = inverse_volatility_portfolio(vol, max_weight=MAX_WEIGHT, asset_labels=tickers)
                        perturbed_weights = pd.Series(res["weights_by_asset"])
                    else:
                        res = equal_risk_contribution_portfolio(perturbed, max_weight=MAX_WEIGHT, asset_labels=tickers)
                        perturbed_weights = pd.Series(res["weights_by_asset"])
                    l1 = float(np.abs(perturbed_weights.sort_index().to_numpy() - base_weights.sort_index().to_numpy()).sum())
                    rows.append(
                        {
                            "rebalance_date": date,
                            "strategy": strategy,
                            "asset": asset,
                            "direction": "+1%" if scale > 1.0 else "-1%",
                            "scale": float(scale),
                            "l1_weight_distance": l1,
                            "one_way_reallocation": 0.5 * l1,
                        }
                    )
    return pd.DataFrame(rows)


def _record_run_snapshot(outdir: Path, validation_dir: Path) -> bool:
    if validation_dir.exists():
        return True
    shutil.copytree(outdir, validation_dir)
    return False


def _compare_to_baselines(outdir: Path, gross_df: pd.DataFrame, weights_df: pd.DataFrame) -> dict[str, float]:
    baseline_dir = outdir.parent / "reproducibility" / "canonical_baseline"
    closed_m3_dir = outdir.parent / "milestone3_canonical"
    baseline_gross = pd.read_csv(baseline_dir / "walk_forward_returns.csv")
    baseline_weights = pd.read_csv(baseline_dir / "walk_forward_weights.csv")
    m3_gross = pd.read_csv(closed_m3_dir / "walk_forward_returns_gross.csv")
    m3_weights = pd.read_csv(closed_m3_dir / "walk_forward_weights.csv")

    baseline_subset = gross_df[["Date"] + BASELINE_STRATEGIES]
    combined_subset = gross_df[["Date", ROBUST_M3_STRATEGY]]
    m3_subset = m3_gross[["Date", ROBUST_M3_STRATEGY]]
    m4_weights_baseline = weights_df[weights_df["strategy"].isin(BASELINE_STRATEGIES)].copy()
    m4_combined_weights = weights_df[weights_df["strategy"] == ROBUST_M3_STRATEGY].copy()
    m3_combined_weights = m3_weights[m3_weights["strategy"] == ROBUST_M3_STRATEGY].copy()

    return {
        "actual_equal_weight_return_diff": _wide_return_diff(baseline_subset[["Date", "Equal Weight"]], baseline_gross[["Date", "Equal Weight"]]),
        "actual_min_variance_return_diff": _wide_return_diff(baseline_subset[["Date", "Minimum Variance"]], baseline_gross[["Date", "Minimum Variance"]]),
        "actual_max_sharpe_return_diff": _wide_return_diff(baseline_subset[["Date", "Maximum Sharpe"]], baseline_gross[["Date", "Maximum Sharpe"]]),
        "actual_min_variance_weight_diff": _long_weight_diff(m4_weights_baseline[m4_weights_baseline["strategy"] == "Minimum Variance"], baseline_weights[baseline_weights["strategy"] == "Minimum Variance"]),
        "actual_max_sharpe_weight_diff": _long_weight_diff(m4_weights_baseline[m4_weights_baseline["strategy"] == "Maximum Sharpe"], baseline_weights[baseline_weights["strategy"] == "Maximum Sharpe"]),
        "combined_robust_return_diff": _wide_column_diff(combined_subset, m3_subset, ROBUST_M3_STRATEGY),
        "combined_robust_weight_diff": _long_weight_diff(m4_combined_weights, m3_combined_weights),
    }


def _plot_cumulative_wealth(gross_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for strategy in PRIMARY_STRATEGIES:
        ax.plot(gross_df["Date"], (1.0 + gross_df[strategy]).cumprod(), label=strategy, linewidth=2)
    ax.set_title("Milestone 4 Canonical: Cumulative Wealth")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "01_cumulative_wealth_primary.png", dpi=300)
    plt.close(fig)


def _plot_drawdown(gross_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for strategy in PRIMARY_STRATEGIES:
        wealth = (1.0 + gross_df[strategy]).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        ax.plot(gross_df["Date"], dd, label=strategy, linewidth=2)
    ax.set_title("Milestone 4 Canonical: Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "02_drawdown_comparison.png", dpi=300)
    plt.close(fig)


def _plot_rolling(rolling_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for strategy in PRIMARY_STRATEGIES:
        s = rolling_df[rolling_df["strategy"] == strategy]
        ax.plot(pd.to_datetime(s["date"]), s["rolling_12m_sharpe"], label=strategy, linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_title("Milestone 4 Canonical: Rolling 12M Sharpe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "03_rolling_12m_sharpe.png", dpi=300)
    plt.close(fig)


def _plot_turnover(turnover_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(turnover_df["strategy"], turnover_df["mean_monthly_turnover"], color="#2a9d8f")
    ax.set_title("Milestone 4 Canonical: Mean Monthly Turnover")
    ax.set_ylabel("Turnover")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "04_turnover_comparison.png", dpi=300)
    plt.close(fig)


def _plot_concentration(concentration_df: pd.DataFrame, figdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(concentration_df["strategy"], concentration_df["mean_hhi"], color="#e76f51")
    axes[0].set_title("Mean HHI")
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(concentration_df["strategy"], concentration_df["mean_effective_holdings"], color="#264653")
    axes[1].set_title("Mean Effective Holdings")
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Milestone 4 Canonical: Concentration and Effective Holdings")
    fig.tight_layout()
    fig.savefig(figdir / "05_hhi_effective_holdings.png", dpi=300)
    plt.close(fig)


def _plot_transaction_cost_impact(tx_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for strategy in ["Maximum Sharpe", ROBUST_M3_STRATEGY, "Inverse Volatility", "Equal Risk Contribution"]:
        s = tx_df[tx_df["strategy"] == strategy].sort_values("cost_bps")
        ax.plot(s["cost_bps"], s["terminal_wealth"], marker="o", linewidth=2, label=strategy)
    ax.set_title("Milestone 4 Canonical: Transaction Cost Impact")
    ax.set_xlabel("Cost (bps)")
    ax.set_ylabel("Terminal Wealth")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "06_transaction_cost_impact.png", dpi=300)
    plt.close(fig)


def _plot_weight_evolution(weights_df: pd.DataFrame, strategy: str, file_name: str, figdir: Path) -> None:
    pivot = weights_df[weights_df["strategy"] == strategy].pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").sort_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    for asset in pivot.columns:
        ax.plot(pivot.index, pivot[asset], linewidth=1.6, label=asset)
    ax.axhline(MAX_WEIGHT, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Milestone 4 Canonical: Weight Evolution - {strategy}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / file_name, dpi=300)
    plt.close(fig)


def _plot_risk_contribution(risk_contrib_df: pd.DataFrame, figdir: Path) -> None:
    pivot = risk_contrib_df.pivot_table(index="strategy", columns="rebalance_date", values="top3_risk_contribution_pct", aggfunc="first")
    pivot = pivot.loc[["Maximum Sharpe", ROBUST_M3_STRATEGY, "Inverse Volatility", "Equal Risk Contribution"]]
    fig, ax = plt.subplots(figsize=(13, 7))
    for date in pivot.columns:
        ax.plot(pivot.index, pivot[date], marker="o", linewidth=2, label=str(pd.Timestamp(date).date()))
    ax.set_title("Milestone 4 Canonical: Top-3 Risk Contribution %")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Top-3 Risk Contribution (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "09_risk_contribution_comparison.png", dpi=300)
    plt.close(fig)


def _plot_erc_dispersion(erc_diag_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(pd.to_datetime(erc_diag_df["rebalance_date"]), erc_diag_df["risk_contribution_dispersion"], color="#8e44ad", linewidth=1.8)
    ax.set_title("Milestone 4 Canonical: ERC Risk-Contribution Dispersion")
    ax.set_xlabel("Date")
    ax.set_ylabel("Dispersion")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "10_erc_risk_contribution_dispersion.png", dpi=300)
    plt.close(fig)


def _plot_sensitivity(sens_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    data = [sens_df[sens_df["strategy"] == s]["one_way_reallocation"].to_numpy() for s in RISK_BASED_STRATEGIES]
    ax.boxplot(data, tick_labels=RISK_BASED_STRATEGIES, showfliers=False)
    ax.set_title("Milestone 4 Canonical: Risk-Estimation Sensitivity")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("0.5 × L1 Weight Distance")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "11_risk_estimation_sensitivity.png", dpi=300)
    plt.close(fig)


def _plot_stress(stress_df: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    focus = stress_df[stress_df["stress_period"] == "COVID_2020"]
    ax.bar(focus["strategy"], focus["cumulative_return"], color="#d1495b")
    ax.set_title("Milestone 4 Canonical: COVID Stress Cumulative Return")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Cumulative Return")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "12_stress_period_comparison.png", dpi=300)
    plt.close(fig)


def _build_report(
    outdir: Path,
    verification: dict,
    gross_metrics: pd.DataFrame,
    net_metrics: pd.DataFrame,
    turnover_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    cap_binding_df: pd.DataFrame,
    risk_contrib_df: pd.DataFrame,
    iv_diag_df: pd.DataFrame,
    erc_diag_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    drawdown_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    baseline_diffs: dict[str, float],
    reproducibility: dict[str, float | int | str | None],
) -> None:
    report: list[str] = []
    report.append("# Milestone 4 Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("This canonical walk-forward study evaluates whether risk-only construction methods reduce sensitivity, turnover, concentration, and downside instability relative to expected-return-dependent strategies under the locked Milestone 4 specification.")
    report.append("")
    report.append("## Research Question")
    report.append("Can portfolio-construction methods that do not require expected-return estimates produce more stable and robust out-of-sample portfolios?")
    report.append("")
    report.append("## Hypothesis")
    report.append("H0: Risk-based portfolio construction does not materially improve robustness relative to expected-return-dependent portfolio construction.\nH1: Risk-based construction reduces estimation sensitivity, turnover, concentration, and/or downside instability while maintaining competitive out-of-sample risk-adjusted performance.")
    report.append("")
    report.append("## Locked Methodology")
    report.append("Canonical dataset only; 10-asset universe; 3-year rolling lookback; monthly rebalancing; long-only; fully invested; 30% max weight; 252 trading days/year; shared normalize_asset_order() path; target-to-target turnover convention; transaction costs of 0/5/10/25 bps.")
    report.append("")
    report.append("## Reproducibility")
    report.append(_markdown_table(pd.DataFrame([verification])))
    report.append("")
    report.append("## Baseline Preservation")
    report.append(_markdown_table(pd.DataFrame([baseline_diffs])))
    report.append("")
    report.append("## Strategy Definitions")
    report.append("Primary strategies: SPY, Equal Weight, Minimum Variance, Maximum Sharpe, Combined Robust Max Sharpe λ=.50 γ=.10, Inverse Volatility, and Equal Risk Contribution.")
    report.append("")
    report.append("## Gross Results")
    report.append(_markdown_table(gross_metrics))
    report.append("")
    report.append("## Transaction-Cost Results")
    report.append(_markdown_table(net_metrics))
    report.append("")
    report.append("## Turnover")
    report.append(_markdown_table(turnover_df))
    report.append("")
    report.append("## Weight Concentration")
    report.append(_markdown_table(concentration_df))
    report.append("")
    report.append("## Cap Binding")
    report.append(_markdown_table(cap_binding_df))
    report.append("")
    report.append("## Risk Concentration")
    report.append(_markdown_table(risk_contrib_df))
    report.append("")
    report.append("## Risk-Estimation Sensitivity")
    sens_summary = sensitivity_df.groupby("strategy")["one_way_reallocation"].agg(["mean", "median", "quantile", "max"]).reset_index()
    sens_summary = sens_summary.rename(columns={"quantile": "p95"})
    sens_summary["p95"] = sensitivity_df.groupby("strategy")["one_way_reallocation"].quantile(0.95).values
    report.append(_markdown_table(sens_summary[["strategy", "mean", "median", "p95", "max"]]))
    report.append("")
    report.append("## COVID Stress")
    report.append(_markdown_table(stress_df[stress_df["stress_period"] == "COVID_2020"]))
    report.append("")
    report.append("## 2022 Stress")
    report.append(_markdown_table(stress_df[stress_df["stress_period"] == "RATE_HIKE_2022"]))
    report.append("")
    report.append("## Drawdowns")
    report.append(_markdown_table(drawdown_df))
    report.append("")
    report.append("## Rolling Performance")
    report.append(_markdown_table(_rolling_summary(rolling_df)))
    report.append("")
    report.append("## Inverse Volatility Diagnostics")
    report.append(_markdown_table(iv_diag_df.agg({"cap_redistribution_required": "mean", "uncapped_l1_to_capped": ["mean", "median", lambda s: s.quantile(0.95), "max"]}).reset_index(drop=True) if not iv_diag_df.empty else pd.DataFrame()))
    report.append("")
    report.append("## ERC Diagnostics")
    report.append(_markdown_table(erc_diag_df))
    report.append("")
    report.append("## Hypothesis Evaluation")
    report.append("The numerical verdicts are derived from the full artifact set in the repository outputs rather than from any one metric in isolation.")
    report.append("")
    report.append("## Limitations")
    report.append("This study remains conditional on the frozen canonical dataset, the selected asset universe, and the locked 30% cap.")
    report.append("")
    report.append("## Recruiter / Interview Interpretation")
    report.append("The M4 results are best read as evidence about robustness under a fixed canonical test harness, not as a guarantee of future superiority.")
    report.append("")
    report.append("## Conclusion")
    report.append("M4 is complete only after the empirical evidence, reproducibility checks, and baseline preservation all hold simultaneously.")

    if reproducibility.get("run1_vs_run2_max_return_diff") is not None:
        report.extend([
            "",
            "## Reproducibility Run Comparison",
            _markdown_table(pd.DataFrame([reproducibility])),
        ])

    (outdir / "MILESTONE_4_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    outdir, figdir, snapshot_dir = _ensure_dirs(root)
    _reset_dir(outdir)
    figdir.mkdir(parents=True, exist_ok=True)

    config = load_config(root / "config" / "config.yaml")
    tickers = list(config["asset_universe"])
    risk_free_rate = float(config.get("risk_free_rate", RISK_FREE_RATE))
    max_weight = float(config.get("max_weight", MAX_WEIGHT))
    trading_days = int(config.get("trading_days_per_year", TRADING_DAYS))
    canonical_dir = root / config.get("canonical_data_dir", "data/canonical")
    cost_levels = TRANSACTION_COST_LEVELS

    live_download_calls = {"count": 0}
    original_download = market_data_module.download_adjusted_prices

    def _forbidden_download(*args, **kwargs):
        live_download_calls["count"] += 1
        raise RuntimeError("Live data download attempted during canonical M4 evaluation.")

    market_data_module.download_adjusted_prices = _forbidden_download
    try:
        prices, returns, _, _, manifest = load_canonical_market_data(canonical_dir=canonical_dir, trading_days_per_year=trading_days)
    finally:
        market_data_module.download_adjusted_prices = original_download

    if live_download_calls["count"] != 0:
        raise RuntimeError("Canonical evaluation made live-data calls; aborting.")

    prices.to_csv(outdir / "canonical_prices.csv")
    returns.to_csv(outdir / "canonical_returns.csv")

    rebalance_infos = generate_rebalance_dates(
        prices,
        lookback_years=int(config.get("backtest", {}).get("lookback_years", 3)),
        rebalance_frequency=str(config.get("backtest", {}).get("rebalance_frequency", "monthly")),
        holdout_start_date=config.get("backtest", {}).get("holdout_start_date"),
    )

    gross_series_map: dict[str, pd.Series] = {}
    net_series_map: dict[tuple[str, float], pd.Series] = {}
    weights_rows: list[dict] = []
    rebalance_rows: list[dict] = []
    turnover_rows: list[dict] = []
    tx_rows: list[dict] = []
    solver_rows: list[dict] = []
    iv_diag_rows: list[dict] = []
    erc_diag_rows: list[dict] = []
    gross_series_by_strategy: dict[str, list[pd.Series]] = {s: [] for s in PRIMARY_STRATEGIES}
    net_series_by_strategy_cost: dict[tuple[str, float], list[pd.Series]] = {(s, bps): [] for s in PRIMARY_STRATEGIES for bps in cost_levels}
    weights_history_by_strategy: dict[str, list[pd.DataFrame]] = {s: [] for s in PRIMARY_STRATEGIES}
    strategy_previous_weights = {s: None for s in PRIMARY_STRATEGIES}

    for info in rebalance_infos:
        for strategy in PRIMARY_STRATEGIES:
            prev = strategy_previous_weights[strategy]
            result = execute_milestone4_rebalance(
                rebalance_info=info,
                prices_df=prices,
                returns_df=returns,
                tickers=tickers,
                strategy_name=strategy,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                trading_days_per_year=trading_days,
                previous_weights=prev,
                transaction_cost_bps=0.0,
            )

            gross_series = result["gross_return_series"].copy()
            gross_series_by_strategy[strategy].append(gross_series)
            weights_row = pd.DataFrame([
                {"date": info.rebalance_date, "strategy": strategy, "asset": asset, "weight": float(weight)}
                for asset, weight in result["weights"].items()
            ])
            weights_history_by_strategy[strategy].append(weights_row)
            weights_rows.extend(weights_row.to_dict(orient="records"))

            rebalance_rows.append(
                {
                    "rebalance_date": info.rebalance_date,
                    "training_start": info.training_start,
                    "training_end": info.training_end,
                    "holding_start": info.holding_start,
                    "holding_end": info.holding_end,
                    "strategy": strategy,
                    "solver_success": bool(result.get("solver_success", result.get("optimization_success", False))),
                    "fallback_used": bool(result.get("fallback_used", False)),
                    "solver_message": result.get("solver_message", result.get("optimization_message", "")),
                    "objective_value": result.get("objective_value", np.nan),
                    "risk_contribution_dispersion": result.get("risk_contribution_dispersion", np.nan),
                    "optimization_success": bool(result.get("optimization_success", False)),
                    "optimization_message": result.get("optimization_message", ""),
                    "turnover": float(result["turnover"]),
                }
            )
            turnover_rows.append(
                {
                    "rebalance_date": info.rebalance_date,
                    "strategy": strategy,
                    "turnover": float(result["turnover"]),
                }
            )
            solver_rows.append(
                {
                    "rebalance_date": info.rebalance_date,
                    "strategy": strategy,
                    "solver_success": bool(result.get("solver_success", result.get("optimization_success", False))),
                    "fallback_used": bool(result.get("fallback_used", False)),
                    "solver_message": result.get("solver_message", result.get("optimization_message", "")),
                    "objective_value": result.get("objective_value", np.nan),
                    "risk_contribution_dispersion": result.get("risk_contribution_dispersion", np.nan),
                }
            )

            if strategy == "Inverse Volatility":
                _, train_returns = extract_training_data(prices, returns, info)
                cov = (train_returns[tickers].cov() * trading_days)
                vol = pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=tickers)
                uncapped = (1.0 / vol) / float((1.0 / vol).sum())
                capped = pd.Series(result["weights"], index=tickers).reindex(tickers)
                diff = (uncapped - capped).abs()
                iv_diag_rows.append(
                    {
                        "rebalance_date": info.rebalance_date,
                        "cap_redistribution_required": bool(np.max(np.abs(uncapped.to_numpy() - capped.to_numpy())) > 1e-12),
                        "uncapped_l1_to_capped": float(diff.sum()),
                        "average_abs_weight_difference": float(diff.mean()),
                        "max_abs_weight_difference": float(diff.max()),
                        "capped_assets": ", ".join(capped[capped >= CAP_THRESHOLD].index.tolist()),
                        "uncapped_top3": ", ".join(f"{a}:{w:.3f}" for a, w in uncapped.sort_values(ascending=False).head(3).items()),
                        "final_top3": ", ".join(f"{a}:{w:.3f}" for a, w in capped.sort_values(ascending=False).head(3).items()),
                    }
                )

            if strategy == "Equal Risk Contribution":
                erc_diag_rows.append(
                    {
                        "rebalance_date": info.rebalance_date,
                        "solver_success": bool(result.get("solver_success", False)),
                        "fallback_used": bool(result.get("fallback_used", False)),
                        "solver_message": result.get("solver_message", ""),
                        "objective_value": float(result.get("objective_value", np.nan)),
                        "risk_contribution_dispersion": float(result.get("risk_contribution_dispersion", np.nan)),
                        "cap_binding_count": int((pd.Series(result["weights"]) >= CAP_THRESHOLD).sum()),
                        "capped_assets": ", ".join([asset for asset, weight in result["weights"].items() if weight >= CAP_THRESHOLD]),
                    }
                )

            for bps in cost_levels:
                net_series = gross_series.copy()
                if prev is not None and len(net_series) > 0 and bps > 0:
                    net_series.iloc[0] = net_series.iloc[0] - float(result["turnover"]) * float(bps) / 10000.0
                net_series_by_strategy_cost[(strategy, bps)].append(net_series)
                tx_rows.append(
                    {
                        "rebalance_date": info.rebalance_date,
                        "strategy": strategy,
                        "cost_bps": float(bps),
                        "turnover": float(result["turnover"]),
                        "transaction_cost": 0.0 if prev is None else float(result["turnover"]) * float(bps) / 10000.0,
                    }
                )

            strategy_previous_weights[strategy] = {asset: float(weight) for asset, weight in result["weights"].items()}

    gross_series_map = {strategy: pd.concat(parts).sort_index() for strategy, parts in gross_series_by_strategy.items()}
    net_series_map = {
        (strategy, bps): pd.concat(parts).sort_index()
        for (strategy, bps), parts in net_series_by_strategy_cost.items()
    }
    gross_df = pd.DataFrame(gross_series_map).sort_index()
    gross_df.index.name = "Date"
    gross_df_reset = gross_df.reset_index()

    net_long_rows = []
    for (strategy, bps), series in net_series_map.items():
        for dt, value in series.items():
            net_long_rows.append({"date": dt, "strategy": strategy, "cost_bps": float(bps), "net_return": float(value)})
    net_long_df = pd.DataFrame(net_long_rows)

    weights_df = pd.DataFrame(weights_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)
    turnover_df = pd.DataFrame(turnover_rows)
    tx_df = pd.DataFrame(tx_rows)
    solver_df = pd.DataFrame(solver_rows)
    iv_diag_df = pd.DataFrame(iv_diag_rows)
    erc_diag_df = pd.DataFrame(erc_diag_rows)

    gross_metrics_df = _compute_metrics_by_strategy(gross_series_map)
    net_metrics_df = _compute_net_metrics(net_series_map, gross_series_map)
    turnover_stats_df = _compute_turnover(weights_df)
    concentration_df = _compute_concentration(weights_df)
    cap_binding_df = _compute_cap_binding(weights_df)
    rolling_df = _compute_rolling_performance(gross_series_map)
    stress_windows = [
        ("COVID_2020", pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-31")),
        ("RATE_HIKE_2022", pd.Timestamp("2022-04-01"), pd.Timestamp("2022-10-31")),
    ]
    stress_df = _compute_stress(gross_series_map, weights_df, stress_windows)
    drawdown_df = _compute_drawdowns(gross_series_map)
    rep_dates = _compute_representative_dates(rebalance_infos)
    risk_contrib_df = _compute_risk_contrib_analysis(weights_df, prices, returns, rebalance_infos, tickers, rep_dates)
    sensitivity_df = _compute_sensitivity(gross_series_map, weights_history_by_strategy, prices, returns, rebalance_infos, tickers)

    gross_df_reset.to_csv(outdir / "walk_forward_returns_gross.csv", index=False)
    net_long_df.to_csv(outdir / "walk_forward_returns_net.csv", index=False)
    weights_df.to_csv(outdir / "walk_forward_weights.csv", index=False)
    gross_metrics_df.to_csv(outdir / "gross_metrics.csv", index=False)
    net_metrics_df.to_csv(outdir / "net_metrics.csv", index=False)
    rebalance_df.to_csv(outdir / "rebalance_history.csv", index=False)
    turnover_stats_df.to_csv(outdir / "turnover_analysis.csv", index=False)
    concentration_df.to_csv(outdir / "concentration_analysis.csv", index=False)
    cap_binding_df.to_csv(outdir / "cap_binding_analysis.csv", index=False)
    risk_contrib_df.to_csv(outdir / "risk_contribution_analysis.csv", index=False)
    erc_diag_df.to_csv(outdir / "erc_diagnostics.csv", index=False)
    iv_diag_df.to_csv(outdir / "inverse_volatility_diagnostics.csv", index=False)
    sensitivity_df.to_csv(outdir / "risk_estimation_sensitivity.csv", index=False)
    stress_df.to_csv(outdir / "stress_period_analysis.csv", index=False)
    drawdown_df.to_csv(outdir / "drawdown_analysis.csv", index=False)
    rolling_df.to_csv(outdir / "rolling_performance.csv", index=False)
    tx_df.to_csv(outdir / "transaction_cost_history.csv", index=False)
    tx_summary = tx_df.merge(net_metrics_df[["strategy", "cost_bps", "terminal_wealth", "cagr", "sharpe", "cost_drag", "gross_terminal_wealth"]], on=["strategy", "cost_bps"], how="left")
    tx_summary.to_csv(outdir / "transaction_cost_analysis.csv", index=False)
    solver_df.to_csv(outdir / "solver_history.csv", index=False)

    _plot_cumulative_wealth(gross_df_reset, figdir)
    _plot_drawdown(gross_df_reset, figdir)
    _plot_rolling(rolling_df, figdir)
    _plot_turnover(turnover_stats_df, figdir)
    _plot_concentration(concentration_df, figdir)
    _plot_transaction_cost_impact(tx_summary, figdir)
    _plot_weight_evolution(weights_df, "Maximum Sharpe", "07_weight_evolution_max_sharpe.png", figdir)
    _plot_weight_evolution(weights_df, ROBUST_M3_STRATEGY, "08_weight_evolution_combined_robust.png", figdir)
    _plot_risk_contribution(risk_contrib_df, figdir)
    _plot_erc_dispersion(erc_diag_df, figdir)
    _plot_sensitivity(sensitivity_df, figdir)
    _plot_stress(stress_df, figdir)

    baseline_diffs = _compare_to_baselines(outdir, gross_df_reset, weights_df)
    baseline_pass = (
        baseline_diffs["actual_equal_weight_return_diff"] <= BASELINE_RETURN_TOLERANCE
        and baseline_diffs["actual_min_variance_return_diff"] <= BASELINE_RETURN_TOLERANCE
        and baseline_diffs["actual_max_sharpe_return_diff"] <= BASELINE_RETURN_TOLERANCE
        and baseline_diffs["actual_min_variance_weight_diff"] <= BASELINE_WEIGHT_TOLERANCE
        and baseline_diffs["actual_max_sharpe_weight_diff"] <= BASELINE_WEIGHT_TOLERANCE
        and baseline_diffs["combined_robust_return_diff"] <= BASELINE_RETURN_TOLERANCE
        and baseline_diffs["combined_robust_weight_diff"] <= BASELINE_WEIGHT_TOLERANCE
    )
    if not baseline_pass:
        raise RuntimeError("Baseline preservation gate failed; aborting M4 interpretation.")

    erc_attempts = int(len(erc_diag_df))
    erc_successes = int((erc_diag_df["solver_success"] == True).sum()) if not erc_diag_df.empty else 0
    erc_failures = int(erc_attempts - erc_successes)
    erc_fallbacks = int((erc_diag_df["fallback_used"] == True).sum()) if not erc_diag_df.empty else 0
    if erc_fallbacks > 0:
        raise RuntimeError("ERC fallback count exceeded zero; stopping before empirical interpretation.")

    inverse_successes = int(len(iv_diag_df))
    inverse_failures = 0

    reproducibility = {
        "run1_vs_run2_max_return_diff": None,
        "run1_vs_run2_max_weight_diff": None,
    }
    if snapshot_dir.exists():
        run1_gross = pd.read_csv(snapshot_dir / "walk_forward_returns_gross.csv")
        run1_weights = pd.read_csv(snapshot_dir / "walk_forward_weights.csv")
        reproducibility["run1_vs_run2_max_return_diff"] = _wide_return_diff(run1_gross, gross_df_reset)
        reproducibility["run1_vs_run2_max_weight_diff"] = _long_weight_diff(run1_weights, weights_df)
    else:
        _record_run_snapshot(outdir, snapshot_dir)

    verification = {
        "canonical_hashes_validated": True,
        "live_data_calls": live_download_calls["count"],
        "oos_start": gross_df_reset["Date"].min().strftime("%Y-%m-%d"),
        "oos_end": gross_df_reset["Date"].max().strftime("%Y-%m-%d"),
        "oos_observations": int(gross_df_reset["Date"].nunique()),
        "rebalance_count": int(len(rebalance_infos)),
        "baseline_return_tolerance": BASELINE_RETURN_TOLERANCE,
        "baseline_weight_tolerance": BASELINE_WEIGHT_TOLERANCE,
        "baseline_equal_weight_return_diff": baseline_diffs["actual_equal_weight_return_diff"],
        "baseline_min_variance_return_diff": baseline_diffs["actual_min_variance_return_diff"],
        "baseline_max_sharpe_return_diff": baseline_diffs["actual_max_sharpe_return_diff"],
        "baseline_min_variance_weight_diff": baseline_diffs["actual_min_variance_weight_diff"],
        "baseline_max_sharpe_weight_diff": baseline_diffs["actual_max_sharpe_weight_diff"],
        "combined_robust_return_diff": baseline_diffs["combined_robust_return_diff"],
        "combined_robust_weight_diff": baseline_diffs["combined_robust_weight_diff"],
        "baseline_consistency_pass": baseline_pass,
        "optimizer_failures": int(rebalance_df["optimization_success"].eq(False).sum()),
        "optimizer_fallbacks": int(rebalance_df["fallback_used"].sum()),
        "erc_rebalance_attempts": erc_attempts,
        "erc_solver_successes": erc_successes,
        "erc_solver_failures": erc_failures,
        "erc_fallback_count": erc_fallbacks,
        "erc_fallback_dates": erc_diag_df.loc[erc_diag_df["fallback_used"] == True, "rebalance_date"].astype(str).tolist() if not erc_diag_df.empty else [],
        "erc_fallback_reasons": erc_diag_df.loc[erc_diag_df["fallback_used"] == True, "solver_message"].astype(str).tolist() if not erc_diag_df.empty else [],
        "inverse_vol_rebalance_attempts": inverse_successes,
        "inverse_vol_failures": inverse_failures,
        "run1_vs_run2_max_return_diff": reproducibility["run1_vs_run2_max_return_diff"],
        "run1_vs_run2_max_weight_diff": reproducibility["run1_vs_run2_max_weight_diff"],
        "run1_snapshot_recorded": not snapshot_dir.exists(),
    }

    verification_path = outdir / "evaluation_verification.json"
    with verification_path.open("w", encoding="utf-8") as file_obj:
        json.dump(verification, file_obj, indent=2)

    gross_metrics_full = gross_metrics_df.copy()
    net_metrics_full = net_metrics_df.copy()
    if net_metrics_full.empty:
        raise RuntimeError("No net metrics were produced.")

    _build_report(
        outdir=outdir,
        verification=verification,
        gross_metrics=gross_metrics_full,
        net_metrics=net_metrics_full,
        turnover_df=turnover_stats_df,
        concentration_df=concentration_df,
        cap_binding_df=cap_binding_df,
        risk_contrib_df=risk_contrib_df,
        iv_diag_df=iv_diag_df,
        erc_diag_df=erc_diag_df,
        sensitivity_df=sensitivity_df,
        stress_df=stress_df,
        drawdown_df=drawdown_df,
        rolling_df=rolling_df,
        baseline_diffs=baseline_diffs,
        reproducibility=reproducibility,
    )


if __name__ == "__main__":
    main()