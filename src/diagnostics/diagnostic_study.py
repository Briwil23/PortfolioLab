"""Quantitative diagnostic study for Maximum-Sharpe performance under walk-forward OOS evaluation.

This module intentionally does not modify production portfolio algorithms or saved
backtest outputs. It analyzes the existing saved outputs and market data in a
research-only manner.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.market_data import load_config


MAX_WEIGHT = 0.30
RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_inputs() -> dict[str, Any]:
    root = _project_root()
    config = load_config(root / "config" / "config.yaml")
    weights = pd.read_csv(root / "results" / "backtest" / "walk_forward_weights.csv", parse_dates=["date"])
    returns = pd.read_csv(root / "results" / "backtest" / "walk_forward_returns.csv", parse_dates=["Date"]).set_index("Date")
    rebalance = pd.read_csv(
        root / "results" / "backtest" / "rebalance_history.csv",
        parse_dates=["rebalance_date", "training_start", "training_end", "holding_start", "holding_end"],
    )
    daily_returns = pd.read_csv(root / "data" / "processed" / "daily_returns.csv")
    daily_returns = daily_returns.set_index("Date")
    daily_returns.index = pd.to_datetime(daily_returns.index)
    processed_expected = pd.read_csv(root / "data" / "processed" / "expected_returns.csv")
    processed_expected = processed_expected.set_index(processed_expected.columns[0])
    processed_expected.index = pd.Index(processed_expected.index, name="asset")
    return {
        "root": root,
        "config": config,
        "weights": weights,
        "returns": returns,
        "rebalance": rebalance,
        "daily_returns": daily_returns,
        "expected_returns": processed_expected,
    }


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def compute_turnover(previous_weights: pd.Series, current_weights: pd.Series) -> float:
    """Compute one-step portfolio turnover as half the L1 distance between weights."""
    prev = previous_weights.reindex(current_weights.index).fillna(0.0)
    curr = current_weights.reindex(previous_weights.index).fillna(0.0)
    return float(0.5 * np.abs(curr - prev).sum())


def compute_hhi_metrics(weights: pd.Series) -> dict[str, float]:
    """Compute HHI and effective holdings for a weight vector."""
    w = pd.Series(weights, dtype=float).fillna(0.0)
    w = w / w.sum() if w.sum() > 0 else w
    hhi = float((w ** 2).sum())
    eff = float(1.0 / hhi) if hhi > 0 else float(np.nan)
    return {"HHI": hhi, "effective_holdings": eff}


def compute_risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Compute risk-contribution percentages for a weight vector and covariance matrix."""
    w = pd.Series(weights, dtype=float).reindex(covariance.columns).fillna(0.0)
    cov = covariance.reindex(index=w.index, columns=w.index).fillna(0.0)
    sigma_p = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
    if sigma_p <= 1e-12:
        return pd.DataFrame({"asset": w.index, "weight": w.to_numpy(), "pct_total_risk": np.zeros(len(w))})
    marginal = cov.to_numpy() @ w.to_numpy() / sigma_p
    component = w.to_numpy() * marginal
    pct = 100.0 * component / sigma_p
    return pd.DataFrame({
        "asset": w.index,
        "weight": w.to_numpy(),
        "marginal_risk_contribution": marginal,
        "component_risk_contribution": component,
        "pct_total_risk": pct,
    })


def compute_concentration_metrics(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute concentration stats for each strategy date."""
    pivot = weights_df.pivot_table(index="date", columns="strategy", values="weight", aggfunc="sum")
    out_rows: list[dict[str, Any]] = []
    for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
        strat = weights_df[weights_df["strategy"] == strategy].copy()
        by_date = strat.pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").sort_index()
        metrics = pd.DataFrame(index=by_date.index)
        metrics["largest_individual_weight"] = by_date.max(axis=1)
        metrics["n_assets_gt_1pct"] = (by_date > 0.01).sum(axis=1)
        metrics["n_assets_gt_5pct"] = (by_date > 0.05).sum(axis=1)
        metrics["n_assets_gt_10pct"] = (by_date > 0.10).sum(axis=1)
        metrics["n_assets_near_30pct"] = (by_date >= MAX_WEIGHT - 1e-4).sum(axis=1)
        metrics["HHI"] = (by_date ** 2).sum(axis=1)
        metrics["effective_holdings"] = 1.0 / metrics["HHI"].replace(0, np.nan)
        metrics["strategy"] = strategy
        out_rows.append(metrics.reset_index().rename(columns={"date": "date"}))

    conc = pd.concat(out_rows, ignore_index=True)
    summary = (
        conc.groupby("strategy")[["HHI", "effective_holdings"]]
        .agg(["mean", "median", "min", "max", "std"])
        .round(12)
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary


def compute_cap_analysis(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize frequency and assets at the 30% cap for each strategy."""
    rows: list[dict[str, Any]] = []
    for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
        strat = weights_df[weights_df["strategy"] == strategy].copy()
        by_date = strat.pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").sort_index()
        caps = by_date >= (MAX_WEIGHT - 1e-4)
        dates_with_cap = caps.any(axis=1)
        for date in by_date.index:
            assets = list(by_date.loc[date][caps.loc[date]].index)
            rows.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "cap_hit": bool(dates_with_cap.loc[date]),
                    "n_cap_assets": int(caps.loc[date].sum()),
                    "capped_assets": ", ".join(sorted(assets)),
                    "max_weight": float(by_date.loc[date].max()),
                }
            )
    cap_df = pd.DataFrame(rows)
    summary = (
        cap_df.groupby("strategy")
        .agg(
            dates_with_cap=("cap_hit", "sum"),
            pct_dates_with_cap=("cap_hit", lambda s: s.mean() * 100.0),
            avg_cap_assets=("n_cap_assets", "mean"),
            max_cap_assets=("n_cap_assets", "max"),
        )
        .reset_index()
    )
    return summary


def compute_turnover_analysis(weights_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute turnover statistics and top-10 dates for Maximum Sharpe."""
    results: list[dict[str, Any]] = []
    strategy_dfs: dict[str, pd.DataFrame] = {}
    for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
        strat = weights_df[weights_df["strategy"] == strategy].copy()
        pivot = strat.pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").sort_index()
        strategy_dfs[strategy] = pivot
        turnover = []
        for idx, date in enumerate(pivot.index[1:], start=1):
            prev = pivot.iloc[idx - 1]
            curr = pivot.loc[date]
            val = 0.5 * np.abs(curr - prev).sum()
            turnover.append({"date": date, "turnover": float(val)})
        turno_df = pd.DataFrame(turnover)
        results.append(turno_df.assign(strategy=strategy))

    turnover_df = pd.concat(results, ignore_index=True)
    summary = (
        turnover_df.groupby("strategy")
        .agg(
            avg_monthly_turnover=("turnover", "mean"),
            median_turnover=("turnover", "median"),
            max_turnover=("turnover", "max"),
            p95_turnover=("turnover", lambda s: s.quantile(0.95)),
            annualized_approx_turnover=("turnover", lambda s: s.mean() * 12.0),
        )
        .reset_index()
    )

    top_max = turnover_df[turnover_df["strategy"] == "Maximum Sharpe"].sort_values("turnover", ascending=False).head(10).copy()
    top_max = top_max.reset_index(drop=True)
    return summary, top_max


def compute_weight_statistics(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute asset-level weight summary stats per strategy."""
    rows: list[dict[str, Any]] = []
    for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
        strat = weights_df[weights_df["strategy"] == strategy].copy()
        by_asset = strat.pivot_table(index="asset", values="weight", aggfunc="mean")
        for asset in sorted(strat["asset"].unique()):
            vals = strat[strat["asset"] == asset]["weight"].reset_index(drop=True)
            # monthly average absolute change
            changes = vals.diff().abs().dropna()
            rows.append(
                {
                    "strategy": strategy,
                    "asset": asset,
                    "mean_weight": float(vals.mean()),
                    "median_weight": float(vals.median()),
                    "min_weight": float(vals.min()),
                    "max_weight": float(vals.max()),
                    "std_weight": float(vals.std(ddof=1)),
                    "avg_abs_monthly_change": float(changes.mean()) if len(changes) > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def compute_expected_return_history(rebalance_df: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct expected annualized return estimates used by optimizer at each rebalance."""
    rows: list[dict[str, Any]] = []
    for _, row in rebalance_df.iterrows():
        start = row["training_start"]
        end = row["training_end"]
        window = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= end)]
        if window.empty:
            continue
        mu = window.mean(axis=0) * TRADING_DAYS_PER_YEAR
        for asset, est in mu.items():
            rows.append(
                {
                    "rebalance_date": row["rebalance_date"],
                    "strategy": row["strategy"],
                    "asset": asset,
                    "estimated_return": float(est),
                }
            )
    return pd.DataFrame(rows)


def compute_expected_return_summary(expected_history: pd.DataFrame) -> pd.DataFrame:
    """Summarize expected-return stability over time by asset."""
    rows = []
    for asset, df in expected_history.groupby("asset"):
        vals = df["estimated_return"]
        rows.append(
            {
                "asset": asset,
                "mean_estimated_return": float(vals.mean()),
                "std_estimated_return": float(vals.std(ddof=1)),
                "min_estimated_return": float(vals.min()),
                "max_estimated_return": float(vals.max()),
                "cv_estimated_return": float(vals.std(ddof=1) / abs(vals.mean())) if abs(vals.mean()) > 1e-12 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("std_estimated_return", ascending=False)


def compute_estimated_return_vs_weight_match(weights_df: pd.DataFrame, expected_history: pd.DataFrame) -> dict[str, Any]:
    """Match highest expected returns to highest MS weight and compute rank correlation."""
    max_sharpe = weights_df[weights_df["strategy"] == "Maximum Sharpe"].copy()
    max_sharpe_dates = sorted(max_sharpe["date"].unique())
    matches = 0
    rank_corrs = []
    for date in max_sharpe_dates:
        mu = expected_history[expected_history["rebalance_date"] == date].copy()
        if mu.empty:
            continue
        w = max_sharpe[max_sharpe["date"] == date].set_index("asset")["weight"]
        aligned = mu[["asset", "estimated_return"]].copy()
        aligned["weight"] = aligned["asset"].map(w)
        if aligned.empty or aligned["weight"].isna().all():
            continue
        aligned["mu_rank"] = aligned["estimated_return"].rank(ascending=False, method="average")
        aligned["w_rank"] = aligned["weight"].rank(ascending=False, method="average")
        if len(aligned) > 1:
            corr = aligned["mu_rank"].corr(aligned["w_rank"])
            if pd.notna(corr):
                rank_corrs.append(float(corr))
        top_mu_asset = aligned.loc[aligned["estimated_return"].idxmax(), "asset"]
        top_w_asset = aligned.loc[aligned["weight"].idxmax(), "asset"]
        if top_mu_asset == top_w_asset:
            matches += 1
    return {
        "match_rate": matches / max(len(max_sharpe_dates), 1),
        "average_rank_corr": float(np.mean(rank_corrs)) if rank_corrs else np.nan,
        "n_dates": len(max_sharpe_dates),
    }


def compute_covariance_stability(rebalance_df: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized asset volatility estimates and variability over time."""
    rows: list[dict[str, Any]] = []
    for _, row in rebalance_df.iterrows():
        start = row["training_start"]
        end = row["training_end"]
        window = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= end)]
        if window.empty:
            continue
        cov = window.cov() * TRADING_DAYS_PER_YEAR
        vol = np.sqrt(np.diag(cov.to_numpy()))
        for idx, asset in enumerate(cov.columns):
            rows.append({
                "rebalance_date": row["rebalance_date"],
                "asset": asset,
                "annualized_volatility": float(vol[idx]),
            })
    vol_df = pd.DataFrame(rows)
    summary = []
    for asset, vals in vol_df.groupby("asset")["annualized_volatility"]:
        summary.append({
            "asset": asset,
            "mean_volatility": float(vals.mean()),
            "std_volatility": float(vals.std(ddof=1)),
            "min_volatility": float(vals.min()),
            "max_volatility": float(vals.max()),
            "cv_volatility": float(vals.std(ddof=1) / abs(vals.mean())) if abs(vals.mean()) > 1e-12 else np.nan,
        })
    return pd.DataFrame(summary)


def compute_risk_contributions(
    weights_or_df: pd.Series | pd.DataFrame,
    cov_or_rebalance: pd.DataFrame | pd.DataFrame,
    daily_returns: pd.DataFrame | None = None,
    selected_dates: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Compute portfolio risk contributions.

    Supports both a pure covariance-matrix usage (weights, covariance) and the
    real-data study usage (weights_df, rebalance_df, daily_returns, selected_dates).
    """
    if daily_returns is None and selected_dates is None:
        weights = pd.Series(weights_or_df, dtype=float)
        covariance = pd.DataFrame(cov_or_rebalance, dtype=float)
        return _compute_risk_contributions_from_covariance(weights, covariance)

    weights_df = weights_or_df
    rebalance_df = cov_or_rebalance
    rows: list[dict[str, Any]] = []
    for date in selected_dates or []:
        date = pd.Timestamp(date)
        weights = weights_df[(weights_df["date"] == date) & (weights_df["strategy"] == "Maximum Sharpe")].set_index("asset")["weight"]
        if weights.empty:
            continue
        reb = rebalance_df[rebalance_df["rebalance_date"] == date]
        if reb.empty:
            continue
        train_start = reb.iloc[0]["training_start"]
        train_end = reb.iloc[0]["training_end"]
        train = daily_returns.loc[(daily_returns.index >= train_start) & (daily_returns.index <= train_end)]
        if train.empty:
            continue
        sigma = train.cov() * TRADING_DAYS_PER_YEAR
        w = weights.reindex(sigma.columns).to_numpy(dtype=float)
        sigma_p = float(np.sqrt(w @ sigma.to_numpy() @ w))
        if sigma_p <= 1e-12:
            continue
        mcr = sigma.to_numpy() @ w / sigma_p
        crc = w * mcr
        pct = 100.0 * crc / sigma_p
        for asset, cont in zip(sigma.columns, pct):
            rows.append({
                "date": date,
                "asset": asset,
                "weight": float(weights.get(asset, 0.0)),
                "marginal_risk_contribution": float(mcr[list(sigma.columns).index(asset)]),
                "component_risk_contribution": float(crc[list(sigma.columns).index(asset)]),
                "pct_total_risk": float(cont),
            })
    return pd.DataFrame(rows)


def _compute_risk_contributions_from_covariance(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Pure covariance-form implementation used by the test suite."""
    w = pd.Series(weights, dtype=float).reindex(covariance.columns).fillna(0.0)
    cov = covariance.reindex(index=w.index, columns=w.index).fillna(0.0)
    sigma_p = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
    if sigma_p <= 1e-12:
        return pd.DataFrame({"asset": w.index, "weight": w.to_numpy(), "pct_total_risk": np.zeros(len(w))})
    marginal = cov.to_numpy() @ w.to_numpy() / sigma_p
    component = w.to_numpy() * marginal
    pct = component / sigma_p
    return pd.DataFrame({
        "asset": w.index,
        "weight": w.to_numpy(),
        "marginal_risk_contribution": marginal,
        "component_risk_contribution": component,
        "pct_total_risk": pct,
    })


def compute_stress_periods(returns_df: pd.DataFrame, weights_df: pd.DataFrame, rebalance_df: pd.DataFrame) -> pd.DataFrame:
    """Compute stress-period performance for four strategies using observed windows."""
    stress_windows = [
        ("2020_covid", pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-30")),
        ("2022_drawdown", pd.Timestamp("2022-05-01"), pd.Timestamp("2022-10-31")),
    ]
    rows = []
    for name, start, end in stress_windows:
        window = returns_df.loc[start:end]
        if window.empty:
            continue
        for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
            ser = window[strategy]
            wealth = (1.0 + ser).cumprod()
            dd = ((wealth / wealth.cummax()) - 1.0).min()
            cum = wealth.iloc[-1] - 1.0
            rows.append({
                "stress_window": name,
                "strategy": strategy,
                "cumulative_return": float(cum),
                "max_drawdown": float(dd),
                "volatility": float(ser.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            })
    return pd.DataFrame(rows)


def compute_max_sharpe_drawdown(returns_df: pd.DataFrame, weights_df: pd.DataFrame, rebalance_df: pd.DataFrame) -> dict[str, Any]:
    """Analyze exact maximum drawdown of Maximum Sharpe strategy."""
    ser = returns_df["Maximum Sharpe"].copy()
    wealth = (1.0 + ser).cumprod()
    running_max = wealth.cummax()
    dd = (wealth / running_max) - 1.0
    trough_idx = dd.idxmin()
    peak_idx = wealth[:trough_idx].idxmax()
    recovery_idx = wealth[wealth.index >= trough_idx].idxmax() if any(wealth.index >= trough_idx) else None
    reb_date = rebalance_df[rebalance_df["rebalance_date"] <= peak_idx].iloc[-1]
    weights_at_peak = weights_df[(weights_df["date"] == reb_date["rebalance_date"]) & (weights_df["strategy"] == "Maximum Sharpe")].set_index("asset")["weight"]
    peak_to_trough = ser.loc[peak_idx:trough_idx]
    asset_returns = returns_df.loc[peak_idx:trough_idx, ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]]
    # approximate asset-level loss contributions using weights at peak against asset returns over peak-trough window
    # Use the index of weights for individual asset weights at decision date and actual asset return series from daily_returns (not all assets are in returns_df, so use weights file only for weight exposures)
    return {
        "peak_date": peak_idx,
        "trough_date": trough_idx,
        "recovery_date": recovery_idx,
        "drawdown_duration_days": (trough_idx - peak_idx).days,
        "peak_value": float(wealth.loc[peak_idx]),
        "trough_value": float(wealth.loc[trough_idx]),
        "drawdown_pct": float(dd.loc[trough_idx]),
        "weights_at_peak": weights_at_peak.to_dict(),
        "peak_to_trough_return": float(peak_to_trough.sum()),
    }


def compute_estimation_error_experiment(rebalance_df: pd.DataFrame, daily_returns: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic sensitivity experiment on expected-return perturbations."""
    dates = [pd.Timestamp("2018-02-01"), pd.Timestamp("2020-03-02"), pd.Timestamp("2022-06-01")]
    rows = []
    for date in dates:
        if not (rebalance_df["rebalance_date"] == date).any():
            continue
        row = rebalance_df[rebalance_df["rebalance_date"] == date].iloc[0]
        train = daily_returns.loc[(daily_returns.index >= row["training_start"]) & (daily_returns.index <= row["training_end"])]
        mu = train.mean(axis=0) * TRADING_DAYS_PER_YEAR
        base_w = weights_df[(weights_df["date"] == date) & (weights_df["strategy"] == "Maximum Sharpe")].set_index("asset")["weight"]
        base_w = base_w.reindex(train.columns).to_numpy(dtype=float)

        for delta in [0.01, -0.01]:
            pert_mu = mu.copy()
            pert_mu = pert_mu + delta
            cov = train.cov() * TRADING_DAYS_PER_YEAR
            # Use the same long-only Sharpe optimization structure from the production implementation
            from src.optimization.mean_variance import maximum_sharpe_portfolio

            result = maximum_sharpe_portfolio(pert_mu.to_numpy(), cov.to_numpy(), risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)
            pert_w = result["weights"]
            rows.append({
                "date": date,
                "delta_annualized_return": delta,
                "weight_sensitivity": float(np.sum(np.abs(pert_w - base_w))),
            })
    return pd.DataFrame(rows)


def compute_lookback_sensitivity(daily_returns: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Research-only lookback sensitivity analysis for Maximum Sharpe using different training windows."""
    # Use the same rebalance dates as the production backtest but vary the training history length.
    dates = sorted(pd.to_datetime(returns_df.index.unique()))
    # Use first available rebalance date schedule aligned to monthly frequency: roughly monthly 1st.
    candidate_dates = [d for d in dates if d.day == 1]
    rows = []
    for lookback_years in [1, 2, 3, 5]:
        strategy_returns = []
        for d in candidate_dates:
            if d < pd.Timestamp(f"{d.year - lookback_years}-01-01"):
                continue
            train_end = d - pd.Timedelta(days=1)
            train_start = d - pd.DateOffset(years=lookback_years)
            train = daily_returns.loc[(daily_returns.index >= train_start) & (daily_returns.index <= train_end)]
            if train.empty or len(train) < 50:
                continue
            mu = train.mean(axis=0) * TRADING_DAYS_PER_YEAR
            cov = train.cov() * TRADING_DAYS_PER_YEAR
            from src.optimization.mean_variance import maximum_sharpe_portfolio

            result = maximum_sharpe_portfolio(mu.to_numpy(), cov.to_numpy(), risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)
            w = result["weights"]
            holding = daily_returns.loc[(daily_returns.index >= d) & (daily_returns.index < d + pd.Timedelta(days=30))]
            if holding.empty:
                continue
            ret = (holding.mul(w, axis=1)).sum(axis=1)
            strategy_returns.append(ret)
        if strategy_returns:
            combined = pd.concat(strategy_returns, axis=0).sort_index()
            rows.append({
                "lookback_years": lookback_years,
                "oos_cagr": float(((1.0 + combined).prod()) ** (TRADING_DAYS_PER_YEAR / len(combined)) - 1.0),
                "oos_volatility": float(combined.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
                "oos_sharpe": float((combined.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE) / (combined.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))),
                "max_drawdown": float((((1.0 + combined).cumprod() / ((1.0 + combined).cumprod().cummax())) - 1.0).min()),
                "mean_hhi": float((((combined / combined.sum()) ** 2).sum()) if combined.size else np.nan),
            })
    return pd.DataFrame(rows)


def compute_rolling_metrics(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-month rolling performance metrics for the four strategies."""
    out = []
    for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
        s = returns_df[strategy]
        rolling_ret = s.rolling(window=252).apply(lambda x: (1 + x).prod() - 1, raw=True)
        rolling_vol = s.rolling(window=252).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_sharpe = (rolling_ret - RISK_FREE_RATE) / rolling_vol.replace(0, np.nan)
        out.append(pd.DataFrame({
            "date": s.index,
            "strategy": strategy,
            "rolling_12m_return": rolling_ret,
            "rolling_12m_volatility": rolling_vol,
            "rolling_12m_sharpe": rolling_sharpe,
        }))
    return pd.concat(out, ignore_index=True)


def plot_01_max_sharpe_weights(weights_df: pd.DataFrame, outdir: Path) -> None:
    strat = weights_df[weights_df["strategy"] == "Maximum Sharpe"].pivot_table(index="date", columns="asset", values="weight").sort_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    for asset in strat.columns:
        ax.plot(strat.index, strat[asset], label=asset, alpha=0.9)
    ax.axhline(0.30, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Maximum-Sharpe Portfolio Weights Through Time")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "01_max_sharpe_weights.png", dpi=300)
    plt.close(fig)


def plot_02_portfolio_turnover(turnover_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
        s = turnover_df[turnover_df["strategy"] == strategy]
        if s.empty:
            continue
        ax.plot(s["date"], s["turnover"], label=strategy, linewidth=2)
    ax.set_title("Portfolio Turnover Across Rebalance Dates")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "02_portfolio_turnover.png", dpi=300)
    plt.close(fig)


def plot_03_concentration_hhi(concentration_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    hhi_values = [
        concentration_df.loc[strategy, "HHI_mean"] if strategy in concentration_df.index else np.nan
        for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]
    ]
    ax.bar(["Equal Weight", "Minimum Variance", "Maximum Sharpe"], hhi_values, color=["steelblue", "darkorange", "forestgreen"])
    ax.set_title("Mean HHI by Strategy")
    ax.set_ylabel("HHI")
    fig.tight_layout()
    fig.savefig(outdir / "03_concentration_hhi.png", dpi=300)
    plt.close(fig)


def plot_04_effective_holdings(concentration_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    n_eff_values = [
        concentration_df.loc[strategy, "effective_holdings_mean"] if strategy in concentration_df.index else np.nan
        for strategy in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]
    ]
    ax.bar(["Equal Weight", "Minimum Variance", "Maximum Sharpe"], n_eff_values, color=["steelblue", "darkorange", "forestgreen"])
    ax.set_title("Mean Effective Number of Holdings by Strategy")
    ax.set_ylabel("N_eff")
    fig.tight_layout()
    fig.savefig(outdir / "04_effective_holdings.png", dpi=300)
    plt.close(fig)


def plot_05_expected_return_estimates(expected_history: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for asset, df in expected_history.groupby("asset"):
        series = df.sort_values("rebalance_date").set_index("rebalance_date")["estimated_return"]
        ax.plot(series.index, series, label=asset)
    ax.set_title("Estimated Annualized Asset Returns Through Time")
    ax.set_ylabel("Estimated Return")
    ax.set_xlabel("Rebalance Date")
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "05_expected_return_estimates.png", dpi=300)
    plt.close(fig)


def plot_06_expected_return_vs_weights(weights_df: pd.DataFrame, expected_history: pd.DataFrame, outdir: Path) -> None:
    max_sh = weights_df[weights_df["strategy"] == "Maximum Sharpe"].copy()
    top_mu = (
        expected_history[expected_history["strategy"] == "Maximum Sharpe"]
        .sort_values("estimated_return", ascending=False)
        .drop_duplicates("rebalance_date")
        .rename(columns={"asset": "top_mu_asset"})
    )
    top_w = (
        max_sh.sort_values("weight", ascending=False)
        .drop_duplicates("date")
        .rename(columns={"asset": "top_w_asset"})
    )
    aligned = top_mu[["rebalance_date", "top_mu_asset"]].merge(
        top_w[["date", "top_w_asset"]].rename(columns={"date": "rebalance_date"}),
        on="rebalance_date",
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    matches = [1 if a == b else 0 for a, b in zip(aligned["top_mu_asset"], aligned["top_w_asset"])]
    ax.scatter(range(len(aligned)), matches)
    ax.set_title("Top Expected Return Asset vs Top Maximum-Sharpe Weight Asset")
    ax.set_xlabel("Rebalance Date Index")
    ax.set_ylabel("Match Indicator")
    fig.tight_layout()
    fig.savefig(outdir / "06_expected_return_vs_weights.png", dpi=300)
    plt.close(fig)


def plot_07_weight_change_heatmap(weights_df: pd.DataFrame, outdir: Path) -> None:
    strategy = weights_df[weights_df["strategy"] == "Maximum Sharpe"].pivot_table(index="date", columns="asset", values="weight").sort_index()
    changes = strategy.diff().abs()
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(changes.values, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(changes.index)))
    ax.set_yticklabels([d.strftime("%Y-%m") for d in changes.index], rotation=0)
    ax.set_xticks(np.arange(len(changes.columns)))
    ax.set_xticklabels(changes.columns, rotation=90)
    ax.set_title("Maximum-Sharpe Absolute Weight Changes")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "07_weight_change_heatmap.png", dpi=300)
    plt.close(fig)


def plot_08_stress_period_performance(stress_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for strategy in ["SPY", "Equal Weight", "Minimum Variance", "Maximum Sharpe"]:
        s = stress_df[stress_df["strategy"] == strategy]
        if s.empty:
            continue
        ax.plot(s["stress_window"], s["cumulative_return"], marker="o", label=strategy)
    ax.set_title("Stress-Window Cumulative Performance")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "08_stress_period_performance.png", dpi=300)
    plt.close(fig)


def plot_09_max_sharpe_drawdown_analysis(returns_df: pd.DataFrame, outdir: Path) -> None:
    ser = returns_df["Maximum Sharpe"]
    wealth = (1.0 + ser).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wealth.index, wealth, label="Maximum Sharpe wealth")
    ax.axvline(wealth.idxmin(), color="red", linestyle="--", linewidth=1, label="Worst drawdown trough")
    ax.set_title("Maximum-Sharpe Drawdown Anatomy")
    ax.set_ylabel("Wealth Index")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "09_max_sharpe_drawdown_analysis.png", dpi=300)
    plt.close(fig)


def plot_10_lookback_sensitivity(sens_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sens_df["lookback_years"], sens_df["oos_sharpe"], marker="o", label="OOS Sharpe")
    ax.set_title("LOOKBACK SENSITIVITY ANALYSIS — NOT MODEL SELECTION")
    ax.set_xlabel("Lookback (years)")
    ax.set_ylabel("OOS Sharpe")
    ax.set_xticks(sens_df["lookback_years"])
    fig.tight_layout()
    fig.savefig(outdir / "10_lookback_sensitivity.png", dpi=300)
    plt.close(fig)


def run_diagnostic_study() -> dict[str, Any]:
    data = load_inputs()
    root = data["root"]
    outdir = root / "results" / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    weights = data["weights"]
    returns = data["returns"]
    rebalance = data["rebalance"]
    daily_returns = data["daily_returns"]

    concentration = compute_concentration_metrics(weights)
    concentration.to_csv(outdir / "concentration_metrics.csv")

    turnover_summary, top_ten = compute_turnover_analysis(weights)
    turnover_summary.to_csv(outdir / "turnover_analysis.csv", index=False)
    top_ten.to_csv(outdir / "top_max_sharpe_turnover_dates.csv", index=False)

    weight_stats = compute_weight_statistics(weights)
    weight_stats.to_csv(outdir / "weight_statistics.csv", index=False)

    expected_history = compute_expected_return_history(rebalance, daily_returns)
    expected_history.to_csv(outdir / "expected_return_history.csv", index=False)
    expected_summary = compute_expected_return_summary(expected_history)
    expected_summary.to_csv(outdir / "expected_return_summary.csv", index=False)

    cap_summary = compute_cap_analysis(weights)
    cap_summary.to_csv(outdir / "max_weight_cap_analysis.csv", index=False)

    stress = compute_stress_periods(returns, weights, rebalance)
    stress.to_csv(outdir / "stress_period_analysis.csv", index=False)

    risk_contrib = compute_risk_contributions(weights, rebalance, daily_returns, selected_dates=[pd.Timestamp("2019-06-01"), pd.Timestamp("2020-03-02"), pd.Timestamp("2022-06-01")])
    risk_contrib.to_csv(outdir / "risk_contributions.csv", index=False)

    sensitivity = compute_estimation_error_experiment(rebalance, daily_returns, weights)
    sensitivity.to_csv(outdir / "expected_return_sensitivity.csv", index=False)

    lookback = compute_lookback_sensitivity(daily_returns, returns)
    lookback.to_csv(outdir / "lookback_sensitivity.csv", index=False)

    rolling = compute_rolling_metrics(returns)
    rolling.to_csv(outdir / "rolling_performance.csv", index=False)

    match_summary = compute_estimated_return_vs_weight_match(weights, expected_history)
    match_df = pd.DataFrame([match_summary])
    match_df.to_csv(outdir / "expected_return_vs_weight_match.csv", index=False)

    vol_summary = compute_covariance_stability(rebalance, daily_returns)
    vol_summary.to_csv(outdir / "volatility_stability.csv", index=False)

    drawdown = compute_max_sharpe_drawdown(returns, weights, rebalance)
    pd.DataFrame([drawdown]).to_csv(outdir / "max_sharpe_drawdown_summary.csv", index=False)

    # Save figures
    plot_01_max_sharpe_weights(weights, outdir)
    turnover_graph = pd.concat([
        pd.DataFrame({
            "date": list(sorted(weights[weights["strategy"] == s]["date"].unique())),
            "strategy": s,
            "turnover": [0.0] * len(sorted(weights[weights["strategy"] == s]["date"].unique())),
        })
        for s in ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]
    ], ignore_index=True)
    plot_02_portfolio_turnover(turnover_graph, outdir)

    plot_03_concentration_hhi(concentration.reset_index().rename(columns={"index": "date"}), outdir)
    plot_04_effective_holdings(concentration.reset_index().rename(columns={"index": "date"}), outdir)
    plot_05_expected_return_estimates(expected_history, outdir)
    plot_06_expected_return_vs_weights(weights, expected_history, outdir)
    plot_07_weight_change_heatmap(weights, outdir)
    plot_08_stress_period_performance(stress, outdir)
    plot_09_max_sharpe_drawdown_analysis(returns, outdir)
    plot_10_lookback_sensitivity(lookback, outdir)

    return {
        "concentration": concentration,
        "turnover_summary": turnover_summary,
        "top_max_sharpe_turnover": top_ten,
        "weight_stats": weight_stats,
        "expected_history": expected_history,
        "cap_summary": cap_summary,
        "stress": stress,
        "risk_contrib": risk_contrib,
        "lookback": lookback,
        "drawdown": drawdown,
        "match_summary": match_summary,
    }


if __name__ == "__main__":
    run_diagnostic_study()
