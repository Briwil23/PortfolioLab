"""Build canonical dataset snapshot and reproducible canonical Milestone 2 baseline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from run_analysis import run_milestone_2_backtest
from src.data.market_data import (
    ReproducibilityError,
    create_canonical_dataset_snapshot,
    fetch_and_prepare_market_data,
    load_config,
    write_environment_manifest,
)
from src.reproducibility.checks import (
    max_keyed_return_difference,
    max_keyed_weight_difference,
)


def _run_canonical_baseline_once(
    root: Path,
    config: dict,
    output_dir: Path,
    enforce_no_live_download: bool = True,
) -> tuple[dict, int]:
    """Run the classical backtest in canonical mode and return results + download-call count."""
    from src.data import market_data as market_data_module

    download_call_count = 0
    original_download = market_data_module.download_adjusted_prices

    def _guarded_download(*args, **kwargs):
        nonlocal download_call_count
        download_call_count += 1
        raise ReproducibilityError(
            "Canonical mode attempted live download; this is not allowed."
        )

    if enforce_no_live_download:
        market_data_module.download_adjusted_prices = _guarded_download

    try:
        prices, returns, covariance, expected_returns = fetch_and_prepare_market_data(
            tickers=config["asset_universe"],
            start_date=config["start_date"],
            end_date=config.get("end_date"),
            trading_days_per_year=int(config.get("trading_days_per_year", 252)),
            save_output=False,
            data_mode="canonical",
            canonical_dir=root / config.get("canonical_data_dir", "data/canonical"),
        )

        result = run_milestone_2_backtest(
            root=root,
            tickers=config["asset_universe"],
            prices=prices,
            returns=returns,
            trading_days_per_year=int(config.get("trading_days_per_year", 252)),
            risk_free_rate=float(config.get("risk_free_rate", 0.02)),
            max_weight=float(config.get("max_weight", 0.30)),
            benchmark_ticker=config.get("benchmark_ticker", "SPY"),
            backtest_config=config.get("backtest", {}),
            backtest_output_dir=output_dir,
        )
    finally:
        market_data_module.download_adjusted_prices = original_download

    return result, download_call_count


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config" / "config.yaml")

    canonical_dir = root / config.get("canonical_data_dir", "data/canonical")
    reproducibility_dir = root / "results" / "reproducibility"
    baseline_root = reproducibility_dir / "canonical_baseline"
    run1_dir = reproducibility_dir / "canonical_baseline_run_1"
    run2_dir = reproducibility_dir / "canonical_baseline_run_2"

    canonical_dir.mkdir(parents=True, exist_ok=True)
    reproducibility_dir.mkdir(parents=True, exist_ok=True)

    # Build a fresh canonical snapshot from current live dataset.
    prices, returns, _, _ = fetch_and_prepare_market_data(
        tickers=config["asset_universe"],
        start_date=config["start_date"],
        end_date=config.get("end_date"),
        trading_days_per_year=int(config.get("trading_days_per_year", 252)),
        save_output=False,
        data_mode="live",
    )

    manifest = create_canonical_dataset_snapshot(
        prices=prices,
        returns=returns,
        output_dir=canonical_dir,
        dataset_name="portfoliolab_canonical_dataset_v1",
        tickers=config["asset_universe"],
        trading_days_per_year=int(config.get("trading_days_per_year", 252)),
        source_provider="yfinance",
        download_settings={
            "interval": "1d",
            "auto_adjust": False,
            "actions": "ignore",
            "group_by": "ticker",
        },
        price_adjustment_convention="Yahoo Finance Adj Close",
        missing_value_policy="drop rows with any NaN after coverage validation; then ffill/bfill",
    )

    environment_manifest = write_environment_manifest(
        output_path=reproducibility_dir / "environment_manifest.json",
        dependency_names=["numpy", "pandas", "scipy", "yfinance", "pytest", "PyYAML"],
    )

    # Canonical baseline runs (clean directories first)
    for path in (baseline_root, run1_dir, run2_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    run1_result, run1_download_calls = _run_canonical_baseline_once(
        root=root,
        config=config,
        output_dir=run1_dir,
        enforce_no_live_download=True,
    )
    run2_result, run2_download_calls = _run_canonical_baseline_once(
        root=root,
        config=config,
        output_dir=run2_dir,
        enforce_no_live_download=True,
    )

    # Publish canonical baseline as run 1 artifact set.
    for file_name in [
        "walk_forward_returns.csv",
        "walk_forward_weights.csv",
        "walk_forward_metrics.csv",
        "rebalance_history.csv",
    ]:
        shutil.copy2(run1_dir / file_name, baseline_root / file_name)

    returns_diff = max_keyed_return_difference(
        run1_dir / "walk_forward_returns.csv",
        run2_dir / "walk_forward_returns.csv",
    )
    weights_diff = max_keyed_weight_difference(
        run1_dir / "walk_forward_weights.csv",
        run2_dir / "walk_forward_weights.csv",
    )

    run1_returns = pd.read_csv(run1_dir / "walk_forward_returns.csv")
    run1_weights = pd.read_csv(run1_dir / "walk_forward_weights.csv")
    run1_rebalance = pd.read_csv(run1_dir / "rebalance_history.csv")

    date_col = "Date" if "Date" in run1_returns.columns else (
        "date" if "date" in run1_returns.columns else run1_returns.columns[0]
    )
    run1_dates = pd.to_datetime(run1_returns[date_col])

    failures = int((~run1_rebalance["optimization_success"].astype(bool)).sum())

    report = {
        "historical_m2_note": (
            "Historical Milestone 2 outputs generated from live market data; "
            "exact source-data snapshot was not preserved."
        ),
        "canonical_dataset": {
            "directory": str(canonical_dir.relative_to(root)),
            "manifest": manifest,
        },
        "environment_manifest": environment_manifest,
        "canonical_baseline": {
            "oos_first_date": run1_dates.min().strftime("%Y-%m-%d"),
            "oos_last_date": run1_dates.max().strftime("%Y-%m-%d"),
            "oos_observations": int(run1_dates.nunique()),
            "rebalance_count": int(pd.to_datetime(run1_rebalance["rebalance_date"]).nunique()),
            "return_columns": [c for c in run1_returns.columns if c != date_col],
            "weights_rows": int(len(run1_weights)),
            "optimizer_failures": failures,
        },
        "reproducibility_check": {
            "run_1_vs_run_2_max_return_diff": returns_diff,
            "run_1_vs_run_2_max_weight_diff": weights_diff,
            "run_1_live_download_calls": run1_download_calls,
            "run_2_live_download_calls": run2_download_calls,
        },
        "paths": {
            "run_1": str(run1_dir.relative_to(root)),
            "run_2": str(run2_dir.relative_to(root)),
            "canonical_baseline": str(baseline_root.relative_to(root)),
        },
    }

    report_path = reproducibility_dir / "reproducibility_report.json"
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)


if __name__ == "__main__":
    main()
