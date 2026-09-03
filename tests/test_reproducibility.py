from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_analysis import run_milestone_2_backtest
from src.backtesting.engine import WalkForwardBacktest
from src.backtesting.walk_forward import generate_rebalance_dates
from src.data.market_data import (
    ReproducibilityError,
    compute_file_sha256,
    create_canonical_dataset_snapshot,
    fetch_and_prepare_market_data,
)
from src.reproducibility.checks import (
    max_keyed_return_difference,
    max_keyed_weight_difference,
)


def _make_prices(tickers: list[str], start: str = "2015-01-01", end: str = "2020-12-31") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.00008 + i * 0.00001
        data[ticker] = 100 * (1 + drift) ** np.arange(len(dates))
    return pd.DataFrame(data, index=dates)


def _build_weight_rows(backtest: WalkForwardBacktest) -> pd.DataFrame:
    rows = []
    for item in backtest.portfolio_weights_history:
        for asset, weight in item.weights.items():
            rows.append(
                {
                    "date": item.rebalance_date,
                    "strategy": item.strategy_name,
                    "asset": asset,
                    "weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def test_live_loader_behavior_unchanged(monkeypatch):
    tickers = ["AAA", "BBB", "CCC"]
    prices = _make_prices(tickers)

    called = {"n": 0}

    def fake_download(**kwargs):
        called["n"] += 1
        return prices

    monkeypatch.setattr("src.data.market_data.download_adjusted_prices", fake_download)

    out_prices, out_returns, out_cov, out_mu = fetch_and_prepare_market_data(
        tickers=tickers,
        start_date="2015-01-01",
        end_date="2015-12-31",
        save_output=False,
        data_mode="live",
    )

    assert called["n"] == 1
    assert not out_prices.empty
    assert not out_returns.empty
    assert out_cov.shape[0] == len(tickers)
    assert out_mu.shape[0] == len(tickers)


def test_canonical_loader_no_download(monkeypatch, tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    prices = _make_prices(tickers)
    returns = prices.pct_change().dropna()

    canonical_dir = tmp_path / "canonical"
    create_canonical_dataset_snapshot(
        prices=prices,
        returns=returns,
        output_dir=canonical_dir,
        dataset_name="test_snapshot",
        tickers=tickers,
        trading_days_per_year=252,
        source_provider="unit-test",
        download_settings={"interval": "1d"},
        price_adjustment_convention="Adj Close",
        missing_value_policy="dropna-any",
    )

    called = {"n": 0}

    def forbidden_download(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("Canonical mode should not call live download")

    monkeypatch.setattr("src.data.market_data.download_adjusted_prices", forbidden_download)

    out_prices, out_returns, out_cov, out_mu = fetch_and_prepare_market_data(
        tickers=tickers,
        start_date="2015-01-01",
        end_date="2020-12-31",
        save_output=False,
        data_mode="canonical",
        canonical_dir=canonical_dir,
    )

    assert called["n"] == 0
    pd.testing.assert_frame_equal(out_prices, prices, check_freq=False)
    pd.testing.assert_frame_equal(out_returns, returns, check_freq=False)
    assert out_cov.shape == (3, 3)
    assert out_mu.shape[0] == 3


def test_canonical_mode_missing_file_fails(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    prices = _make_prices(tickers)
    returns = prices.pct_change().dropna()

    canonical_dir = tmp_path / "canonical"
    create_canonical_dataset_snapshot(
        prices=prices,
        returns=returns,
        output_dir=canonical_dir,
        dataset_name="test_snapshot",
        tickers=tickers,
        trading_days_per_year=252,
        source_provider="unit-test",
        download_settings={"interval": "1d"},
        price_adjustment_convention="Adj Close",
        missing_value_policy="dropna-any",
    )

    (canonical_dir / "canonical_returns.csv").unlink()

    with pytest.raises(ReproducibilityError):
        fetch_and_prepare_market_data(
            tickers=tickers,
            start_date="2015-01-01",
            end_date="2020-12-31",
            save_output=False,
            data_mode="canonical",
            canonical_dir=canonical_dir,
        )


def test_canonical_mode_hash_mismatch_fails(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    prices = _make_prices(tickers)
    returns = prices.pct_change().dropna()

    canonical_dir = tmp_path / "canonical"
    create_canonical_dataset_snapshot(
        prices=prices,
        returns=returns,
        output_dir=canonical_dir,
        dataset_name="test_snapshot",
        tickers=tickers,
        trading_days_per_year=252,
        source_provider="unit-test",
        download_settings={"interval": "1d"},
        price_adjustment_convention="Adj Close",
        missing_value_policy="dropna-any",
    )

    tampered = pd.read_csv(canonical_dir / "canonical_returns.csv", index_col=0)
    tampered.iloc[0, 0] += 0.000001
    tampered.to_csv(canonical_dir / "canonical_returns.csv")

    with pytest.raises(ReproducibilityError):
        fetch_and_prepare_market_data(
            tickers=tickers,
            start_date="2015-01-01",
            end_date="2020-12-31",
            save_output=False,
            data_mode="canonical",
            canonical_dir=canonical_dir,
        )


def test_manifest_hashes_match_files(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    prices = _make_prices(tickers)
    returns = prices.pct_change().dropna()

    canonical_dir = tmp_path / "canonical"
    manifest = create_canonical_dataset_snapshot(
        prices=prices,
        returns=returns,
        output_dir=canonical_dir,
        dataset_name="test_snapshot",
        tickers=tickers,
        trading_days_per_year=252,
        source_provider="unit-test",
        download_settings={"interval": "1d"},
        price_adjustment_convention="Adj Close",
        missing_value_policy="dropna-any",
    )

    prices_hash = compute_file_sha256(canonical_dir / "canonical_prices.csv")
    returns_hash = compute_file_sha256(canonical_dir / "canonical_returns.csv")

    assert prices_hash == manifest["file_hashes"]["canonical_prices.csv"]
    assert returns_hash == manifest["file_hashes"]["canonical_returns.csv"]


def test_keyed_comparison_helpers(tmp_path):
    left_returns = pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03"],
            "Maximum Sharpe": [0.01, 0.02],
            "Minimum Variance": [0.005, 0.015],
        }
    )
    right_returns = pd.DataFrame(
        {
            "Date": ["2020-01-03", "2020-01-02"],
            "Maximum Sharpe": [0.02, 0.01],
            "Minimum Variance": [0.015, 0.005],
        }
    )

    left_weights = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31"],
            "strategy": ["Maximum Sharpe", "Maximum Sharpe"],
            "asset": ["AAA", "BBB"],
            "weight": [0.6, 0.4],
        }
    )
    right_weights = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31"],
            "strategy": ["Maximum Sharpe", "Maximum Sharpe"],
            "asset": ["BBB", "AAA"],
            "weight": [0.4, 0.6],
        }
    )

    left_returns_path = tmp_path / "left_returns.csv"
    right_returns_path = tmp_path / "right_returns.csv"
    left_weights_path = tmp_path / "left_weights.csv"
    right_weights_path = tmp_path / "right_weights.csv"

    left_returns.to_csv(left_returns_path, index=False)
    right_returns.to_csv(right_returns_path, index=False)
    left_weights.to_csv(left_weights_path, index=False)
    right_weights.to_csv(right_weights_path, index=False)

    assert max_keyed_return_difference(left_returns_path, right_returns_path) == 0.0
    assert max_keyed_weight_difference(left_weights_path, right_weights_path) == 0.0


def test_canonical_baseline_rerun_reproducibility(tmp_path):
    tickers = ["SPY", "QQQ", "IWM", "EFA"]
    prices = _make_prices(tickers)
    returns = prices.pct_change().dropna()

    rebalance_infos = generate_rebalance_dates(
        prices,
        lookback_years=2,
        rebalance_frequency="monthly",
        holdout_start_date="2018-01-01",
    )

    run1 = WalkForwardBacktest(
        prices_df=prices,
        returns_df=returns,
        tickers=tickers,
        rebalance_infos=rebalance_infos,
        strategies=["Equal Weight", "Minimum Variance", "Maximum Sharpe"],
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
    )
    run2 = WalkForwardBacktest(
        prices_df=prices,
        returns_df=returns,
        tickers=tickers,
        rebalance_infos=rebalance_infos,
        strategies=["Equal Weight", "Minimum Variance", "Maximum Sharpe"],
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
    )

    res1 = run1.run()
    res2 = run2.run()

    ret1 = pd.DataFrame(res1["returns_concatenated"])
    ret2 = pd.DataFrame(res2["returns_concatenated"])

    w1 = _build_weight_rows(run1)
    w2 = _build_weight_rows(run2)

    ret1_path = tmp_path / "run1_returns.csv"
    ret2_path = tmp_path / "run2_returns.csv"
    w1_path = tmp_path / "run1_weights.csv"
    w2_path = tmp_path / "run2_weights.csv"

    ret1.to_csv(ret1_path)
    ret2.to_csv(ret2_path)
    w1.to_csv(w1_path, index=False)
    w2.to_csv(w2_path, index=False)

    assert max_keyed_return_difference(ret1_path, ret2_path) <= 1e-12
    assert max_keyed_weight_difference(w1_path, w2_path) <= 1e-12


def test_benchmark_schema_contract(tmp_path):
    tickers = ["SPY", "QQQ", "IWM", "EFA"]
    prices = _make_prices(tickers)
    returns = prices.pct_change().dropna()

    result = run_milestone_2_backtest(
        root=tmp_path,
        tickers=tickers,
        prices=prices,
        returns=returns,
        trading_days_per_year=252,
        risk_free_rate=0.02,
        max_weight=0.30,
        benchmark_ticker="SPY",
        backtest_config={
            "enabled": True,
            "lookback_years": 2,
            "rebalance_frequency": "monthly",
            "holdout_start_date": "2018-01-01",
        },
        backtest_output_dir=tmp_path / "baseline",
    )

    returns_csv = pd.read_csv(tmp_path / "baseline" / "walk_forward_returns.csv")
    metrics_csv = pd.read_csv(tmp_path / "baseline" / "walk_forward_metrics.csv", index_col=0)

    assert "SPY" not in returns_csv.columns
    assert "SPY" in metrics_csv.index
    assert "SPY" in result["metrics_df"].index
