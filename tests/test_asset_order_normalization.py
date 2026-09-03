from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import execute_rebalance
from src.backtesting.walk_forward import generate_rebalance_dates
from src.backtesting.walk_forward import (
    RebalanceInfo,
    normalize_asset_order,
)
from src.data.market_data import load_canonical_market_data, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "config.yaml")


def _make_simple_prices(columns: list[str], start: str = "2015-01-01", end: str = "2018-12-31") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    data = {}
    for idx, column in enumerate(columns):
        data[column] = 100.0 * (1.0 + 0.0001 + idx * 0.00001) ** np.arange(len(dates))
    return pd.DataFrame(data, index=dates)


def test_normalize_asset_order_reorders_columns():
    frame = pd.DataFrame(
        {
            "B": [1.0, 2.0],
            "A": [3.0, 4.0],
            "C": [5.0, 6.0],
        }
    )

    normalized = normalize_asset_order(frame, ["A", "B", "C"])

    assert list(normalized.columns) == ["A", "B", "C"]
    pd.testing.assert_frame_equal(normalized, frame.loc[:, ["A", "B", "C"]])


def test_normalize_asset_order_missing_extra_duplicate_failures():
    missing = pd.DataFrame({"A": [1.0], "B": [2.0]})
    with pytest.raises(ValueError, match="Missing assets"):
        normalize_asset_order(missing, ["A", "B", "C"])

    extra = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]})
    with pytest.raises(ValueError, match="Extra assets"):
        normalize_asset_order(extra, ["A", "B", "C"])

    duplicate = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["A", "A", "B"])
    with pytest.raises(ValueError, match="duplicates"):
        normalize_asset_order(duplicate, ["A", "B"])


def test_classical_pipeline_is_permutation_invariant():
    prices, returns, _, _, _ = load_canonical_market_data(ROOT / "data" / "canonical", trading_days_per_year=int(CONFIG.get("trading_days_per_year", 252)))
    rebalance_info = generate_rebalance_dates(
        prices,
        lookback_years=int(CONFIG["backtest"]["lookback_years"]),
        rebalance_frequency=str(CONFIG["backtest"]["rebalance_frequency"]),
        holdout_start_date=str(CONFIG["backtest"]["holdout_start_date"]),
    )[0]

    canonical_order = list(prices.columns)
    permuted_order = list(reversed(canonical_order))
    permuted_returns = returns.loc[:, permuted_order]
    permuted_returns_2 = returns.loc[:, canonical_order[3:] + canonical_order[:3]]

    kwargs = dict(
        prices_df=prices,
        tickers=canonical_order,
        rebalance_info=rebalance_info,
        risk_free_rate=float(CONFIG.get("risk_free_rate", 0.02)),
        max_weight=float(CONFIG.get("max_weight", 0.30)),
        trading_days_per_year=int(CONFIG.get("trading_days_per_year", 252)),
    )

    base_min = execute_rebalance(strategy_name="Minimum Variance", returns_df=permuted_returns, previous_weights=None, **kwargs)
    alt_min = execute_rebalance(strategy_name="Minimum Variance", returns_df=permuted_returns_2, previous_weights=None, **kwargs)
    base_sharpe = execute_rebalance(strategy_name="Maximum Sharpe", returns_df=permuted_returns, previous_weights=None, **kwargs)
    alt_sharpe = execute_rebalance(strategy_name="Maximum Sharpe", returns_df=permuted_returns_2, previous_weights=None, **kwargs)

    base_min_weights = np.array([base_min[3][ticker] for ticker in canonical_order])
    alt_min_weights = np.array([alt_min[3][ticker] for ticker in canonical_order])
    base_sharpe_weights = np.array([base_sharpe[3][ticker] for ticker in canonical_order])
    alt_sharpe_weights = np.array([alt_sharpe[3][ticker] for ticker in canonical_order])

    assert np.max(np.abs(base_min_weights - alt_min_weights)) < 1e-12
    assert np.max(np.abs(base_sharpe_weights - alt_sharpe_weights)) < 1e-12

    base_min_returns = base_min[2] @ base_min_weights
    alt_min_returns = alt_min[2] @ alt_min_weights
    base_sharpe_returns = base_sharpe[2] @ base_sharpe_weights
    alt_sharpe_returns = alt_sharpe[2] @ alt_sharpe_weights

    assert np.max(np.abs(base_min_returns - alt_min_returns)) < 1e-12
    assert np.max(np.abs(base_sharpe_returns - alt_sharpe_returns)) < 1e-12


def test_classical_pipeline_requires_exact_asset_set():
    tickers = ["AAA", "BBB", "CCC"]
    prices = _make_simple_prices(tickers)
    returns = prices.pct_change().dropna()

    with pytest.raises(ValueError, match="Asset-order mismatch"):
        normalize_asset_order(returns[["AAA", "BBB"]], tickers)

    with pytest.raises(ValueError, match="Asset-order mismatch"):
        normalize_asset_order(returns.assign(DDD=returns["AAA"]), tickers)

    duplicate = pd.DataFrame(np.column_stack([returns["AAA"], returns["AAA"], returns["BBB"]]), columns=["AAA", "AAA", "BBB"], index=returns.index)
    with pytest.raises(ValueError, match="duplicates"):
        normalize_asset_order(duplicate, tickers)
