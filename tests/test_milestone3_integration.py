import numpy as np
import pandas as pd
import pytest

from src.backtesting.milestone3 import (
    execute_milestone3_rebalance,
    run_milestone3_walk_forward,
)
from src.backtesting.walk_forward import RebalanceInfo, generate_rebalance_dates


@pytest.fixture
def synthetic_prices():
    dates = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    prices_data = {
        "Asset_0": 100 * (1.00008) ** np.arange(len(dates)),
        "Asset_1": 100 * (1.00010) ** np.arange(len(dates)),
        "Asset_2": 100 * (1.00012) ** np.arange(len(dates)),
        "Asset_3": 100 * (1.00009) ** np.arange(len(dates)),
    }
    return pd.DataFrame(prices_data, index=dates)


@pytest.fixture
def synthetic_returns(synthetic_prices):
    return synthetic_prices.pct_change().dropna()


@pytest.fixture
def synthetic_tickers():
    return ["Asset_0", "Asset_1", "Asset_2", "Asset_3"]


def test_previous_weight_chaining_for_turnover_aware_strategy(synthetic_prices, synthetic_returns, synthetic_tickers):
    rebalance_infos = generate_rebalance_dates(
        synthetic_prices,
        lookback_years=2,
        rebalance_frequency="quarterly",
        holdout_start_date="2017-01-01",
    )
    if len(rebalance_infos) < 2:
        pytest.skip("Not enough rebalance dates for the chaining check")

    prev = {"Asset_0": 0.25, "Asset_1": 0.25, "Asset_2": 0.25, "Asset_3": 0.25}
    rebalance = rebalance_infos[0]
    first = execute_milestone3_rebalance(
        rebalance_info=rebalance,
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        strategy_name="Turnover-Aware Max Sharpe γ=0.10",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        previous_weights=prev,
    )
    assert first["previous_weights"] == prev
    assert first["turnover"] >= 0.0
    assert first["transaction_cost"] >= 0.0
    assert np.isclose(sum(first["weights"].values()), 1.0, atol=1e-8)

    second_prev = first["weights"]
    second = execute_milestone3_rebalance(
        rebalance_info=rebalance_infos[1],
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        strategy_name="Turnover-Aware Max Sharpe γ=0.10",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        previous_weights=second_prev,
    )
    assert second["previous_weights"] == second_prev
    assert np.isclose(sum(second["weights"].values()), 1.0, atol=1e-8)


def test_first_rebalance_convention_has_no_initial_transaction_cost(synthetic_prices, synthetic_returns, synthetic_tickers):
    rebalance_infos = generate_rebalance_dates(
        synthetic_prices,
        lookback_years=2,
        rebalance_frequency="quarterly",
        holdout_start_date="2017-01-01",
    )
    if len(rebalance_infos) == 0:
        pytest.skip("No rebalance dates")

    result = execute_milestone3_rebalance(
        rebalance_info=rebalance_infos[0],
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        strategy_name="Turnover-Aware Max Sharpe γ=0.10",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        previous_weights=None,
        transaction_cost_bps=10.0,
    )

    assert result["transaction_cost"] == 0.0
    assert result["turnover"] >= 0.0


def test_zero_bps_gross_equals_net(synthetic_prices, synthetic_returns, synthetic_tickers):
    rebalance_infos = generate_rebalance_dates(
        synthetic_prices,
        lookback_years=2,
        rebalance_frequency="quarterly",
        holdout_start_date="2017-01-01",
    )
    if len(rebalance_infos) < 2:
        pytest.skip("Need multiple rebalances")

    result = run_milestone3_walk_forward(
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        rebalance_infos=rebalance_infos[:2],
        strategies=["Turnover-Aware Max Sharpe γ=0.10"],
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        transaction_cost_bps=0.0,
    )

    gross = result["gross_returns"]["Turnover-Aware Max Sharpe γ=0.10"]
    net = result["net_returns"]["Turnover-Aware Max Sharpe γ=0.10"]
    np.testing.assert_allclose(gross.values, net.values, rtol=1e-12, atol=1e-12)


def test_positive_costs_reduce_terminal_wealth(synthetic_prices, synthetic_returns, synthetic_tickers):
    rebalance_infos = generate_rebalance_dates(
        synthetic_prices,
        lookback_years=2,
        rebalance_frequency="quarterly",
        holdout_start_date="2017-01-01",
    )
    if len(rebalance_infos) < 2:
        pytest.skip("Need multiple rebalances")

    zero_cost = run_milestone3_walk_forward(
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        rebalance_infos=rebalance_infos[:2],
        strategies=["Turnover-Aware Max Sharpe γ=0.10"],
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        transaction_cost_bps=0.0,
    )
    pos_cost = run_milestone3_walk_forward(
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        rebalance_infos=rebalance_infos[:2],
        strategies=["Turnover-Aware Max Sharpe γ=0.10"],
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        transaction_cost_bps=10.0,
    )

    gross_zero = (1.0 + zero_cost["gross_returns"]["Turnover-Aware Max Sharpe γ=0.10"]).cumprod()
    gross_pos = (1.0 + pos_cost["gross_returns"]["Turnover-Aware Max Sharpe γ=0.10"]).cumprod()
    net_pos = (1.0 + pos_cost["net_returns"]["Turnover-Aware Max Sharpe γ=0.10"]).cumprod()

    assert net_pos.iloc[-1] <= gross_pos.iloc[-1] + 1e-12
    assert gross_pos.iloc[-1] >= gross_zero.iloc[-1] - 1e-12


def test_milestone3_anti_lookahead_on_shrunk_sharpe(synthetic_prices, synthetic_returns, synthetic_tickers):
    rebalance_infos = generate_rebalance_dates(
        synthetic_prices,
        lookback_years=2,
        rebalance_frequency="quarterly",
        holdout_start_date="2017-01-01",
    )
    if len(rebalance_infos) == 0:
        pytest.skip("No rebalance dates")

    ref = execute_milestone3_rebalance(
        rebalance_info=rebalance_infos[0],
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        strategy_name="Shrunk Max Sharpe λ=0.50",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        previous_weights=None,
    )

    mutated = synthetic_returns.copy()
    mutated.loc[mutated.index > rebalance_infos[0].rebalance_date] = 0.5 * mutated.loc[mutated.index > rebalance_infos[0].rebalance_date]

    alt = execute_milestone3_rebalance(
        rebalance_info=rebalance_infos[0],
        prices_df=synthetic_prices,
        returns_df=mutated,
        tickers=synthetic_tickers,
        strategy_name="Shrunk Max Sharpe λ=0.50",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        previous_weights=None,
    )

    assert np.max(np.abs(np.array(list(ref["weights"].values())) - np.array(list(alt["weights"].values())))) < 1e-8


def test_milestone3_anti_lookahead_on_combined_strategy(synthetic_prices, synthetic_returns, synthetic_tickers):
    rebalance_infos = generate_rebalance_dates(
        synthetic_prices,
        lookback_years=2,
        rebalance_frequency="quarterly",
        holdout_start_date="2017-01-01",
    )
    if len(rebalance_infos) == 0:
        pytest.skip("No rebalance dates")

    ref = execute_milestone3_rebalance(
        rebalance_info=rebalance_infos[0],
        prices_df=synthetic_prices,
        returns_df=synthetic_returns,
        tickers=synthetic_tickers,
        strategy_name="Combined Robust Max Sharpe λ=0.50 γ=0.10",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        previous_weights=None,
    )

    mutated = synthetic_returns.copy()
    mutated.loc[mutated.index > rebalance_infos[0].rebalance_date] = 0.5 * mutated.loc[mutated.index > rebalance_infos[0].rebalance_date]

    alt = execute_milestone3_rebalance(
        rebalance_info=rebalance_infos[0],
        prices_df=synthetic_prices,
        returns_df=mutated,
        tickers=synthetic_tickers,
        strategy_name="Combined Robust Max Sharpe λ=0.50 γ=0.10",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
        previous_weights=None,
    )

    assert np.max(np.abs(np.array(list(ref["weights"].values())) - np.array(list(alt["weights"].values())))) < 1e-8
