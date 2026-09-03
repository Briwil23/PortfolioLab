from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import execute_rebalance
from src.backtesting.milestone4 import execute_milestone4_rebalance
from src.backtesting.walk_forward import generate_rebalance_dates, normalize_asset_order
from src.data.market_data import fetch_and_prepare_market_data, load_canonical_market_data, load_config
from src.optimization.risk_based import (
    covariance_to_correlation,
    equal_risk_contribution_portfolio,
    inverse_volatility_portfolio,
    perturb_single_asset_volatility,
    reconstruct_covariance_from_correlation,
    risk_contribution_report,
    risk_contributions,
)
from src.reproducibility.checks import max_keyed_return_difference, max_keyed_weight_difference


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "config.yaml")


def _make_prices(columns: list[str], start: str = "2015-01-01", end: str = "2020-12-31") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    data = {}
    for idx, column in enumerate(columns):
        data[column] = 100.0 * (1.0 + 0.0001 + idx * 0.00002) ** np.arange(len(dates))
    return pd.DataFrame(data, index=dates)


def _canonical_inputs():
    prices, returns, _, _, _ = load_canonical_market_data(
        ROOT / "data" / "canonical",
        trading_days_per_year=int(CONFIG.get("trading_days_per_year", 252)),
    )
    rebalance_info = generate_rebalance_dates(
        prices,
        lookback_years=int(CONFIG["backtest"]["lookback_years"]),
        rebalance_frequency=str(CONFIG["backtest"]["rebalance_frequency"]),
        holdout_start_date=str(CONFIG["backtest"]["holdout_start_date"]),
    )[0]
    return prices, returns, rebalance_info


def test_inverse_volatility_analytical_weights():
    vol = pd.Series([0.20, 0.10, 0.40], index=["A", "B", "C"])
    result = inverse_volatility_portfolio(vol, max_weight=0.60, asset_labels=vol.index.tolist())
    expected = np.array([1 / 0.20, 1 / 0.10, 1 / 0.40], dtype=float)
    expected = expected / expected.sum()
    np.testing.assert_allclose(result["weights"], expected, atol=1e-12)


def test_inverse_volatility_cap_and_redistribute():
    vol = pd.Series([0.50, 0.20, 0.10, 0.10], index=["A", "B", "C", "D"])
    result = inverse_volatility_portfolio(vol, max_weight=0.30, asset_labels=vol.index.tolist())
    weights = pd.Series(result["weights_by_asset"])
    assert np.isclose(weights.sum(), 1.0, atol=1e-12)
    assert (weights <= 0.30 + 1e-12).all()
    assert np.isclose(weights["C"], 0.30, atol=1e-12)
    assert np.isclose(weights["D"], 0.30, atol=1e-12)


def test_inverse_volatility_infeasible_cap_fails():
    vol = pd.Series([0.20, 0.10, 0.40], index=["A", "B", "C"])
    with pytest.raises(ValueError, match="feasible"):
        inverse_volatility_portfolio(vol, max_weight=0.20, asset_labels=vol.index.tolist())


def test_erc_sum_weights_and_nonnegative():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.05]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    result = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    weights = result["weights"]
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)
    assert np.all(weights >= -1e-10)


def test_erc_respects_cap():
    cov = pd.DataFrame(
        [
            [0.04, 0.01, 0.00, 0.00],
            [0.01, 0.03, 0.01, 0.00],
            [0.00, 0.01, 0.05, 0.01],
            [0.00, 0.00, 0.01, 0.02],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    result = equal_risk_contribution_portfolio(cov, max_weight=0.30, asset_labels=cov.columns.tolist())
    assert np.all(result["weights"] <= 0.30 + 1e-10)


def test_erc_fallback_metadata_is_auditable(monkeypatch):
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00, 0.00], [0.01, 0.03, 0.01, 0.00], [0.00, 0.01, 0.05, 0.01], [0.00, 0.00, 0.01, 0.02]],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )

    class _FailedResult:
        success = False
        status = 7
        message = "forced failure"
        x = np.full(4, 0.25)

    monkeypatch.setattr("src.optimization.risk_based.minimize", lambda *args, **kwargs: _FailedResult())

    result = equal_risk_contribution_portfolio(cov, max_weight=0.30, asset_labels=cov.columns.tolist())
    assert result["solver_success"] is False
    assert result["fallback_used"] is True
    assert result["solver_message"]
    assert result["objective_value"] >= 0.0
    assert result["risk_contribution_dispersion"] >= 0.0


def test_erc_success_metadata_is_explicit():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00, 0.00], [0.01, 0.03, 0.01, 0.00], [0.00, 0.01, 0.05, 0.01], [0.00, 0.00, 0.01, 0.02]],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    result = equal_risk_contribution_portfolio(cov, max_weight=0.30, asset_labels=cov.columns.tolist())
    assert result["solver_success"] is True
    assert result["fallback_used"] is False
    assert result["solver_message"]
    assert result["objective_value"] >= 0.0
    assert result["risk_contribution_dispersion"] >= 0.0


def test_erc_synthetic_equal_risk_case():
    cov = pd.DataFrame(np.eye(4), index=list("ABCD"), columns=list("ABCD"))
    result = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    rc = risk_contributions(result["weights"], cov, asset_labels=cov.columns.tolist())
    np.testing.assert_allclose(rc["normalized_risk_contribution"].to_numpy(), np.full(4, 0.25), atol=1e-8)


def test_risk_contribution_identities():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.05]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    weights = np.array([0.3, 0.4, 0.3])
    rc = risk_contributions(weights, cov, asset_labels=cov.columns.tolist())
    report = risk_contribution_report(weights, cov, asset_labels=cov.columns.tolist())
    portfolio_vol = report["portfolio_volatility"]
    np.testing.assert_allclose(rc["absolute_risk_contribution"].sum(), portfolio_vol, atol=1e-10)
    np.testing.assert_allclose(rc["absolute_risk_contribution"].abs().sum(), portfolio_vol, atol=1e-10)
    np.testing.assert_allclose(rc["normalized_risk_contribution"].sum(), 1.0, atol=1e-10)


def test_erc_anti_lookahead_via_orchestration():
    prices, returns, rebalance_info = _canonical_inputs()
    tickers = list(prices.columns)
    ref = execute_milestone4_rebalance(
        rebalance_info=rebalance_info,
        prices_df=prices,
        returns_df=returns,
        tickers=tickers,
        strategy_name="Equal Risk Contribution",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
    )
    mutated = returns.copy()
    mutated.loc[mutated.index > rebalance_info.rebalance_date] = mutated.loc[mutated.index > rebalance_info.rebalance_date] * 50.0
    alt = execute_milestone4_rebalance(
        rebalance_info=rebalance_info,
        prices_df=prices,
        returns_df=mutated,
        tickers=tickers,
        strategy_name="Equal Risk Contribution",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
    )
    diff = np.max(np.abs(np.array(list(ref["weights"].values())) - np.array(list(alt["weights"].values()))))
    assert diff <= 1e-10


def test_permutation_invariance_inverse_volatility():
    vol = pd.Series([0.20, 0.10, 0.40], index=["A", "B", "C"])
    base = inverse_volatility_portfolio(vol, max_weight=0.60, asset_labels=vol.index.tolist())
    perm = inverse_volatility_portfolio(vol.reindex(["C", "A", "B"]), max_weight=0.60, asset_labels=["C", "A", "B"])
    base_map = pd.Series(base["weights_by_asset"])
    perm_map = pd.Series(perm["weights_by_asset"])
    np.testing.assert_allclose(base_map.sort_index().values, perm_map.sort_index().values, atol=1e-12)


def test_permutation_invariance_erc():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.05]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    base = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    perm_cov = cov.reindex(index=["C", "A", "B"], columns=["C", "A", "B"])
    perm = equal_risk_contribution_portfolio(perm_cov, max_weight=0.60, asset_labels=perm_cov.columns.tolist())
    np.testing.assert_allclose(
        pd.Series(base["weights_by_asset"]).sort_index().values,
        pd.Series(perm["weights_by_asset"]).sort_index().values,
        atol=1e-10,
    )


def test_expected_return_independence():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.05]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    vol = pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=cov.columns)
    inv_1 = inverse_volatility_portfolio(vol, max_weight=0.60, asset_labels=vol.index.tolist())
    inv_2 = inverse_volatility_portfolio(vol, max_weight=0.60, asset_labels=vol.index.tolist())
    erc_1 = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    erc_2 = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    np.testing.assert_allclose(inv_1["weights"], inv_2["weights"], atol=1e-12)
    np.testing.assert_allclose(erc_1["weights"], erc_2["weights"], atol=1e-12)


def test_covariance_perturbation_construction():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.05]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    vol, corr = covariance_to_correlation(cov, asset_labels=cov.columns.tolist())
    reconstructed = reconstruct_covariance_from_correlation(vol, corr, asset_labels=cov.columns.tolist())
    pd.testing.assert_frame_equal(reconstructed, cov)
    perturbed = perturb_single_asset_volatility(cov, "B", 1.01, asset_labels=cov.columns.tolist())
    assert perturbed.loc["B", "B"] > cov.loc["B", "B"]


def test_l1_sensitivity_calculation():
    baseline = np.array([0.2, 0.3, 0.5])
    perturbed = np.array([0.21, 0.29, 0.50])
    l1 = np.sum(np.abs(perturbed - baseline))
    assert np.isclose(l1, 0.02)
    assert np.isclose(0.5 * l1, 0.01)


def test_m4_anti_lookahead():
    prices, returns, rebalance_info = _canonical_inputs()
    ref_inv = execute_milestone4_rebalance(
        rebalance_info=rebalance_info,
        prices_df=prices,
        returns_df=returns,
        tickers=list(prices.columns),
        strategy_name="Inverse Volatility",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
    )
    mutated = returns.copy()
    mutated.loc[mutated.index > rebalance_info.rebalance_date] = mutated.loc[mutated.index > rebalance_info.rebalance_date] * 2.0
    alt_inv = execute_milestone4_rebalance(
        rebalance_info=rebalance_info,
        prices_df=prices,
        returns_df=mutated,
        tickers=list(prices.columns),
        strategy_name="Inverse Volatility",
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
    )
    np.testing.assert_allclose(
        np.array(list(ref_inv["weights"].values())),
        np.array(list(alt_inv["weights"].values())),
        atol=1e-12,
    )


def test_asset_label_mismatch_behavior():
    vol = pd.Series([0.20, 0.10], index=["A", "B"])
    with pytest.raises(ValueError, match="label mismatch"):
        inverse_volatility_portfolio(vol, max_weight=0.60, asset_labels=["A", "C"])
    cov = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="label mismatch"):
        equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=["A", "C"])
    duplicate = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["A", "A"], columns=["A", "A"])
    with pytest.raises(ValueError, match="Duplicate"):
        equal_risk_contribution_portfolio(duplicate, max_weight=0.60, asset_labels=["A", "B"])


def test_deterministic_repeated_solver_result():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.05]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    first = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    second = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    np.testing.assert_allclose(first["weights"], second["weights"], atol=1e-12)


def test_solver_success_and_pathological_failure():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.01], [0.00, 0.01, 0.05]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    result = equal_risk_contribution_portfolio(cov, max_weight=0.60, asset_labels=cov.columns.tolist())
    assert result["success"]
    assert result["solver_success"]
    assert not result["fallback_used"]
    bad_cov = pd.DataFrame(np.full((3, 3), np.nan), index=["A", "B", "C"], columns=["A", "B", "C"])
    with pytest.raises(ValueError):
        equal_risk_contribution_portfolio(bad_cov, max_weight=0.60, asset_labels=bad_cov.columns.tolist())


def test_baseline_preservation_and_transaction_costs():
    prices, returns, rebalance_info = _canonical_inputs()
    tickers = list(prices.columns)
    kwargs = dict(
        rebalance_info=rebalance_info,
        prices_df=prices,
        returns_df=returns,
        tickers=tickers,
        risk_free_rate=0.02,
        max_weight=0.30,
        trading_days_per_year=252,
    )
    baseline = execute_rebalance(strategy_name="Maximum Sharpe", previous_weights=None, **kwargs)
    m4 = execute_milestone4_rebalance(strategy_name="Maximum Sharpe", previous_weights=None, **kwargs)
    np.testing.assert_allclose(
        np.array(list(baseline[3].values())),
        np.array(list(m4["weights"].values())),
        atol=1e-6,
    )
    assert m4["transaction_cost"] == 0.0


def test_canonical_no_live_data_behavior(monkeypatch, tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    prices = _make_prices(tickers)
    returns = prices.pct_change().dropna()
    canonical_dir = tmp_path / "canonical"
    from src.data.market_data import create_canonical_dataset_snapshot

    create_canonical_dataset_snapshot(
        prices=prices,
        returns=returns,
        output_dir=canonical_dir,
        dataset_name="unit-test",
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
    fetch_and_prepare_market_data(
        tickers=tickers,
        start_date="2015-01-01",
        end_date="2020-12-31",
        save_output=False,
        data_mode="canonical",
        canonical_dir=canonical_dir,
    )
    assert called["n"] == 0
