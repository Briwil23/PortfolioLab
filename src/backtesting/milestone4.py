"""Milestone 4 canonical orchestration for risk-based portfolio construction."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from src.backtesting.engine import calculate_portfolio_returns, minimum_variance_portfolio, maximum_sharpe_portfolio
from src.backtesting.milestone3 import execute_milestone3_rebalance
from src.backtesting.walk_forward import (
    RebalanceInfo,
    extract_holding_data,
    extract_training_data,
    normalize_asset_order,
)
from src.data.market_data import calculate_annualized_covariance, calculate_annualized_expected_returns
from src.optimization.mean_variance import compute_equal_weight_portfolio
from src.optimization.risk_based import (
    equal_risk_contribution_portfolio,
    inverse_volatility_portfolio,
)

logger = logging.getLogger(__name__)

MILESTONE4_STRATEGY_NAMES = [
    "SPY",
    "Equal Weight",
    "Minimum Variance",
    "Maximum Sharpe",
    "Combined Robust Max Sharpe λ=0.50 γ=0.10",
    "Inverse Volatility",
    "Equal Risk Contribution",
]


def _weights_to_dict(weights_array: np.ndarray, tickers: list[str]) -> dict[str, float]:
    return {ticker: float(weight) for ticker, weight in zip(tickers, weights_array)}


def _minimum_variance_weights(expected_returns: pd.Series, covariance: pd.DataFrame, max_weight: float) -> dict:
    result = minimum_variance_portfolio(expected_returns.to_numpy(), covariance.to_numpy(), max_weight=max_weight)
    return {"weights": _weights_to_dict(result["weights"], expected_returns.index.tolist()), "success": bool(result.get("success", False)), "message": result.get("message", "")}


def _maximum_sharpe_weights(expected_returns: pd.Series, covariance: pd.DataFrame, risk_free_rate: float, max_weight: float) -> dict:
    result = maximum_sharpe_portfolio(expected_returns.to_numpy(), covariance.to_numpy(), risk_free_rate=risk_free_rate, max_weight=max_weight)
    return {"weights": _weights_to_dict(result["weights"], expected_returns.index.tolist()), "success": bool(result.get("success", False)), "message": result.get("message", "")}


def _combined_robust_weights(
    rebalance_info: RebalanceInfo,
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    tickers: list[str],
    risk_free_rate: float,
    max_weight: float,
    trading_days_per_year: int,
    previous_weights: dict[str, float] | None,
) -> dict:
    result = execute_milestone3_rebalance(
        rebalance_info=rebalance_info,
        prices_df=prices_df,
        returns_df=returns_df,
        tickers=tickers,
        strategy_name="Combined Robust Max Sharpe λ=0.50 γ=0.10",
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        trading_days_per_year=trading_days_per_year,
        previous_weights=previous_weights,
    )
    return {"weights": result["weights"], "success": bool(result.get("optimization_success", False)), "message": result.get("optimization_message", "")}


def _risk_only_weights(
    strategy_name: str,
    covariance: pd.DataFrame,
    tickers: list[str],
    max_weight: float,
) -> dict:
    if strategy_name == "Inverse Volatility":
        volatility = pd.Series(np.sqrt(np.diag(covariance.to_numpy())), index=tickers, name="annualized_volatility")
        result = inverse_volatility_portfolio(volatility, max_weight=max_weight, asset_labels=tickers)
        return {
            "weights": result["weights_by_asset"],
            "success": bool(result.get("success", False)),
            "solver_success": bool(result.get("success", False)),
            "fallback_used": False,
            "message": result.get("message", ""),
            "solver_message": result.get("message", ""),
            "objective_value": np.nan,
            "risk_contribution_report": result.get("risk_contribution_report"),
        }

    if strategy_name == "Equal Risk Contribution":
        result = equal_risk_contribution_portfolio(covariance, max_weight=max_weight, asset_labels=tickers)
        return {
            "weights": result["weights_by_asset"],
            "success": bool(result.get("success", False)),
            "solver_success": bool(result.get("solver_success", result.get("success", False))),
            "fallback_used": bool(result.get("fallback_used", False)),
            "message": result.get("message", ""),
            "solver_message": result.get("solver_message", result.get("message", "")),
            "objective_value": float(result.get("objective_value", np.nan)),
            "risk_contribution_report": result.get("risk_contribution_report"),
            "risk_contribution_dispersion": float(result.get("risk_contribution_dispersion", np.nan)),
        }

    raise ValueError(f"Unsupported M4 risk-only strategy: {strategy_name}")


def execute_milestone4_rebalance(
    rebalance_info: RebalanceInfo,
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    tickers: list[str],
    strategy_name: str,
    risk_free_rate: float,
    max_weight: float,
    trading_days_per_year: int,
    previous_weights: dict[str, float] | None = None,
    transaction_cost_bps: float = 0.0,
) -> dict:
    """Execute one Milestone 4 rebalance using the shared walk-forward framework."""
    training_prices, training_returns = extract_training_data(prices_df, returns_df, rebalance_info)
    holding_returns = extract_holding_data(returns_df, rebalance_info)

    training_returns = normalize_asset_order(training_returns, tickers)
    holding_returns = normalize_asset_order(holding_returns, tickers)

    expected_returns = calculate_annualized_expected_returns(training_returns, trading_days_per_year=trading_days_per_year)
    covariance = calculate_annualized_covariance(training_returns, trading_days_per_year=trading_days_per_year)
    expected_returns = expected_returns.loc[tickers]
    covariance = covariance.loc[tickers, tickers]

    if previous_weights is not None:
        previous_weights = {ticker: float(previous_weights.get(ticker, 0.0)) for ticker in tickers}

    if strategy_name == "Equal Weight":
        weights_array = compute_equal_weight_portfolio(len(tickers))
        result = {"weights": _weights_to_dict(weights_array, tickers), "success": True, "solver_success": True, "fallback_used": False, "message": "Equal weight portfolio", "solver_message": "Equal weight portfolio", "objective_value": np.nan}
    elif strategy_name == "Minimum Variance":
        result = _minimum_variance_weights(expected_returns, covariance, max_weight)
    elif strategy_name == "Maximum Sharpe":
        result = _maximum_sharpe_weights(expected_returns, covariance, risk_free_rate, max_weight)
    elif strategy_name == "SPY":
        result = {"weights": {ticker: 1.0 if idx == 0 else 0.0 for idx, ticker in enumerate(tickers)}, "success": True, "solver_success": True, "fallback_used": False, "message": "SPY benchmark", "solver_message": "SPY benchmark", "objective_value": np.nan}
    elif strategy_name == "Combined Robust Max Sharpe λ=0.50 γ=0.10":
        result = _combined_robust_weights(
            rebalance_info=rebalance_info,
            prices_df=prices_df,
            returns_df=returns_df,
            tickers=tickers,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            trading_days_per_year=trading_days_per_year,
            previous_weights=previous_weights,
        )
    elif strategy_name in {"Inverse Volatility", "Equal Risk Contribution"}:
        result = _risk_only_weights(strategy_name, covariance, tickers, max_weight)
    else:
        raise ValueError(f"Unsupported M4 strategy: {strategy_name}")

    weights = result["weights"]
    if isinstance(weights, np.ndarray):
        weights = {ticker: float(weight) for ticker, weight in zip(tickers, weights)}

    if previous_weights is None:
        turnover = 0.0
        transaction_cost = 0.0
    else:
        turnover = float(np.sum(np.abs(np.array([weights[t] for t in tickers]) - np.array([previous_weights.get(t, 0.0) for t in tickers]))) / 2.0)
        transaction_cost = float(turnover * (float(transaction_cost_bps) / 10_000.0))

    gross_weighted = calculate_portfolio_returns(holding_returns[tickers], weights)
    net_return_series = gross_weighted.copy()
    if previous_weights is not None and len(net_return_series) > 0:
        net_return_series.iloc[0] = net_return_series.iloc[0] - transaction_cost

    return {
        "rebalance_date": rebalance_info.rebalance_date,
        "training_start": rebalance_info.training_start,
        "training_end": rebalance_info.training_end,
        "holding_start": rebalance_info.holding_start,
        "holding_end": rebalance_info.holding_end,
        "strategy_name": strategy_name,
        "weights": weights,
        "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        "turnover": turnover,
        "transaction_cost_bps": float(transaction_cost_bps),
        "transaction_cost": transaction_cost,
        "gross_return": float(gross_weighted.mean()) if len(gross_weighted) else 0.0,
        "net_return": float(net_return_series.mean()) if len(net_return_series) else 0.0,
        "gross_return_series": gross_weighted,
        "net_return_series": net_return_series,
        "optimization_success": bool(result.get("success", False)),
        "optimization_message": result.get("message", ""),
        "solver_success": bool(result.get("solver_success", result.get("success", False))),
        "fallback_used": bool(result.get("fallback_used", False)),
        "solver_message": result.get("solver_message", result.get("message", "")),
        "objective_value": result.get("objective_value", np.nan),
        "risk_contribution_report": result.get("risk_contribution_report"),
        "risk_contribution_dispersion": result.get("risk_contribution_dispersion"),
    }


def run_milestone4_walk_forward(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    tickers: list[str],
    rebalance_infos: list[RebalanceInfo],
    strategies: list[str] | None = None,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.30,
    trading_days_per_year: int = 252,
    transaction_cost_bps: float | list[float] = 0.0,
) -> dict:
    """Run the Milestone 4 strategies across the shared walk-forward schedule."""
    if strategies is None:
        strategies = MILESTONE4_STRATEGY_NAMES

    if isinstance(transaction_cost_bps, (int, float)):
        cost_levels = [float(transaction_cost_bps)]
    else:
        cost_levels = [float(x) for x in transaction_cost_bps]

    results = {
        "gross_returns": defaultdict(dict),
        "net_returns": defaultdict(dict),
        "weights_history": [],
        "rebalance_history": [],
        "turnover_history": [],
        "transaction_cost_history": [],
        "solver_history": [],
        "strategy_previous_weights": {strategy: None for strategy in strategies},
    }

    for rebalance_info in rebalance_infos:
        for strategy in strategies:
            previous_weights = results["strategy_previous_weights"][strategy]
            for cost_bps in cost_levels:
                result = execute_milestone4_rebalance(
                    rebalance_info=rebalance_info,
                    prices_df=prices_df,
                    returns_df=returns_df,
                    tickers=tickers,
                    strategy_name=strategy,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_weight,
                    trading_days_per_year=trading_days_per_year,
                    previous_weights=previous_weights,
                    transaction_cost_bps=cost_bps,
                )

                results["weights_history"].append(
                    {
                        "date": rebalance_info.rebalance_date,
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        **{f"weight_{ticker}": result["weights"].get(ticker, 0.0) for ticker in tickers},
                    }
                )
                results["rebalance_history"].append(
                    {
                        "date": rebalance_info.rebalance_date,
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "turnover": result["turnover"],
                        "transaction_cost": result["transaction_cost"],
                        "optimization_success": result["optimization_success"],
                    }
                )
                results["turnover_history"].append(
                    {
                        "date": rebalance_info.rebalance_date,
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "turnover": result["turnover"],
                    }
                )
                results["transaction_cost_history"].append(
                    {
                        "date": rebalance_info.rebalance_date,
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "transaction_cost": result["transaction_cost"],
                    }
                )
                results["solver_history"].append(
                    {
                        "date": rebalance_info.rebalance_date,
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "solver_success": result["solver_success"],
                        "fallback_used": result["fallback_used"],
                        "solver_message": result["solver_message"],
                        "objective_value": result["objective_value"],
                        "risk_contribution_dispersion": result.get("risk_contribution_dispersion"),
                    }
                )

                if strategy not in results["gross_returns"]:
                    results["gross_returns"][strategy] = []
                if strategy not in results["net_returns"]:
                    results["net_returns"][strategy] = []
                results["gross_returns"][strategy].append(result["gross_return_series"])
                results["net_returns"][strategy].append(result["net_return_series"])

            last_weight_row = next(
                (
                    event
                    for event in reversed(results["weights_history"])
                    if event["strategy"] == strategy and event["date"] == rebalance_info.rebalance_date
                ),
                None,
            )
            if last_weight_row is not None:
                results["strategy_previous_weights"][strategy] = {ticker: last_weight_row[f"weight_{ticker}"] for ticker in tickers}

    for strategy in strategies:
        gross_series_list = results["gross_returns"].get(strategy, [])
        net_series_list = results["net_returns"].get(strategy, [])
        results["gross_returns"][strategy] = pd.concat(gross_series_list, axis=0) if gross_series_list else pd.Series(dtype=float)
        results["net_returns"][strategy] = pd.concat(net_series_list, axis=0) if net_series_list else pd.Series(dtype=float)

    return {
        "gross_returns": {k: v for k, v in results["gross_returns"].items() if isinstance(v, pd.Series)},
        "net_returns": {k: v for k, v in results["net_returns"].items() if isinstance(v, pd.Series)},
        "weights_history": results["weights_history"],
        "rebalance_history": results["rebalance_history"],
        "turnover_history": results["turnover_history"],
        "transaction_cost_history": results["transaction_cost_history"],
        "solver_history": results["solver_history"],
    }