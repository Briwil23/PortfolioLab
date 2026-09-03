"""Milestone 3 walk-forward integration for robust portfolio research strategies."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.engine import calculate_portfolio_returns
from src.backtesting.walk_forward import (
    RebalanceInfo,
    extract_holding_data,
    extract_training_data,
    normalize_asset_order,
)
from src.data.market_data import calculate_annualized_covariance, calculate_annualized_expected_returns
from src.optimization.mean_variance import compute_equal_weight_portfolio, maximum_sharpe_portfolio
from src.optimization.robust import (
    combined_robust_turnover_aware_maximum_sharpe_portfolio,
    robust_maximum_sharpe_portfolio,
    turnover_aware_maximum_sharpe_portfolio,
)

logger = logging.getLogger(__name__)

MILESTONE3_STRATEGY_NAMES = [
    "Maximum Sharpe",
    "Equal Weight",
    "Minimum Variance",
    "SPY",
    "Shrunk Max Sharpe λ=0.25",
    "Shrunk Max Sharpe λ=0.50",
    "Shrunk Max Sharpe λ=0.75",
    "Turnover-Aware Max Sharpe γ=0.05",
    "Turnover-Aware Max Sharpe γ=0.10",
    "Turnover-Aware Max Sharpe γ=0.25",
    "Combined Robust Max Sharpe λ=0.50 γ=0.10",
]


def _weights_to_dict(weights_array: np.ndarray, tickers: list[str]) -> dict[str, float]:
    return {ticker: float(weight) for ticker, weight in zip(tickers, weights_array)}


def _get_strategy_config(strategy_name: str) -> dict:
    if strategy_name == "Maximum Sharpe":
        return {"kind": "classical_max_sharpe"}
    if strategy_name == "Equal Weight":
        return {"kind": "equal_weight"}
    if strategy_name == "Minimum Variance":
        return {"kind": "minimum_variance"}
    if strategy_name == "SPY":
        return {"kind": "spy"}
    if strategy_name == "Shrunk Max Sharpe λ=0.25":
        return {"kind": "shrunk_max_sharpe", "lambda_value": 0.25}
    if strategy_name == "Shrunk Max Sharpe λ=0.50":
        return {"kind": "shrunk_max_sharpe", "lambda_value": 0.50}
    if strategy_name == "Shrunk Max Sharpe λ=0.75":
        return {"kind": "shrunk_max_sharpe", "lambda_value": 0.75}
    if strategy_name == "Turnover-Aware Max Sharpe γ=0.05":
        return {"kind": "turnover_aware", "gamma": 0.05}
    if strategy_name == "Turnover-Aware Max Sharpe γ=0.10":
        return {"kind": "turnover_aware", "gamma": 0.10}
    if strategy_name == "Turnover-Aware Max Sharpe γ=0.25":
        return {"kind": "turnover_aware", "gamma": 0.25}
    if strategy_name == "Combined Robust Max Sharpe λ=0.50 γ=0.10":
        return {"kind": "combined_robust_turnover_aware", "lambda_value": 0.50, "gamma": 0.10}
    raise ValueError(f"Unknown Milestone 3 strategy: {strategy_name}")


def _compute_robust_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    strategy_name: str,
    risk_free_rate: float,
    max_weight: float,
    previous_weights: dict[str, float] | None,
) -> dict:
    config = _get_strategy_config(strategy_name)
    kind = config["kind"]

    if kind == "classical_max_sharpe":
        result = maximum_sharpe_portfolio(
            expected_returns.to_numpy(),
            covariance.to_numpy(),
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
        )
        return {
            "weights": _weights_to_dict(result["weights"], expected_returns.index.tolist()),
            "turnover": 0.0,
            "success": bool(result.get("success", False)),
            "message": result.get("message", ""),
            "strategy_name": strategy_name,
            "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        }

    if kind == "equal_weight":
        weights_array = compute_equal_weight_portfolio(len(expected_returns))
        return {
            "weights": _weights_to_dict(weights_array, expected_returns.index.tolist()),
            "turnover": 0.0,
            "success": True,
            "message": "Equal weight portfolio",
            "strategy_name": strategy_name,
            "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        }

    if kind == "minimum_variance":
        from src.optimization.mean_variance import minimum_variance_portfolio

        result = minimum_variance_portfolio(
            expected_returns.to_numpy(),
            covariance.to_numpy(),
            max_weight=max_weight,
        )
        return {
            "weights": _weights_to_dict(result["weights"], expected_returns.index.tolist()),
            "turnover": 0.0,
            "success": bool(result.get("success", False)),
            "message": result.get("message", ""),
            "strategy_name": strategy_name,
            "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        }

    if kind == "spy":
        weights = {ticker: 1.0 if idx == 0 else 0.0 for idx, ticker in enumerate(expected_returns.index.tolist())}
        return {
            "weights": weights,
            "turnover": 0.0,
            "success": True,
            "message": "SPY benchmark",
            "strategy_name": strategy_name,
            "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        }

    if kind == "shrunk_max_sharpe":
        lambda_value = float(config["lambda_value"])
        result = robust_maximum_sharpe_portfolio(
            expected_returns.to_numpy(),
            covariance.to_numpy(),
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            lambda_value=lambda_value,
            target_type="grand_mean",
        )
        return {
            "weights": _weights_to_dict(result["weights"], expected_returns.index.tolist()),
            "turnover": 0.0,
            "success": bool(result.get("success", False)),
            "message": result.get("message", ""),
            "strategy_name": strategy_name,
            "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        }

    if kind == "turnover_aware":
        gamma = float(config["gamma"])
        prev = None if previous_weights is None else np.asarray([previous_weights[ticker] for ticker in expected_returns.index.tolist()], dtype=float)
        result = turnover_aware_maximum_sharpe_portfolio(
            expected_returns.to_numpy(),
            covariance.to_numpy(),
            previous_weights=prev,
            gamma=gamma,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
        )
        return {
            "weights": _weights_to_dict(result["weights"], expected_returns.index.tolist()),
            "turnover": float(result.get("turnover", 0.0)),
            "success": bool(result.get("success", False)),
            "message": result.get("message", ""),
            "strategy_name": strategy_name,
            "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        }

    if kind == "combined_robust_turnover_aware":
        gamma = float(config["gamma"])
        lambda_value = float(config["lambda_value"])
        prev = None if previous_weights is None else np.asarray([previous_weights[ticker] for ticker in expected_returns.index.tolist()], dtype=float)
        result = combined_robust_turnover_aware_maximum_sharpe_portfolio(
            expected_returns.to_numpy(),
            covariance.to_numpy(),
            previous_weights=prev,
            gamma=gamma,
            lambda_value=lambda_value,
            target_type="grand_mean",
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
        )
        return {
            "weights": _weights_to_dict(result["weights"], expected_returns.index.tolist()),
            "turnover": float(result.get("turnover", 0.0)),
            "success": bool(result.get("success", False)),
            "message": result.get("message", ""),
            "strategy_name": strategy_name,
            "previous_weights": previous_weights.copy() if previous_weights is not None else None,
        }

    raise ValueError(f"Unsupported strategy kind: {kind}")


def execute_milestone3_rebalance(
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
    """Execute one Milestone 3 rebalance under strict anti-lookahead controls."""
    training_prices, training_returns = extract_training_data(prices_df, returns_df, rebalance_info)
    holding_returns = extract_holding_data(returns_df, rebalance_info)

    training_returns = normalize_asset_order(training_returns, tickers)
    holding_returns = normalize_asset_order(holding_returns, tickers)

    expected_returns = calculate_annualized_expected_returns(training_returns, trading_days_per_year=trading_days_per_year)
    covariance = calculate_annualized_covariance(training_returns, trading_days_per_year=trading_days_per_year)

    ordered_tickers = list(tickers)
    expected_returns = expected_returns.loc[ordered_tickers]
    covariance = covariance.loc[ordered_tickers, ordered_tickers]

    if previous_weights is not None:
        previous_weights = {ticker: float(previous_weights.get(ticker, 0.0)) for ticker in ordered_tickers}
    else:
        previous_weights = None

    result = _compute_robust_weights(
        expected_returns=expected_returns,
        covariance=covariance,
        strategy_name=strategy_name,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        previous_weights=previous_weights,
    )
    weights = result["weights"]
    turnover = float(result["turnover"])
    if previous_weights is None:
        turnover = 0.0
        transaction_cost = 0.0
    else:
        turnover = float(np.sum(np.abs(np.array([weights[t] for t in ordered_tickers]) - np.array([previous_weights.get(t, 0.0) for t in ordered_tickers]))) / 2.0)
        transaction_cost = float(turnover * (float(transaction_cost_bps) / 10_000.0))

    gross_weighted = calculate_portfolio_returns(holding_returns[ordered_tickers], weights)
    if previous_weights is None:
        net_return_series = gross_weighted.copy()
    else:
        net_return_series = gross_weighted.copy()
        if len(net_return_series) > 0:
            net_return_series.iloc[0] = net_return_series.iloc[0] - transaction_cost

    portfolio_return = float(gross_weighted.mean()) if len(gross_weighted) else 0.0
    net_return = float(net_return_series.mean()) if len(net_return_series) else 0.0

    output = {
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
        "transaction_cost": float(transaction_cost),
        "gross_return": portfolio_return,
        "net_return": net_return,
        "gross_return_series": gross_weighted,
        "net_return_series": net_return_series,
        "optimization_success": bool(result["success"]),
        "optimization_message": result["message"],
    }
    return output


def run_milestone3_walk_forward(
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
    """Run the Milestone 3 research strategies across the leakage-safe walk-forward schedule."""
    if strategies is None:
        strategies = MILESTONE3_STRATEGY_NAMES

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
        "strategy_previous_weights": {strategy: None for strategy in strategies},
    }

    for rebalance_info in rebalance_infos:
        for strategy in strategies:
            previous_weights = results["strategy_previous_weights"][strategy]
            for cost_bps in cost_levels:
                result = execute_milestone3_rebalance(
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

                gross_series = result["gross_return_series"]
                net_series = result["net_return_series"]
                if strategy not in results["gross_returns"]:
                    results["gross_returns"][strategy] = []
                if strategy not in results["net_returns"]:
                    results["net_returns"][strategy] = []
                results["gross_returns"][strategy].append(gross_series)
                results["net_returns"][strategy].append(net_series)

            if strategy in results["strategy_previous_weights"]:
                strategy_weights = None
                for event in results["weights_history"][::-1]:
                    if event["strategy"] == strategy and event["date"] == rebalance_info.rebalance_date:
                        strategy_weights = {ticker: event[f"weight_{ticker}"] for ticker in tickers}
                        break
                results["strategy_previous_weights"][strategy] = strategy_weights

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
    }


def save_milestone3_outputs(output_dir: str | Path, results: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results["weights_history"]).to_csv(output_dir / "walk_forward_weights.csv", index=False)
    pd.DataFrame(results["rebalance_history"]).to_csv(output_dir / "rebalance_history.csv", index=False)
    pd.DataFrame(results["turnover_history"]).to_csv(output_dir / "turnover_history.csv", index=False)
    pd.DataFrame(results["transaction_cost_history"]).to_csv(output_dir / "transaction_cost_history.csv", index=False)

    gross_df = pd.DataFrame({k: v for k, v in results["gross_returns"].items()})
    net_df = pd.DataFrame({k: v for k, v in results["net_returns"].items()})
    gross_df.to_csv(output_dir / "walk_forward_returns_gross.csv")
    net_df.to_csv(output_dir / "walk_forward_returns_net.csv")
