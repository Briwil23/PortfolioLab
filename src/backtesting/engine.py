"""Walk-forward backtesting engine for executing rebalance-and-hold strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from src.data.market_data import (
    calculate_annualized_covariance,
    calculate_annualized_expected_returns,
)
from src.optimization.mean_variance import (
    compute_equal_weight_portfolio,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
)
from src.backtesting.walk_forward import RebalanceInfo, extract_holding_data, extract_training_data

logger = logging.getLogger(__name__)


@dataclass
class PortfolioWeights:
    """Portfolio weights at a rebalance date."""

    rebalance_date: datetime
    strategy_name: str
    weights: dict[str, float]
    optimization_success: bool
    optimization_message: str = ""


@dataclass
class RebalanceRecord:
    """Record of a single rebalance event with metadata."""

    rebalance_date: datetime
    training_start: datetime
    training_end: datetime
    holding_start: datetime
    holding_end: datetime | None
    strategy: str
    optimization_success: bool
    optimization_message: str = ""
    n_training_observations: int = 0


def portfolio_weights_from_optimization(
    weights_array: np.ndarray,
    tickers: list[str],
) -> dict[str, float]:
    """Convert numpy weight array to ticker-indexed dictionary."""
    return {ticker: float(weight) for ticker, weight in zip(tickers, weights_array)}


def calculate_portfolio_returns(
    holding_returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Calculate time-series portfolio returns.

    Args:
        holding_returns: DataFrame with index=dates, columns=tickers.
        weights: Dictionary of ticker -> weight.

    Returns:
        Series with portfolio returns aligned to holding_returns index.
    """
    weight_array = np.array([weights[ticker] for ticker in holding_returns.columns])
    portfolio_rets = holding_returns @ weight_array
    return portfolio_rets


def execute_rebalance(
    rebalance_info: RebalanceInfo,
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    tickers: list[str],
    strategy_name: str,
    risk_free_rate: float,
    max_weight: float,
    trading_days_per_year: int,
    previous_weights: dict[str, float] | None = None,
) -> tuple[PortfolioWeights, RebalanceRecord, pd.DataFrame, dict[str, float]]:
    """
    Execute one rebalance event for a single strategy.

    Args:
        rebalance_info: RebalanceInfo object.
        prices_df: Full price history.
        returns_df: Full returns history.
        tickers: List of asset tickers.
        strategy_name: Name of strategy ("Equal Weight", "Minimum Variance", "Maximum Sharpe").
        risk_free_rate: Risk-free rate for optimization.
        max_weight: Maximum position weight.
        trading_days_per_year: Trading days for annualization.
        previous_weights: Previous successful weights for fallback on failure.

    Returns:
        Tuple of (PortfolioWeights, RebalanceRecord, holding_returns_df, weights_dict).

    Note:
        If optimization fails, uses previous_weights or equal weights as fallback.
    """
    # Extract training and holding data
    training_prices, training_returns = extract_training_data(prices_df, returns_df, rebalance_info)
    holding_returns = extract_holding_data(returns_df, rebalance_info)

    n_training_obs = len(training_returns)

    # Calculate parameters from training data only
    expected_returns = calculate_annualized_expected_returns(
        training_returns, trading_days_per_year=trading_days_per_year
    )
    covariance = calculate_annualized_covariance(
        training_returns, trading_days_per_year=trading_days_per_year
    )

    # Optimize based on strategy
    success = False
    weights_dict = None
    message = ""

    try:
        if strategy_name == "Equal Weight":
            n_assets = len(tickers)
            weights_array = compute_equal_weight_portfolio(n_assets)
            weights_dict = portfolio_weights_from_optimization(weights_array, tickers)
            success = True
            message = "Equal weight portfolio"

        elif strategy_name == "Minimum Variance":
            result = minimum_variance_portfolio(
                expected_returns.to_numpy(),
                covariance.to_numpy(),
                max_weight=max_weight,
            )
            if result.get("success", False):
                weights_array = result["weights"]
                weights_dict = portfolio_weights_from_optimization(weights_array, tickers)
                success = True
                message = f"Optimization succeeded (status: {result.get('status', 'unknown')})"
            else:
                message = f"Optimization failed: {result.get('message', 'unknown error')}"

        elif strategy_name == "Maximum Sharpe":
            result = maximum_sharpe_portfolio(
                expected_returns.to_numpy(),
                covariance.to_numpy(),
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
            )
            if result.get("success", False):
                weights_array = result["weights"]
                weights_dict = portfolio_weights_from_optimization(weights_array, tickers)
                success = True
                message = f"Optimization succeeded (status: {result.get('status', 'unknown')})"
            else:
                message = f"Optimization failed: {result.get('message', 'unknown error')}"

        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    except Exception as e:
        logger.warning(f"Exception during {strategy_name} optimization: {e}")
        success = False
        message = f"Exception: {str(e)}"

    # Fallback logic if optimization failed
    if not success:
        if previous_weights is not None:
            logger.warning(
                f"Optimization failed for {strategy_name} at {rebalance_info.rebalance_date}. "
                f"Using previous weights."
            )
            weights_dict = previous_weights.copy()
            message = f"Failed; using previous weights. {message}"
        else:
            logger.warning(
                f"Optimization failed for {strategy_name} at {rebalance_info.rebalance_date}. "
                f"No previous weights; using equal weight fallback."
            )
            n_assets = len(tickers)
            weights_array = compute_equal_weight_portfolio(n_assets)
            weights_dict = portfolio_weights_from_optimization(weights_array, tickers)
            message = f"Failed; using equal weight fallback. {message}"
            success = True  # Fallback succeeded

    # Verify weights
    if weights_dict is None:
        raise RuntimeError(f"Could not determine weights for {strategy_name} at {rebalance_info.rebalance_date}")

    # Create portfolio records
    portfolio_weights = PortfolioWeights(
        rebalance_date=rebalance_info.rebalance_date,
        strategy_name=strategy_name,
        weights=weights_dict,
        optimization_success=success,
        optimization_message=message,
    )

    rebalance_record = RebalanceRecord(
        rebalance_date=rebalance_info.rebalance_date,
        training_start=rebalance_info.training_start,
        training_end=rebalance_info.training_end,
        holding_start=rebalance_info.holding_start,
        holding_end=rebalance_info.holding_end,
        strategy=strategy_name,
        optimization_success=success,
        optimization_message=message,
        n_training_observations=n_training_obs,
    )

    return portfolio_weights, rebalance_record, holding_returns, weights_dict


class WalkForwardBacktest:
    """Orchestrates walk-forward backtesting."""

    def __init__(
        self,
        prices_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        tickers: list[str],
        rebalance_infos: list[RebalanceInfo],
        strategies: list[str],
        risk_free_rate: float = 0.02,
        max_weight: float = 0.30,
        trading_days_per_year: int = 252,
    ):
        """
        Args:
            prices_df: DataFrame with adjusted close prices.
            returns_df: DataFrame with daily returns.
            tickers: List of asset tickers.
            rebalance_infos: List of RebalanceInfo objects.
            strategies: List of strategy names to test.
            risk_free_rate: Risk-free rate for optimization.
            max_weight: Maximum position weight constraint.
            trading_days_per_year: Trading days for annualization.
        """
        self.prices_df = prices_df
        self.returns_df = returns_df
        self.tickers = tickers
        self.rebalance_infos = rebalance_infos
        self.strategies = strategies
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.trading_days_per_year = trading_days_per_year

        # Storage for results
        self.portfolio_weights_history: list[PortfolioWeights] = []
        self.rebalance_records: list[RebalanceRecord] = []
        self.portfolio_returns: dict[str, pd.Series] = {strategy: [] for strategy in strategies}
        self.previous_weights: dict[str, dict[str, float]] = {strategy: None for strategy in strategies}

    def run(self) -> dict:
        """
        Execute full walk-forward backtest.

        Returns:
            Dictionary containing:
            - portfolio_weights_history: All weights across all rebalance dates
            - rebalance_records: All rebalance events with metadata
            - portfolio_returns: Time-series returns for each strategy
            - returns_concatenated: Concatenated return series aligned to common dates
        """
        logger.info(f"Starting walk-forward backtest with {len(self.rebalance_infos)} rebalance dates")

        # Iterate through rebalance dates
        for rebalance_idx, rebalance_info in enumerate(self.rebalance_infos):
            logger.info(
                f"Rebalance {rebalance_idx + 1}/{len(self.rebalance_infos)}: {rebalance_info.rebalance_date.date()}"
            )

            for strategy in self.strategies:
                try:
                    portfolio_weights, rebalance_record, holding_returns, weights_dict = execute_rebalance(
                        rebalance_info=rebalance_info,
                        prices_df=self.prices_df,
                        returns_df=self.returns_df,
                        tickers=self.tickers,
                        strategy_name=strategy,
                        risk_free_rate=self.risk_free_rate,
                        max_weight=self.max_weight,
                        trading_days_per_year=self.trading_days_per_year,
                        previous_weights=self.previous_weights[strategy],
                    )

                    # Store records
                    self.portfolio_weights_history.append(portfolio_weights)
                    self.rebalance_records.append(rebalance_record)

                    # Calculate portfolio returns for this holding period
                    portfolio_ret = calculate_portfolio_returns(holding_returns, weights_dict)
                    self.portfolio_returns[strategy].append(portfolio_ret)

                    # Update previous weights for next rebalance
                    self.previous_weights[strategy] = weights_dict

                except Exception as e:
                    logger.error(
                        f"Error processing {strategy} at {rebalance_info.rebalance_date}: {e}",
                        exc_info=True,
                    )
                    raise

        logger.info("Walk-forward backtest completed")

        # Concatenate return series
        returns_concatenated = {}
        for strategy in self.strategies:
            if self.portfolio_returns[strategy]:
                returns_concatenated[strategy] = pd.concat(self.portfolio_returns[strategy])
            else:
                returns_concatenated[strategy] = pd.Series(dtype=float)

        return {
            "portfolio_weights_history": self.portfolio_weights_history,
            "rebalance_records": self.rebalance_records,
            "portfolio_returns": self.portfolio_returns,
            "returns_concatenated": returns_concatenated,
        }
