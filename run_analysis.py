"""Run the in-sample PortfolioLab optimization pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market_data import fetch_and_prepare_market_data, load_config
from src.optimization.mean_variance import (
    compute_equal_weight_portfolio,
    efficient_frontier,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    portfolio_expected_return,
)
from src.risk.metrics import (
    annualized_return,
    annualized_volatility,
    compute_portfolio_return_series,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from src.visualization.plots import (
    plot_cumulative_growth,
    plot_drawdown,
    plot_efficient_frontier,
    plot_portfolio_allocations,
    plot_risk_return_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_portfolio_series(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
    benchmark_ticker: str,
) -> pd.Series:
    """Construct a weighted portfolio return series and convert it to wealth values."""
    portfolio_returns = compute_portfolio_return_series(weights, returns_df)
    wealth_index = (1.0 + portfolio_returns).cumprod()
    return wealth_index


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config" / "config.yaml")

    tickers = config["asset_universe"]
    start_date = config["start_date"]
    end_date = config.get("end_date")
    trading_days_per_year = int(config.get("trading_days_per_year", 252))
    risk_free_rate = float(config.get("risk_free_rate", 0.02))
    max_weight = float(config.get("max_weight", 0.30))
    benchmark_ticker = config.get("benchmark_ticker", "SPY")

    logger.info("Starting PortfolioLab analysis for tickers: %s", tickers)

    prices, returns, covariance, expected_returns = fetch_and_prepare_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        trading_days_per_year=trading_days_per_year,
        save_output=True,
        output_dir=root / "data" / "processed",
    )

    benchmark_returns = returns[benchmark_ticker].copy()
    benchmark_wealth = (1.0 + benchmark_returns).cumprod()

    n_assets = len(tickers)
    equal_weights = compute_equal_weight_portfolio(n_assets)
    equal_portfolio = {
        "weights": equal_weights,
        "expected_return": float(portfolio_expected_return(equal_weights, expected_returns.to_numpy())),
        "volatility": float(np.sqrt(equal_weights @ covariance.to_numpy() @ equal_weights)),
        "sharpe_ratio": float((portfolio_expected_return(equal_weights, expected_returns.to_numpy()) - risk_free_rate) / max(float(np.sqrt(equal_weights @ covariance.to_numpy() @ equal_weights)), 1e-12)),
        "success": True,
    }

    minvar = minimum_variance_portfolio(
        expected_returns.to_numpy(),
        covariance.to_numpy(),
        max_weight=max_weight,
    )
    maxsharpe = maximum_sharpe_portfolio(
        expected_returns.to_numpy(),
        covariance.to_numpy(),
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )

    frontier = efficient_frontier(
        expected_returns.to_numpy(),
        covariance.to_numpy(),
        max_weight=max_weight,
        num_points=30,
    )
    frontier_df = pd.DataFrame(frontier)

    weight_map = {
        "SPY": np.eye(n_assets)[tickers.index(benchmark_ticker)],
        "Equal Weight": equal_weights,
        "Minimum Variance": minvar["weights"],
        "Maximum Sharpe": maxsharpe["weights"],
    }
    metric_rows = {}
    wealth_series = {"SPY": benchmark_wealth}

    for name, weights in weight_map.items():
        if name == "SPY":
            portfolio_returns = returns[benchmark_ticker]
            wealth = benchmark_wealth
        else:
            portfolio_returns = compute_portfolio_return_series(weights, returns)
            wealth = (1.0 + portfolio_returns).cumprod()

        metric_rows[name] = {
            "annualized_return": annualized_return(portfolio_returns, trading_days_per_year=trading_days_per_year),
            "annualized_volatility": annualized_volatility(portfolio_returns, trading_days_per_year=trading_days_per_year),
            "sharpe": sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate, trading_days_per_year=trading_days_per_year),
            "sortino": sortino_ratio(portfolio_returns, risk_free_rate=risk_free_rate, trading_days_per_year=trading_days_per_year),
            "max_drawdown": max_drawdown(wealth),
            "cumulative_return": cumulative_return(wealth),
        }
        wealth_series[name] = wealth

    metrics_df = pd.DataFrame(metric_rows).T
    metrics_df.index.name = "portfolio"

    weights_table = pd.DataFrame(
        [weight_map["Equal Weight"], weight_map["Minimum Variance"], weight_map["Maximum Sharpe"]],
        index=["Equal Weight", "Minimum Variance", "Maximum Sharpe"],
        columns=tickers,
    )
    weights_table.loc["SPY"] = [1.0 if ticker == benchmark_ticker else 0.0 for ticker in tickers]

    weights_table.to_csv(root / "results" / "performance" / "portfolio_weights.csv")
    metrics_df.to_csv(root / "results" / "performance" / "portfolio_metrics.csv")
    frontier_df.to_csv(root / "results" / "performance" / "efficient_frontier.csv")

    plot_efficient_frontier(frontier_df, root / "results" / "figures" / "efficient_frontier.png")
    plot_portfolio_allocations(weights_table, root / "results" / "figures" / "portfolio_allocations.png")
    plot_cumulative_growth(wealth_series, root / "results" / "figures" / "cumulative_growth.png")
    plot_drawdown(
        {
            name: (series / series.cummax() - 1.0)
            for name, series in wealth_series.items()
        },
        root / "results" / "figures" / "drawdown.png",
    )
    plot_risk_return_comparison(metrics_df, root / "results" / "figures" / "risk_return_comparison.png")

    logger.info("PortfolioLab run finished successfully.")
    logger.info("\n%s", metrics_df.round(4))


if __name__ == "__main__":
    main()
