"""Run PortfolioLab analysis: Milestone 1 (in-sample) + Milestone 2 (walk-forward backtest)."""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import asdict

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
    plot_rolling_volatility,
    plot_rolling_sharpe,
    plot_weight_stability,
    plot_weight_statistics,
)
from src.backtesting.walk_forward import generate_rebalance_dates
from src.backtesting.engine import WalkForwardBacktest

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


def run_milestone_1_analysis(
    root: Path,
    tickers: list[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    expected_returns: pd.DataFrame,
    trading_days_per_year: int,
    risk_free_rate: float,
    max_weight: float,
    benchmark_ticker: str,
) -> dict:
    """
    Run Milestone 1 in-sample analysis.

    Returns:
        Dictionary with metrics_df and wealth_series for visualization.
    """
    logger.info("=" * 80)
    logger.info("MILESTONE 1 — IN-SAMPLE ANALYSIS")
    logger.info("=" * 80)

    benchmark_returns = returns[benchmark_ticker].copy()
    benchmark_wealth = (1.0 + benchmark_returns).cumprod()

    n_assets = len(tickers)
    equal_weights = compute_equal_weight_portfolio(n_assets)
    equal_portfolio = {
        "weights": equal_weights,
        "expected_return": float(
            portfolio_expected_return(equal_weights, expected_returns.to_numpy())
        ),
        "volatility": float(
            np.sqrt(equal_weights @ covariance.to_numpy() @ equal_weights)
        ),
        "sharpe_ratio": float(
            (
                portfolio_expected_return(equal_weights, expected_returns.to_numpy())
                - risk_free_rate
            )
            / max(float(np.sqrt(equal_weights @ covariance.to_numpy() @ equal_weights)), 1e-12)
        ),
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
            "annualized_return": annualized_return(
                portfolio_returns, trading_days_per_year=trading_days_per_year
            ),
            "annualized_volatility": annualized_volatility(
                portfolio_returns, trading_days_per_year=trading_days_per_year
            ),
            "sharpe": sharpe_ratio(
                portfolio_returns,
                risk_free_rate=risk_free_rate,
                trading_days_per_year=trading_days_per_year,
            ),
            "sortino": sortino_ratio(
                portfolio_returns,
                risk_free_rate=risk_free_rate,
                trading_days_per_year=trading_days_per_year,
            ),
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

    logger.info("\n%s", metrics_df.round(4))
    logger.info("Outputs saved to results/performance/ and results/figures/")

    return {
        "metrics_df": metrics_df,
        "wealth_series": wealth_series,
        "weights_table": weights_table,
        "frontier_df": frontier_df,
    }


def run_milestone_2_backtest(
    root: Path,
    tickers: list[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    trading_days_per_year: int,
    risk_free_rate: float,
    max_weight: float,
    benchmark_ticker: str,
    backtest_config: dict,
) -> dict:
    """
    Run Milestone 2 walk-forward out-of-sample backtest.

    Returns:
        Dictionary with backtest results.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("MILESTONE 2 — WALK-FORWARD OUT-OF-SAMPLE BACKTEST")
    logger.info("=" * 80)

    # Generate rebalance dates
    rebalance_infos = generate_rebalance_dates(
        prices,
        lookback_years=backtest_config.get("lookback_years", 3),
        rebalance_frequency=backtest_config.get("rebalance_frequency", "monthly"),
        holdout_start_date=backtest_config.get("holdout_start_date"),
    )

    logger.info(f"Generated {len(rebalance_infos)} rebalance dates")

    if len(rebalance_infos) == 0:
        logger.warning("No rebalance dates generated. Skipping Milestone 2.")
        return None

    # Run walk-forward backtest
    strategies = ["Equal Weight", "Minimum Variance", "Maximum Sharpe"]
    backtest = WalkForwardBacktest(
        prices_df=prices,
        returns_df=returns,
        tickers=tickers,
        rebalance_infos=rebalance_infos,
        strategies=strategies,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        trading_days_per_year=trading_days_per_year,
    )

    results = backtest.run()

    # Create output directory
    backtest_dir = root / "results" / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # Save rebalance history
    rebalance_records = [asdict(record) for record in backtest.rebalance_records]
    rebalance_df = pd.DataFrame(rebalance_records)
    rebalance_df.to_csv(backtest_dir / "rebalance_history.csv", index=False)
    logger.info(f"Saved rebalance history: {len(rebalance_df)} events")

    # Save portfolio weights through time
    weights_records = []
    for portfolio_weights in backtest.portfolio_weights_history:
        for asset, weight in portfolio_weights.weights.items():
            weights_records.append({
                "date": portfolio_weights.rebalance_date,
                "strategy": portfolio_weights.strategy_name,
                "asset": asset,
                "weight": weight,
            })
    weights_time_df = pd.DataFrame(weights_records)
    weights_time_df.to_csv(backtest_dir / "walk_forward_weights.csv", index=False)

    # Calculate out-of-sample metrics for each strategy
    spy_returns = returns[benchmark_ticker].copy()

    # Align returns across all strategies
    all_dates = set()
    for strategy_rets in results["returns_concatenated"].values():
        all_dates.update(strategy_rets.index)

    all_dates = sorted(all_dates)

    metrics_rows = {}
    wealth_series_oos = {}

    # Add SPY benchmark
    spy_wealth = (1.0 + spy_returns[spy_returns.index >= min(all_dates)]).cumprod()
    spy_rets_subset = spy_returns[spy_returns.index >= min(all_dates)]

    metrics_rows["SPY"] = {
        "annualized_return": annualized_return(spy_rets_subset, trading_days_per_year),
        "annualized_volatility": annualized_volatility(spy_rets_subset, trading_days_per_year),
        "sharpe": sharpe_ratio(spy_rets_subset, risk_free_rate, trading_days_per_year),
        "sortino": sortino_ratio(spy_rets_subset, risk_free_rate, trading_days_per_year),
        "max_drawdown": max_drawdown(spy_wealth),
        "cumulative_return": cumulative_return(spy_wealth),
        "calmar": annualized_return(spy_rets_subset, trading_days_per_year) / abs(max_drawdown(spy_wealth)) if max_drawdown(spy_wealth) != 0 else 0,
    }
    wealth_series_oos["SPY"] = spy_wealth

    # Process each strategy
    for strategy in strategies:
        strategy_rets = results["returns_concatenated"].get(strategy)
        if strategy_rets is None or len(strategy_rets) == 0:
            logger.warning(f"No returns for {strategy}")
            continue

        # Align with common date range
        strategy_rets_aligned = strategy_rets[strategy_rets.index >= min(all_dates)]

        if len(strategy_rets_aligned) == 0:
            logger.warning(f"No aligned returns for {strategy}")
            continue

        wealth_oos = (1.0 + strategy_rets_aligned).cumprod()

        metrics_rows[strategy] = {
            "annualized_return": annualized_return(strategy_rets_aligned, trading_days_per_year),
            "annualized_volatility": annualized_volatility(strategy_rets_aligned, trading_days_per_year),
            "sharpe": sharpe_ratio(strategy_rets_aligned, risk_free_rate, trading_days_per_year),
            "sortino": sortino_ratio(strategy_rets_aligned, risk_free_rate, trading_days_per_year),
            "max_drawdown": max_drawdown(wealth_oos),
            "cumulative_return": cumulative_return(wealth_oos),
            "calmar": annualized_return(strategy_rets_aligned, trading_days_per_year) / abs(max_drawdown(wealth_oos)) if max_drawdown(wealth_oos) != 0 else 0,
        }
        wealth_series_oos[strategy] = wealth_oos

    oos_metrics_df = pd.DataFrame(metrics_rows).T
    oos_metrics_df.index.name = "portfolio"

    # Save metrics
    oos_metrics_df.to_csv(backtest_dir / "walk_forward_metrics.csv")

    # Save returns
    returns_df_oos = pd.DataFrame(results["returns_concatenated"])
    returns_df_oos.to_csv(backtest_dir / "walk_forward_returns.csv")

    logger.info("\nOut-of-Sample Metrics:")
    logger.info("\n%s", oos_metrics_df.round(4))

    logger.info(f"\nBacktest outputs saved to results/backtest/")

    return {
        "metrics_df": oos_metrics_df,
        "wealth_series": wealth_series_oos,
        "returns_df": returns_df_oos,
        "rebalance_df": rebalance_df,
        "weights_time_df": weights_time_df,
    }


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
    backtest_config = config.get("backtest", {})

    logger.info("Starting PortfolioLab analysis for tickers: %s", tickers)

    # Fetch market data once (used by both Milestone 1 and 2)
    prices, returns, covariance, expected_returns = fetch_and_prepare_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        trading_days_per_year=trading_days_per_year,
        save_output=True,
        output_dir=root / "data" / "processed",
    )

    # ========================================================================
    # MILESTONE 1: In-sample analysis
    # ========================================================================
    m1_results = run_milestone_1_analysis(
        root=root,
        tickers=tickers,
        prices=prices,
        returns=returns,
        covariance=covariance,
        expected_returns=expected_returns,
        trading_days_per_year=trading_days_per_year,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        benchmark_ticker=benchmark_ticker,
    )

    # Generate Milestone 1 visualizations
    plot_efficient_frontier(
        m1_results["frontier_df"], root / "results" / "figures" / "efficient_frontier.png"
    )
    plot_portfolio_allocations(
        m1_results["weights_table"], root / "results" / "figures" / "portfolio_allocations.png"
    )
    plot_cumulative_growth(
        m1_results["wealth_series"], root / "results" / "figures" / "cumulative_growth.png"
    )
    plot_drawdown(
        {
            name: (series / series.cummax() - 1.0)
            for name, series in m1_results["wealth_series"].items()
        },
        root / "results" / "figures" / "drawdown.png",
    )
    plot_risk_return_comparison(
        m1_results["metrics_df"], root / "results" / "figures" / "risk_return_comparison.png"
    )

    # ========================================================================
    # MILESTONE 2: Walk-forward out-of-sample backtest
    # ========================================================================
    if backtest_config.get("enabled", False):
        m2_results = run_milestone_2_backtest(
            root=root,
            tickers=tickers,
            prices=prices,
            returns=returns,
            trading_days_per_year=trading_days_per_year,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            benchmark_ticker=benchmark_ticker,
            backtest_config=backtest_config,
        )

        if m2_results is not None:
            # Generate Milestone 2 visualizations
            plot_cumulative_growth(
                m2_results["wealth_series"],
                root / "results" / "figures" / "walk_forward_cumulative_growth.png",
            )
            plot_drawdown(
                {
                    name: (series / series.cummax() - 1.0)
                    for name, series in m2_results["wealth_series"].items()
                },
                root / "results" / "figures" / "walk_forward_drawdown.png",
            )
            
            # Additional Milestone 2 analytics
            plot_rolling_volatility(
                m2_results["returns_df"],
                window=252,
                output_path=root / "results" / "figures" / "rolling_volatility.png",
            )
            plot_rolling_sharpe(
                m2_results["returns_df"],
                risk_free_rate=risk_free_rate,
                window=252,
                output_path=root / "results" / "figures" / "rolling_sharpe.png",
            )
            plot_weight_stability(
                m2_results["weights_time_df"],
                output_path=root / "results" / "figures" / "weight_stability.png",
            )
            plot_weight_statistics(
                m2_results["weights_time_df"],
                output_path=root / "results" / "figures" / "weight_statistics.png",
            )
    else:
        logger.info("Milestone 2 (backtest) disabled in config.")

    logger.info("\nPortfolioLab analysis completed successfully.")


if __name__ == "__main__":
    main()

