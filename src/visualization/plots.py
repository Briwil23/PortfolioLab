"""Visualization utilities for portfolio metrics and efficient-frontier analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_figure(fig: plt.Figure, file_path: str | Path) -> None:
    """Save a Matplotlib figure to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    output_path: str | Path = "results/figures/efficient_frontier.png",
) -> None:
    """Plot the constrained efficient frontier."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frontier_df["volatility"], frontier_df["target_return"], marker="o", linewidth=2)
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Portfolio Volatility")
    ax.set_ylabel("Expected Return")
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_path)


def plot_portfolio_allocations(
    weights_df: pd.DataFrame,
    output_path: str | Path = "results/figures/portfolio_allocations.png",
) -> None:
    """Create a grouped bar chart of allocation weights for each portfolio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    weights_df.plot(kind="bar", ax=ax)
    ax.set_title("Portfolio Allocations")
    ax.set_xlabel("Portfolio")
    ax.set_ylabel("Weight")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Asset")
    save_figure(fig, output_path)


def plot_cumulative_growth(
    series_dict: dict[str, pd.Series],
    output_path: str | Path = "results/figures/cumulative_growth.png",
) -> None:
    """Plot cumulative growth of portfolio and benchmark series."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in series_dict.items():
        ax.plot(values.index, values, label=name)
    ax.set_title("Cumulative Portfolio Growth")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_path)


def plot_drawdown(
    drawdown_dict: dict[str, pd.Series],
    output_path: str | Path = "results/figures/drawdown.png",
) -> None:
    """Plot portfolio drawdown trajectories."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in drawdown_dict.items():
        ax.plot(values.index, values, label=name)
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_path)


def plot_risk_return_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path = "results/figures/risk_return_comparison.png",
) -> None:
    """Plot annualized return against annualized volatility."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in metrics_df.iterrows():
        ax.scatter(row["annualized_volatility"], row["annualized_return"], label=row.name)
        ax.annotate(row.name, (row["annualized_volatility"], row["annualized_return"]))
    ax.set_title("Risk vs Return")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_path)


# ============================================================================
# MILESTONE 2 — WALK-FORWARD BACKTESTING VISUALIZATIONS
# ============================================================================


def plot_rolling_volatility(
    returns_dict: dict[str, pd.Series],
    window: int = 252,
    output_path: str | Path = "results/figures/rolling_volatility.png",
) -> None:
    """Plot rolling 12-month (252-day) volatility for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, returns in returns_dict.items():
        if len(returns) > 0:
            rolling_vol = returns.rolling(window=window).std() * (252 ** 0.5)
            ax.plot(rolling_vol.index, rolling_vol, label=name, alpha=0.8)

    ax.set_title("Walk-Forward Rolling 12-Month Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_path)


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    risk_free_rate: float = 0.02,
    window: int = 252,
    output_path: str | Path = "results/figures/rolling_sharpe.png",
) -> None:
    """Plot rolling 12-month Sharpe ratio for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    for name, returns in returns_dict.items():
        if len(returns) > 0:
            rolling_ret = returns.rolling(window=window).mean() * 252
            rolling_vol = returns.rolling(window=window).std() * (252 ** 0.5)
            rolling_sharpe = (rolling_ret - risk_free_rate) / rolling_vol.clip(lower=1e-6)
            ax.plot(rolling_sharpe.index, rolling_sharpe, label=name, alpha=0.8)

    ax.set_title("Walk-Forward Rolling 12-Month Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    save_figure(fig, output_path)


def plot_weight_stability(
    weights_df: pd.DataFrame,
    output_path: str | Path = "results/figures/weight_stability.png",
) -> None:
    """Plot weight evolution through time for each asset across all rebalance dates."""
    # weights_df should have columns: date, strategy, asset, weight
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique strategies (excluding SPY benchmark)
    strategies = [s for s in weights_df["strategy"].unique() if s != "SPY"]

    colors = plt.cm.Set2(range(len(strategies)))

    for idx, strategy in enumerate(strategies):
        strategy_data = weights_df[weights_df["strategy"] == strategy]

        # Get unique assets
        assets = strategy_data["asset"].unique()

        for asset in assets:
            asset_data = strategy_data[strategy_data["asset"] == asset].sort_values("date")
            ax.plot(
                asset_data["date"],
                asset_data["weight"],
                marker=".",
                label=f"{strategy} - {asset}",
                alpha=0.6,
                color=colors[idx],
            )

    ax.set_title("Walk-Forward Portfolio Weight Evolution")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    save_figure(fig, output_path)


def plot_weight_statistics(
    weights_df: pd.DataFrame,
    output_path: str | Path = "results/figures/weight_statistics.png",
) -> None:
    """Plot weight statistics (mean, min, max, std) by asset and strategy."""
    # Compute statistics by strategy and asset
    stats = weights_df.groupby(["strategy", "asset"])["weight"].agg(
        ["mean", "min", "max", "std"]
    ).reset_index()

    strategies = sorted([s for s in stats["strategy"].unique() if s != "SPY"])
    n_strategies = len(strategies)
    n_assets = len(stats[stats["strategy"] == strategies[0]])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metrics = ["mean", "min", "max", "std"]

    for ax, metric in zip(axes, metrics):
        for strategy in strategies:
            strategy_stats = stats[stats["strategy"] == strategy]
            ax.bar(
                strategy_stats["asset"],
                strategy_stats[metric],
                alpha=0.7,
                label=strategy,
            )

        ax.set_title(f"Portfolio Weight {metric.capitalize()} Across Rebalances")
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel("Asset")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

