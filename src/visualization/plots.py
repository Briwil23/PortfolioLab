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
