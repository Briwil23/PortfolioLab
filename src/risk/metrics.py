"""Risk and performance metrics for portfolio evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_return(series: pd.Series) -> float:
    """Compute cumulative return from a price or wealth series."""
    if series.empty:
        raise ValueError("Series is empty.")
    if len(series) < 2:
        return 0.0
    start_value = float(series.iloc[0])
    end_value = float(series.iloc[-1])
    if start_value <= 0:
        raise ValueError("Initial value must be positive to compute cumulative return.")
    return end_value / start_value - 1.0


def annualized_return(
    series: pd.Series,
    trading_days_per_year: int = 252,
) -> float:
    """Annualize a series of daily returns."""
    if series.empty:
        raise ValueError("Series is empty.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")

    total_returns = 1.0 + series
    cumulative_growth = float(total_returns.prod())
    n_periods = len(series)
    if n_periods <= 0:
        return 0.0
    return cumulative_growth ** (trading_days_per_year / n_periods) - 1.0


def annualized_volatility(
    series: pd.Series,
    trading_days_per_year: int = 252,
) -> float:
    """Annualize the volatility of daily returns."""
    if series.empty:
        raise ValueError("Series is empty.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")
    return float(series.std(ddof=1) * np.sqrt(trading_days_per_year))


def sharpe_ratio(
    series: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio for a return series."""
    ann_ret = annualized_return(series, trading_days_per_year=trading_days_per_year)
    vol = annualized_volatility(series, trading_days_per_year=trading_days_per_year)
    if vol <= 0:
        return 0.0
    return (ann_ret - risk_free_rate) / vol


def sortino_ratio(
    series: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio using downside deviation."""
    if series.empty:
        raise ValueError("Series is empty.")
    negative_returns = series[series < 0]
    if negative_returns.empty:
        return np.inf

    downside_dev = float(np.sqrt(np.mean(np.square(negative_returns))))
    annual_downside = downside_dev * np.sqrt(trading_days_per_year)
    if annual_downside <= 0:
        return 0.0

    ann_ret = annualized_return(series, trading_days_per_year=trading_days_per_year)
    return (ann_ret - risk_free_rate) / annual_downside


def max_drawdown(series: pd.Series) -> float:
    """Compute maximum drawdown from a cumulative wealth index."""
    if series.empty:
        raise ValueError("Series is empty.")
    if len(series) < 2:
        return 0.0

    wealth = pd.Series(series, dtype=float).copy()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return float(drawdown.min())


def compute_portfolio_return_series(
    weights: np.ndarray,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Compute the return series for a portfolio given asset return paths."""
    if asset_returns.empty:
        raise ValueError("Asset-return table is empty.")
    if len(weights) != asset_returns.shape[1]:
        raise ValueError("Portfolio weights do not match the asset-return columns.")
    return (asset_returns.mul(weights, axis=1).sum(axis=1))
