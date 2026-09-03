"""Walk-forward backtesting framework with strict anti-lookahead controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd


def normalize_asset_order(frame: pd.DataFrame, configured_assets: list[str]) -> pd.DataFrame:
    """Validate and reorder asset columns to match the configured asset list."""
    if frame.empty:
        raise ValueError("Cannot normalize asset order for an empty DataFrame.")

    frame_columns = list(frame.columns)
    configured_set = set(configured_assets)
    frame_set = set(frame_columns)

    if len(frame_columns) != len(frame_set):
        duplicates = sorted({column for column in frame_columns if frame_columns.count(column) > 1})
        raise ValueError(
            "Asset-order normalization requires unique columns, but duplicates were found: "
            f"{duplicates}"
        )

    missing = sorted(configured_set - frame_set)
    extra = sorted(frame_set - configured_set)
    if missing or extra:
        raise ValueError(
            "Asset-order mismatch. "
            f"Missing assets: {missing}. Extra assets: {extra}."
        )

    return frame.loc[:, configured_assets].copy()


@dataclass
class RebalanceInfo:
    """Information about a single rebalance event."""

    rebalance_date: datetime
    training_start: datetime
    training_end: datetime
    holding_start: datetime
    holding_end: datetime | None = None

    def validate_anti_lookahead(self) -> None:
        """Verify that training window ends strictly before holding period begins."""
        if self.training_end >= self.holding_start:
            raise ValueError(
                f"Anti-lookahead violation: training_end ({self.training_end}) "
                f">= holding_start ({self.holding_start})"
            )

    def __str__(self) -> str:
        return (
            f"Rebalance: {self.rebalance_date.date()} | "
            f"Training: {self.training_start.date()} to {self.training_end.date()} | "
            f"Holding: {self.holding_start.date()}"
        )


def generate_rebalance_dates(
    prices_df: pd.DataFrame,
    lookback_years: int,
    rebalance_frequency: str,
    holdout_start_date: str | None = None,
) -> list[RebalanceInfo]:
    """
    Generate rebalance dates for walk-forward backtesting.

    Args:
        prices_df: DataFrame with dates as index and assets as columns.
        lookback_years: Number of years of training data before each rebalance.
        rebalance_frequency: "monthly", "quarterly", or "annual".
        holdout_start_date: Earliest rebalance date (ensures minimum training).

    Returns:
        List of RebalanceInfo objects defining each rebalance event.

    Note:
        The training window ends at the day BEFORE the rebalance date to prevent
        information leakage. Weights calculated at the rebalance date are applied
        starting the next trading day.

        Example with monthly rebalancing and 3-year lookback:
        - Rebalance date: 2018-01-31
        - Training: 2015-01-01 through 2018-01-30 (3 years, strictly before rebalance)
        - Holding period: 2018-01-31 through 2018-02-28 (weights applied to Feb returns)
    """
    if rebalance_frequency not in ("monthly", "quarterly", "annual"):
        raise ValueError(
            f"Invalid rebalance_frequency: {rebalance_frequency}. "
            "Must be 'monthly', 'quarterly', or 'annual'."
        )

    all_dates = prices_df.index.to_pydatetime()
    if len(all_dates) == 0:
        return []

    data_start = all_dates[0]
    data_end = all_dates[-1]

    # Earliest possible rebalance date (after minimum training window)
    earliest_rebalance = data_start + timedelta(days=365 * lookback_years)

    # Apply user-specified holdout start if provided
    if holdout_start_date:
        specified_date = pd.to_datetime(holdout_start_date)
        earliest_rebalance = max(earliest_rebalance, specified_date)

    # Align earliest rebalance to month/quarter/year boundary
    if rebalance_frequency == "monthly":
        earliest_rebalance = earliest_rebalance.replace(day=1)
    elif rebalance_frequency == "quarterly":
        quarter = (earliest_rebalance.month - 1) // 3
        earliest_rebalance = earliest_rebalance.replace(month=quarter * 3 + 1, day=1)
    elif rebalance_frequency == "annual":
        earliest_rebalance = earliest_rebalance.replace(month=1, day=1)

    # Generate rebalance dates
    rebalance_dates = []
    current_date = earliest_rebalance

    while current_date <= data_end:
        if current_date in all_dates:
            rebalance_dates.append(current_date)

        # Advance to next period
        if rebalance_frequency == "monthly":
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        elif rebalance_frequency == "quarterly":
            if current_date.month == 10:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 3)
        elif rebalance_frequency == "annual":
            current_date = current_date.replace(year=current_date.year + 1)

    # Build RebalanceInfo objects
    rebalance_infos = []
    for rebalance_date in rebalance_dates:
        # Training window: lookback_years years ending the day before rebalance
        training_end = rebalance_date - timedelta(days=1)
        training_start = training_end - timedelta(days=365 * lookback_years)

        # Holding period: starts at rebalance date
        holding_start = rebalance_date

        # Find the next rebalance date to determine holding_end
        holding_end = None
        rebalance_idx = rebalance_dates.index(rebalance_date)
        if rebalance_idx + 1 < len(rebalance_dates):
            holding_end = rebalance_dates[rebalance_idx + 1] - timedelta(days=1)

        rebalance_info = RebalanceInfo(
            rebalance_date=rebalance_date,
            training_start=training_start,
            training_end=training_end,
            holding_start=holding_start,
            holding_end=holding_end,
        )
        rebalance_info.validate_anti_lookahead()
        rebalance_infos.append(rebalance_info)

    return rebalance_infos


def extract_training_data(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    rebalance_info: RebalanceInfo,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract training data for a given rebalance window.

    Args:
        prices_df: Full price history.
        returns_df: Full daily returns.
        rebalance_info: Rebalance information object.

    Returns:
        Tuple of (training_prices_df, training_returns_df) with data strictly
        before the rebalance date.

    Raises:
        ValueError: If training window is empty or contains future data.
    """
    # Extract data strictly within training window (inclusive on both ends)
    training_prices = prices_df[
        (prices_df.index >= rebalance_info.training_start)
        & (prices_df.index <= rebalance_info.training_end)
    ].copy()

    training_returns = returns_df[
        (returns_df.index >= rebalance_info.training_start)
        & (returns_df.index <= rebalance_info.training_end)
    ].copy()

    if len(training_prices) == 0:
        raise ValueError(
            f"Training window for {rebalance_info.rebalance_date} is empty. "
            f"Window: {rebalance_info.training_start} to {rebalance_info.training_end}"
        )

    # Verify no future data leaked
    if training_prices.index.max() >= rebalance_info.rebalance_date:
        raise ValueError(
            f"Training data contains rebalance date or later. "
            f"Max training date: {training_prices.index.max()}, "
            f"Rebalance date: {rebalance_info.rebalance_date}"
        )

    return training_prices, training_returns


def extract_holding_data(
    returns_df: pd.DataFrame,
    rebalance_info: RebalanceInfo,
) -> pd.DataFrame:
    """
    Extract holding-period returns for a given rebalance window.

    Args:
        returns_df: Full daily returns.
        rebalance_info: Rebalance information object.

    Returns:
        DataFrame with returns from rebalance_date through holding_end (inclusive).

    Raises:
        ValueError: If holding window is empty.
    """
    holding_end = rebalance_info.holding_end
    if holding_end is None:
        # Use all remaining data if no next rebalance date
        holding_end = returns_df.index.max()

    holding_returns = returns_df[
        (returns_df.index >= rebalance_info.holding_start)
        & (returns_df.index <= holding_end)
    ].copy()

    if len(holding_returns) == 0:
        raise ValueError(
            f"Holding window for {rebalance_info.rebalance_date} is empty. "
            f"Window: {rebalance_info.holding_start} to {holding_end}"
        )

    # Verify holding period is strictly after training
    if holding_returns.index.min() <= rebalance_info.training_end:
        raise ValueError(
            f"Holding period overlaps with training. "
            f"Training end: {rebalance_info.training_end}, "
            f"Holding start: {holding_returns.index.min()}"
        )

    return holding_returns
