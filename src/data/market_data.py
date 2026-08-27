"""Market data acquisition and statistical preprocessing for PortfolioLab."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def download_adjusted_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted historical prices from Yahoo Finance.

    Parameters
    ----------
    tickers : Iterable[str]
        Ticker symbols to download.
    start_date : str
        Download start date in ISO format such as ``YYYY-MM-DD``.
    end_date : Optional[str]
        Download end date; defaults to the current date if omitted.
    interval : str, default "1d"
        Price interval.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with one column per ticker and values in
        adjusted close prices.
    """
    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("No tickers were supplied for market-data retrieval.")

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info("Downloading %s from %s to %s.", ticker_list, start_date, end_date)
    raw = yf.download(
        ticker_list,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        actions="ignore",
        progress=False,
        threads=True,
        group_by="ticker",
    )

    if raw.empty:
        raise ValueError(f"No market data returned for tickers: {ticker_list}")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(1):
            raise ValueError("Downloaded data does not contain adjusted-close prices.")
        prices = raw.xs("Adj Close", level=1, axis=1).copy()
        prices.columns = [str(col).replace("^", "") for col in prices.columns]
    else:
        if "Adj Close" not in raw.columns:
            raise ValueError("Downloaded data does not contain adjusted-close prices.")
        prices = raw[["Adj Close"]].copy()
        prices.columns = [ticker_list[0]] if len(ticker_list) == 1 else prices.columns

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.sort_index()
    return prices


def clean_and_validate_prices(
    prices: pd.DataFrame,
    minimum_coverage: float = 0.80,
) -> pd.DataFrame:
    """Validate market-price data and align it on common trading dates.

    The function checks for unusable data, missing observations, and poor
    coverage before returning the cleaned price table.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw adjusted-close price table.
    minimum_coverage : float, default 0.80
        Minimum fraction of non-missing observations required for each ticker.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame with only valid tickers and aligned dates.
    """
    if prices.empty:
        raise ValueError("The price table is empty.")

    if prices.isna().all().all():
        raise ValueError("All price observations are missing.")

    coverage = prices.notna().mean()
    invalid_tickers = coverage[coverage < minimum_coverage].index.tolist()

    if invalid_tickers:
        message = (
            "The following assets do not have sufficient usable data: "
            f"{invalid_tickers}. Minimum required coverage: {minimum_coverage}."
        )
        raise ValueError(message)

    cleaned = prices.copy().sort_index()
    cleaned = cleaned.dropna(axis=0, how="any")
    cleaned = cleaned.ffill().bfill()

    if cleaned.empty:
        raise ValueError("No usable price observations remain after validation.")

    return cleaned


def calculate_daily_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """Compute daily simple or log returns from a price table.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted-close price DataFrame indexed by date.
    method : str, default "simple"
        Return method: ``"simple"`` or ``"log"``.

    Returns
    -------
    pd.DataFrame
        Daily returns with the same columns as the input price table.
    """
    if prices.empty:
        raise ValueError("Cannot compute returns from an empty price table.")

    if method not in {"simple", "log"}:
        raise ValueError("method must be either 'simple' or 'log'.")

    if method == "simple":
        returns = prices.pct_change().dropna()
    else:
        returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def calculate_annualized_expected_returns(
    returns: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Annualize expected daily returns."""
    if returns.empty:
        raise ValueError("Returns are empty.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")
    return returns.mean(axis=0) * trading_days_per_year


def calculate_annualized_covariance(
    returns: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Annualize the sample covariance matrix of returns."""
    if returns.empty:
        raise ValueError("Returns are empty.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")
    covariance = returns.cov() * trading_days_per_year
    if not np.all(np.isfinite(covariance.to_numpy())):
        raise ValueError("Annualized covariance matrix contains non-finite values.")
    return covariance


def save_processed_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    output_dir: str | Path,
) -> None:
    """Save processed market data to CSV files for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prices.to_csv(output_dir / "prices.csv")
    returns.to_csv(output_dir / "daily_returns.csv")
    covariance.to_csv(output_dir / "covariance_matrix.csv")
    expected_returns.to_frame(name="expected_return").to_csv(
        output_dir / "expected_returns.csv"
    )


def fetch_and_prepare_market_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
    trading_days_per_year: int = 252,
    save_output: bool = True,
    output_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Download, clean, and transform a market-data set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]
        Cleaned prices, daily returns, annualized covariance matrix, and annualized
        expected returns.
    """
    prices = download_adjusted_prices(tickers=tickers, start_date=start_date, end_date=end_date)
    cleaned_prices = clean_and_validate_prices(prices)
    daily_returns = calculate_daily_returns(cleaned_prices, method="simple")
    expected_returns = calculate_annualized_expected_returns(
        daily_returns, trading_days_per_year=trading_days_per_year
    )
    covariance = calculate_annualized_covariance(
        daily_returns, trading_days_per_year=trading_days_per_year
    )

    if save_output:
        save_processed_data(
            prices=cleaned_prices,
            returns=daily_returns,
            covariance=covariance,
            expected_returns=expected_returns,
            output_dir=output_dir,
        )

    return cleaned_prices, daily_returns, covariance, expected_returns
