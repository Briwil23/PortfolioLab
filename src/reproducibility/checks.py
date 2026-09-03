"""Keyed comparison helpers for reproducibility checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_returns_long(csv_path: str | Path) -> pd.DataFrame:
    """Read walk-forward returns CSV and normalize to date+strategy+value rows."""
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        date_col = "Date"
    elif "date" in df.columns:
        date_col = "date"
    else:
        date_col = df.columns[0]

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized[date_col])
    strategy_columns = [c for c in normalized.columns if c not in {date_col, "date"}]

    return normalized[["date"] + strategy_columns].melt(
        id_vars=["date"],
        var_name="strategy",
        value_name="return_value",
    )


def read_weights_long(csv_path: str | Path) -> pd.DataFrame:
    """Read walk-forward weights CSV in long date+strategy+asset+weight form."""
    df = pd.read_csv(csv_path)
    required = {"date", "strategy", "asset", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Weights file missing required columns: {sorted(missing)}")

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    return normalized[["date", "strategy", "asset", "weight"]]


def max_keyed_return_difference(csv_a: str | Path, csv_b: str | Path) -> float:
    """Return maximum absolute return difference after date+strategy join."""
    left = read_returns_long(csv_a).rename(columns={"return_value": "return_left"})
    right = read_returns_long(csv_b).rename(columns={"return_value": "return_right"})

    joined = left.merge(right, on=["date", "strategy"], how="inner")
    if joined.empty:
        raise ValueError("No overlapping date+strategy rows found in return files.")

    return float((joined["return_left"] - joined["return_right"]).abs().max())


def max_keyed_weight_difference(csv_a: str | Path, csv_b: str | Path) -> float:
    """Return maximum absolute weight difference after date+strategy+asset join."""
    left = read_weights_long(csv_a).rename(columns={"weight": "weight_left"})
    right = read_weights_long(csv_b).rename(columns={"weight": "weight_right"})

    joined = left.merge(right, on=["date", "strategy", "asset"], how="inner")
    if joined.empty:
        raise ValueError("No overlapping date+strategy+asset rows found in weight files.")

    return float((joined["weight_left"] - joined["weight_right"]).abs().max())
