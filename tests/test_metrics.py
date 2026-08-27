import numpy as np
import pandas as pd

from src.risk.metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_cumulative_return():
    series = pd.Series([1.0, 1.1, 1.21])
    assert np.isclose(cumulative_return(series), 0.21)


def test_annualized_return_and_volatility():
    series = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
    ret = annualized_return(series, trading_days_per_year=252)
    vol = annualized_volatility(series, trading_days_per_year=252)
    assert ret > -1.0
    assert vol >= 0.0


def test_sharpe_ratio():
    series = pd.Series([0.005, 0.008, -0.002, 0.010, 0.006])
    s = sharpe_ratio(series, risk_free_rate=0.01, trading_days_per_year=252)
    assert np.isfinite(s)


def test_sortino_ratio():
    series = pd.Series([0.02, 0.01, -0.03, 0.04, -0.01, 0.03])
    s = sortino_ratio(series, risk_free_rate=0.01, trading_days_per_year=252)
    assert np.isfinite(s)


def test_max_drawdown():
    series = pd.Series([100.0, 110.0, 105.0, 90.0, 95.0])
    assert np.isclose(max_drawdown(series), -0.18181818181818182)


def test_max_drawdown_is_negative():
    """Verify max drawdown is returned as negative value (convention)."""
    series = pd.Series([1.0, 1.5, 1.2, 0.8, 0.9])
    dd = max_drawdown(series)
    assert dd < 0, "Max drawdown should be negative (represents loss)"


def test_annualized_return_vs_arithmetic():
    """Verify that annualized_return computes CAGR, not arithmetic mean * 252."""
    # Simple test: if returns are constant at 1% per day
    daily_return = 0.01
    n_days = 100
    series = pd.Series([daily_return] * n_days)
    
    # CAGR: (1.01)^100 ^ (252/100) - 1
    cagr = annualized_return(series, trading_days_per_year=252)
    
    # Arithmetic annualization (what optimizer uses for expected returns)
    arithmetic_ann = daily_return * 252
    
    # They should be different
    assert not np.isclose(cagr, arithmetic_ann), "CAGR and arithmetic annualization should differ"
    # CAGR should be higher for positive returns
    assert cagr > arithmetic_ann, "CAGR should exceed arithmetic annualization for positive returns"


def test_annualized_volatility_formula():
    """Verify annualized volatility follows: daily_std * sqrt(252)."""
    series = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01])
    
    daily_std = series.std(ddof=1)
    expected_ann_vol = daily_std * np.sqrt(252)
    computed_ann_vol = annualized_volatility(series, trading_days_per_year=252)
    
    np.testing.assert_allclose(expected_ann_vol, computed_ann_vol, rtol=1e-10)


def test_sharpe_calculation_consistency():
    """Verify Sharpe ratio calculation: (ann_return - rf) / ann_volatility."""
    series = pd.Series([0.005, 0.008, -0.002, 0.010, 0.006, 0.003, -0.001])
    rf = 0.02
    
    ann_ret = annualized_return(series, trading_days_per_year=252)
    ann_vol = annualized_volatility(series, trading_days_per_year=252)
    sharpe_manual = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    sharpe_func = sharpe_ratio(series, risk_free_rate=rf, trading_days_per_year=252)
    
    np.testing.assert_allclose(sharpe_manual, sharpe_func, rtol=1e-10)


def test_sortino_downside_deviation():
    """Verify Sortino uses downside deviation (negative returns only)."""
    # Mix of positive and negative returns
    positive_returns = pd.Series([0.01, 0.02, 0.015])
    mixed_returns = pd.Series([0.01, -0.05, 0.02, -0.03, 0.015])
    
    # Sortino of all-positive series should be high or infinite
    sortino_pos = sortino_ratio(positive_returns, risk_free_rate=0.0, trading_days_per_year=252)
    # Sortino of mixed series should be lower (has downside)
    sortino_mixed = sortino_ratio(mixed_returns, risk_free_rate=0.0, trading_days_per_year=252)
    
    assert sortino_pos > sortino_mixed, "Positive returns should have higher Sortino"


def test_cumulative_return_formula():
    """Verify cumulative return: end / start - 1."""
    prices = pd.Series([100.0, 110.0, 99.0, 120.0])
    
    expected_ret = (120.0 / 100.0) - 1.0
    computed_ret = cumulative_return(prices)
    
    np.testing.assert_allclose(expected_ret, computed_ret, rtol=1e-10)


def test_max_drawdown_extreme_case():
    """Test max drawdown with single peak and continuous decline."""
    prices = pd.Series([1.0, 2.0, 1.5, 1.0, 0.5, 0.25])
    dd = max_drawdown(prices)
    
    # From peak of 2.0 to trough of 0.25
    expected_dd = (0.25 - 2.0) / 2.0
    np.testing.assert_allclose(dd, expected_dd, rtol=1e-10)
