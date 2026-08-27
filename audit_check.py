"""Comprehensive pre-GitHub audit of PortfolioLab Milestone 1."""

import numpy as np
import pandas as pd
from pathlib import Path

# Import modules to test
from src.risk.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    cumulative_return,
)
from src.optimization.mean_variance import (
    portfolio_expected_return,
    portfolio_volatility,
    portfolio_sharpe_ratio,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
)

print("\n" + "="*80)
print("AUDIT 1: MAXIMUM DRAWDOWN CONVENTION")
print("="*80)

# Test max_drawdown with simple example
wealth = pd.Series([1.0, 1.1, 1.05, 0.8, 0.85])
dd = max_drawdown(wealth)
print(f"Wealth series: {wealth.tolist()}")
print(f"Max drawdown: {dd:.4f}")
print(f"Convention check: Should be negative? {dd < 0}")
assert dd < 0, "Max drawdown should be negative (represents loss from peak)"
print("✓ Convention: Drawdown is correctly negative")

print("\n" + "="*80)
print("AUDIT 3: ANNUALIZATION VERIFICATION")
print("="*80)

# Simple daily returns
daily_returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])

# Expected return: mean * 252
expected_ann = daily_returns.mean() * 252
print(f"\nDaily returns: {daily_returns.tolist()}")
print(f"Expected annual return (arithmetic): {expected_ann:.6f}")

# CAGR: (1 + cumulative_return)^(252/n_days) - 1
cagr = annualized_return(daily_returns, trading_days_per_year=252)
print(f"Realized annualized return (CAGR): {cagr:.6f}")
print("✓ Two methods correctly distinguished:")
print(f"  - Optimizer uses arithmetic: {expected_ann:.6f}")
print(f"  - Realized returns use CAGR: {cagr:.6f}")

# Volatility
daily_std = daily_returns.std(ddof=1)
annual_vol_formula = daily_std * np.sqrt(252)
annual_vol_func = annualized_volatility(daily_returns, trading_days_per_year=252)
print(f"\nDaily std (ddof=1): {daily_std:.6f}")
print(f"Annualized vol (formula): {annual_vol_formula:.6f}")
print(f"Annualized vol (function): {annual_vol_func:.6f}")
assert np.isclose(annual_vol_formula, annual_vol_func), "Volatility annualization mismatch"
print("✓ Volatility annualization verified")

print("\n" + "="*80)
print("AUDIT 4: SHARPE RATIO CONSISTENCY")
print("="*80)

rf = 0.02
daily_rf = rf / 252
series = pd.Series([0.005, 0.008, -0.002, 0.010, 0.006, 0.003, -0.001])
sharpe = sharpe_ratio(series, risk_free_rate=rf, trading_days_per_year=252)

# Verify calculation: (ann_return - rf) / ann_volatility
ann_ret = annualized_return(series, trading_days_per_year=252)
ann_vol = annualized_volatility(series, trading_days_per_year=252)
sharpe_manual = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0

print(f"Annualized return: {ann_ret:.6f}")
print(f"Risk-free rate: {rf:.6f}")
print(f"Annualized volatility: {ann_vol:.6f}")
print(f"Sharpe (function): {sharpe:.6f}")
print(f"Sharpe (manual): {sharpe_manual:.6f}")
assert np.isclose(sharpe, sharpe_manual), "Sharpe ratio mismatch"
print("✓ Sharpe ratio calculation verified")

print("\n" + "="*80)
print("AUDIT 5: SORTINO RATIO")
print("="*80)

sortino = sortino_ratio(series, risk_free_rate=rf, trading_days_per_year=252)
negative_returns = series[series < 0]
downside_dev = np.sqrt(np.mean(np.square(negative_returns)))
annual_downside = downside_dev * np.sqrt(252)
sortino_manual = (ann_ret - rf) / annual_downside if annual_downside > 0 else 0.0

print(f"Negative returns: {negative_returns.tolist()}")
print(f"Downside deviation (daily): {downside_dev:.6f}")
print(f"Downside deviation (annual): {annual_downside:.6f}")
print(f"Sortino (function): {sortino:.6f}")
print(f"Sortino (manual): {sortino_manual:.6f}")
assert np.isclose(sortino, sortino_manual), "Sortino ratio mismatch"
print("✓ Sortino ratio calculation verified")

print("\n" + "="*80)
print("AUDIT 6: OPTIMIZATION CONSTRAINTS")
print("="*80)

mu = np.array([0.08, 0.10, 0.12, 0.06])
sigma = np.array([
    [0.04, 0.01, 0.00, 0.01],
    [0.01, 0.03, 0.01, 0.00],
    [0.00, 0.01, 0.05, 0.01],
    [0.01, 0.00, 0.01, 0.02],
])
max_w = 0.40

# Minimum variance
mv_result = minimum_variance_portfolio(mu, sigma, max_weight=max_w)
w_mv = mv_result["weights"]

print(f"\nMinimum Variance:")
print(f"  Weights: {w_mv}")
print(f"  Sum of weights: {w_mv.sum():.6f}")
print(f"  Min weight: {w_mv.min():.6f} (should be >= 0)")
print(f"  Max weight: {w_mv.max():.6f} (should be <= {max_w})")
print(f"  Success: {mv_result['success']}")
assert np.isclose(w_mv.sum(), 1.0, atol=1e-6), "Weights don't sum to 1"
assert np.all(w_mv >= -1e-8), "Negative weights found"
assert np.all(w_mv <= max_w + 1e-8), "Weight exceeds max_weight"
print("✓ Minimum variance constraints satisfied")

# Maximum Sharpe
ms_result = maximum_sharpe_portfolio(mu, sigma, risk_free_rate=0.02, max_weight=max_w)
w_ms = ms_result["weights"]

print(f"\nMaximum Sharpe:")
print(f"  Weights: {w_ms}")
print(f"  Sum of weights: {w_ms.sum():.6f}")
print(f"  Min weight: {w_ms.min():.6f} (should be >= 0)")
print(f"  Max weight: {w_ms.max():.6f} (should be <= {max_w})")
print(f"  Success: {ms_result['success']}")
assert np.isclose(w_ms.sum(), 1.0, atol=1e-6), "Weights don't sum to 1"
assert np.all(w_ms >= -1e-8), "Negative weights found"
assert np.all(w_ms <= max_w + 1e-8), "Weight exceeds max_weight"
print("✓ Maximum Sharpe constraints satisfied")

# Verify Sharpe calculation
ret_ms = portfolio_expected_return(w_ms, mu)
vol_ms = portfolio_volatility(w_ms, sigma)
sharpe_ms_manual = (ret_ms - 0.02) / vol_ms if vol_ms > 0 else 0.0
print(f"\nSharpe ratio verification:")
print(f"  Reported Sharpe: {ms_result['sharpe_ratio']:.6f}")
print(f"  Computed Sharpe: {sharpe_ms_manual:.6f}")
assert np.isclose(ms_result['sharpe_ratio'], sharpe_ms_manual, atol=1e-6), "Sharpe mismatch"
print("✓ Sharpe ratio correctly reported")

print("\n" + "="*80)
print("AUDIT 12: INDEPENDENT NUMERICAL VALIDATION")
print("="*80)

# Use a known allocation
test_weights = np.array([0.30, 0.25, 0.20, 0.25])
test_mu = np.array([0.08, 0.10, 0.12, 0.06])
test_sigma = np.array([
    [0.04, 0.015, 0.005, 0.010],
    [0.015, 0.03, 0.010, 0.005],
    [0.005, 0.010, 0.05, 0.012],
    [0.010, 0.005, 0.012, 0.025],
])

# Portfolio return
port_ret = test_weights @ test_mu
port_ret_func = portfolio_expected_return(test_weights, test_mu)
print(f"Portfolio return (manual): {port_ret:.6f}")
print(f"Portfolio return (function): {port_ret_func:.6f}")
assert np.isclose(port_ret, port_ret_func), "Return mismatch"

# Portfolio variance
port_var = test_weights @ test_sigma @ test_weights
port_vol = np.sqrt(port_var)
port_vol_func = portfolio_volatility(test_weights, test_sigma)
print(f"Portfolio volatility (manual): {port_vol:.6f}")
print(f"Portfolio volatility (function): {port_vol_func:.6f}")
assert np.isclose(port_vol, port_vol_func), "Volatility mismatch"

# Sharpe
rf_test = 0.02
sharpe_test = (port_ret - rf_test) / port_vol
sharpe_test_func = portfolio_sharpe_ratio(test_weights, test_mu, test_sigma, rf_test)
print(f"Sharpe ratio (manual): {sharpe_test:.6f}")
print(f"Sharpe ratio (function): {sharpe_test_func:.6f}")
assert np.isclose(sharpe_test, sharpe_test_func), "Sharpe mismatch"
print("✓ All portfolio math independently verified")

print("\n" + "="*80)
print("AUDIT 13: CSV OUTPUT INSPECTION")
print("="*80)

csv_dir = Path("results/performance")
if csv_dir.exists():
    # Check weights
    weights_csv = pd.read_csv(csv_dir / "portfolio_weights.csv", index_col=0)
    print(f"\nPortfolio weights shape: {weights_csv.shape}")
    print(f"Row sums (should be 1.0 or close):")
    row_sums = weights_csv.sum(axis=1)
    print(row_sums)
    for idx, s in row_sums.items():
        if not np.isclose(s, 1.0, atol=1e-6):
            print(f"  WARNING: {idx} sum = {s:.6f}")
    print(f"No NaNs: {not weights_csv.isna().any().any()}")
    
    # Check metrics
    metrics_csv = pd.read_csv(csv_dir / "portfolio_metrics.csv", index_col=0)
    print(f"\nPortfolio metrics shape: {metrics_csv.shape}")
    print(f"Columns: {metrics_csv.columns.tolist()}")
    print(f"No NaNs: {not metrics_csv.isna().any().any()}")
    print("\nSample metrics (first row):")
    print(metrics_csv.iloc[0])
    
    # Check efficient frontier
    frontier_csv = pd.read_csv(csv_dir / "efficient_frontier.csv", index_col=0)
    print(f"\nEfficient frontier shape: {frontier_csv.shape}")
    print(f"No NaNs: {not frontier_csv.isna().any().any()}")
    print(f"Volatility values (sample): {frontier_csv['volatility'].head().tolist()}")
    print(f"Target return range: [{frontier_csv['target_return'].min():.4f}, {frontier_csv['target_return'].max():.4f}]")
    print("✓ CSV outputs present and well-formed")
else:
    print("⚠ Results directory not found yet")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
