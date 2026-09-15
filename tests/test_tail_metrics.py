import math

import numpy as np
import pytest

from src.risk.tail_metrics import (
    downside_deviation,
    empirical_cvar,
    empirical_var,
    worst_compounded_n_day_return,
    worst_daily_return,
)


def test_empirical_var_ordered_losses():
    losses = np.array([0.10, 0.20, 0.50, 0.80, 1.10], dtype=float)
    assert np.isclose(empirical_var(losses, 0.95), 1.10, atol=1e-12)


def test_empirical_var_ties_and_zero_based_index_rule():
    losses = np.array([0.10, 0.30, 0.30, 0.90, 0.90], dtype=float)
    assert np.isclose(empirical_var(losses, 0.80), 0.90, atol=1e-12)


def test_empirical_cvar_matches_reference_for_ordered_sample():
    losses = np.array([0.10, 0.20, 0.50, 0.80, 1.10], dtype=float)
    assert np.isclose(empirical_cvar(losses, 0.95), 1.10, atol=1e-12)


def test_empirical_cvar_tie_at_var_threshold():
    losses = np.array([0.10, 0.20, 0.50, 0.50, 0.50, 0.90], dtype=float)
    alpha = 0.80
    expected = 0.50 + (1.0 / ((1.0 - alpha) * len(losses))) * np.sum(np.maximum(losses - 0.50, 0.0))
    assert np.isclose(empirical_var(losses, alpha), 0.50, atol=1e-12)
    assert np.isclose(empirical_cvar(losses, alpha), expected, atol=1e-12)
    assert empirical_cvar(losses, alpha) > empirical_var(losses, alpha)
    strict_tail_mean = losses[losses > 0.50].mean() if np.any(losses > 0.50) else 0.50
    assert np.isclose(strict_tail_mean, 0.90, atol=1e-12)
    assert np.isclose((losses[losses >= 0.50].mean()), 0.60, atol=1e-12)


def test_empirical_cvar_strict_tail_average_differs_from_ru():
    losses = np.array([0.10, 0.20, 0.50, 0.50, 0.50, 0.90], dtype=float)
    alpha = 0.80
    vaR = empirical_var(losses, alpha)
    strict_tail_mean = losses[losses > vaR].mean() if np.any(losses > vaR) else vaR
    ru = empirical_cvar(losses, alpha)
    assert strict_tail_mean != ru
    assert np.isclose(ru, 0.8333333333333333, atol=1e-12)


def test_all_gains_loss_vector():
    returns = np.array([0.05, 0.10, 0.20], dtype=float)
    losses = -returns
    assert np.isclose(empirical_var(losses, 0.95), -0.05, atol=1e-12)
    assert np.isclose(empirical_cvar(losses, 0.95), -0.05, atol=1e-12)


def test_all_losses():
    losses = np.array([0.10, 0.20, 0.30, 0.50], dtype=float)
    assert np.isclose(empirical_var(losses, 0.95), 0.50, atol=1e-12)
    assert np.isclose(empirical_cvar(losses, 0.95), 0.50, atol=1e-12)


def test_repeated_identical_losses():
    losses = np.array([0.30, 0.30, 0.30, 0.30], dtype=float)
    alpha = 0.95
    assert np.isclose(empirical_var(losses, alpha), 0.30, atol=1e-12)
    assert np.isclose(empirical_cvar(losses, alpha), 0.30, atol=1e-12)


def test_single_outlier_is_worst_tail():
    losses = np.array([0.01, 0.02, 0.03, 0.04, 2.00], dtype=float)
    assert np.isclose(empirical_var(losses, 0.95), 2.00, atol=1e-12)
    assert np.isclose(empirical_cvar(losses, 0.95), 2.00, atol=1e-12)


def test_small_sample_alpha_not_valid():
    losses = np.array([0.10, 0.20, 0.30])
    with pytest.raises(ValueError):
        empirical_var(losses, 1.0)
    with pytest.raises(ValueError):
        empirical_cvar(losses, 1.0)


def test_invalid_alpha_values_rejected():
    losses = np.array([0.10, 0.20, 0.30])
    with pytest.raises(ValueError):
        empirical_var(losses, -0.1)
    with pytest.raises(ValueError):
        empirical_cvar(losses, 1.2)


def test_nan_and_inf_rejected():
    arr = np.array([0.10, np.nan, 0.30], dtype=float)
    with pytest.raises(ValueError):
        empirical_var(arr, 0.95)
    with pytest.raises(ValueError):
        empirical_cvar(arr, 0.95)

    arr2 = np.array([0.10, np.inf, 0.30], dtype=float)
    with pytest.raises(ValueError):
        empirical_var(arr2, 0.95)
    with pytest.raises(ValueError):
        empirical_cvar(arr2, 0.95)


def test_empty_series_rejected():
    with pytest.raises(ValueError):
        empirical_var(np.array([], dtype=float), 0.95)
    with pytest.raises(ValueError):
        empirical_cvar(np.array([], dtype=float), 0.95)


def test_translational_invariance():
    losses = np.array([0.10, 0.20, 0.50, 0.80], dtype=float)
    c = 0.25
    assert np.isclose(empirical_var(losses + c, 0.95), empirical_var(losses, 0.95) + c, atol=1e-12)
    assert np.isclose(empirical_cvar(losses + c, 0.95), empirical_cvar(losses, 0.95) + c, atol=1e-12)


def test_positive_homogeneity():
    losses = np.array([0.10, 0.20, 0.50, 0.80], dtype=float)
    lam = 2.0
    assert np.isclose(empirical_var(lam * losses, 0.95), lam * empirical_var(losses, 0.95), atol=1e-12)
    assert np.isclose(empirical_cvar(lam * losses, 0.95), lam * empirical_cvar(losses, 0.95), atol=1e-12)


def test_cvar_ge_var():
    samples = [
        np.array([0.05, 0.10, 0.20, 0.40, 0.80], dtype=float),
        np.array([0.00, 0.10, 0.10, 0.20, 0.60], dtype=float),
        np.array([0.10, 0.20, 0.50, 0.50, 0.90], dtype=float),
    ]
    for losses in samples:
        assert empirical_cvar(losses, 0.95) >= empirical_var(losses, 0.95) - 1e-12


def test_downside_deviation_matches_negative_moment_convention():
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05], dtype=float)
    expected = math.sqrt(np.mean(np.square(np.array([-0.02, -0.04], dtype=float))))
    assert np.isclose(downside_deviation(returns), expected, atol=1e-12)


def test_worst_daily_return_signed():
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05], dtype=float)
    assert np.isclose(worst_daily_return(returns), -0.04, atol=1e-12)


def test_worst_compounded_5_day_and_21_day_returns():
    returns = np.array([0.10, -0.20, 0.15, -0.30, 0.25, -0.10, 0.08, 0.02], dtype=float)
    expected_5 = min(
        [
            np.prod(1.0 + returns[i : i + 5]) - 1.0
            for i in range(len(returns) - 4)
        ]
    )
    assert np.isclose(worst_compounded_n_day_return(returns, 5), expected_5, atol=1e-12)

    with pytest.raises(ValueError):
        worst_compounded_n_day_return(returns, 21)


def test_insufficient_history_for_window_rejected():
    returns = np.array([0.05, -0.01, 0.02], dtype=float)
    with pytest.raises(ValueError):
        worst_compounded_n_day_return(returns, 5)


def test_bad_input_types_rejected():
    with pytest.raises(ValueError):
        empirical_var(["a", "b", "c"], 0.95)
    with pytest.raises(ValueError):
        empirical_cvar(["a", "b", "c"], 0.95)
    with pytest.raises(ValueError):
        downside_deviation(["a", "b"])
    with pytest.raises(ValueError):
        worst_daily_return(["a", "b"])
    with pytest.raises(ValueError):
        worst_compounded_n_day_return(["a", "b"], 2)
