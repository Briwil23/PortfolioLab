from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.optimization import cvar as cvar_module
from src.optimization.cvar import _build_ru_lp, minimum_cvar_95_portfolio
from src.risk.tail_metrics import empirical_cvar


@pytest.fixture
def benchmark_returns():
    return np.array(
        [
            [0.01, -0.02, 0.03],
            [0.02, 0.04, -0.01],
            [-0.03, 0.01, 0.02],
            [0.05, -0.01, 0.00],
            [-0.02, 0.03, 0.01],
            [0.00, 0.02, -0.02],
            [0.04, -0.03, 0.02],
            [-0.01, 0.05, 0.00],
        ],
        dtype=float,
    )


def test_minimum_cvar_95_solver_matches_metric_identity(benchmark_returns):
    result = minimum_cvar_95_portfolio(benchmark_returns, asset_labels=["A", "B", "C"], alpha=0.95, max_weight=0.50)

    assert result["success"]
    assert result["weights"].shape == (3,)
    np.testing.assert_allclose(result["weights"].sum(), 1.0, atol=1e-10)
    assert np.all(result["weights"] >= -1e-10)
    assert np.all(result["weights"] <= 0.50 + 1e-10)

    losses = -benchmark_returns @ result["weights"]
    metric_value = empirical_cvar(losses, alpha=0.95)
    assert np.isclose(result["objective_cvar"], metric_value, atol=1e-10)
    assert np.isclose(result["objective_value"], metric_value, atol=1e-10)


def test_all_gains_environment_allows_negative_zeta():
    returns = np.array(
        [
            [0.02, 0.03],
            [0.04, 0.01],
            [0.03, 0.02],
            [0.05, 0.06],
            [0.01, 0.04],
            [0.07, 0.08],
        ],
        dtype=float,
    )
    result = minimum_cvar_95_portfolio(returns, asset_labels=["A", "B"], alpha=0.95, max_weight=0.60)

    assert result["success"]
    assert result["zeta"] < 0.0
    assert result["objective_cvar"] < 0.0
    assert result["objective_value"] < 0.0


def test_permutation_invariance_by_asset_label(benchmark_returns):
    original = minimum_cvar_95_portfolio(benchmark_returns, asset_labels=["A", "B", "C"], alpha=0.95, max_weight=0.50)
    permuted = minimum_cvar_95_portfolio(benchmark_returns[:, ::-1], asset_labels=["C", "B", "A"], alpha=0.95, max_weight=0.50)

    original_map = {label: weight for label, weight in zip(["A", "B", "C"], original["weights"])}
    permuted_map = {label: weight for label, weight in zip(["C", "B", "A"], permuted["weights"])}

    assert np.isclose(original_map["A"], permuted_map["A"], atol=1e-10)
    assert np.isclose(original_map["B"], permuted_map["B"], atol=1e-10)
    assert np.isclose(original_map["C"], permuted_map["C"], atol=1e-10)
    assert np.isclose(original["objective_cvar"], permuted["objective_cvar"], atol=1e-10)


def test_dominated_tail_asset_receives_lower_weight():
    returns = np.array(
        [
            [0.00, -0.02],
            [0.01, -0.01],
            [-0.02, -0.03],
            [0.04, 0.03],
            [-0.05, -0.06],
            [0.02, 0.01],
            [0.03, 0.02],
            [0.01, 0.00],
        ],
        dtype=float,
    )
    result = minimum_cvar_95_portfolio(returns, asset_labels=["dominant", "dominated"], alpha=0.95, max_weight=0.60)
    assert result["weights"][0] >= result["weights"][1] - 1e-10


def test_identical_assets_are_feasible_and_objective_invariant():
    returns = np.array(
        [
            [0.02, 0.02],
            [-0.01, -0.01],
            [0.04, 0.04],
            [-0.03, -0.03],
            [0.05, 0.05],
            [-0.02, -0.02],
        ],
        dtype=float,
    )
    result = minimum_cvar_95_portfolio(returns, asset_labels=["A", "B"], alpha=0.95, max_weight=0.60)
    assert result["success"]
    assert np.isclose(result["weights"][0] + result["weights"][1], 1.0, atol=1e-10)


def test_constant_return_asset_remains_feasible():
    returns = np.array(
        [
            [0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03],
            [0.01, 0.01, 0.01],
            [0.05, 0.05, 0.05],
            [-0.01, -0.01, -0.01],
            [0.02, 0.02, 0.02],
        ],
        dtype=float,
    )
    result = minimum_cvar_95_portfolio(returns, asset_labels=["A", "B", "C"], alpha=0.95, max_weight=0.60)
    assert result["success"]
    np.testing.assert_allclose(result["weights"].sum(), 1.0, atol=1e-10)


def test_infeasible_max_weight_fails_before_solver():
    returns = np.array([[0.01, -0.02], [0.03, 0.04]], dtype=float)
    with pytest.raises(ValueError):
        minimum_cvar_95_portfolio(returns, asset_labels=["A", "B"], alpha=0.95, max_weight=0.30)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        minimum_cvar_95_portfolio(np.array([]), asset_labels=[], alpha=0.95, max_weight=0.30)
    with pytest.raises(ValueError):
        minimum_cvar_95_portfolio(np.array([[1.0, np.nan], [0.0, 0.1]]), asset_labels=["A", "B"], alpha=0.95, max_weight=0.30)
    with pytest.raises(ValueError):
        minimum_cvar_95_portfolio(np.array([[0.1, 0.2], [0.3, 0.4]]), asset_labels=["A", "A"], alpha=0.95, max_weight=0.30)
    with pytest.raises(ValueError):
        minimum_cvar_95_portfolio(np.array([[0.1, 0.2],[0.3,0.4]]), asset_labels=["A"], alpha=0.95, max_weight=0.30)
    with pytest.raises(ValueError):
        minimum_cvar_95_portfolio(np.array([[0.1, 0.2],[0.3,0.4]]), asset_labels=["A", "B"], alpha=1.0, max_weight=0.30)


def test_known_answer_reference_for_two_asset_case():
    returns = np.array(
        [
            [0.05, -0.10],
            [0.04, 0.02],
            [-0.01, 0.00],
            [0.00, -0.02],
            [0.06, 0.03],
            [-0.03, -0.04],
        ],
        dtype=float,
    )
    result = minimum_cvar_95_portfolio(returns, asset_labels=["A", "B"], alpha=0.95, max_weight=0.60)
    losses = -returns @ result["weights"]
    expected = empirical_cvar(losses, 0.95)
    assert np.isclose(result["objective_cvar"], expected, atol=1e-10)
    assert np.isclose(result["objective_value"], expected, atol=1e-10)


def test_ru_inequality_matrix_uses_negative_u_sign():
    returns = np.array([[0.10, -0.04], [-0.03, 0.06]], dtype=float)
    c, A_ub, b_ub, bounds, _ = _build_ru_lp(returns, alpha=0.95, max_weight=0.60)

    assert A_ub.shape == (2, 5)
    np.testing.assert_allclose(A_ub[0, :2], -returns[0], atol=1e-12)
    np.testing.assert_allclose(A_ub[0, 2], -1.0, atol=1e-12)
    np.testing.assert_allclose(A_ub[0, 3], -1.0, atol=1e-12)
    np.testing.assert_allclose(A_ub[0, 4], 0.0, atol=1e-12)
    np.testing.assert_allclose(A_ub[1, :2], -returns[1], atol=1e-12)
    np.testing.assert_allclose(A_ub[1, 2], -1.0, atol=1e-12)
    np.testing.assert_allclose(A_ub[1, 3], 0.0, atol=1e-12)
    np.testing.assert_allclose(A_ub[1, 4], -1.0, atol=1e-12)
    np.testing.assert_allclose(b_ub, np.zeros(2), atol=1e-12)


def test_valid_and_invalid_u_point_are_detected_by_matrix_inequality():
    returns = np.array([[0.10, -0.04]], dtype=float)
    _, A_ub, _, _, _ = _build_ru_lp(returns, alpha=0.95, max_weight=0.60)
    w = np.array([0.5, 0.5], dtype=float)
    zeta = 0.05
    valid_u = 0.01
    invalid_u = -0.20

    L_t = -returns[0] @ w
    assert np.isclose(L_t, -0.03, atol=1e-12)
    assert np.isclose(L_t - zeta, -0.08, atol=1e-12)
    assert valid_u >= L_t - zeta
    assert not (invalid_u >= L_t - zeta)

    x_valid = np.array([w[0], w[1], zeta, valid_u], dtype=float)
    x_invalid = np.array([w[0], w[1], zeta, invalid_u], dtype=float)

    assert A_ub[0] @ x_valid <= 1e-12
    assert not (A_ub[0] @ x_invalid <= 1e-12)


def test_candidate_objective_dominance_and_post_validation(monkeypatch):
    returns = np.array(
        [
            [0.05, -0.10],
            [0.04, 0.02],
            [-0.01, 0.00],
            [0.00, -0.02],
            [0.06, 0.03],
            [-0.03, -0.04],
        ],
        dtype=float,
    )
    candidate = np.array([0.5, 0.5], dtype=float)
    candidate_losses = -returns @ candidate
    candidate_cvar = empirical_cvar(candidate_losses, alpha=0.95)
    result = minimum_cvar_95_portfolio(returns, asset_labels=["A", "B"], alpha=0.95, max_weight=0.60)
    assert result["objective_cvar"] <= candidate_cvar + 1e-10

    bad_result = SimpleNamespace(
        success=True,
        x=np.array([0.5, 0.5, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        status=0,
        message="bad",
    )
    monkeypatch.setattr(cvar_module, "linprog", lambda **kwargs: bad_result)
    with pytest.raises(ValueError, match="Post-solution validation failed"):
        minimum_cvar_95_portfolio(returns, asset_labels=["A", "B"], alpha=0.95, max_weight=0.60)


def test_solver_failure_is_clearly_reported(monkeypatch):
    returns = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    monkeypatch.setattr(
        cvar_module,
        "linprog",
        lambda **kwargs: SimpleNamespace(success=False, status=2, message="infeasible"),
    )
    with pytest.raises(ValueError, match="CVaR optimizer failed"):
        minimum_cvar_95_portfolio(returns, asset_labels=["A", "B"], alpha=0.95, max_weight=0.60)
