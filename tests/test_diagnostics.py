import numpy as np
import pandas as pd

from src.diagnostics.diagnostic_study import (
    compute_hhi_metrics,
    compute_turnover,
    compute_risk_contributions,
)


def test_turnover_is_nonnegative_and_symmetric():
    w0 = pd.Series({"A": 0.5, "B": 0.5})
    w1 = pd.Series({"A": 0.8, "B": 0.2})
    turnover = compute_turnover(w0, w1)
    assert np.isclose(turnover, 0.3), turnover


def test_hhi_and_effective_holdings_values():
    weights = pd.Series({"A": 0.5, "B": 0.25, "C": 0.25})
    hhi = compute_hhi_metrics(weights)
    assert np.isclose(hhi["HHI"], 0.375), hhi
    assert np.isclose(hhi["effective_holdings"], 1.0 / 0.375), hhi


def test_risk_contributions_sum_to_portfolio_risk():
    w = pd.Series({"A": 0.6, "B": 0.4})
    sigma = pd.DataFrame([[0.20, 0.05], [0.05, 0.12]], index=["A", "B"], columns=["A", "B"])
    contrib = compute_risk_contributions(w, sigma)
    assert contrib["pct_total_risk"].sum() > 0.99
    assert contrib["pct_total_risk"].sum() < 1.01
