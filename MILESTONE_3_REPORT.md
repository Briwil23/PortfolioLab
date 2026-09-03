# Milestone 3 Report

## 1. Executive Summary
This study was validated in two stages. Reproducibility testing first exposed a canonical baseline discrepancy caused by asset-order labeling. After shared ticker-order normalization was implemented, the corrected canonical baseline and the Milestone 3 classical strategies reproduced within the declared numerical tolerances, and the robust-strategy conclusions were rechecked against the regenerated outputs.

## 2. Research Motivation
Classical Maximum Sharpe showed high turnover, concentration, cap-binding, and expected-return sensitivity in prior diagnostics. Milestone 3 evaluates whether robust mechanisms mitigate these failure modes.

## 3. Hypotheses
H0: Shrinkage and turnover penalties do not materially improve robustness.
H1: Shrinkage and turnover-aware optimization reduce instability while preserving or improving out-of-sample risk-adjusted performance.

## 4. Canonical Dataset and Reproducibility
Canonical dataset: 2015-01-02 to 2026-09-02.
Live-download calls during canonical run: 0.
Historical Milestone 2 note: Historical Milestone 2 outputs generated from live market data; exact source-data snapshot was not preserved.
Validation sequence: reproducibility testing exposed a canonical baseline discrepancy, root cause analysis traced it to asset-order labeling, shared ticker-order normalization was implemented, permutation-invariance tests were added, and the baseline and experimental pipelines were regenerated under the corrected shared path.

## 5. Experimental Design
Leakage-safe monthly walk-forward with 3-year lookback, long-only full-investment constraints, and max weight 30%.
Strategies include baseline, shrinkage grid, turnover-aware grid, and combined center-point variant.

## 6. Classical Baseline
| strategy | cagr | annualized_volatility | sharpe | sortino | max_drawdown | calmar | cumulative_return | terminal_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 0.140891 | 0.191280 | 0.632011 | 0.594807 | -0.337173 | 0.417860 | 2.091791 | 3.091791 |
| Equal Weight | 0.085794 | 0.125086 | 0.525993 | 0.495136 | -0.255194 | 0.336193 | 1.023599 | 2.023599 |
| Minimum Variance | 0.057471 | 0.080486 | 0.465564 | 0.447203 | -0.221996 | 0.258885 | 0.613704 | 1.613704 |
| Maximum Sharpe | 0.097030 | 0.131589 | 0.585388 | 0.553651 | -0.255800 | 0.379322 | 1.210101 | 2.210101 |

## 7. Expected-Return Shrinkage Results
| strategy | mean_monthly_turnover | pct_reduction_vs_max_sharpe |
| --- | --- | --- |
| Maximum Sharpe | 0.101372 |  |
| Shrunk Max Sharpe λ=0.25 | 0.095925 | 5.373024 |
| Shrunk Max Sharpe λ=0.50 | 0.086662 | 14.510697 |
| Shrunk Max Sharpe λ=0.75 | 0.081984 | 19.125035 |

## 8. Turnover-Aware Results
| strategy | mean_monthly_turnover | pct_reduction_vs_max_sharpe |
| --- | --- | --- |
| Maximum Sharpe | 0.101372 |  |
| Turnover-Aware Max Sharpe γ=0.05 | 0.073639 | 27.357751 |
| Turnover-Aware Max Sharpe γ=0.10 | 0.053435 | 47.287554 |
| Turnover-Aware Max Sharpe γ=0.25 | 0.032137 | 68.297753 |

## 9. Combined Robust Strategy
| strategy | cagr | annualized_volatility | sharpe | sortino | max_drawdown | calmar | cumulative_return | terminal_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maximum Sharpe | 0.097030 | 0.131589 | 0.585388 | 0.553651 | -0.255800 | 0.379322 | 1.210101 | 2.210101 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.096777 | 0.108820 | 0.705538 | 0.684938 | -0.223658 | 0.432700 | 1.205727 | 2.205727 |

## 10. Transaction-Cost Analysis
| strategy | cost_bps | terminal_wealth | cagr | sharpe | total_estimated_transaction_cost_drag | gross_terminal_wealth |
| --- | --- | --- | --- | --- | --- | --- |
| Maximum Sharpe | 0.000000 | 2.210101 | 0.097030 | 0.585388 | 0.000000 | 2.210101 |
| Maximum Sharpe | 5.000000 | 2.202613 | 0.096596 | 0.582087 | 0.003396 | 2.210101 |
| Maximum Sharpe | 10.000000 | 2.195149 | 0.096161 | 0.578785 | 0.006792 | 2.210101 |
| Maximum Sharpe | 25.000000 | 2.172903 | 0.094858 | 0.568876 | 0.016980 | 2.210101 |
| Shrunk Max Sharpe λ=0.50 | 0.000000 | 2.068792 | 0.088599 | 0.633824 | 0.000000 | 2.068792 |
| Shrunk Max Sharpe λ=0.50 | 5.000000 | 2.062798 | 0.088230 | 0.630420 | 0.002903 | 2.068792 |
| Shrunk Max Sharpe λ=0.50 | 10.000000 | 2.056820 | 0.087861 | 0.627016 | 0.005806 | 2.068792 |
| Shrunk Max Sharpe λ=0.50 | 25.000000 | 2.038988 | 0.086755 | 0.616796 | 0.014516 | 2.068792 |
| Turnover-Aware Max Sharpe γ=0.10 | 0.000000 | 2.288588 | 0.101510 | 0.621638 | 0.000000 | 2.288588 |
| Turnover-Aware Max Sharpe γ=0.10 | 5.000000 | 2.284494 | 0.101280 | 0.619877 | 0.001790 | 2.288588 |
| Turnover-Aware Max Sharpe γ=0.10 | 10.000000 | 2.280407 | 0.101049 | 0.618115 | 0.003580 | 2.288588 |
| Turnover-Aware Max Sharpe γ=0.10 | 25.000000 | 2.268187 | 0.100359 | 0.612827 | 0.008950 | 2.288588 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.000000 | 2.205727 | 0.096777 | 0.705538 | 0.000000 | 2.205727 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 5.000000 | 2.202447 | 0.096586 | 0.703773 | 0.001487 | 2.205727 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 10.000000 | 2.199172 | 0.096396 | 0.702008 | 0.002973 | 2.205727 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 25.000000 | 2.189374 | 0.095824 | 0.696707 | 0.007433 | 2.205727 |

## 11. Concentration and Risk Diversification
| strategy | mean_hhi | median_hhi | max_hhi | mean_effective_holdings | median_effective_holdings | min_effective_holdings | average_largest_position | maximum_largest_position |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maximum Sharpe | 0.263796 | 0.263271 | 0.280000 | 3.802906 | 3.798363 | 3.571429 | 0.300000 | 0.300000 |
| Shrunk Max Sharpe λ=0.50 | 0.255375 | 0.260868 | 0.280000 | 3.959471 | 3.833362 | 3.571429 | 0.300000 | 0.300000 |
| Turnover-Aware Max Sharpe γ=0.10 | 0.262470 | 0.263557 | 0.280000 | 3.823388 | 3.794239 | 3.571429 | 0.300000 | 0.300000 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.251352 | 0.258822 | 0.280000 | 4.021201 | 3.863663 | 3.571429 | 0.300000 | 0.300000 |

## 12. Parameter Sensitivity
| strategy | mean | median | min | max |
| --- | --- | --- | --- | --- |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.737919 | 0.713853 | 0.538412 | 0.900000 |
| Maximum Sharpe | 0.721353 | 0.730350 | 0.444021 | 0.901314 |
| Shrunk Max Sharpe λ=0.50 | 0.731102 | 0.706539 | 0.510223 | 0.940966 |
| Turnover-Aware Max Sharpe γ=0.10 | 0.715609 | 0.727689 | 0.390581 | 0.909127 |

## 13. Stress-Period Results
| stress_period | start_date | end_date | strategy | cumulative_return | annualized_volatility | max_drawdown | entry_rebalance_date | top_positions_at_entry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | SPY | -0.233713 | 0.792570 | -0.334439 | 2019-11-01 | SPY:1.000, EEM:0.000, EFA:0.000 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Maximum Sharpe | -0.082049 | 0.381412 | -0.177489 | 2019-11-01 | LQD:0.300, SPY:0.275, TLT:0.184 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Shrunk Max Sharpe λ=0.50 | -0.055697 | 0.292989 | -0.143205 | 2019-11-01 | IEF:0.300, LQD:0.300, SPY:0.273 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Turnover-Aware Max Sharpe γ=0.10 | -0.084077 | 0.386960 | -0.179480 | 2019-11-01 | LQD:0.300, SPY:0.290, TLT:0.189 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Combined Robust Max Sharpe λ=0.50 γ=0.10 | -0.056737 | 0.295114 | -0.144038 | 2019-11-01 | IEF:0.300, LQD:0.300, SPY:0.277 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | SPY | -0.137650 | 0.256123 | -0.212847 | 2022-04-01 | SPY:1.000, EEM:0.000, EFA:0.000 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Maximum Sharpe | -0.168121 | 0.186961 | -0.212058 | 2022-04-01 | QQQ:0.300, GLD:0.300, TLT:0.269 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Shrunk Max Sharpe λ=0.50 | -0.146076 | 0.144794 | -0.174622 | 2022-04-01 | IEF:0.300, GLD:0.256, SPY:0.175 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Turnover-Aware Max Sharpe γ=0.10 | -0.168086 | 0.176476 | -0.206744 | 2022-04-01 | QQQ:0.300, GLD:0.300, TLT:0.225 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Combined Robust Max Sharpe λ=0.50 γ=0.10 | -0.150276 | 0.139725 | -0.176197 | 2022-04-01 | IEF:0.300, GLD:0.237, SPY:0.177 |

## 14. Limitations
This is a historical study on one canonical dataset and does not guarantee future performance.

## 15. Hypothesis Assessment
{
  "turnover_reduction_combined_vs_max_sharpe": true,
  "concentration_reduction_combined_vs_max_sharpe": true,
  "sensitivity_reduction_combined_vs_max_sharpe": false,
  "net_cost_drag_reduction_10bps": true
}

Baseline consistency after the ordering fix: the corrected canonical baseline reproduces the Milestone 3 classical strategies within the predeclared numerical tolerances.

## 16. Research Conclusions
Within this historical canonical walk-forward study, evidence supports meaningful robustness gains from turnover control and combined robust construction across multiple instability dimensions. The earlier baseline discrepancy was a validation artifact caused by asset-order labeling, not a different economic solution. These results should be interpreted as historical evidence on this canonical dataset only and not as a guarantee of future alpha.