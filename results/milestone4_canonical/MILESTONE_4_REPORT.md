# Milestone 4 Report

## Executive Summary
This canonical walk-forward study evaluates whether risk-only construction methods reduce sensitivity, turnover, concentration, and downside instability relative to expected-return-dependent strategies under the locked Milestone 4 specification.

## Research Question
Can portfolio-construction methods that do not require expected-return estimates produce more stable and robust out-of-sample portfolios?

## Hypothesis
H0: Risk-based portfolio construction does not materially improve robustness relative to expected-return-dependent portfolio construction.
H1: Risk-based construction reduces estimation sensitivity, turnover, concentration, and/or downside instability while maintaining competitive out-of-sample risk-adjusted performance.

## Locked Methodology
Canonical dataset only; 10-asset universe; 3-year rolling lookback; monthly rebalancing; long-only; fully invested; 30% max weight; 252 trading days/year; shared normalize_asset_order() path; target-to-target turnover convention; transaction costs of 0/5/10/25 bps.

## Reproducibility
| canonical_hashes_validated | live_data_calls | oos_start | oos_end | oos_observations | rebalance_count | baseline_return_tolerance | baseline_weight_tolerance | baseline_equal_weight_return_diff | baseline_min_variance_return_diff | baseline_max_sharpe_return_diff | baseline_min_variance_weight_diff | baseline_max_sharpe_weight_diff | combined_robust_return_diff | combined_robust_weight_diff | baseline_consistency_pass | optimizer_failures | optimizer_fallbacks | erc_rebalance_attempts | erc_solver_successes | erc_solver_failures | erc_fallback_count | erc_fallback_dates | erc_fallback_reasons | inverse_vol_rebalance_attempts | inverse_vol_failures | run1_vs_run2_max_return_diff | run1_vs_run2_max_weight_diff | run1_snapshot_recorded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 2018-02-01 | 2026-09-02 | 2158 | 67 | 0.000000 | 0.000001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1 | 0 | 0 | 67 | 67 | 0 | 0 | [] | [] | 67 | 0 | 0.000000 | 0.000000 | 0 |

## Baseline Preservation
| actual_equal_weight_return_diff | actual_min_variance_return_diff | actual_max_sharpe_return_diff | actual_min_variance_weight_diff | actual_max_sharpe_weight_diff | combined_robust_return_diff | combined_robust_weight_diff |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

The corrected canonical baseline walk-forward return artifact does not contain SPY as a saved strategy-return column. SPY preservation is therefore verified through the documented benchmark/schema contract rather than through the keyed numeric return comparison used for Equal Weight, Minimum Variance, and Maximum Sharpe.

Combined Robust Max Sharpe λ=0.50 γ=.10 reproduced the closed Milestone 3 canonical artifact exactly: maximum daily return difference = 0.0 and maximum portfolio weight difference = 0.0. This confirms that Milestone 4 did not alter the previously closed robust strategy.

## Strategy Definitions
Primary strategies: SPY, Equal Weight, Minimum Variance, Maximum Sharpe, Combined Robust Max Sharpe λ=.50 γ=.10, Inverse Volatility, and Equal Risk Contribution.

## Gross Results
| strategy | cagr | annualized_volatility | sharpe | sortino | max_drawdown | calmar | cumulative_return | terminal_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 0.140891 | 0.191280 | 0.632011 | 0.594807 | -0.337173 | 0.417860 | 2.091791 | 3.091791 |
| Equal Weight | 0.085794 | 0.125086 | 0.525993 | 0.495136 | -0.255194 | 0.336193 | 1.023599 | 2.023599 |
| Minimum Variance | 0.057471 | 0.080486 | 0.465564 | 0.447203 | -0.221996 | 0.258885 | 0.613704 | 1.613704 |
| Maximum Sharpe | 0.097030 | 0.131589 | 0.585388 | 0.553651 | -0.255800 | 0.379322 | 1.210101 | 2.210101 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.096777 | 0.108820 | 0.705538 | 0.684938 | -0.223658 | 0.432700 | 1.205727 | 2.205727 |
| Inverse Volatility | 0.069586 | 0.098693 | 0.502432 | 0.471943 | -0.235633 | 0.295316 | 0.779057 | 1.779057 |
| Equal Risk Contribution | 0.063781 | 0.092940 | 0.471068 | 0.443458 | -0.237624 | 0.268412 | 0.698044 | 1.698044 |

## Transaction-Cost Results
| strategy | cost_bps | cagr | annualized_volatility | sharpe | sortino | max_drawdown | calmar | cumulative_return | terminal_wealth | gross_terminal_wealth | cost_drag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 0.000000 | 0.140891 | 0.191280 | 0.632011 | 0.594807 | -0.337173 | 0.417860 | 2.091791 | 3.091791 | 3.091791 | 0.000000 |
| SPY | 5.000000 | 0.140891 | 0.191280 | 0.632011 | 0.594807 | -0.337173 | 0.417860 | 2.091791 | 3.091791 | 3.091791 | 0.000000 |
| SPY | 10.000000 | 0.140891 | 0.191280 | 0.632011 | 0.594807 | -0.337173 | 0.417860 | 2.091791 | 3.091791 | 3.091791 | 0.000000 |
| SPY | 25.000000 | 0.140891 | 0.191280 | 0.632011 | 0.594807 | -0.337173 | 0.417860 | 2.091791 | 3.091791 | 3.091791 | 0.000000 |
| Equal Weight | 0.000000 | 0.085794 | 0.125086 | 0.525993 | 0.495136 | -0.255194 | 0.336193 | 1.023599 | 2.023599 | 2.023599 | 0.000000 |
| Equal Weight | 5.000000 | 0.085794 | 0.125086 | 0.525993 | 0.495136 | -0.255194 | 0.336193 | 1.023599 | 2.023599 | 2.023599 | 0.000000 |
| Equal Weight | 10.000000 | 0.085794 | 0.125086 | 0.525993 | 0.495136 | -0.255194 | 0.336193 | 1.023599 | 2.023599 | 2.023599 | 0.000000 |
| Equal Weight | 25.000000 | 0.085794 | 0.125086 | 0.525993 | 0.495136 | -0.255194 | 0.336193 | 1.023599 | 2.023599 | 2.023599 | 0.000000 |
| Minimum Variance | 0.000000 | 0.057471 | 0.080486 | 0.465564 | 0.447203 | -0.221996 | 0.258885 | 0.613704 | 1.613704 | 1.613704 | 0.000000 |
| Minimum Variance | 5.000000 | 0.057372 | 0.080487 | 0.464321 | 0.445980 | -0.222050 | 0.258372 | 0.612401 | 1.612401 | 1.613704 | 0.001302 |
| Minimum Variance | 10.000000 | 0.057272 | 0.080487 | 0.463077 | 0.444756 | -0.222105 | 0.257859 | 0.611100 | 1.611100 | 1.613704 | 0.002604 |
| Minimum Variance | 25.000000 | 0.056973 | 0.080490 | 0.459345 | 0.441085 | -0.222269 | 0.256324 | 0.607202 | 1.607202 | 1.613704 | 0.006502 |
| Maximum Sharpe | 0.000000 | 0.097030 | 0.131589 | 0.585388 | 0.553651 | -0.255800 | 0.379322 | 1.210101 | 2.210101 | 2.210101 | 0.000000 |
| Maximum Sharpe | 5.000000 | 0.096596 | 0.131588 | 0.582087 | 0.550482 | -0.256062 | 0.377235 | 1.202613 | 2.202613 | 2.210101 | 0.007489 |
| Maximum Sharpe | 10.000000 | 0.096161 | 0.131588 | 0.578785 | 0.547596 | -0.256324 | 0.375154 | 1.195149 | 2.195149 | 2.210101 | 0.014953 |
| Maximum Sharpe | 25.000000 | 0.094858 | 0.131589 | 0.568876 | 0.538081 | -0.257111 | 0.368939 | 1.172903 | 2.172903 | 2.210101 | 0.037198 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.000000 | 0.096777 | 0.108820 | 0.705538 | 0.684938 | -0.223658 | 0.432700 | 1.205727 | 2.205727 | 2.205727 | 0.000000 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 5.000000 | 0.096586 | 0.108822 | 0.703773 | 0.683195 | -0.223865 | 0.431449 | 1.202447 | 2.202447 | 2.205727 | 0.003280 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 10.000000 | 0.096396 | 0.108824 | 0.702008 | 0.681449 | -0.224071 | 0.430200 | 1.199172 | 2.199172 | 2.205727 | 0.006555 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 25.000000 | 0.095824 | 0.108832 | 0.696707 | 0.676206 | -0.224692 | 0.426469 | 1.189374 | 2.189374 | 2.205727 | 0.016353 |
| Inverse Volatility | 0.000000 | 0.069586 | 0.098693 | 0.502432 | 0.471943 | -0.235633 | 0.295316 | 0.779057 | 1.779057 | 1.779057 | 0.000000 |
| Inverse Volatility | 5.000000 | 0.069556 | 0.098694 | 0.502118 | 0.471638 | -0.235645 | 0.295172 | 0.778624 | 1.778624 | 1.779057 | 0.000433 |
| Inverse Volatility | 10.000000 | 0.069526 | 0.098695 | 0.501804 | 0.471333 | -0.235657 | 0.295028 | 0.778191 | 1.778191 | 1.779057 | 0.000866 |
| Inverse Volatility | 25.000000 | 0.069434 | 0.098698 | 0.500862 | 0.470418 | -0.235693 | 0.294596 | 0.776892 | 1.776892 | 1.779057 | 0.002164 |
| Equal Risk Contribution | 0.000000 | 0.063781 | 0.092940 | 0.471068 | 0.443458 | -0.237624 | 0.268412 | 0.698044 | 1.698044 | 1.698044 | 0.000000 |
| Equal Risk Contribution | 5.000000 | 0.063744 | 0.092941 | 0.470663 | 0.443064 | -0.237637 | 0.268241 | 0.697537 | 1.697537 | 1.698044 | 0.000507 |
| Equal Risk Contribution | 10.000000 | 0.063707 | 0.092942 | 0.470258 | 0.442671 | -0.237650 | 0.268070 | 0.697030 | 1.697030 | 1.698044 | 0.001014 |
| Equal Risk Contribution | 25.000000 | 0.063595 | 0.092946 | 0.469042 | 0.441491 | -0.237691 | 0.267556 | 0.695510 | 1.695510 | 1.698044 | 0.002535 |

## Turnover
| strategy | mean_monthly_turnover | median_turnover | p95_turnover | maximum_turnover | annualized_approx_turnover | pct_reduction_vs_max_sharpe | pct_reduction_vs_combined_robust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 100.000000 |
| Equal Weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 100.000000 | 100.000000 |
| Minimum Variance | 0.024456 | 0.017892 | 0.057592 | 0.232707 | 0.293469 | 76.235244 | 45.715519 |
| Maximum Sharpe | 0.102908 | 0.068715 | 0.295680 | 0.575056 | 1.234891 | 0.000000 | -128.424312 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.045051 | 0.020718 | 0.155315 | 0.577053 | 0.540613 | 56.221823 | 0.000000 |
| Inverse Volatility | 0.007348 | 0.004440 | 0.012749 | 0.088928 | 0.088181 | 92.859189 | 83.688652 |
| Equal Risk Contribution | 0.009027 | 0.005037 | 0.016634 | 0.109307 | 0.108326 | 91.227884 | 79.962354 |

## Weight Concentration
| strategy | mean_hhi | median_hhi | max_hhi | mean_effective_holdings | median_effective_holdings | min_effective_holdings | average_largest_position | maximum_largest_position |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Equal Weight | 0.100000 | 0.100000 | 0.100000 | 10.000000 | 10.000000 | 10.000000 | 0.100000 | 0.100000 |
| Minimum Variance | 0.228390 | 0.232542 | 0.271467 | 4.409988 | 4.300300 | 3.683695 | 0.300000 | 0.300000 |
| Maximum Sharpe | 0.263796 | 0.263271 | 0.280000 | 3.802906 | 3.798363 | 3.571429 | 0.300000 | 0.300000 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 0.251352 | 0.258822 | 0.280000 | 4.021201 | 3.863663 | 3.571429 | 0.300000 | 0.300000 |
| Inverse Volatility | 0.127404 | 0.131058 | 0.140452 | 7.896719 | 7.630228 | 7.119888 | 0.221719 | 0.271615 |
| Equal Risk Contribution | 0.139899 | 0.143211 | 0.162126 | 7.260985 | 6.982696 | 6.168057 | 0.254177 | 0.300000 |

## Cap Binding
| strategy | n_rebalance_dates | dates_with_cap_hit | pct_rebalance_dates_with_cap_hit | avg_cap_assets | max_cap_assets | most_frequently_capped_assets |
| --- | --- | --- | --- | --- | --- | --- |
| SPY | 67 | 67 | 100.000000 | 1.000000 | 1 | SPY:67 |
| Equal Weight | 67 | 0 | 0.000000 | 0.000000 | 0 |  |
| Minimum Variance | 67 | 67 | 100.000000 | 1.656716 | 2 | IEF:67, LQD:44 |
| Maximum Sharpe | 67 | 67 | 100.000000 | 2.059701 | 3 | QQQ:44, SPY:33, GLD:29, IEF:17, LQD:10 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 67 | 67 | 100.000000 | 1.925373 | 3 | IEF:44, SPY:27, GLD:25, LQD:15, QQQ:15 |
| Inverse Volatility | 67 | 0 | 0.000000 | 0.000000 | 0 |  |
| Equal Risk Contribution | 67 | 23 | 34.328358 | 0.343284 | 1 | IEF:23 |

## Risk Concentration
| rebalance_date | strategy | largest_risk_contributor | largest_risk_contribution_pct | top3_risk_contribution_pct | risk_contribution_dispersion | cap_binding_count |
| --- | --- | --- | --- | --- | --- | --- |
| 2018-02-01 00:00:00 | Maximum Sharpe | QQQ | 53.101269 | 98.882013 | 0.361084 | 3 |
| 2018-02-01 00:00:00 | Combined Robust Max Sharpe λ=0.50 γ=0.10 | QQQ | 36.386146 | 87.026314 | 0.190400 | 1 |
| 2018-02-01 00:00:00 | Inverse Volatility | VNQ | 12.940814 | 38.250145 | 0.008858 | 0 |
| 2018-02-01 00:00:00 | Equal Risk Contribution | LQD | 10.000002 | 30.000004 | 0.000000 | 0 |
| 2022-02-01 00:00:00 | Maximum Sharpe | QQQ | 58.498833 | 87.263763 | 0.293117 | 1 |
| 2022-02-01 00:00:00 | Combined Robust Max Sharpe λ=0.50 γ=0.10 | SPY | 27.211506 | 69.234747 | 0.099943 | 1 |
| 2022-02-01 00:00:00 | Inverse Volatility | SPY | 13.553567 | 39.853958 | 0.019582 | 0 |
| 2022-02-01 00:00:00 | Equal Risk Contribution | TLT | 10.979026 | 31.827861 | 0.000666 | 1 |
| 2026-09-01 00:00:00 | Maximum Sharpe | GLD | 43.552819 | 93.631772 | 0.229572 | 2 |
| 2026-09-01 00:00:00 | Combined Robust Max Sharpe λ=0.50 γ=0.10 | SPY | 36.829265 | 85.350349 | 0.183599 | 1 |
| 2026-09-01 00:00:00 | Inverse Volatility | EFA | 11.959483 | 34.391691 | 0.002696 | 0 |
| 2026-09-01 00:00:00 | Equal Risk Contribution | VNQ | 10.000006 | 30.000014 | 0.000000 | 0 |

## Risk-Estimation Sensitivity
| strategy | mean | median | p95 | max |
| --- | --- | --- | --- | --- |
| Equal Risk Contribution | 0.000793 | 0.000642 | 0.001578 | 0.002083 |
| Inverse Volatility | 0.000873 | 0.000735 | 0.001707 | 0.001993 |

## COVID Stress
| stress_period | start_date | end_date | strategy | cumulative_return | annualized_volatility | max_drawdown | entry_rebalance_date | top_positions_at_entry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | SPY | -0.233713 | 0.792570 | -0.334439 | 2019-11-01 | SPY:1.000, QQQ:0.000, IWM:0.000 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Equal Weight | -0.135823 | 0.478800 | -0.218684 | 2019-11-01 | SPY:0.100, QQQ:0.100, IWM:0.100 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Minimum Variance | -0.052186 | 0.281311 | -0.149356 | 2019-11-01 | IEF:0.300, LQD:0.300, GLD:0.105 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Maximum Sharpe | -0.082049 | 0.381412 | -0.177489 | 2019-11-01 | LQD:0.300, SPY:0.275, TLT:0.184 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Combined Robust Max Sharpe λ=0.50 γ=0.10 | -0.056737 | 0.295114 | -0.144038 | 2019-11-01 | IEF:0.300, LQD:0.300, SPY:0.277 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Inverse Volatility | -0.085806 | 0.355732 | -0.174851 | 2019-11-01 | LQD:0.215, IEF:0.206, TLT:0.090 |
| COVID_2020 | 2020-02-20 00:00:00 | 2020-03-31 00:00:00 | Equal Risk Contribution | -0.067893 | 0.313358 | -0.154401 | 2019-11-01 | IEF:0.263, LQD:0.167, TLT:0.108 |

## 2022 Stress
| stress_period | start_date | end_date | strategy | cumulative_return | annualized_volatility | max_drawdown | entry_rebalance_date | top_positions_at_entry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | SPY | -0.137650 | 0.256123 | -0.212847 | 2022-04-01 | SPY:1.000, QQQ:0.000, IWM:0.000 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Equal Weight | -0.174629 | 0.169021 | -0.211914 | 2022-04-01 | SPY:0.100, QQQ:0.100, IWM:0.100 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Minimum Variance | -0.152943 | 0.115762 | -0.170083 | 2022-04-01 | IEF:0.300, LQD:0.239, GLD:0.143 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Maximum Sharpe | -0.168121 | 0.186961 | -0.212058 | 2022-04-01 | QQQ:0.300, GLD:0.300, TLT:0.269 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Combined Robust Max Sharpe λ=0.50 γ=0.10 | -0.150276 | 0.139725 | -0.176197 | 2022-04-01 | IEF:0.300, GLD:0.237, SPY:0.177 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Inverse Volatility | -0.162220 | 0.137297 | -0.188291 | 2022-04-01 | IEF:0.255, LQD:0.147, GLD:0.104 |
| RATE_HIKE_2022 | 2022-04-01 00:00:00 | 2022-10-31 00:00:00 | Equal Risk Contribution | -0.164743 | 0.128873 | -0.186760 | 2022-04-01 | IEF:0.300, TLT:0.158, GLD:0.114 |

## Drawdowns
| strategy | max_drawdown | peak_date | trough_date | recovery_date | drawdown_duration_calendar_days | drawdown_duration_observations |
| --- | --- | --- | --- | --- | --- | --- |
| Maximum Sharpe | -0.255800 | 2021-12-27 | 2022-10-14 | 2024-03-01 | 291 | 203 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | -0.223658 | 2021-12-27 | 2022-11-03 | 2023-12-01 | 311 | 217 |
| Inverse Volatility | -0.235633 | 2021-11-09 | 2022-10-20 | 2024-07-16 | 345 | 239 |
| Equal Risk Contribution | -0.237624 | 2021-11-09 | 2022-10-20 | 2024-08-19 | 345 | 239 |

## Rolling Performance
| strategy | best_rolling_12m_sharpe | worst_rolling_12m_sharpe | median_rolling_12m_sharpe | std_rolling_12m_sharpe |
| --- | --- | --- | --- | --- |
| SPY | 3.451765 | -0.902476 | 0.735383 | 1.007528 |
| Equal Weight | 3.404878 | -1.716739 | 0.818765 | 1.009617 |
| Minimum Variance | 3.895491 | -2.326368 | 0.822677 | 1.283227 |
| Maximum Sharpe | 3.546681 | -1.668478 | 0.777963 | 1.036970 |
| Combined Robust Max Sharpe λ=0.50 γ=0.10 | 3.870482 | -1.921138 | 1.071092 | 1.167122 |
| Inverse Volatility | 3.834744 | -2.071982 | 0.864619 | 1.098740 |
| Equal Risk Contribution | 3.758431 | -2.197943 | 0.834007 | 1.145528 |

## Inverse Volatility Diagnostics
| cap_redistribution_required | uncapped_l1_to_capped |
| --- | --- |
| 0.000000 | 0.000000 |
|  | 0.000000 |
|  | 0.000000 |
|  | 0.000000 |

## ERC Diagnostics
| rebalance_date | solver_success | fallback_used | solver_message | objective_value | risk_contribution_dispersion | cap_binding_count | capped_assets |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-02-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2018-03-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2018-05-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2018-06-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2018-08-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2018-10-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2018-11-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-02-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-03-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-04-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-05-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-07-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-08-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-10-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2019-11-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2020-04-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000792 | 0.000792 | 1 | IEF |
| 2020-05-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001088 | 0.001088 | 1 | IEF |
| 2020-06-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001123 | 0.001123 | 1 | IEF |
| 2020-07-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001171 | 0.001171 | 1 | IEF |
| 2020-09-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001325 | 0.001325 | 1 | IEF |
| 2020-10-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001417 | 0.001417 | 1 | IEF |
| 2020-12-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001446 | 0.001446 | 1 | IEF |
| 2021-02-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001496 | 0.001496 | 1 | IEF |
| 2021-03-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001262 | 0.001262 | 1 | IEF |
| 2021-04-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001118 | 0.001118 | 1 | IEF |
| 2021-06-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001104 | 0.001104 | 1 | IEF |
| 2021-07-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.001037 | 0.001037 | 1 | IEF |
| 2021-09-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000953 | 0.000953 | 1 | IEF |
| 2021-10-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000935 | 0.000935 | 1 | IEF |
| 2021-11-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000885 | 0.000885 | 1 | IEF |
| 2021-12-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000798 | 0.000798 | 1 | IEF |
| 2022-02-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000666 | 0.000666 | 1 | IEF |
| 2022-03-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000559 | 0.000559 | 1 | IEF |
| 2022-04-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000404 | 0.000404 | 1 | IEF |
| 2022-06-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000279 | 0.000279 | 1 | IEF |
| 2022-07-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000114 | 0.000114 | 1 | IEF |
| 2022-08-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000066 | 0.000066 | 1 | IEF |
| 2022-09-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000022 | 0.000022 | 1 | IEF |
| 2022-11-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2022-12-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-02-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-03-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-05-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-06-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-08-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-09-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-11-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2023-12-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-02-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-03-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-04-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-05-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-07-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-08-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-10-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2024-11-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2025-04-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2025-05-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2025-07-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2025-08-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2025-10-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2025-12-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2026-04-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2026-05-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2026-06-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2026-07-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |
| 2026-09-01 00:00:00 | 1 | 0 | Optimization terminated successfully | 0.000000 | 0.000000 | 0 |  |

## Hypothesis Evaluation
The numerical verdicts are derived from the full artifact set in the repository outputs rather than from any one metric in isolation.

## Limitations
This study remains conditional on the frozen canonical dataset, the selected asset universe, and the locked 30% cap.

## Recruiter / Interview Interpretation
The M4 results are best read as evidence about robustness under a fixed canonical test harness, not as a guarantee of future superiority.

## Conclusion
M4 is complete only after the empirical evidence, reproducibility checks, and baseline preservation all hold simultaneously.

In this canonical out-of-sample study, removing expected-return estimates materially improved portfolio stability, diversification, and trading efficiency, but those improvements did not translate into superior risk-adjusted returns. The Combined Robust Max Sharpe model provided the strongest overall balance between performance and implementation stability among the tested approaches.

## Reproducibility Run Comparison
| run1_vs_run2_max_return_diff | run1_vs_run2_max_weight_diff |
| --- | --- |
| 0.000000 | 0.000000 |