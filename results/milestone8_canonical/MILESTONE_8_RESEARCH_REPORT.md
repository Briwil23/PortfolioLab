# Portfolio Construction Is a Trade-Off, Not a Ranking: A Cross-Strategy Synthesis of Return, Risk, Tail Loss, Turnover, Stress Behavior, and Factor Exposure

## 1. Executive Summary

This M8 synthesis is intentionally framed as an empirical trade-off map rather than a ranking exercise. The source of truth is the certified M8 canonical evidence produced under the public-repository contract: the strategy comparison, trade-off summary, personality features, supplemental evidence, provenance, and verification artifacts generated under Phase 8.3. We are not recreating the historical optimization process or altering the underlying methodology. The objective is to interpret the already-certified cross-strategy evidence in a transparent, evidence-first way.

The eight-strategy common comparison universe is defined by the certified M7 result set and is composed of: Minimum CVaR, SPY, Equal Weight, Minimum Variance, Maximum Sharpe, Combined Robust Max Sharpe λ=0.50 γ=0.10, Inverse Volatility, and Equal Risk Contribution. The evidence base is leakage-safe, out-of-sample relative to the historical optimization design used in the certified prior milestones, and carried forward without modification. The synthesis does not claim that one strategy is universally best; rather, the central finding is that portfolio construction is a multi-objective frontier problem.

SPY produced the highest observed annualized return at approximately 14.09%, but it also carried the highest realized annualized volatility among the eight strategies at approximately 19.13%. The Combined Robust Max Sharpe λ=0.50 γ=0.10 strategy had the strongest observed risk-adjusted profile, with a gross Sharpe of approximately 0.706 and a Sortino of approximately 0.685. Minimum CVaR had the shallowest full-period maximum drawdown at approximately -21.22%, while Minimum Variance delivered the lowest realized 95% VaR and CVaR, at approximately 0.7798% and 1.1489% respectively. The distinction is important: Minimum CVaR was optimized for historical tail-risk minimization in its objective function, but Minimum Variance generated slightly lower realized out-of-sample tail loss under the certified post-optimization evidence.

Transaction costs were modest under the 10 bps implementation convention, but not zero. The cost drag was largest in the higher-turnover return-seeking strategies and smallest in the more stable low-turnover designs, while the cost-drag magnitudes remained far smaller than the cross-strategy differences in return, volatility, and tail risk. Stress-period evidence reinforced the same conclusion from a different angle: COVID strongly favored more defensive constructions relative to SPY, whereas 2022 did not produce the same ordering, demonstrating regime dependence. The M5 factor evidence remains explanatory rather than causal; the certified static alpha tests did not establish statistically significant alpha after multiple-testing correction. The synthesis therefore treats factor exposures as context, not proof of a permanent superior implementation.

The principal conclusion is straightforward: the strategy universe is not reducible to a single winner. It is a set of different portfolio personalities designed around different objectives. Some strategies maximize realized return, others maximize risk-adjusted performance, others minimize volatility and drawdown, and others are better suited to low-turnover or lower-tail-risk implementation. The Phase 8.4 report presents those trade-offs directly and without rhetorical inflation.

## 2. Research Question and Motivation

The core research question for M8 is empirical and intentionally modest. We do not ask which strategy is objectively best in the abstract; instead, we ask how different portfolio objectives change the realized risk-return frontier once the strategy is evaluated on the certified evidence set. The synthesis asks five things directly: how return and volatility differ across strategies; whether risk-adjusted performance survives implementation costs; whether tail-risk optimization maps to realized tail-loss outcomes; how drawdown and stress regimes alter the inference; and how factor-specific explanatory evidence should be interpreted without overstating its causal power.

The motivation is not to rank strategies in a marketing sense. The objective is to identify the empirical frontier of trade-offs: what gains in return are paid for by volatility, drawdown, turnover, and tail-loss exposure; and which strategy profiles are better aligned with specific portfolio objectives. This is the appropriate lens for cross-strategy synthesis because the strategies are not homogeneous ETFs or naive portfolios with the same mission. They embody visibly different optimization objectives, from baseline market exposure to variance minimization and tail-risk control.

## 3. Experimental Evidence Base

The quantitative source of truth for the report is the M8 canonical output set generated under Phase 8.3:

- strategy_comparison.csv
- tradeoff_summary.csv
- personality_features.csv
- supplemental_evidence.csv
- provenance.json
- milestone8_verification.json

These artifacts are read as the authoritative evidence for the report. Prior milestone outputs remain useful context for interpretation and methodology, but they do not replace the certified M8 values. The deterministic M8 verification metadata reports a deterministic max difference of 0.0 and cost-drag reconstruction errors of 0.0. The M3/M4/M5/M7 mutation counts remained zero. Public repository compatibility is true and the frozen factor snapshot is not required for the public synthesis contract.

The strategy universe is the eight-strategy common comparison set created under M7 and preserved in the M8 canonical comparison matrix. The report therefore works with a common, comparable strategy set rather than mixing separate historical datasets or re-optimizing strategies. That keeps the evidence coherent and makes the synthesis interpretable as a cross-strategy comparison rather than an ad hoc portfolio narrative.

## 4. Strategy Universe

The certified comparison universe includes eight strategies:

1. Minimum CVaR
2. SPY
3. Equal Weight
4. Minimum Variance
5. Maximum Sharpe
6. Combined Robust Max Sharpe λ=0.50 γ=0.10
7. Inverse Volatility
8. Equal Risk Contribution

These names are preserved exactly from the canonical M8 strategy table and are used in all figures and narrative references. The strategy set is not deliberately reduced to a subset for narrative convenience, and no strategy is treated as “the best” because that would collapse the evidence into a single unsupported statement. The report instead describes the strategy profiles in terms of the dimensions each strategy optimizes and the metrics it delivers under realized conditions.

## 5. Cross-Strategy Return and Volatility Trade-Offs

The empirical risk-return frontier is shown in Figure 1, which plots volatility on the x-axis and annualized return on the y-axis. The visual story is consistent with the canonical values. SPY had the highest observed annualized return of 14.0891%, but it also had the highest annualized volatility at 19.1280%. Combined Robust Max Sharpe λ=0.50 γ=0.10 landed in the middle of the return spectrum with roughly 9.6777% annualized return and 10.8820% volatility. Minimum Variance produced 5.7471% annualized return with the lowest realized volatility at 8.0486%, while Minimum CVaR produced 5.9540% annualized return at 8.1614% volatility.

This is not a theoretical efficient frontier and should not be misread as a claim about an optimal set of portfolios. It is an empirical trade-off map. The chart demonstrates two important things. First, returns increase materially as one moves toward higher-beta or return-seeking exposures. Second, the lower-volatility constructions are materially less volatile, but they do so at a meaningful reduction in realized return. In other words, the return-volatility frontier has clear curvature: a strategy can meaningfully reduce volatility without fully sacrificing return, but the large price of higher return is higher realized variance and broader drawdown exposure.

The return-volatility distinction also matters for interpretation of the Sharpe result. SPY is not low-volatility, and it is not the highest Sharpe in the M8 evidence. The Combined Robust strategy had the highest Sharpe at approximately 0.7055 and the highest Sortino at approximately 0.6849, while SPY delivered different but naturally more return-dominant outcomes at 0.6320 Sharpe. The empirical frontier is therefore not a single line of best risk-adjusted performance; it is a family of strategy profiles with different objectives.

Figure 1 is therefore best read as a descriptive decomposition of how strategy objectives map to realized return and volatility rather than a normative optimization chart.

## 6. Risk-Adjusted Performance

The certified Sharpe and Sortino evidence is one of the clearest illustrations that a strategy can have a higher return without dominating risk-adjusted performance. The Combined Robust Max Sharpe λ=0.50 γ=0.10 strategy had the highest Sharpe at 0.705538 and the highest Sortino at 0.684938. SPY had the highest return, but not the highest Sharpe; its Sharpe was 0.632011 and Sortino 0.594807. Maximum Sharpe had 0.585388 Sharpe and 0.553651 Sortino, which placed it behind SPY and behind Combined Robust. Equal Weight had lower volatility than SPY but weaker realized risk-adjusted performance, with Sharpe 0.525993 and Sortino 0.495136. The lower-volatility structures such as Minimum Variance and Minimum CVaR had lower Sharpe ratios, but their realized outcomes reflect a different portfolio objective: they are not necessarily designed to maximize Sharpe over the observed sample.

This is the core M8 nuance: Sharpe ranking is an empirical ordering, not an overall personal judgment that a strategy is “better.” A strategy can have a lower Sharpe and still be attractive if the objective is drawdown or tail-risk reduction. The cross-strategy evidence indicates that risk-adjusted performance and return-maximizing behavior are meaningfully different objectives. The report therefore avoids reducing the analysis to an absolute Sharpe champion.

## 7. Drawdown and Tail-Risk Protection

The drawdown map in Figure 3 is especially important because it demonstrates the distinction between raw growth and downside resilience. The key result is that Minimum CVaR had the shallowest full-period maximum drawdown at -21.2206%, while Minimum Variance had -22.1996% and Combined Robust had -22.3658%. SPY had a maximum drawdown of -33.7173%, and Maximum Sharpe had -25.5800%. In other words, the more defensive design did not necessarily maximize return, but it did materially reduce the depth of the realized downside path over the full sample.

This interpretation is also revealed by the realized tail-risk metrics. Minimum Variance produced the lowest realized 95% VaR and CVaR: 0.7798% and 1.1489% respectively. Minimum CVaR did not have the lowest realized out-of-sample CVaR; its realized 95% CVaR loss was 1.1566%, very slightly above Minimum Variance. This directly addresses a central M8 question: optimization objective and realized tail-loss outcome are not necessarily interchangeable. Historical training-window CVaR minimization may deliver attractive in-sample risk control without translating into the absolute lowest realized OOS tail loss. The difference is small in absolute terms, but it is real and important. The evidence does not support any claim that the discrepancy is statistically significant; the safe interpretation is that optimization objective and realized OOS outcome diverged modestly under the certified evidence set.

The evidence is therefore cautious and technically honest. A strategy can be designed around tail-risk containment and still not deliver the lowest realized tail loss in an out-of-sample period. This has direct implications for portfolio policy: tail-risk optimization is a design lens, not a guarantee of realized dominance over all other strategies.

## 8. Transaction Costs and Turnover

The transaction-cost evidence is stored in the trade-off summary and is visualized in Figure 5. The gross vs net Sharpe comparison reveals that the effect of the 10 bps cost convention is measurable but modest. For Maximum Sharpe, gross Sharpe was 0.585388 and net Sharpe was 0.578785, a drag of 0.006603. For Minimum CVaR, the gross Sharpe was 0.484480 and net Sharpe 0.477101, a drag of 0.007380. For Combined Robust, gross Sharpe 0.705538 versus net Sharpe 0.702008 reflected a drag of 0.003530. For Minimum Variance, the drag was 0.002487. For Inverse Volatility, it was 0.000628. For Equal Risk Contribution, 0.000811. SPY and Equal Weight had zero drag under the certified convention, because the relevant implementation convention preserved their gross and net values in the canonical net matrix.

This does not mean transaction costs were irrelevant. It means they were important in a measured way. The more turnover-intensive strategies showed larger divergences between gross and net performance, but the differences remained secondary relative to the more substantial strategy-level differences in return, volatility, drawdown, and stress behavior. It would be misleading to interpret the cost-drag evidence as a universal “winner” metric. Instead, it should be understood as a necessary implementation-context measure: higher turnover or more active rebalancing can change realized performance without necessarily changing the basic strategic profile.

The canonical turnover evidence is compatible with this interpretation because the strategy-level turnover values differ materially across strategies. The report therefore discusses turnover as a context variable rather than as a causal explanation for all observed differences. It matters because implementation friction is not uniform across strategies.

## 9. Stress-Regime Behavior

The stress-period evidence is a crucial M8 finding because it shows that risk behavior is highly regime-dependent. Figure 6 compares the certified COVID and 2022 cumulative return outcomes across the strategy set. During the COVID period, SPY fell roughly -13.6407% cumulatively, while Minimum CVaR was roughly -0.3039% and Minimum Variance was roughly -0.5583%. This is a striking contrast. Relative to SPY, the defensive constructions were far more resilient in the acute COVID drawdown window.

The 2022 period is different. The 2022 cumulative performance ordering was not the same as the COVID ordering. SPY was approximately -17.7443%, which was less negative than several defensive portfolio constructions. Minimum CVaR was -19.8230%, while Minimum Variance was -20.1753%. Combined Robust was -19.4944%. This tells a precise story: defensive constructions did not uniformly outperform in every adverse regime. Risk control matters, but it does not guarantee superior return in every stress environment. The primary M8 message is that “defensive” does not mean “superior return in every adverse regime.”

This is a central reason not to collapse the evidence into a single strategy score. Regime dependence is one of the most important findings from the M8 synthesis: stress behavior can change the preferred strategy profile depending on the type of risk shock and the time window being considered.

## 10. Robustness, Diversification, and Concentration

The M3 and M4 supplemental evidence continues to be valuable as context, but it is not incorporated into the primary performance matrix. The evidence helps explain why some strategies look different in realized risk and turnover terms. M3 concentration evidence shows the largest difference in portfolio concentration: SPY had a mean HHI of 1.0, Equal Weight had 0.10, and Minimum Variance had 0.2283904315866191. These values are meaningful because they show that concentration and diversification differ materially across strategies and help explain why realized risk behavior differs even when realized return is not dramatically different. A concentrated benchmark like SPY is a very different risk profile from a diversified, low-turnover minimum-variance construction.

M4 stress-period evidence reinforces that the realized advantage of a defensive strategy is regime-specific. A strategy that is strong during COVID may not be superior during the 2022 stress episode. The evidence is observational rather than causal, but it reinforces the interpretation that robustness does not mean an unambiguous universal optimum. It means a strategy behaves differently under different stress conditions and with different exposure structures. That is a valid and central M8 conclusion.

## 11. Factor Evidence and Performance Interpretation

The M5 evidence is handled carefully under the public explanatory contract. The M8 supplemental evidence includes rolling-factor exposures, but they are clearly labeled as explanatory and not a direct performance comparison. This is important because the certified static alpha tests did not establish statistically significant alpha after multiple-testing correction. The M5 evidence therefore does not support strong causal claims such as “this strategy’s performance is explained by factor timing” or “alpha is statistically established.”

The safer framing is: factor exposures provide explanatory context consistent with observed cross-strategy differences. The evidence is useful for understanding why different strategies may have different realized outcomes, especially in terms of market beta, concentration, and related factor-driven behavior. However, these are observational patterns, not causal proof. This distinction is crucial for a serious research report because it preserves scientific honesty and avoids overstating what the evidence can support.

## 12. Strategy Profiles / Portfolio Personalities

The objective personality table yields descriptive strategy profiles that can be used without turning them into rankings. These are evidence-based, descriptive personas rather than marketing labels. They are designed to summarize the dominant empirical characteristics of each strategy profile.

- SPY: higher-return / higher-risk benchmark. It has the highest realized return, but also the highest annualized volatility and one of the largest drawdowns. It is a strong growth-oriented benchmark, not a universal winner.
- Combined Robust Max Sharpe λ=0.50 γ=0.10: strong observed risk-adjusted profile. It has the highest Sharpe and Sortino, with moderate volatility and relatively controlled drawdown. It is not the highest-return strategy, but it is the strongest observed risk-adjusted performer.
- Minimum Variance: low-volatility construction. It has the lowest volatility and the lowest realized 95% VaR and CVaR, but it gives up meaningful return relative to the high-return benchmark. Its profile is stability-focused.
- Minimum CVaR: tail-risk-focused construction. It has the shallowest full-period maximum drawdown and very low realized tail loss, but it does not produce the lowest post-optimization realized VaR and CVaR. It is better understood as a downside-defense design than a universal tail-loss champion.
- Maximum Sharpe: return-seeking risk-adjusted benchmark. It has a high Sharpe but not the highest return, and it retains some drawdown risk.
- Inverse Volatility: lower-volatility policy mix with moderate return and moderate drawdown.
- Equal Risk Contribution: balanced allocation with lower realized volatility than the return-seeking benchmarks but not the extreme low-volatility performance of Minimum Variance.
- Equal Weight: diversified, simple benchmark with moderate return and moderate drawdown. It is not “the best” in any unambiguous sense, but it is a useful baseline against which more actively optimized strategies can be compared.

These profiles are not ordinal rankings. Their purpose is descriptive clarity: they allow a portfolio reviewer to understand the observed characteristics of each strategy without collapsing the evidence into a single score.

## 13. Cross-Strategy Hypothesis Evaluation

The M8 evidence supports a careful evaluation of the principal hypotheses using the terminology specified for the phase:

- Hypothesis: higher-return strategies also deliver higher Sharpe. PARTIALLY SUPPORTED. SPY had the highest return, but Combined Robust had the highest Sharpe. This shows that return and risk-adjusted performance are not interchangeable.
- Hypothesis: a strategy optimized for historical tail risk will deliver the lowest realized out-of-sample tail loss. NOT SUPPORTED. Minimum CVaR did not produce the lowest realized OOS VaR or CVaR. Minimum Variance did.
- Hypothesis: lower-volatility strategies robustly dominate the full-period downside profile. SUPPORTED in a qualified sense. Minimum CVaR and Minimum Variance had shallower drawdowns than the return-seeking benchmark, although not all adverse regimes favored the same strategy.
- Hypothesis: implementation costs materially change strategy rankings. PARTIALLY SUPPORTED. Cost drag is modest in absolute terms but nonzero, and it is larger for the more turnover-intensive strategies. The cost drag is meaningful for interpretation but not enough to overturn the broad strategic story.
- Hypothesis: factor exposures explain most of the observed cross-strategy differences. INCONCLUSIVE or PARTIALLY SUPPORTED at best. The M5 evidence is explanatory only, and the report does not treat it as causal proof. It helps frame differences, but it does not establish significant alpha or a unique underlying driver for each strategy profile.

This is the appropriate framing for the M8 hypothesis set: evidence is strong on objective trade-offs and much weaker on causal claims about explanatory drivers.

## 14. Practical Portfolio-Construction Lessons

Several practical lessons follow from the evidence without becoming personalized investment advice:

- Optimizing one risk metric does not guarantee dominance on the same metric out of sample. The realized VaR and CVaR evidence show that a historical tail-risk optimization did not produce the lowest realized tail loss.
- Return, volatility, drawdown, and tail loss are related but distinct phenomena. A strategy can have a higher return while still being less attractive on a drawdown or tail-risk basis.
- Turnover affects implementation quality. The cost drag was small under the 10 bps convention but nonzero, and higher-turnover constructions merit closer cost scrutiny.
- Stress behavior is regime-dependent. COVID and 2022 did not rank strategies the same way.
- A clear objective matters. A return-seeking strategy is not misleadingly “wrong” simply because it is not the highest-Sharpe strategy; it is designed to serve a different objective.
- Full-sample Sharpe alone is not sufficient. The risk-return frontier, drawdown evidence, and stress-period behavior must all be inspected together.

These lessons remain consistent with the M8 evidence and do not rely on speculative future assumptions.

## 15. Limitations

The limitations of the synthesis are important and should be preserved explicitly.

- The evidence is based on a single historical sample, not a universal law of portfolio behavior.
- The asset universe is the one used in the certified historical architecture, not a generalized universe across all potential investments.
- The implementation convention is an ETF-like historical convention with transaction costs at 10 bps, which is a design assumption rather than a universal market reality.
- Estimation windows and rebalance assumptions affect the realized results, especially in the tail-risk and turnover-sensitive strategies.
- The M8 synthesis does not guarantee future performance and does not convert historical empirical evidence into a certainty claim.
- The M5 factor evidence is observational and explanatory; it does not provide a stable causal mechanism for future outperformance.
- The report does not claim any statistically significant alpha after certified multiple-testing control.
- The strategy synthesis is cross-sectional and empirical; it does not imply any economic superiority of a small metric difference over another.

## 16. Conclusion

The M8 evidence supports a straightforward conclusion: the strategy set spans a real empirical frontier of trade-offs. SPY dominates the full-sample return dimension, Combined Robust dominates the observed risk-adjusted dimension, Minimum Variance dominates realized tail-risk and volatility dimensions, and Minimum CVaR dominates the full-period drawdown-defense dimension. These are different portfolio personalities, not different embodiments of one universal optimum.

The evidence therefore does not justify a universal winner. It justifies a more informed understanding of how portfolio objectives change the realized outcome. That is exactly the right conclusion for a cross-strategy synthesis. The goal is not to tell a recruiter or portfolio reviewer that one strategy is “the best,” but to show that different portfolios optimize different dimensions of performance and that those dimensions interact with risk, turnover, stress, and explanatory factor context.

## 17. Reproducibility and Provenance

The report is fully traceable to the certified artifacts created in Phase 8.3. The relevant source artifacts are the canonical M7 and supporting M3/M4/M5 files listed in provenance.json and milestone8_verification.json. The M8 verification metadata confirms:

- public repository compatibility = true
- frozen factor snapshot requirement = false
- deterministic max difference = 0.0
- return cost-drag reconstruction error = 0.0
- Sharpe cost-drag reconstruction error = 0.0
- M3/M4/M5/M7 mutation counts = 0

The public-repository contract remains valid for the M8 synthesis because the certified public evidence, not the local frozen snapshot, is the basis for the report. The report therefore preserves the same constraints used throughout the M8 certification process: no methodology rewrite, no re-optimization, no silent replacement of evidence, no subjective ranking, and no universal winner claim.

## 18. Figure and Artifact Guide

The report uses the following figures and relative paths:

- Figure 1: [figures/01_return_vs_volatility.png](figures/01_return_vs_volatility.png)
- Figure 2: [figures/02_sharpe_vs_turnover.png](figures/02_sharpe_vs_turnover.png)
- Figure 3: [figures/03_return_vs_max_drawdown.png](figures/03_return_vs_max_drawdown.png)
- Figure 4: [figures/04_return_vs_realized_cvar.png](figures/04_return_vs_realized_cvar.png)
- Figure 5: [figures/05_gross_vs_net_sharpe.png](figures/05_gross_vs_net_sharpe.png)
- Figure 6: [figures/06_stress_period_comparison.png](figures/06_stress_period_comparison.png)
- Figure 7: [figures/07_strategy_feature_heatmap.png](figures/07_strategy_feature_heatmap.png)

These are the canonical M8 visual outputs created for the final research report and are treated as read-only evidence in the same way as the underlying CSV and JSON artifacts.

---

This report is intended as a quantitative synthesis rather than a marketing or ranking document. It is written to be usable by a technically literate investment professional, financial engineer, portfolio reviewer, or recruiter examining the evidence quality and interpretive discipline of the PortfolioLab M8 work.
