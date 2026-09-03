# PortfolioLab

**Institutional Portfolio Optimization, Risk & Backtesting Engine**

Python • SciPy • pandas • NumPy • Quantitative Finance • Mathematical Optimization

## Overview

PortfolioLab is a production-quality quantitative-finance research framework for constructing, analyzing, and visualizing diversified portfolios using classical mean-variance optimization and rigorous statistical methodology.

The project now spans four completed research milestones: classical portfolio optimization, walk-forward out-of-sample backtesting, robust portfolio construction, and risk-based portfolio construction under a frozen canonical evaluation harness.

This is **not a predictive model or claims any investment outperformance**. Rather, it is a rigorous demonstration of how to implement portfolio optimization mathematics correctly, document assumptions clearly, and structure quantitative research code for professional use.

## Research Progression

**Milestone 1 — Classical Portfolio Optimization**
Implemented transparent mean-variance allocation, baseline performance reporting, and explicit optimization math.

**Milestone 2 — Walk-Forward Out-of-Sample Backtesting**
Added rolling training/holding periods, benchmark comparison, and reproducible out-of-sample evaluation.

**Milestone 3 — Robust Portfolio Construction & Turnover Control**
Introduced robust expected-return handling and stronger implementation discipline around turnover and reproducibility.

**Milestone 4 — Risk-Based Portfolio Construction**
Added inverse-volatility and equal-risk-contribution portfolios, a frozen canonical dataset, locked sensitivity experiments, and deterministic canonical evaluation.

Milestone 4’s main empirical result is balanced rather than dominating: Combined Robust Max Sharpe achieved the strongest out-of-sample risk-adjusted performance in the canonical test, with gross Sharpe of about 0.706. Maximum Sharpe followed at about 0.585, Inverse Volatility at about 0.502, and Equal Risk Contribution at about 0.471. The risk-based portfolios reduced turnover dramatically versus Maximum Sharpe, with Inverse Volatility down about 92.9% and ERC down about 91.2%, and they typically held about 7 to 8 effective positions versus roughly 4 for the optimized return-dependent portfolios. ERC also completed all 67 of 67 optimizations without fallback on the canonical run.

The conclusion is therefore conservative: risk-based construction materially improved robustness, diversification, and implementation stability, but it did not produce the highest out-of-sample risk-adjusted performance.

## Problem & Motivation

Portfolio optimization is a cornerstone of quantitative finance, but many implementations obscure the underlying mathematics behind black-box libraries or make undocumented assumptions about data, constraints, and methodology.

This project addresses three goals:

1. **Mathematical Transparency** — All optimization, statistics, and performance calculations are implemented explicitly and verified against manual independent calculations.

2. **Research-Grade Methodology** — Data is validated, annualization is handled carefully to distinguish between expected returns (used by the optimizer) and realized annualized returns (used for historical performance reporting), and in-sample versus out-of-sample results are clearly separated.

3. **Professional Code Quality** — The codebase demonstrates modular architecture, comprehensive unit tests, type hints, explicit error handling, and reproducible configuration suitable for a quantitative trading or risk-management environment.

### Why Explicit Implementation?

Using an off-the-shelf portfolio-optimization library would be faster, but would obscure understanding. By implementing the core math manually, this project demonstrates:

- ✓ Deep understanding of the Markowitz mean-variance framework
- ✓ Competence with constrained optimization (scipy.optimize)
- ✓ Careful handling of covariance matrices and numerical stability
- ✓ Rigorous statistical methodology
- ✓ Software engineering discipline in research code

## Current Features

- **Market Data**: Download and validation of adjusted historical prices from Yahoo Finance
- **Return Calculations**: Daily simple/log returns with robust annualization
- **Expected Returns**: Arithmetic annualization for parameter estimation (optimizer input)
- **Realized Returns**: Geometric annualization (CAGR) for historical performance reporting
- **Covariance Estimation**: Sample covariance matrix with proper annualization
- **Portfolio Optimization**:
  - Equal-weight baseline
  - Minimum-variance (risk minimization)
  - Maximum-Sharpe (risk-adjusted return maximization)
  - Constrained efficient frontier
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, cumulative return
- **Benchmark Comparison**: Against SPY with detailed performance breakdown
- **Reproducible Outputs**: CSV files for all weights and metrics, publication-quality PNG figures
- **Unit Tests**: 21 comprehensive tests verifying mathematical correctness

## Asset Universe

The initial diversified universe consists of 10 liquid ETFs:

**Equities** (US and international):
- SPY: US large-cap (benchmark)
- QQQ: US technology growth
- IWM: US small-cap
- EFA: Developed international
- EEM: Emerging markets

**Fixed Income**:
- TLT: US long-duration Treasuries
- IEF: US intermediate Treasuries
- LQD: Investment-grade corporate bonds

**Alternatives**:
- GLD: Gold
- VNQ: US real estate (REIT)

**Benchmark**: SPY

## Mathematical Framework

All portfolio calculations follow classical Markowitz mean-variance theory with explicit mathematical documentation.

### Portfolio Return

$$R_p = w^T \mu$$

where $w$ is the weight vector and $\mu$ is the expected-return vector (annualized).

### Portfolio Variance & Volatility

$$\sigma_p^2 = w^T \Sigma w$$
$$\sigma_p = \sqrt{w^T \Sigma w}$$

where $\Sigma$ is the annualized covariance matrix.

### Sharpe Ratio

$$\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}$$

where $R_f$ is the annual risk-free rate (currently 2% by default, configurable).

### Optimization Problems

**Minimum Variance**:
$$\min_w w^T \Sigma w$$
$$\text{subject to:} \sum w_i = 1, \quad 0 \le w_i \le w_{\max}$$

**Maximum Sharpe**:
$$\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$$
$$\text{subject to:} \sum w_i = 1, \quad 0 \le w_i \le w_{\max}$$

Both use long-only constraints and position weight caps (default 30% per asset).

### Efficient Frontier

For a grid of target portfolio returns, solve:
$$\min_w w^T \Sigma w$$
$$\text{subject to:} \sum w_i = 1, \quad w^T \mu = R_{\text{target}}, \quad 0 \le w_i \le w_{\max}$$

### Annualization Distinctions

- **Expected Annual Return** (optimizer): $\mu_{\text{annual}} = \text{mean}(\text{daily returns}) \times 252$
- **Realized Annualized Return** (historical): $r_{\text{ann}} = \left(\prod (1 + r_i)\right)^{252/n} - 1$ (CAGR)
- **Annualized Volatility**: $\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$
- **Maximum Drawdown**: Negative value representing peak-to-trough loss (e.g., -0.25 = 25% loss)

The optimization uses constrained quadratic programming and numerical optimization with SciPy. The framework is intentionally transparent, with the underlying objective and constraints written directly in code rather than hidden behind a black-box library.

## Repository Structure

```text
PortfolioLab/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── market_data.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── mean_variance.py
│   ├── risk/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── backtesting/
│   │   └── __init__.py
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
├── notebooks/
│   └── 01_portfolio_optimization.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   └── test_optimization.py
├── results/
│   ├── figures/
│   └── performance/
└── run_analysis.py
```

## Installation

### Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/PortfolioLab.git
cd PortfolioLab

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python run_analysis.py
```

This command:
1. Loads configuration from `config/config.yaml`
2. Downloads 11 years of ETF price data from Yahoo Finance
3. Validates and cleans the data
4. Calculates daily returns
5. Estimates expected returns and covariance matrix
6. Solves optimization problems (equal-weight, min-var, max-Sharpe, frontier)
7. Generates 5 publication-quality PNG figures
8. Saves 3 CSV result files with weights and metrics
9. Prints performance summary to console

**Typical execution time**: 2–5 seconds (download may vary)

### Configuration

Edit `config/config.yaml` to customize:

```yaml
start_date: "2015-01-01"          # Analysis start date
end_date: null                     # null = today
trading_days_per_year: 252         # Trading calendar
risk_free_rate: 0.02               # Risk-free rate for Sharpe
max_weight: 0.30                   # Maximum position size
benchmark_ticker: "SPY"            # Benchmark for comparison
asset_universe:                    # Investable tickers
  - SPY
  - QQQ
  - ...
```

### Run Tests

```bash
# Full test suite (21 tests)
python -m pytest -v

# Specific test file
python -m pytest tests/test_metrics.py -v
python -m pytest tests/test_optimization.py -v

# Run with coverage
python -m pytest --cov=src tests/
```

### Audit & Verification

```bash
# Run mathematical audit checks
python audit_check.py
```

This script independently verifies:
- Maximum drawdown convention
- Annualization calculations
- Sharpe and Sortino ratios
- Optimization constraints
- Portfolio math (return, volatility, Sharpe)
- CSV output quality

## Example Outputs

### Generated Files

**Performance Metrics** (`results/performance/`):
- `portfolio_weights.csv` — Optimal allocations for each strategy
- `portfolio_metrics.csv` — Annualized return, volatility, Sharpe, Sortino, max drawdown
- `efficient_frontier.csv` — 30-point constrained efficient frontier

**Visualizations** (`results/figures/`):
- `efficient_frontier.png` — Constrained frontier curve
- `portfolio_allocations.png` — Asset weight comparison
- `cumulative_growth.png` — Wealth index trajectories
- `drawdown.png` — Drawdown paths over time
- `risk_return_comparison.png` — Return vs risk scatter

**Processed Data** (`data/processed/`):
- `prices.csv` — Clean adjusted-close prices
- `daily_returns.csv` — Daily return matrix
- `covariance_matrix.csv` — Annualized covariance
- `expected_returns.csv` — Annualized expected returns

## Limitations & Caveats

### By Design (Milestone 1)

- **In-Sample Only**: All results use the same historical window for estimation and evaluation
- **No Walk-Forward Testing**: Milestone 2 will introduce rebalancing and forward validation
- **No Transaction Costs**: Assumes frictionless trading
- **No Turnover Constraints**: Allocation changes are unconstrained
- **Static Rebalancing Frequency**: Portfolio is held constant throughout the sample
- **Simple Risk Model**: Uses sample covariance; no factor model or shrinkage
- **Flat Risk-Free Rate**: 2% constant rate; future versions may use Treasury curve

### Known Sensitivities

- Optimization results are sensitive to the historical window used for estimation
- Parameter estimates (especially covariance) are subject to sampling error
- Past performance does not guarantee future results
- Efficient frontier assumes forward returns match historical estimates

### Out-of-Scope (Future Milestones)

- Walk-forward backtesting (Milestone 2)
- Transaction costs and slippage (Milestone 3)
- Risk parity and factor tilts (Milestone 4)
- Black-Litterman views (Milestone 5)
- VaR/CVaR and non-linear risk (Milestone 6)
- Factor attribution and decomposition (Milestone 7)
- Live trading and risk monitoring (Future)

## Methodology

### In-Sample Construction

This milestone is **intentionally an in-sample demonstration**. The workflow is:

1. Download historical price data (2015–2026, ~11 years)
2. Calculate daily returns from adjusted prices
3. Estimate sample mean and covariance from the same historical window
4. Solve optimization problems using these estimated parameters
5. Report portfolio metrics computed from realized returns in the same period
6. Compare results against benchmark (SPY)

All analysis uses a single historical window for both parameter estimation and performance evaluation. **This does not validate predictive or out-of-sample performance.**

### Key Assumptions

- **Trading Days**: 252 per year (excludes weekends/holidays)
- **Risk-Free Rate**: 2.0% annually, constant throughout the sample
- **Long-Only Constraints**: No short selling
- **Position Limits**: Maximum 30% per asset (configurable)
- **Price Adjustment**: Adjusted close prices account for splits and distributions
- **Return Calculation**: Simple daily returns (percentage changes)
- **Covariance**: Sample covariance; no shrinkage or factor model adjustments
- **Rebalancing**: Portfolio is static (no rebalancing during the sample)

### Historical Performance (Illustrative)

In-sample results from 2015–2026 data:

| Strategy | Return | Volatility | Sharpe | Max Drawdown |
|----------|--------|------------|--------|--------------|
| SPY (Benchmark) | 13.85% | 17.58% | 0.674 | −33.72% |
| Equal Weight | 8.60% | 11.46% | 0.575 | −25.52% |
| Minimum Variance | 5.56% | 7.20% | 0.494 | −20.99% |
| **Maximum Sharpe** | 13.04% | 11.59% | **0.952** | −21.59% |

**Interpretation**: The Maximum-Sharpe portfolio achieved the highest risk-adjusted return (Sharpe 0.952) with 36% lower volatility than SPY. **This is a historical in-sample result, not a forward-looking prediction.**

## Repository Structure

```text
PortfolioLab/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── config/
│   └── config.yaml                    # Configurable parameters
│
├── data/
│   ├── raw/                           # Downloaded prices (not committed)
│   └── processed/                     # Processed returns & covariance
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── market_data.py             # Data download & preprocessing
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── mean_variance.py           # Portfolio optimization
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   └── metrics.py                 # Risk & performance metrics
│   │
│   ├── backtesting/
│   │   └── __init__.py                # Placeholder for Milestone 2
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                   # Figure generation
│
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py                # 12 tests for risk metrics
│   └── test_optimization.py           # 9 tests for optimization
│
├── results/
│   ├── performance/                   # CSV outputs (weights, metrics, frontier)
│   └── figures/                       # PNG visualizations
│
├── run_analysis.py                    # Main execution script
└── audit_check.py                     # Audit & verification script
```

## Code Quality & Testing

### Test Coverage

The project includes **21 comprehensive unit tests** covering:

**Metrics Tests** (12 tests):
- Cumulative return calculation
- Annualized return (CAGR)
- Annualized volatility (sqrt rule)
- Sharpe ratio consistency
- Sortino ratio (downside deviation)
- Maximum drawdown (convention: negative)
- Return vs arithmetic annualization distinction
- Extreme cases and edge conditions

**Optimization Tests** (9 tests):
- Equal-weight portfolio
- Minimum-variance optimization
- Maximum-Sharpe optimization
- Long-only constraint enforcement
- Weight sum constraint (= 1.0)
- Portfolio return and volatility correctness
- Sharpe ratio recomputation
- Efficient frontier feasibility
- Constraint satisfaction

### Mathematical Verification

All portfolio calculations are independently verified using manual computation:
- Portfolio return: $w^T \mu$
- Portfolio variance: $w^T \Sigma w$
- Sharpe ratio: $(R_p - R_f) / \sigma_p$
- Sortino ratio and downside deviation

See `audit_check.py` for full verification script.

### Code Standards

- **Type Hints**: All function signatures include type annotations
- **Docstrings**: All public functions documented with parameter and return descriptions
- **Error Handling**: Explicit validation with meaningful error messages
- **PEP 8**: Code follows Python style guidelines
- **Reproducibility**: Configuration externalized, no hardcoded paths or seeds
- **Modularity**: Clear separation of concerns across 5 modules

## Roadmap

### Milestone 1 (Current) ✓
In-sample portfolio optimization with mean-variance framework.

### Milestone 2
**Walk-Forward Backtesting**
- Rolling optimization window
- Portfolio rebalancing at fixed intervals
- Out-of-sample validation
- Turnover and implementation cost tracking

### Milestone 3
**Transaction Costs & Constraints**
- Bid-ask spread modeling
- Rebalancing frequency and costs
- Turnover limits and constraints
- Hold periods and manager restrictions

### Milestone 4
**Risk Parity & Factor Tilts**
- Risk contribution allocation
- Factor model integration
- Style-aware portfolio construction
- Momentum and value tilts

### Milestone 5
**Black-Litterman Framework**
- Prior distribution from market prices
- Investor views integration
- Posterior return estimation
- Posterior covariance adjustment

### Milestone 6
**Non-Linear Risk Metrics**
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Non-normal return distributions
- Tail risk optimization

### Milestone 7
**Factor Attribution & Decomposition**
- Fama-French factors
- Cross-sectional regression
- Factor exposure analysis
- Performance attribution

### Milestone 8
**Interactive Research Dashboard**
- Web-based portfolio visualization
- Real-time metric updates
- Sensitivity analysis interface
- Strategy backtesting UI

## Benchmark Results

This section shows **illustrative in-sample performance only**.

### 2015–2026 Annualized Metrics

```
                  Return   Vol   Sharpe   Sortino   MaxDD
SPY               13.85%  17.58%  0.674   0.638   -33.72%
Equal-Weight       8.60%  11.46%  0.575   0.542   -25.52%
Min-Variance       5.56%   7.20%  0.494   0.478   -20.99%
Max-Sharpe        13.04%  11.59%  0.952   0.910   -21.59%
```

**Key Observations**:
- Max-Sharpe achieved highest risk-adjusted return despite lower absolute return than SPY
- Min-Variance portfolio produced lowest volatility at the cost of lower return
- Equal-weight was middle ground between max-Sharpe and min-Variance
- All portfolios experienced drawdowns but Max-Sharpe and Min-Variance reduced downside

**Critical Note**: This is **one historical window with a specific start/end date**. Different periods will produce different results. Do not extrapolate to future performance.

## For Recruiters

This project demonstrates:

✓ **Deep quantitative knowledge**: Mean-variance framework, optimization theory, risk modeling  
✓ **Advanced Python**: Type hints, vectorization, pandas/numpy/scipy usage  
✓ **Software engineering discipline**: Modular architecture, unit tests, clear documentation  
✓ **Research rigor**: Transparent methodology, explicit assumptions, in-sample/out-of-sample distinction  
✓ **Communication skills**: Clear README with mathematical exposition and honest limitations  

The code is intentionally **not production-grade**—the goal is to demonstrate understanding, not to build an investment product.

## For Quantitative Researchers

This implementation serves as a reference for:

- Implementing Markowitz mean-variance optimization cleanly
- Handling daily-to-annual return annualization correctly
- Distinguishing expected returns (optimizer) from realized returns (performance reporting)
- Building efficient frontiers with binding constraints
- Verifying optimization results independently
- Documenting assumptions and methodology rigorously

See `audit_check.py` and inline docstrings for mathematical details.

## For Software Engineers

This project demonstrates professional practices:

- Modular structure with clear separation of concerns
- Type annotations for improved maintainability
- Comprehensive unit tests with edge case coverage
- Configuration management (YAML externalization)
- Clear error messages and input validation
- Reproducible execution and logging
- Documentation via README and docstrings
- Git-friendly structure (.gitignore, no secrets, clean history)

## Known Limitations & Assumptions

**Critical Disclaimer**: This is a research demonstration, not investment advice. Backtested results do not guarantee future performance. Optimization is sensitive to parameter estimates and market conditions.

**Assumptions**:
- 252 trading days per year
- 2% constant risk-free rate
- Simple daily returns (not log returns)
- Sample covariance (no shrinkage, factor model, or other adjustments)
- Long-only constraints (no short selling)
- Frictionless market (no costs, slippage, or bid-ask spreads)

## Contributing

This is a personal research project. Suggestions and corrections are welcome via GitHub issues or pull requests.

## License

MIT License — See LICENSE file for details.

## Resources & References

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*.
- Bodie, Z., Kane, A., & Marcus, A. (2021). *Investments* (12th ed.)
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- scipy.optimize documentation: https://docs.scipy.org/doc/scipy/reference/optimize.html
- pandas documentation: https://pandas.pydata.org/docs/

---

**Status**: Milestone 1 complete and production-ready for demonstration. Ready for Milestone 2 (walk-forward backtesting) on request.
