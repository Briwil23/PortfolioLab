# Reproducibility Contract

## Historical Milestone 2 Artifacts

`results/backtest/` contains valid historical Milestone 2 research outputs.

These files were generated from live market data at run time, and the exact
source-data snapshot used at that time was not preserved in the repository.
Because of this, exact bitwise rerun equality for those historical files is
not guaranteed.

Historical labeling statement:

"Historical Milestone 2 outputs generated from live market data; exact
source-data snapshot was not preserved."

## Canonical Reproducibility Baseline

Reproducible research now uses canonical frozen inputs under `data/canonical/`:

- `canonical_prices.csv`
- `canonical_returns.csv`
- `dataset_manifest.json`

The manifest stores metadata and SHA256 hashes for canonical files. Canonical
mode verifies these hashes at runtime and fails closed on missing files or hash
mismatch.

### Data Modes

Configured in `config/config.yaml`:

- `data_mode: canonical`
- `canonical_data_dir: data/canonical`

Supported modes:

- `canonical`: load only canonical files; no live download allowed
- `live`: download market data from provider using existing retrieval logic

## Environment Manifest

`results/reproducibility/environment_manifest.json` captures versions of key
runtime dependencies needed for numerical and optimization reproducibility.

## Baseline Output Contract

Canonical baseline outputs are written to:

- `results/reproducibility/canonical_baseline/`

with standard files:

- `walk_forward_returns.csv`
- `walk_forward_weights.csv`
- `walk_forward_metrics.csv`
- `rebalance_history.csv`

### Return-Schema Clarification

`walk_forward_returns.csv` contains portfolio strategy return series only,
unless benchmark inclusion is explicitly implemented and enabled.

This means benchmark `SPY` may appear in metrics artifacts while being absent
from the returns CSV.

## Deterministic Comparison Rules

When validating reproducibility, compare by explicit keys:

- Returns: `date + strategy`
- Weights: `date + strategy + asset`

Do not compare by row order alone.
