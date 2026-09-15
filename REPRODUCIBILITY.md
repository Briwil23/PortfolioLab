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

## Factor Data Provenance and Publication Policy

PortfolioLab's Milestone 5 factor pipeline uses daily Fama/French factor data from the Kenneth R. French Data Library. The repository records the official source entries and the exact transformation path in `src/factors/data.py`, `data/canonical_factors/factor_manifest.json`, and `data/canonical_factors/source_metadata.json`.

### Provider and sources

Provider: Kenneth R. French Data Library

Official source archives recorded by the repository:

- Fama/French 5 Factors [Daily]
  - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip
- Momentum Factor (Mom) [Daily]
  - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip

These are the source inputs referenced by the canonical build logic in `src/factors/data.py`.

### Publication policy

The public repository intentionally excludes the raw archives and the reconstructed `data/canonical_factors/canonical_factors_daily.csv` under a conservative redistribution policy. Explicit redistribution permission for the underlying third-party factor files was not established in the repository metadata, and the project therefore avoids redistributing those files while retaining provenance, hash, and build metadata in the public repo.

This is not a claim that the data are illegal to use; it is a conservative publication decision to avoid distributing source files whose redistribution rights are not explicitly established.

### Canonical construction workflow

The canonical factor dataset is built using the repository's supported Python API in `src/factors/data.py`:

```python
from src.factors.data import build_canonical_factor_dataset
build_canonical_factor_dataset("data/canonical_factors")
```

This function does the following:

1. acquire the official raw archives from the Kenneth R. French Data Library
2. read the expected CSV member from each ZIP archive
3. detect the expected daily header row
4. stop at the first copyright/footer line after data begins
5. reject sentinel values such as `-99.99` or `-999`
6. convert percentage values to decimals by dividing by 100
7. merge the FF5 daily file with the separate daily momentum factor on the `date` key
8. normalize the canonical factor columns to `date`, `MKT_RF`, `SMB`, `HML`, `RMW`, `CMA`, `MOM`, and `RF`
9. validate the final schema and hash integrity against the manifest

These steps are implemented in `build_canonical_factor_dataset(...)` and `validate_canonical_factor_directory(...)` in `src/factors/data.py`.

### Provenance and hash validation

The public metadata files are deliberately retained and are part of the public repo:

- `data/canonical_factors/factor_manifest.json`
- `data/canonical_factors/source_metadata.json`

Their roles are:

- `factor_manifest.json`: stores the canonical file name, canonical SHA256, date range, row count, columns, parser version, and per-archive raw hashes from the official source downloads.
- `source_metadata.json`: stores the provider, factor definitions, units, date convention, parser behavior, and validation notes used to explain the transformed canonical dataset.

The code validates that the locally present canonical file matches the expected hash and that each raw archive still matches its recorded raw hash before accepting the factor directory as valid.

### M5 test dependency

The following tests require the canonical factor dataset to exist and be validated before they can run meaningfully:

- `tests/test_factor_data.py`
- `tests/test_factor_regression.py`
- `tests/test_milestone5_static.py`
- `tests/test_milestone5_rolling_mechanics.py`
- `tests/test_milestone5_rolling_empirical.py`

These are M5-specific factor-analysis tests. They are not representative of the entire repository. The rest of PortfolioLab does not require the factor-data acquisition/build step unless the relevant M5 workflow is being exercised.

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
