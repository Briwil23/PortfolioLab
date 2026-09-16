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

PortfolioLab's Milestone 5 factor pipeline uses daily Fama/French factor data from the Kenneth R. French Data Library. The repository records the official source entries and the exact transformation path in [src/factors/data.py](src/factors/data.py), [data/canonical_factors/factor_manifest.json](data/canonical_factors/factor_manifest.json), and [data/canonical_factors/source_metadata.json](data/canonical_factors/source_metadata.json).

### Provider and sources

Provider: Kenneth R. French Data Library

Official source archives recorded by the repository:

- Fama/French 5 Factors [Daily]
  - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip
- Momentum Factor (Mom) [Daily]
  - https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip

These are the source inputs referenced by the canonical build logic in `src/factors/data.py`.

### Publication policy

PORTFOLIOLAB M5 FACTOR DATA POLICY

1. The M5 empirical study remains tied to the frozen canonical factor snapshot used during certification.
2. The frozen canonical snapshot is identified by the date range 1963-07-01 through 2026-06-30, row count 15854, schema `date, MKT_RF, SMB, HML, RMW, CMA, MOM, RF`, canonical SHA256 `4d402ebf191167dc4af45dd5ccc0525453354c4759e2485891840a1780386c94`, and the source provenance metadata stored in [data/canonical_factors/factor_manifest.json](data/canonical_factors/factor_manifest.json) and [data/canonical_factors/source_metadata.json](data/canonical_factors/source_metadata.json).
3. The public repository does not redistribute the original downloaded Kenneth French archives.
4. The public repository does not redistribute the frozen reconstructed canonical factor CSV.
5. The repository does publish acquisition and build code, provider/source metadata, frozen hashes, transformation methodology, tests, and derived research outputs.
6. Kenneth French source archives are rolling upstream datasets and may change after a PortfolioLab research snapshot is frozen.
7. Current upstream archives can therefore reproduce the pipeline and transformation methodology, but they are not guaranteed to reproduce a previously frozen canonical snapshot byte-for-byte.
8. Exact reproduction of the certified M5 empirical outputs requires the exact frozen factor snapshot identified by the committed provenance record.
9. The frozen M5 results remain historical research artifacts and must not be silently recomputed against revised upstream factor history.

This is not a claim that the data are illegal to use; it is a conservative publication decision to avoid distributing source files whose redistribution rights are not explicitly established.

### Canonical construction workflow

The canonical factor dataset is built using the repository's supported Python API in [src/factors/data.py](src/factors/data.py):

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

These steps are implemented in `build_canonical_factor_dataset(...)` and `validate_canonical_factor_directory(...)` in [src/factors/data.py](src/factors/data.py).

### Reproducibility terminology

Pipeline reproducibility means the public code can reacquire the current official factor sources and execute the documented transformations, validations, merging, and schema construction.

Snapshot reproducibility means the ability to reconstruct the exact historical canonical bytes and SHA256 used for the certified M5 experiment.

Pipeline reproducibility remains available. Snapshot reproducibility is tied to the frozen provenance record and is not guaranteed from current upstream archives alone. The committed canonical hash identifies the frozen research vintage.

A clean-source provenance comparison found that current official Kenneth R. French archives extend beyond the frozen cutoff and also contain historical revisions inside the overlap window, so applying the cutoff restores the frozen row and date dimensions but does not restore the frozen canonical SHA256.

### Provenance and hash validation

The public metadata files are deliberately retained and are part of the public repo:

- `data/canonical_factors/factor_manifest.json`
- `data/canonical_factors/source_metadata.json`

Their roles are:

- `factor_manifest.json`: stores the canonical file name, canonical SHA256, date range, row count, columns, parser version, and per-archive raw hashes from the official source downloads.
- `source_metadata.json`: stores the provider, factor definitions, units, date convention, parser behavior, and validation notes used to explain the transformed canonical dataset.

The code validates that the locally present canonical file matches the expected hash and that each raw archive still matches its recorded raw hash before accepting the factor directory as valid.

The frozen contract is:

- date range: 1963-07-01 through 2026-06-30
- rows: 15854
- columns: `date, MKT_RF, SMB, HML, RMW, CMA, MOM, RF`
- canonical SHA256: `4d402ebf191167dc4af45dd5ccc0525453354c4759e2485891840a1780386c94`

Later official-source vintages were observed to differ historically from that frozen research snapshot.

### M5 test dependency

The following tests require the canonical factor dataset to exist and be validated before they can run meaningfully:

- `tests/test_factor_data.py`
- `tests/test_factor_regression.py`
- `tests/test_milestone5_static.py`
- `tests/test_milestone5_rolling_mechanics.py`
- `tests/test_milestone5_rolling_empirical.py`

These are M5-specific factor-analysis tests. They are not representative of the entire repository. The rest of PortfolioLab does not require the factor-data acquisition/build step unless the relevant M5 workflow is being exercised.

## Public/core vs frozen-snapshot certification

PortfolioLab separates the repository's public validation path from the historical certified Milestone 5 snapshot path.

Public/core validation:

```bash
python -m pytest -m "not frozen_snapshot" -q
```

Historical frozen-snapshot certification:

```bash
python -m pytest -m frozen_snapshot -q
```

Full local certification when the exact frozen snapshot is available:

```bash
python -m pytest tests/ -q
```

Preflight capability check:

```bash
python -m src.reproducibility.preflight
```

Behavior:

- If the exact frozen M5 snapshot is absent or intentionally unavailable in a public clone, `frozen_snapshot` tests are skipped with an explicit reason instead of failing generically.
- If the snapshot is present but invalid/corrupt, the loader fails closed and the tests do not silently masquerade as absent.
- The public/core path remains meaningful for clean clones without claiming historical M5 certification.
- Current upstream Kenneth R. French data is not treated as a substitute for the exact historical frozen M5 dataset.

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
