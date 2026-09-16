from __future__ import annotations

from pathlib import Path

import pytest

from src.factors.data import FactorDataError, load_canonical_factor_dataset

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data" / "canonical_factors"
SKIP_REASON = (
    "Frozen M5 snapshot is not available in this public environment. "
    "Historical frozen_snapshot certification requires the exact locally supplied canonical factor snapshot "
    "described in REPRODUCIBILITY.md."
)


def frozen_snapshot_status(canonical_dir: str | Path | None = None) -> str:
    target = Path(canonical_dir) if canonical_dir is not None else CANONICAL_DIR
    required = [
        target / "factor_manifest.json",
        target / "source_metadata.json",
        target / "canonical_factors_daily.csv",
        target / "raw",
    ]
    if any(not path.exists() for path in required):
        return "UNAVAILABLE"

    try:
        load_canonical_factor_dataset(target)
    except FactorDataError as exc:
        missing_patterns = (
            "Missing factor manifest",
            "Missing source metadata",
            "Missing canonical factor file",
            "Missing raw archive directory",
            "Missing raw archive file",
        )
        if any(pattern in str(exc) for pattern in missing_patterns):
            return "UNAVAILABLE"
        return "INVALID"
    return "AVAILABLE"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "frozen_snapshot: Historical Milestone 5 certification tests requiring the exact frozen canonical factor snapshot; this is a local historical snapshot contract, not a public general factor-data requirement.",
    )


def pytest_collection_modifyitems(config, items):
    if not any(item.get_closest_marker("frozen_snapshot") for item in items):
        return
    status = frozen_snapshot_status()
    if status == "UNAVAILABLE":
        skip = pytest.mark.skip(reason=SKIP_REASON)
        for item in items:
            if item.get_closest_marker("frozen_snapshot"):
                item.add_marker(skip)
    elif status == "INVALID":
        raise FactorDataError("Frozen M5 snapshot is present but invalid or corrupt; fail closed rather than treating it as unavailable.")
    else:
        return


@pytest.fixture
def require_frozen_snapshot():
    status = frozen_snapshot_status()
    if status == "UNAVAILABLE":
        pytest.skip(SKIP_REASON)
    if status == "INVALID":
        raise FactorDataError("Frozen M5 snapshot is present but invalid or corrupt; fail closed rather than treating it as unavailable.")
    return status
