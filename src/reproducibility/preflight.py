from __future__ import annotations

from pathlib import Path

from src.factors.data import FactorDataError, load_canonical_factor_dataset

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "data" / "canonical_factors"


def frozen_snapshot_status(canonical_dir: str | Path | None = None) -> str:
    """Classify the availability of the exact frozen M5 canonical factor snapshot."""
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


def main() -> int:
    public_core = "AVAILABLE"
    snapshot_status = frozen_snapshot_status()
    certification_status = "AVAILABLE" if snapshot_status == "AVAILABLE" else "UNAVAILABLE"
    full_local_status = "AVAILABLE" if snapshot_status == "AVAILABLE" else "UNAVAILABLE"

    print(f"PUBLIC CORE VALIDATION: {public_core}")
    print(f"FROZEN M5 SNAPSHOT: {snapshot_status}")
    print(f"FROZEN M5 CERTIFICATION: {certification_status}")
    print(f"FULL LOCAL CERTIFICATION: {full_local_status}")
    return 0 if snapshot_status in {"AVAILABLE", "UNAVAILABLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
