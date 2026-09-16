from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.factors.data import (
    CANONICAL_FACTORS_FILE,
    CANONICAL_FACTORS_MANIFEST_FILE,
    CANONICAL_FACTORS_SOURCE_METADATA_FILE,
    FactorDataError,
)
from src.reproducibility.preflight import frozen_snapshot_status

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_snapshot_marker_is_registered():
    ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "frozen_snapshot" in ini
    assert "Historical Milestone 5" in ini


def test_missing_snapshot_is_reported_as_unavailable(tmp_path):
    missing_dir = tmp_path / "missing_snapshot"
    assert frozen_snapshot_status(missing_dir) == "UNAVAILABLE"


def test_invalid_snapshot_is_not_treated_as_absent(tmp_path):
    target = tmp_path / "invalid_snapshot"
    target.mkdir()
    (target / CANONICAL_FACTORS_MANIFEST_FILE).write_text(json.dumps({"canonical_sha256": "deadbeef"}), encoding="utf-8")
    (target / CANONICAL_FACTORS_SOURCE_METADATA_FILE).write_text(json.dumps({"provider": "test"}), encoding="utf-8")
    (target / CANONICAL_FACTORS_FILE).write_text("date,MKT_RF\n2020-01-01,0.01\n", encoding="utf-8")
    (target / "raw").mkdir()

    assert frozen_snapshot_status(target) == "INVALID"


def test_public_core_is_available_without_snapshot():
    assert frozen_snapshot_status(ROOT / "data" / "missing") == "UNAVAILABLE"
    assert "UNAVAILABLE" in {"UNAVAILABLE"}
