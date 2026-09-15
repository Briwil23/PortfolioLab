from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from src.factors.data import (
    CANONICAL_FACTORS_FILE,
    CANONICAL_FACTORS_MANIFEST_FILE,
    CANONICAL_FACTORS_RAW_DIRNAME,
    CANONICAL_FACTORS_SOURCE_METADATA_FILE,
    FF5_DAILY_SOURCE_SPEC,
    MOMENTUM_DAILY_SOURCE_SPEC,
    FACTOR_CANONICAL_COLUMNS,
    FactorDataError,
    _parse_daily_archive,
    _spec,
    align_factor_dates_with_returns,
    build_canonical_factor_dataset,
    compute_file_sha256,
    load_canonical_factor_dataset,
    validate_canonical_factor_frame,
    write_factor_validation_artifacts,
)
from src.reproducibility.checks import max_keyed_return_difference


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data" / "canonical_factors"
M4_GROSS = ROOT / "results" / "milestone4_canonical" / "walk_forward_returns_gross.csv"


def _synthetic_zip(tmp_path: Path, filename: str, content: str) -> Path:
    archive_path = tmp_path / filename
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr(filename.replace(".zip", ".csv"), content)
    return archive_path


def _daily_ff5_sample() -> str:
    return """This file was created by using the 202606 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.34,   -0.01,    0.16,    0.01
19630702,    0.79,   -0.26,    0.26,   -0.07,   -0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def _daily_mom_sample() -> str:
    return """This file was created by using the 202606 CRSP database.

,Mom
19630701,    0.10
19630702,   -0.20

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def _daily_ff5_with_sentinel() -> str:
    return """This file was created by using the 202606 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.34,   -0.01,    0.16,    0.01
19630702,  -99.99,    0.00,    0.26,   -0.07,   -0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def _daily_ff5_with_footer_noise() -> str:
    return """This file was created by using the 202606 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.34,   -0.01,    0.16,    0.01
19630702,    0.79,   -0.26,    0.26,   -0.07,   -0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def _copy_canonical_tree(tmp_path: Path) -> Path:
    target = tmp_path / "canonical_copy"
    shutil.copytree(CANONICAL_DIR, target)
    return target


def test_percent_to_decimal_conversion(tmp_path):
    archive = _synthetic_zip(tmp_path, "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", _daily_ff5_sample())
    frame = _parse_daily_archive(archive, _spec(FF5_DAILY_SOURCE_SPEC))
    assert frame.loc[0, "MKT_RF"] == pytest.approx(-0.0067)
    assert frame.loc[0, "RF"] == pytest.approx(0.0001)
    assert frame.loc[1, "CMA"] == pytest.approx(-0.0020)


def test_deliberate_100x_unit_error_rejection(tmp_path):
    archive = _synthetic_zip(tmp_path, "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", _daily_ff5_with_sentinel())
    with pytest.raises(FactorDataError, match="Sentinel value"):
        _parse_daily_archive(archive, _spec(FF5_DAILY_SOURCE_SPEC))


def test_parser_header_detection(tmp_path):
    archive = _synthetic_zip(tmp_path, "F-F_Momentum_Factor_daily_CSV.zip", _daily_mom_sample())
    frame = _parse_daily_archive(archive, _spec(MOMENTUM_DAILY_SOURCE_SPEC))
    assert list(frame.columns) == ["date", "MOM"]
    assert len(frame) == 2


def test_parser_footer_termination(tmp_path):
    archive = _synthetic_zip(tmp_path, "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", _daily_ff5_with_footer_noise())
    frame = _parse_daily_archive(archive, _spec(FF5_DAILY_SOURCE_SPEC))
    assert len(frame) == 2


def test_malformed_row_rejection(tmp_path):
    content = """This file was created by using the 202606 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.34,   -0.01,    0.16,    0.01
BADROW,    0.79,   -0.26,    0.26,   -0.07,   -0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""
    archive = _synthetic_zip(tmp_path, "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", content)
    with pytest.raises(FactorDataError, match="Malformed date token"):
        _parse_daily_archive(archive, _spec(FF5_DAILY_SOURCE_SPEC))


def test_required_factor_label_validation():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "MKT_RF": [0.1, 0.2],
            "SMB": [0.1, 0.2],
            "HML": [0.1, 0.2],
            "RMW": [0.1, 0.2],
            "CMA": [0.1, 0.2],
            "MOM": [0.1, 0.2],
            "RF": [0.1, 0.2],
        }
    )
    validated = validate_canonical_factor_frame(frame)
    assert list(validated.columns) == FACTOR_CANONICAL_COLUMNS


def test_factor_permutation_invariance():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "RF": [0.1, 0.2],
            "MOM": [0.1, 0.2],
            "CMA": [0.1, 0.2],
            "RMW": [0.1, 0.2],
            "HML": [0.1, 0.2],
            "SMB": [0.1, 0.2],
            "MKT_RF": [0.1, 0.2],
        }
    )
    validated = validate_canonical_factor_frame(frame)
    assert list(validated.columns) == FACTOR_CANONICAL_COLUMNS


def test_duplicate_factor_date_rejection():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "MKT_RF": [0.1, 0.2],
            "SMB": [0.1, 0.2],
            "HML": [0.1, 0.2],
            "RMW": [0.1, 0.2],
            "CMA": [0.1, 0.2],
            "MOM": [0.1, 0.2],
            "RF": [0.1, 0.2],
        }
    )
    with pytest.raises(FactorDataError, match="unique"):
        validate_canonical_factor_frame(frame)


def test_missing_nonfinite_rejection():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "MKT_RF": [0.1, float("nan")],
            "SMB": [0.1, 0.2],
            "HML": [0.1, 0.2],
            "RMW": [0.1, 0.2],
            "CMA": [0.1, 0.2],
            "MOM": [0.1, 0.2],
            "RF": [0.1, 0.2],
        }
    )
    with pytest.raises(FactorDataError, match="non-finite"):
        validate_canonical_factor_frame(frame)


def test_deterministic_parsing(tmp_path):
    archive = _synthetic_zip(tmp_path, "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", _daily_ff5_sample())
    first = _parse_daily_archive(archive, _spec(FF5_DAILY_SOURCE_SPEC))
    second = _parse_daily_archive(archive, _spec(FF5_DAILY_SOURCE_SPEC))
    pd.testing.assert_frame_equal(first, second)


def test_canonical_factor_sha256_validation():
    _, manifest, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    assert manifest["canonical_sha256"] == compute_file_sha256(CANONICAL_DIR / CANONICAL_FACTORS_FILE)


def test_fail_closed_hash_mismatch(tmp_path):
    target = _copy_canonical_tree(tmp_path)
    canonical_file = target / CANONICAL_FACTORS_FILE
    frame = pd.read_csv(canonical_file)
    frame.loc[0, "MKT_RF"] += 0.000001
    frame.to_csv(canonical_file, index=False)
    with pytest.raises(FactorDataError, match="hash mismatch"):
        load_canonical_factor_dataset(target)


def test_source_metadata_completeness():
    _, _, source_metadata = load_canonical_factor_dataset(CANONICAL_DIR)
    required = {
        "provider",
        "factor_definitions",
        "source_units",
        "canonical_units",
        "date_convention",
        "parser_behavior",
        "missing_sentinel_behavior_observed",
        "factor_geography",
        "momentum_source",
        "known_multi_asset_limitation",
        "optional_validation_source",
    }
    assert required.issubset(source_metadata)


def test_raw_source_hash_validation():
    _, manifest, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    raw_dir = CANONICAL_DIR / CANONICAL_FACTORS_RAW_DIRNAME
    for source_file in manifest["source_files"]:
        raw_file = raw_dir / source_file["filename"]
        assert compute_file_sha256(raw_file) == source_file["raw_sha256"]


def test_canonical_schema_validation():
    canonical, manifest, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    assert list(canonical.columns) == FACTOR_CANONICAL_COLUMNS
    assert manifest["columns"] == FACTOR_CANONICAL_COLUMNS


def test_exact_date_intersection():
    matched, report = align_factor_dates_with_returns(M4_GROSS, CANONICAL_DIR)
    assert matched["date"].min().date().isoformat() == report["matched_start"]
    assert matched["date"].max().date().isoformat() == report["matched_end"]
    assert report["exact_inner_join_only"] is True
    assert report["no_fill_behavior"] is True


def test_no_fill_behavior():
    matched, report = align_factor_dates_with_returns(M4_GROSS, CANONICAL_DIR)
    assert not matched.empty
    assert report["no_fill_behavior"] is True
    assert report["exact_inner_join_only"] is True


def test_unmatched_date_reporting():
    _, report = align_factor_dates_with_returns(M4_GROSS, CANONICAL_DIR)
    assert isinstance(report["unmatched_returns_dates"], list)
    assert isinstance(report["unmatched_factor_dates"], list)
    assert report["matched_observation_count"] > 0


def test_m4_return_preservation():
    diff = max_keyed_return_difference(M4_GROSS, M4_GROSS)
    assert diff <= 1e-12


def test_m4_date_strategy_uniqueness():
    gross = pd.read_csv(M4_GROSS)
    assert gross["Date"].is_unique
    assert gross.columns.is_unique


def test_no_optimizer_call_from_data_layer(monkeypatch):
    import src.optimization.risk_based as risk_based

    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer should not be called from the factor data layer")

    monkeypatch.setattr(risk_based, "equal_risk_contribution_portfolio", forbidden)
    canonical, manifest, source_metadata = load_canonical_factor_dataset(CANONICAL_DIR)
    assert not canonical.empty
    assert manifest["canonical_file"] == CANONICAL_FACTORS_FILE
    assert source_metadata["provider"]


def test_no_live_call_canonical_load(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Canonical load must not call live downloads")

    monkeypatch.setattr("src.factors.data._download_url_bytes", forbidden)
    canonical, manifest, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    assert not canonical.empty
    assert manifest["canonical_file"] == CANONICAL_FACTORS_FILE


def test_repeated_canonical_load_determinism():
    first = load_canonical_factor_dataset(CANONICAL_DIR)
    second = load_canonical_factor_dataset(CANONICAL_DIR)
    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_canonical_preferred_output_ordering_without_positional_dependence():
    canonical, _, _ = load_canonical_factor_dataset(CANONICAL_DIR)
    shuffled = canonical[["date", "RF", "MOM", "CMA", "RMW", "HML", "SMB", "MKT_RF"]]
    validated = validate_canonical_factor_frame(shuffled)
    assert list(validated.columns) == FACTOR_CANONICAL_COLUMNS


def test_build_and_validation_artifacts(tmp_path):
    output_dir = tmp_path / "validation"
    result = write_factor_validation_artifacts(CANONICAL_DIR, M4_GROSS, output_dir)
    assert (output_dir / "factor_ingestion_validation.json").exists()
    assert (output_dir / "factor_alignment_validation.json").exists()
    assert (output_dir / "factor_data_preview.csv").exists()
    assert result["validation_report"]["no_live_call"] is True


def test_build_canonical_data_contract_is_consistent():
    canonical, manifest, source_metadata = load_canonical_factor_dataset(CANONICAL_DIR)
    assert canonical["date"].is_monotonic_increasing
    assert manifest["units"] == "decimal"
    assert source_metadata["factor_geography"] == "US"
    assert source_metadata["momentum_source"] == "separate daily series"
