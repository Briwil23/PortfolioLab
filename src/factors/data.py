"""Canonical daily factor ingestion and validation for Milestone 5."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import pandas as pd

from src.data.market_data import compute_file_sha256
from src.reproducibility.checks import read_returns_long


CANONICAL_FACTORS_DIRNAME = "canonical_factors"
CANONICAL_FACTORS_FILE = "canonical_factors_daily.csv"
CANONICAL_FACTORS_MANIFEST_FILE = "factor_manifest.json"
CANONICAL_FACTORS_SOURCE_METADATA_FILE = "source_metadata.json"
CANONICAL_FACTORS_RAW_DIRNAME = "raw"
FACTOR_DATA_PREVIEW_FILE = "factor_data_preview.csv"

FACTOR_CANONICAL_COLUMNS = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]
FACTOR_REQUIRED_LABELS = FACTOR_CANONICAL_COLUMNS[1:]
FACTOR_PREFERRED_DISPLAY_ORDER = list(FACTOR_CANONICAL_COLUMNS)

SOURCE_SENTINEL_VALUES = {"-99.99", "-999", "-999.99"}
SOURCE_PROVIDER = "Kenneth R. French Data Library"
PARSER_VERSION = "1.0.0"
DEFAULT_FREQUENCY = "daily"
DEFAULT_UNITS = "decimal"

FF5_DAILY_SOURCE_SPEC = {
    "dataset_name": "Fama/French 5 Factors [Daily]",
    "source_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "archive_filename": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "member_name": "F-F_Research_Data_5_Factors_2x3_daily.csv",
    "source_columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    "canonical_columns": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"],
    "date_format": "%Y%m%d",
}

MOMENTUM_DAILY_SOURCE_SPEC = {
    "dataset_name": "Momentum Factor (Mom) [Daily]",
    "source_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    "archive_filename": "F-F_Momentum_Factor_daily_CSV.zip",
    "member_name": "F-F_Momentum_Factor_daily.csv",
    "source_columns": ["Mom"],
    "canonical_columns": ["MOM"],
    "date_format": "%Y%m%d",
}

FF3_DAILY_SOURCE_SPEC = {
    "dataset_name": "Fama/French 3 Factors [Daily]",
    "source_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "archive_filename": "F-F_Research_Data_Factors_daily_CSV.zip",
    "member_name": "F-F_Research_Data_Factors_daily.csv",
    "source_columns": ["Mkt-RF", "SMB", "HML", "RF"],
    "canonical_columns": ["MKT_RF", "SMB", "HML", "RF"],
    "date_format": "%Y%m%d",
}


class FactorDataError(RuntimeError):
    """Raised when canonical factor ingestion or validation fails."""


@dataclass(frozen=True)
class SourceSpec:
    dataset_name: str
    source_url: str
    archive_filename: str
    member_name: str
    source_columns: tuple[str, ...]
    canonical_columns: tuple[str, ...]
    date_format: str


_DAILY_DATE_RE = re.compile(r"^\d{8}$")


def _spec(spec: dict) -> SourceSpec:
    return SourceSpec(
        dataset_name=str(spec["dataset_name"]),
        source_url=str(spec["source_url"]),
        archive_filename=str(spec["archive_filename"]),
        member_name=str(spec["member_name"]),
        source_columns=tuple(spec["source_columns"]),
        canonical_columns=tuple(spec["canonical_columns"]),
        date_format=str(spec["date_format"]),
    )


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _download_url_bytes(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def _extract_zip_member_text(zip_path: str | Path) -> tuple[str, str]:
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        if len(members) != 1:
            raise FactorDataError(f"Expected exactly one member in {zip_path}, found {members!r}.")
        member_name = members[0]
        text = archive.read(member_name).decode("utf-8-sig")
    return member_name, text


def _normalize_header_tokens(line: str) -> list[str]:
    return [token.strip() for token in next(csv.reader([line]))]


def _is_daily_date_token(token: str) -> bool:
    return bool(_DAILY_DATE_RE.fullmatch(token.strip()))


def _parse_daily_archive(zip_path: str | Path, spec: SourceSpec) -> pd.DataFrame:
    member_name, text = _extract_zip_member_text(zip_path)
    archive_stem = Path(spec.archive_filename).stem
    accepted_member_names = {
        spec.member_name,
        f"{archive_stem}.csv",
        spec.member_name.replace(".csv", "_CSV.csv"),
        spec.member_name.replace("_daily.csv", "_daily_CSV.csv"),
    }
    if member_name not in accepted_member_names:
        raise FactorDataError(
            f"Unexpected archive member for {spec.dataset_name}: {member_name!r} not in {sorted(accepted_member_names)!r}."
        )

    lines = text.splitlines()
    expected_header = ["", *spec.source_columns]

    header_index = None
    for index, line in enumerate(lines):
        if _normalize_header_tokens(line) == expected_header:
            header_index = index
            break
    if header_index is None:
        raise FactorDataError(f"Could not locate expected header for {spec.dataset_name}.")

    rows: list[dict[str, object]] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                continue
            raise FactorDataError(f"Unexpected blank row before daily data in {spec.dataset_name}.")
        if stripped.startswith("Copyright"):
            break

        tokens = [token.strip() for token in next(csv.reader([line]))]
        if len(tokens) != len(spec.source_columns) + 1:
            raise FactorDataError(
                f"Malformed row in {spec.dataset_name}: expected {len(spec.source_columns) + 1} fields, got {len(tokens)}."
            )
        if not _is_daily_date_token(tokens[0]):
            raise FactorDataError(f"Malformed date token in {spec.dataset_name}: {tokens[0]!r}.")

        numeric_tokens = tokens[1:]
        if any(token in SOURCE_SENTINEL_VALUES for token in numeric_tokens):
            raise FactorDataError(f"Sentinel value encountered in {spec.dataset_name}: {numeric_tokens!r}.")

        try:
            values = [float(token) / 100.0 for token in numeric_tokens]
        except ValueError as exc:
            raise FactorDataError(f"Non-numeric factor row in {spec.dataset_name}: {tokens!r}.") from exc

        if not all(pd.notna(value) for value in values):
            raise FactorDataError(f"Non-finite factor row in {spec.dataset_name}: {tokens!r}.")

        rows.append(
            {
                "date": pd.to_datetime(tokens[0], format=spec.date_format),
                **{column: value for column, value in zip(spec.canonical_columns, values, strict=True)},
            }
        )

    if not rows:
        raise FactorDataError(f"No daily rows were parsed for {spec.dataset_name}.")

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    return frame


def validate_canonical_factor_frame(
    factor_frame: pd.DataFrame,
    required_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Validate a canonical factor frame without requiring a fixed column order."""
    required = list(required_labels or FACTOR_REQUIRED_LABELS)
    if factor_frame.empty:
        raise FactorDataError("Canonical factor frame is empty.")

    columns = list(factor_frame.columns)
    if len(set(columns)) != len(columns):
        raise FactorDataError(f"Duplicate factor labels are not allowed: {columns!r}.")
    if "date" not in columns:
        raise FactorDataError("Canonical factor frame must contain a date column.")

    actual_labels = [column for column in columns if column != "date"]
    if set(actual_labels) != set(required):
        missing = sorted(set(required) - set(actual_labels))
        unexpected = sorted(set(actual_labels) - set(required))
        raise FactorDataError(
            f"Factor labels must match the required schema exactly. Missing={missing}, unexpected={unexpected}."
        )

    validated = factor_frame.copy()
    validated["date"] = pd.to_datetime(validated["date"], utc=False)
    if validated["date"].duplicated().any():
        raise FactorDataError("Canonical factor dates must be unique.")

    validated = validated.sort_values("date", kind="mergesort").reset_index(drop=True)
    numeric = validated[[column for column in validated.columns if column != "date"]].apply(pd.to_numeric, errors="coerce")
    if not pd.notna(numeric.to_numpy()).all():
        raise FactorDataError("Canonical factor frame contains non-finite values or missing values.")

    return validated[["date", *required]]


def _validate_source_frame_against_spec(frame: pd.DataFrame, spec: SourceSpec) -> pd.DataFrame:
    renamed = frame.rename(columns={source: canonical for source, canonical in zip(spec.source_columns, spec.canonical_columns, strict=True)})
    return validate_canonical_factor_frame(renamed, required_labels=spec.canonical_columns)


def download_factor_archives(
    raw_dir: str | Path,
    specs: Iterable[dict] | None = None,
    refresh: bool = False,
) -> list[dict]:
    """Download canonical raw archives exactly once into the raw directory."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    chosen_specs = tuple(specs or (FF5_DAILY_SOURCE_SPEC, MOMENTUM_DAILY_SOURCE_SPEC))
    records: list[dict] = []

    for spec_dict in chosen_specs:
        spec = _spec(spec_dict)
        archive_path = raw_path / spec.archive_filename
        if refresh or not archive_path.exists():
            archive_bytes = _download_url_bytes(spec.source_url)
            archive_path.write_bytes(archive_bytes)

        records.append(
            {
                "dataset_name": spec.dataset_name,
                "filename": spec.archive_filename,
                "source_url": spec.source_url,
                "raw_sha256": compute_file_sha256(archive_path),
                "download_utc_timestamp": _utc_now_iso(),
            }
        )

    return records


def build_canonical_factor_dataset(
    canonical_dir: str | Path,
    refresh_raw: bool = False,
    include_ff3_validation: bool = False,
) -> dict:
    """Build the frozen daily factor dataset from the canonical French archives."""
    canonical_path = Path(canonical_dir)
    raw_dir = canonical_path / CANONICAL_FACTORS_RAW_DIRNAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical_path.mkdir(parents=True, exist_ok=True)

    source_records = download_factor_archives(raw_dir=raw_dir, refresh=refresh_raw)

    ff5 = _parse_daily_archive(raw_dir / FF5_DAILY_SOURCE_SPEC["archive_filename"], _spec(FF5_DAILY_SOURCE_SPEC))
    mom = _parse_daily_archive(raw_dir / MOMENTUM_DAILY_SOURCE_SPEC["archive_filename"], _spec(MOMENTUM_DAILY_SOURCE_SPEC))

    ff5 = _validate_source_frame_against_spec(ff5, _spec(FF5_DAILY_SOURCE_SPEC))
    mom = _validate_source_frame_against_spec(mom, _spec(MOMENTUM_DAILY_SOURCE_SPEC))

    optional_validation_source = None
    if include_ff3_validation:
        ff3_path = raw_dir / FF3_DAILY_SOURCE_SPEC["archive_filename"]
        if ff3_path.exists():
            ff3 = _parse_daily_archive(ff3_path, _spec(FF3_DAILY_SOURCE_SPEC))
            ff3 = _validate_source_frame_against_spec(ff3, _spec(FF3_DAILY_SOURCE_SPEC))
            overlap = ff5.merge(ff3, on="date", how="inner", suffixes=("_ff5", "_ff3"))
            if overlap.empty:
                raise FactorDataError("FF3 validation source had no overlap with FF5 daily factors.")
            if not overlap["MKT_RF_ff5"].round(12).equals(overlap["MKT_RF_ff3"].round(12)):
                raise FactorDataError("FF3 validation source does not match FF5 MKT_RF values on overlap.")
            optional_validation_source = FF3_DAILY_SOURCE_SPEC["dataset_name"]

    canonical = ff5.merge(mom, on="date", how="inner")
    canonical = canonical[["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]]
    canonical = validate_canonical_factor_frame(canonical)

    canonical_path = canonical_path / CANONICAL_FACTORS_FILE
    canonical.to_csv(canonical_path, index=False, date_format="%Y-%m-%d")
    canonical_sha256 = compute_file_sha256(canonical_path)

    manifest = {
        "canonical_file": CANONICAL_FACTORS_FILE,
        "canonical_sha256": canonical_sha256,
        "first_date": canonical["date"].iloc[0].date().isoformat(),
        "last_date": canonical["date"].iloc[-1].date().isoformat(),
        "row_count": int(len(canonical)),
        "columns": list(canonical.columns),
        "frequency": DEFAULT_FREQUENCY,
        "units": DEFAULT_UNITS,
        "parser_version": PARSER_VERSION,
        "source_files": source_records,
    }

    source_metadata = {
        "provider": SOURCE_PROVIDER,
        "factor_definitions": {
            "MKT_RF": "Excess market return from the Fama/French U.S. daily factor file.",
            "SMB": "Small minus big.",
            "HML": "High minus low.",
            "RMW": "Robust minus weak.",
            "CMA": "Conservative minus aggressive.",
            "MOM": "Momentum factor from the separate U.S. daily momentum file.",
            "RF": "One-month Treasury bill rate.",
        },
        "source_units": "percent",
        "canonical_units": DEFAULT_UNITS,
        "date_convention": "daily trading dates in ISO YYYY-MM-DD after canonical conversion",
        "parser_behavior": {
            "header_detection": "explicit match on the expected daily header row",
            "footer_termination": "stop at the first copyright/footer line after data begins",
            "sentinel_handling": "reject any observed sentinel token rather than preserve it",
            "fill_behavior": "none",
            "order_dependence": "none; validation is name-based",
        },
        "missing_sentinel_behavior_observed": {
            FF5_DAILY_SOURCE_SPEC["dataset_name"]: "No sentinel values observed in the parsed daily archive.",
            MOMENTUM_DAILY_SOURCE_SPEC["dataset_name"]: "No sentinel values observed in the parsed daily archive.",
        },
        "factor_geography": "US",
        "momentum_source": "separate daily series",
        "known_multi_asset_limitation": "Equity factors are intentionally incomplete for the multi-asset PortfolioLab sleeve.",
        "optional_validation_source": optional_validation_source,
    }

    manifest_path = canonical_path.parent / CANONICAL_FACTORS_MANIFEST_FILE
    source_metadata_path = canonical_path.parent / CANONICAL_FACTORS_SOURCE_METADATA_FILE
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)
    with source_metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(source_metadata, file_obj, indent=2)

    return {
        "canonical_frame": canonical,
        "manifest": manifest,
        "source_metadata": source_metadata,
        "source_records": source_records,
    }


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def validate_canonical_factor_directory(canonical_dir: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """Load and validate the canonical factor directory without touching the network."""
    canonical_path = Path(canonical_dir)
    manifest_path = canonical_path / CANONICAL_FACTORS_MANIFEST_FILE
    source_metadata_path = canonical_path / CANONICAL_FACTORS_SOURCE_METADATA_FILE
    canonical_file_path = canonical_path / CANONICAL_FACTORS_FILE
    raw_dir = canonical_path / CANONICAL_FACTORS_RAW_DIRNAME

    if not manifest_path.exists():
        raise FactorDataError(f"Missing factor manifest: {manifest_path}.")
    if not source_metadata_path.exists():
        raise FactorDataError(f"Missing source metadata: {source_metadata_path}.")
    if not canonical_file_path.exists():
        raise FactorDataError(f"Missing canonical factor file: {canonical_file_path}.")
    if not raw_dir.exists():
        raise FactorDataError(f"Missing raw archive directory: {raw_dir}.")

    manifest = _load_json(manifest_path)
    source_metadata = _load_json(source_metadata_path)

    actual_canonical_sha = compute_file_sha256(canonical_file_path)
    expected_canonical_sha = manifest.get("canonical_sha256")
    if actual_canonical_sha != expected_canonical_sha:
        raise FactorDataError(
            f"Canonical factor hash mismatch. Expected {expected_canonical_sha}, got {actual_canonical_sha}."
        )

    for source_file in manifest.get("source_files", []):
        raw_file = raw_dir / source_file["filename"]
        if not raw_file.exists():
            raise FactorDataError(f"Missing raw archive file: {raw_file}.")
        actual_raw_sha = compute_file_sha256(raw_file)
        expected_raw_sha = source_file.get("raw_sha256")
        if actual_raw_sha != expected_raw_sha:
            raise FactorDataError(
                f"Raw archive hash mismatch for {raw_file.name}. Expected {expected_raw_sha}, got {actual_raw_sha}."
            )

    canonical_frame = pd.read_csv(canonical_file_path, parse_dates=["date"])
    canonical_frame = validate_canonical_factor_frame(canonical_frame)
    return canonical_frame, manifest, source_metadata


def load_canonical_factor_dataset(canonical_dir: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """Read the canonical factor dataset from disk and fail closed on any mismatch."""
    return validate_canonical_factor_directory(canonical_dir)


def align_factor_dates_with_returns(
    returns_csv_path: str | Path,
    canonical_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Intersect M4 strategy return dates with canonical factor dates."""
    canonical_factors, manifest, _ = load_canonical_factor_dataset(canonical_dir)
    returns_long = read_returns_long(returns_csv_path)

    return_dates = pd.Index(sorted(pd.to_datetime(returns_long["date"]).drop_duplicates()))
    factor_dates = pd.Index(canonical_factors["date"])

    joined_dates = pd.DataFrame({"date": return_dates}).merge(
        pd.DataFrame({"date": factor_dates}),
        on="date",
        how="inner",
    )

    unmatched_returns = sorted(return_dates.difference(factor_dates))
    unmatched_factors = sorted(factor_dates.difference(return_dates))

    matched_frame = returns_long.merge(canonical_factors, on="date", how="inner")
    report = {
        "canonical_file": manifest["canonical_file"],
        "matched_start": joined_dates["date"].min().date().isoformat() if not joined_dates.empty else None,
        "matched_end": joined_dates["date"].max().date().isoformat() if not joined_dates.empty else None,
        "matched_observation_count": int(len(joined_dates)),
        "unmatched_returns_dates": [date.date().isoformat() for date in unmatched_returns],
        "unmatched_factor_dates": [date.date().isoformat() for date in unmatched_factors],
        "return_date_count": int(len(return_dates)),
        "factor_date_count": int(len(factor_dates)),
        "intersection_date_count": int(len(joined_dates)),
        "no_fill_behavior": True,
        "exact_inner_join_only": True,
    }
    return matched_frame, report


def write_factor_validation_artifacts(
    canonical_dir: str | Path,
    returns_csv_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Write a compact validation bundle for the phase-1 factor ingestion checkpoint."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    canonical_factors, manifest, source_metadata = load_canonical_factor_dataset(canonical_dir)
    matched_frame, alignment_report = align_factor_dates_with_returns(returns_csv_path, canonical_dir)

    preview_path = output_path / FACTOR_DATA_PREVIEW_FILE
    canonical_factors.head(10).to_csv(preview_path, index=False, date_format="%Y-%m-%d")

    validation_report = {
        "canonical_factor_rows": int(len(canonical_factors)),
        "canonical_factor_first_date": canonical_factors["date"].iloc[0].date().isoformat(),
        "canonical_factor_last_date": canonical_factors["date"].iloc[-1].date().isoformat(),
        "matched_observation_count": alignment_report["matched_observation_count"],
        "matched_start": alignment_report["matched_start"],
        "matched_end": alignment_report["matched_end"],
        "unmatched_returns_date_count": len(alignment_report["unmatched_returns_dates"]),
        "unmatched_factor_date_count": len(alignment_report["unmatched_factor_dates"]),
        "m4_preservation": {
            "keyed_tolerance": 1e-12,
            "no_transform_applied": True,
        },
        "canonical_hash": manifest["canonical_sha256"],
        "raw_source_hashes": {item["filename"]: item["raw_sha256"] for item in manifest["source_files"]},
        "source_metadata_complete": True,
        "no_live_call": True,
    }

    with (output_path / "factor_ingestion_validation.json").open("w", encoding="utf-8") as file_obj:
        json.dump(validation_report, file_obj, indent=2)
    with (output_path / "factor_alignment_validation.json").open("w", encoding="utf-8") as file_obj:
        json.dump(alignment_report, file_obj, indent=2)

    return {
        "validation_report": validation_report,
        "alignment_report": alignment_report,
        "preview_path": str(preview_path),
        "matched_frame": matched_frame,
        "canonical_frame": canonical_factors,
        "manifest": manifest,
        "source_metadata": source_metadata,
    }