"""Validation rules for independent-manifest labeling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any, cast

import pandas as pd

from .schema import (
    ACCESSION_PATTERN,
    CONTRACT_VERSION,
    DATASET_ROLES,
    EVIDENCE_TIERS,
    FEATURE_JOIN_KEY,
    MANIFEST_COLUMNS,
    MANIFEST_JOIN_KEY,
    REQUIRED_VALUE_COLUMNS,
    TARGET_LABELS,
)


class LabelingValidationError(ValueError):
    """Raised when a dataset violates a fatal labeling contract rule."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        errors = self.report.get("validation_errors", [])
        message = "; ".join(str(error) for error in errors) or "Validation failed"
        super().__init__(message)


def blank_mask(series: pd.Series) -> pd.Series:
    """Return a mask for null, empty, or whitespace-only values."""

    return series.isna() | series.astype(str).str.strip().eq("")


def _trim_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = cast(pd.DataFrame, frame.copy())

    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column].dtype) or isinstance(
            cleaned[column].dtype,
            pd.StringDtype,
        ):
            cleaned[column] = (
                cleaned[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    return cleaned


def _invalid_values(
    series: pd.Series,
    allowed: Iterable[str],
) -> list[str]:
    allowed_values = set(allowed)
    return sorted(set(series.astype(str)) - allowed_values)


def _invalid_dates(series: pd.Series) -> list[str]:
    invalid: set[str] = set()

    for value in series.astype(str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            invalid.add(value)
            continue

        if parsed.isoformat() != value:
            invalid.add(value)

    return sorted(invalid)


def distribution(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    """Return a stable value-count mapping for a report column."""

    if column not in frame.columns:
        return {}

    counts = frame[column].value_counts(dropna=False).sort_index()

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def _duplicate_excess_count(series: pd.Series) -> int:
    return int(series.duplicated(keep="first").sum())


def base_report(
    feature_rows: int,
    label_rows: int,
) -> dict[str, Any]:
    """Create the initial machine-readable validation report."""

    return {
        "contract_version": CONTRACT_VERSION,
        "feature_join_key": FEATURE_JOIN_KEY,
        "manifest_join_key": MANIFEST_JOIN_KEY,
        "input_feature_rows": int(feature_rows),
        "input_label_rows": int(label_rows),
        "matched_rows": 0,
        "unmatched_feature_count": 0,
        "unmatched_feature_accessions": [],
        "unmatched_label_count": 0,
        "unmatched_label_accessions": [],
        "duplicate_key_count": 0,
        "duplicate_feature_key_count": 0,
        "duplicate_label_key_count": 0,
        "label_distribution": {},
        "evidence_tier_distribution": {},
        "dataset_role_distribution": {},
        "excluded_from_training_count": 0,
        "training_exclusion_reasons": {},
        "training_ready_rows": 0,
        "validation_errors": [],
        "validation_warnings": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(
    labels: pd.DataFrame,
    allowed_evidence_sources: frozenset[str],
    report: dict[str, Any],
) -> pd.DataFrame:
    """Validate and normalize the curated label manifest."""

    missing_columns = [
        column
        for column in MANIFEST_COLUMNS
        if column not in labels
    ]

    if missing_columns:
        report["validation_errors"].append(
            "Label manifest is missing required column(s): "
            + ", ".join(missing_columns)
        )
        return cast(pd.DataFrame, labels.copy())

    cleaned = _trim_string_columns(labels)

    for column in REQUIRED_VALUE_COLUMNS:
        missing_rows = cleaned.index[
            blank_mask(cleaned[column])
        ].tolist()

        if missing_rows:
            report["validation_errors"].append(
                f"Required manifest field '{column}' is blank at row(s): "
                + ", ".join(
                    str(index + 2)
                    for index in missing_rows
                )
            )

    accessions = cleaned[MANIFEST_JOIN_KEY].astype(str)

    invalid_accessions = sorted(
        value
        for value in set(accessions)
        if not ACCESSION_PATTERN.fullmatch(value)
    )

    if invalid_accessions:
        report["validation_errors"].append(
            "Manifest assembly_accession must be a versioned "
            "NCBI assembly accession; invalid value(s): "
            + ", ".join(invalid_accessions)
        )

    duplicate_count = _duplicate_excess_count(accessions)
    report["duplicate_label_key_count"] = duplicate_count

    if duplicate_count:
        duplicates = sorted(
            accessions[
                accessions.duplicated(keep=False)
            ].unique()
        )

        report["validation_errors"].append(
            "Duplicate assembly_accession value(s) in manifest: "
            + ", ".join(duplicates)
        )

    allowed_columns = (
        ("target_label", TARGET_LABELS),
        ("evidence_tier", EVIDENCE_TIERS),
        ("evidence_source", allowed_evidence_sources),
        ("dataset_role", DATASET_ROLES),
    )

    for column, allowed in allowed_columns:
        invalid = _invalid_values(cleaned[column], allowed)

        if invalid:
            report["validation_errors"].append(
                f"Invalid {column} value(s): {', '.join(invalid)}; "
                "allowed: "
                + ", ".join(sorted(allowed))
            )

    invalid_dates = _invalid_dates(cleaned["label_date"])

    if invalid_dates:
        report["validation_errors"].append(
            "label_date must use valid ISO YYYY-MM-DD values; "
            "invalid value(s): "
            + ", ".join(invalid_dates)
        )

    return cleaned


def validate_features(
    features: pd.DataFrame,
    report: dict[str, Any],
) -> pd.DataFrame:
    """Validate and normalize the feature-matrix join key."""

    if FEATURE_JOIN_KEY not in features:
        report["validation_errors"].append(
            f"Feature matrix is missing required join key "
            f"'{FEATURE_JOIN_KEY}'"
        )
        return cast(pd.DataFrame, features.copy())

    cleaned = cast(pd.DataFrame, features.copy())
    join_values = cleaned[FEATURE_JOIN_KEY]

    blank_rows = cleaned.index[
        blank_mask(join_values)
    ].tolist()

    if blank_rows:
        report["validation_errors"].append(
            f"Feature join key '{FEATURE_JOIN_KEY}' is blank "
            "at row(s): "
            + ", ".join(
                str(index + 2)
                for index in blank_rows
            )
        )

    cleaned[FEATURE_JOIN_KEY] = (
        join_values
        .fillna("")
        .astype(str)
        .str.strip()
    )

    accessions = cleaned[FEATURE_JOIN_KEY]

    invalid_accessions = sorted(
        value
        for value in set(accessions)
        if not ACCESSION_PATTERN.fullmatch(value)
    )

    if invalid_accessions:
        report["validation_errors"].append(
            f"Feature join key '{FEATURE_JOIN_KEY}' contains "
            "invalid or unversioned accession(s): "
            + ", ".join(invalid_accessions)
        )

    duplicate_count = _duplicate_excess_count(accessions)
    report["duplicate_feature_key_count"] = duplicate_count

    if duplicate_count:
        duplicates = sorted(
            accessions[
                accessions.duplicated(keep=False)
            ].unique()
        )

        report["validation_errors"].append(
            "Duplicate feature join key value(s) would permit "
            "a many-to-many join: "
            + ", ".join(duplicates)
        )

    return cleaned