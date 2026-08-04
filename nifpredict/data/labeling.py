"""Manifest-backed labeling for NifPredict feature matrices.

Labels in this module are always supplied by an independently curated manifest.
No biological feature is inspected or used to derive a target label.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

CONTRACT_VERSION = "1.0.0"
FEATURE_JOIN_KEY = "accession_id"
MANIFEST_JOIN_KEY = "assembly_accession"
ACCESSION_PATTERN = re.compile(r"^GC[AF]_\d{9}\.\d+$")

MANIFEST_COLUMNS = (
    "assembly_accession",
    "organism_name",
    "strain",
    "target_label",
    "evidence_tier",
    "evidence_source",
    "evidence_reference",
    "taxonomy_group",
    "source_genome_id",
    "dataset_role",
    "label_curator",
    "label_date",
    "notes",
)
REQUIRED_VALUE_COLUMNS = (
    "assembly_accession",
    "organism_name",
    "target_label",
    "evidence_tier",
    "evidence_source",
    "source_genome_id",
    "dataset_role",
    "label_curator",
    "label_date",
)
TARGET_LABELS = frozenset({"positive", "negative", "uncertain"})
EVIDENCE_TIERS = frozenset({"A", "B", "C"})
DEFAULT_EVIDENCE_SOURCES = frozenset(
    {"literature", "curated_database", "experimental_record", "gene_inference"}
)
DATASET_ROLES = frozenset(
    {"train_candidate", "external_test", "acceptance_test", "edge_case"}
)
DEFAULT_TRAINING_EVIDENCE_TIERS = frozenset({"A", "B"})


@dataclass(frozen=True)
class LabelingResult:
    """Validated outputs and their machine-readable validation report."""

    full_dataset: pd.DataFrame
    training_dataset: pd.DataFrame
    report: dict[str, Any]


class LabelingValidationError(ValueError):
    """Raised when a dataset violates a fatal labeling contract rule."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        errors = self.report.get("validation_errors", [])
        message = "; ".join(str(error) for error in errors) or "Validation failed"
        super().__init__(message)


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported table format without making path/CWD assumptions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input table not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported table format '{suffix}' for {path}; use .csv or .parquet"
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write CSV or Parquet atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame.to_csv(temporary_path, index=False)
        elif suffix == ".parquet":
            frame.to_parquet(temporary_path, engine="pyarrow", index=False)
        else:
            raise ValueError(
                f"Unsupported table format '{suffix}' for {path}; use .csv or .parquet"
            )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Write a JSON report atomically with stable key ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(report), handle, indent=2, sort_keys=True, ensure_ascii=False
            )
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _trim_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = cast(pd.DataFrame, frame.copy())
    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column].dtype) or isinstance(
            cleaned[column].dtype, pd.StringDtype
        ):
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    return cleaned


def _invalid_values(series: pd.Series, allowed: Iterable[str]) -> list[str]:
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


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _duplicate_excess_count(series: pd.Series) -> int:
    return int(series.duplicated(keep="first").sum())


def _base_report(feature_rows: int, label_rows: int) -> dict[str, Any]:
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


def _validate_manifest(
    labels: pd.DataFrame,
    allowed_evidence_sources: frozenset[str],
    report: dict[str, Any],
) -> pd.DataFrame:
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in labels]
    if missing_columns:
        report["validation_errors"].append(
            "Label manifest is missing required column(s): "
            + ", ".join(missing_columns)
        )
        return cast(pd.DataFrame, labels.copy())

    cleaned = _trim_string_columns(labels)
    for column in REQUIRED_VALUE_COLUMNS:
        missing_rows = cleaned.index[_blank_mask(cleaned[column])].tolist()
        if missing_rows:
            report["validation_errors"].append(
                f"Required manifest field '{column}' is blank at row(s): "
                + ", ".join(str(index + 2) for index in missing_rows)
            )

    accessions = cleaned[MANIFEST_JOIN_KEY].astype(str)
    invalid_accessions = sorted(
        value for value in set(accessions) if not ACCESSION_PATTERN.fullmatch(value)
    )
    if invalid_accessions:
        report["validation_errors"].append(
            "Manifest assembly_accession must be a versioned NCBI assembly accession; "
            "invalid value(s): " + ", ".join(invalid_accessions)
        )

    duplicate_count = _duplicate_excess_count(accessions)
    report["duplicate_label_key_count"] = duplicate_count
    if duplicate_count:
        duplicates = sorted(accessions[accessions.duplicated(keep=False)].unique())
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
                f"Invalid {column} value(s): {', '.join(invalid)}; allowed: "
                + ", ".join(sorted(allowed))
            )

    invalid_dates = _invalid_dates(cleaned["label_date"])
    if invalid_dates:
        report["validation_errors"].append(
            "label_date must use valid ISO YYYY-MM-DD values; invalid value(s): "
            + ", ".join(invalid_dates)
        )

    return cleaned


def _validate_features(features: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    if FEATURE_JOIN_KEY not in features:
        report["validation_errors"].append(
            f"Feature matrix is missing required join key '{FEATURE_JOIN_KEY}'"
        )
        return cast(pd.DataFrame, features.copy())

    cleaned = cast(pd.DataFrame, features.copy())
    join_values = cleaned[FEATURE_JOIN_KEY]
    blank_rows = cleaned.index[_blank_mask(join_values)].tolist()
    if blank_rows:
        report["validation_errors"].append(
            f"Feature join key '{FEATURE_JOIN_KEY}' is blank at row(s): "
            + ", ".join(str(index + 2) for index in blank_rows)
        )

    cleaned[FEATURE_JOIN_KEY] = join_values.fillna("").astype(str).str.strip()
    accessions = cleaned[FEATURE_JOIN_KEY]
    invalid_accessions = sorted(
        value for value in set(accessions) if not ACCESSION_PATTERN.fullmatch(value)
    )
    if invalid_accessions:
        report["validation_errors"].append(
            f"Feature join key '{FEATURE_JOIN_KEY}' contains invalid or unversioned "
            "accession(s): " + ", ".join(invalid_accessions)
        )

    duplicate_count = _duplicate_excess_count(accessions)
    report["duplicate_feature_key_count"] = duplicate_count
    if duplicate_count:
        duplicates = sorted(accessions[accessions.duplicated(keep=False)].unique())
        report["validation_errors"].append(
            "Duplicate feature join key value(s) would permit a many-to-many join: "
            + ", ".join(duplicates)
        )
    return cleaned


def build_labeled_datasets(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    strict_mismatches: bool = False,
    training_evidence_tiers: Sequence[str] = ("A", "B"),
    additional_evidence_sources: Sequence[str] = (),
) -> LabelingResult:
    """Validate, one-to-one join, and split full/training-ready datasets.

    Raises:
        LabelingValidationError: if schema, key, controlled-vocabulary, or strict
            mismatch validation fails.
    """
    report = _base_report(len(features), len(labels))
    if features.empty:
        report["validation_errors"].append("Feature matrix contains no rows")
    if labels.empty:
        report["validation_errors"].append(
            "Label manifest contains no curated rows; populate the template before "
            "running the labeling pipeline"
        )
    training_tiers = frozenset(training_evidence_tiers)
    invalid_training_tiers = sorted(training_tiers - EVIDENCE_TIERS)
    if invalid_training_tiers:
        report["validation_errors"].append(
            "Invalid configured training evidence tier(s): "
            + ", ".join(invalid_training_tiers)
        )

    normalized_sources = frozenset(
        source.strip() for source in additional_evidence_sources if source.strip()
    )
    allowed_sources = DEFAULT_EVIDENCE_SOURCES | normalized_sources
    report["training_evidence_tiers"] = sorted(training_tiers)
    report["allowed_evidence_sources"] = sorted(allowed_sources)

    cleaned_labels = _validate_manifest(labels, allowed_sources, report)
    cleaned_features = _validate_features(features, report)
    report["duplicate_key_count"] = int(
        report["duplicate_feature_key_count"] + report["duplicate_label_key_count"]
    )

    if all(column in cleaned_labels for column in MANIFEST_COLUMNS):
        report["label_distribution"] = _distribution(cleaned_labels, "target_label")
        report["evidence_tier_distribution"] = _distribution(
            cleaned_labels, "evidence_tier"
        )
        report["dataset_role_distribution"] = _distribution(
            cleaned_labels, "dataset_role"
        )

    if report["validation_errors"]:
        raise LabelingValidationError(report)

    feature_keys = set(cleaned_features[FEATURE_JOIN_KEY])
    label_keys = set(cleaned_labels[MANIFEST_JOIN_KEY])
    unmatched_features = sorted(feature_keys - label_keys)
    unmatched_labels = sorted(label_keys - feature_keys)
    report["unmatched_feature_accessions"] = unmatched_features
    report["unmatched_feature_count"] = len(unmatched_features)
    report["unmatched_label_accessions"] = unmatched_labels
    report["unmatched_label_count"] = len(unmatched_labels)

    mismatch_messages = []
    if unmatched_features:
        mismatch_messages.append(
            f"{len(unmatched_features)} feature accession(s) have no manifest label"
        )
    if unmatched_labels:
        mismatch_messages.append(
            f"{len(unmatched_labels)} manifest accession(s) have no feature row"
        )
    if strict_mismatches:
        report["validation_errors"].extend(mismatch_messages)
    else:
        report["validation_warnings"].extend(mismatch_messages)
    if report["validation_errors"]:
        raise LabelingValidationError(report)

    full_dataset = cleaned_features.merge(
        cleaned_labels,
        how="inner",
        left_on=FEATURE_JOIN_KEY,
        right_on=MANIFEST_JOIN_KEY,
        validate="one_to_one",
        suffixes=("_feature", ""),
        sort=False,
    )
    full_dataset = full_dataset.sort_values(
        MANIFEST_JOIN_KEY, kind="mergesort"
    ).reset_index(drop=True)
    report["matched_rows"] = len(full_dataset)

    label_ok = full_dataset["target_label"].isin({"positive", "negative"})
    evidence_ok = full_dataset["evidence_tier"].isin(training_tiers)
    role_ok = full_dataset["dataset_role"].eq("train_candidate")
    reference_ok = ~_blank_mask(full_dataset["evidence_reference"])
    source_ok = ~_blank_mask(full_dataset["source_genome_id"])
    training_mask = label_ok & evidence_ok & role_ok & reference_ok & source_ok

    candidate_without_reference = label_ok & evidence_ok & role_ok & ~reference_ok
    if candidate_without_reference.any():
        blocked = sorted(
            full_dataset.loc[candidate_without_reference, MANIFEST_JOIN_KEY].astype(str)
        )
        report["validation_warnings"].append(
            "Training-eligible accession(s) were excluded because "
            "evidence_reference is blank: " + ", ".join(blocked)
        )

    exclusion_reasons = {
        "target_label_not_binary": int((~label_ok).sum()),
        "evidence_tier_not_allowed": int((~evidence_ok).sum()),
        "dataset_role_not_train_candidate": int((~role_ok).sum()),
        "missing_evidence_reference": int((~reference_ok).sum()),
        "missing_source_genome_id": int((~source_ok).sum()),
    }
    report["training_exclusion_reasons"] = exclusion_reasons
    report["excluded_from_training_count"] = int((~training_mask).sum())

    training_dataset = full_dataset.loc[training_mask].copy().reset_index(drop=True)
    report["training_ready_rows"] = len(training_dataset)
    return LabelingResult(full_dataset, training_dataset, report)


def run_labeling_pipeline(
    feature_path: Path,
    label_path: Path,
    output_path: Path,
    training_output_path: Path,
    report_path: Path,
    *,
    strict_mismatches: bool = False,
    training_evidence_tiers: Sequence[str] = ("A", "B"),
    additional_evidence_sources: Sequence[str] = (),
) -> LabelingResult:
    """Read, validate, label, and atomically persist all pipeline outputs."""
    features = read_table(feature_path)
    labels = read_table(label_path)
    try:
        result = build_labeled_datasets(
            features,
            labels,
            strict_mismatches=strict_mismatches,
            training_evidence_tiers=training_evidence_tiers,
            additional_evidence_sources=additional_evidence_sources,
        )
    except LabelingValidationError as exc:
        write_report(exc.report, report_path)
        raise

    write_table(result.full_dataset, output_path)
    write_table(result.training_dataset, training_output_path)
    write_report(result.report, report_path)
    return result