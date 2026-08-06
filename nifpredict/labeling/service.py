"""Application service for joining labels and persisting labeled datasets."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    DEFAULT_EVIDENCE_SOURCES,
    EVIDENCE_TIERS,
    FEATURE_JOIN_KEY,
    MANIFEST_COLUMNS,
    MANIFEST_JOIN_KEY,
    LabelingResult,
)
from .validation import (
    LabelingValidationError,
    base_report,
    blank_mask,
    distribution,
    validate_features,
    validate_manifest,
)


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
        f"Unsupported table format '{suffix}' for {path}; "
        "use .csv or .parquet"
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write CSV or Parquet atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            frame.to_csv(temporary_path, index=False)
        elif suffix == ".parquet":
            frame.to_parquet(
                temporary_path,
                engine="pyarrow",
                index=False,
            )
        else:
            raise ValueError(
                f"Unsupported table format '{suffix}' for {path}; "
                "use .csv or .parquet"
            )

        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_report(
    report: Mapping[str, Any],
    path: Path,
) -> None:
    """Write a JSON report atomically with stable key ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(report),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")

        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_labeled_datasets(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    strict_mismatches: bool = False,
    training_evidence_tiers: Sequence[str] = ("A", "B"),
    additional_evidence_sources: Sequence[str] = (),
) -> LabelingResult:
    """Validate, join, and split full and training-ready datasets.

    Raises:
        LabelingValidationError: If schema, key, controlled-vocabulary,
            or strict mismatch validation fails.
    """
    report = base_report(
        feature_rows=len(features),
        label_rows=len(labels),
    )

    if features.empty:
        report["validation_errors"].append(
            "Feature matrix contains no rows"
        )

    if labels.empty:
        report["validation_errors"].append(
            "Label manifest contains no curated rows; populate the "
            "template before running the labeling pipeline"
        )

    training_tiers = frozenset(training_evidence_tiers)
    invalid_training_tiers = sorted(
        training_tiers - EVIDENCE_TIERS
    )

    if invalid_training_tiers:
        report["validation_errors"].append(
            "Invalid configured training evidence tier(s): "
            + ", ".join(invalid_training_tiers)
        )

    normalized_sources = frozenset(
        source.strip()
        for source in additional_evidence_sources
        if source.strip()
    )
    allowed_sources = (
        DEFAULT_EVIDENCE_SOURCES | normalized_sources
    )

    report["training_evidence_tiers"] = sorted(training_tiers)
    report["allowed_evidence_sources"] = sorted(allowed_sources)

    cleaned_labels = validate_manifest(
        labels,
        allowed_sources,
        report,
    )
    cleaned_features = validate_features(features, report)

    report["duplicate_key_count"] = int(
        report["duplicate_feature_key_count"]
        + report["duplicate_label_key_count"]
    )

    if all(
        column in cleaned_labels
        for column in MANIFEST_COLUMNS
    ):
        report["label_distribution"] = distribution(
            cleaned_labels,
            "target_label",
        )
        report["evidence_tier_distribution"] = distribution(
            cleaned_labels,
            "evidence_tier",
        )
        report["dataset_role_distribution"] = distribution(
            cleaned_labels,
            "dataset_role",
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

    mismatch_messages: list[str] = []

    if unmatched_features:
        mismatch_messages.append(
            f"{len(unmatched_features)} feature accession(s) "
            "have no manifest label"
        )

    if unmatched_labels:
        mismatch_messages.append(
            f"{len(unmatched_labels)} manifest accession(s) "
            "have no feature row"
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

    full_dataset = (
        full_dataset
        .sort_values(
            MANIFEST_JOIN_KEY,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    report["matched_rows"] = len(full_dataset)

    label_ok = full_dataset["target_label"].isin(
        {"positive", "negative"}
    )
    evidence_ok = full_dataset["evidence_tier"].isin(
        training_tiers
    )
    role_ok = full_dataset["dataset_role"].eq(
        "train_candidate"
    )
    reference_ok = ~blank_mask(
        full_dataset["evidence_reference"]
    )
    source_ok = ~blank_mask(
        full_dataset["source_genome_id"]
    )

    training_mask = (
        label_ok
        & evidence_ok
        & role_ok
        & reference_ok
        & source_ok
    )

    candidate_without_reference = (
        label_ok
        & evidence_ok
        & role_ok
        & ~reference_ok
    )

    if candidate_without_reference.any():
        blocked = sorted(
            full_dataset.loc[
                candidate_without_reference,
                MANIFEST_JOIN_KEY,
            ].astype(str)
        )

        report["validation_warnings"].append(
            "Training-eligible accession(s) were excluded because "
            "evidence_reference is blank: "
            + ", ".join(blocked)
        )

    exclusion_reasons = {
        "target_label_not_binary": int((~label_ok).sum()),
        "evidence_tier_not_allowed": int((~evidence_ok).sum()),
        "dataset_role_not_train_candidate": int((~role_ok).sum()),
        "missing_evidence_reference": int((~reference_ok).sum()),
        "missing_source_genome_id": int((~source_ok).sum()),
    }

    report["training_exclusion_reasons"] = exclusion_reasons
    report["excluded_from_training_count"] = int(
        (~training_mask).sum()
    )

    training_dataset = (
        full_dataset.loc[training_mask]
        .copy()
        .reset_index(drop=True)
    )

    report["training_ready_rows"] = len(training_dataset)

    return LabelingResult(
        full_dataset=full_dataset,
        training_dataset=training_dataset,
        report=report,
    )


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
    """Read, validate, label, and persist all pipeline outputs."""
    features = read_table(feature_path)
    labels = read_table(label_path)

    try:
        result = build_labeled_datasets(
            features,
            labels,
            strict_mismatches=strict_mismatches,
            training_evidence_tiers=training_evidence_tiers,
            additional_evidence_sources=(
                additional_evidence_sources
            ),
        )
    except LabelingValidationError as exc:
        write_report(exc.report, report_path)
        raise

    write_table(result.full_dataset, output_path)
    write_table(
        result.training_dataset,
        training_output_path,
    )
    write_report(result.report, report_path)

    return result