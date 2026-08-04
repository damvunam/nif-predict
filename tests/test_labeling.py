"""Offline tests for the independent-manifest labeling pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from nifpredict.data.labeling import (
    MANIFEST_COLUMNS,
    LabelingValidationError,
    build_labeled_datasets,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "feature_matrix.csv")


@pytest.fixture
def labels() -> pd.DataFrame:
    return pd.read_csv(
        FIXTURES / "label_manifest.csv", dtype=str, keep_default_na=False
    )


def _assert_error(
    features: pd.DataFrame, labels: pd.DataFrame, expected: str, **kwargs: object
) -> dict[str, object]:
    with pytest.raises(LabelingValidationError) as captured:
        build_labeled_datasets(features, labels, **kwargs)
    report = captured.value.report
    assert any(expected in message for message in report["validation_errors"])
    return report


def test_valid_join_and_stable_accession_order(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    result = build_labeled_datasets(features, labels)

    assert result.full_dataset["assembly_accession"].tolist() == [
        "GCF_000000001.1",
        "GCF_000000002.1",
        "GCF_000000003.1",
    ]
    assert result.report["matched_rows"] == 3
    assert result.report["validation_errors"] == []


def test_missing_required_column_is_fatal(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    _assert_error(
        features, labels.drop(columns="label_curator"), "missing required column"
    )


def test_duplicate_manifest_accession_is_fatal(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    duplicated = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    report = _assert_error(features, duplicated, "Duplicate assembly_accession")
    assert report["duplicate_label_key_count"] == 1


def test_unversioned_manifest_accession_is_fatal(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    invalid = labels.copy()
    invalid.loc[0, "assembly_accession"] = "GCF_000000001"
    _assert_error(features, invalid, "versioned NCBI assembly accession")


def test_invalid_target_label_is_fatal(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    invalid = labels.copy()
    invalid.loc[0, "target_label"] = "yes"
    _assert_error(features, invalid, "Invalid target_label")


def test_invalid_evidence_tier_is_fatal(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    invalid = labels.copy()
    invalid.loc[0, "evidence_tier"] = "D"
    _assert_error(features, invalid, "Invalid evidence_tier")


def test_invalid_evidence_source_is_fatal(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    invalid = labels.copy()
    invalid.loc[0, "evidence_source"] = "guess"
    _assert_error(features, invalid, "Invalid evidence_source")


def test_uncertain_stays_full_but_is_not_training_ready(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    result = build_labeled_datasets(features, labels)
    assert "uncertain" in set(result.full_dataset["target_label"])
    assert "uncertain" not in set(result.training_dataset["target_label"])


def test_edge_case_stays_full_but_is_not_training_ready(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    result = build_labeled_datasets(features, labels)
    edge_accession = "GCF_000000003.1"
    assert edge_accession in set(result.full_dataset["assembly_accession"])
    assert edge_accession not in set(result.training_dataset["assembly_accession"])


def test_label_without_feature_is_reported(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    extra = labels.iloc[[0]].copy()
    extra.loc[:, "assembly_accession"] = "GCF_000000004.1"
    extra.loc[:, "source_genome_id"] = "GCF_000000004.1"
    result = build_labeled_datasets(features, pd.concat([labels, extra]))

    assert result.report["unmatched_label_accessions"] == ["GCF_000000004.1"]
    assert result.report["unmatched_label_count"] == 1


def test_feature_without_label_is_reported(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    extra = pd.DataFrame([{"accession_id": "GCF_000000004.1", "status": "SUCCESS"}])
    result = build_labeled_datasets(pd.concat([features, extra]), labels)

    assert result.report["unmatched_feature_accessions"] == ["GCF_000000004.1"]
    assert result.report["unmatched_feature_count"] == 1


def test_strict_mismatch_mode_is_fatal(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    _assert_error(
        features.iloc[:-1],
        labels,
        "manifest accession(s) have no feature row",
        strict_mismatches=True,
    )


def test_duplicate_feature_key_blocks_many_to_many_join(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    report = _assert_error(features=duplicated, labels=labels, expected="many-to-many")
    assert report["duplicate_feature_key_count"] == 1


def test_derivatives_preserve_shared_source_genome_id(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    derivative_labels = labels.iloc[:2].copy()
    derivative_labels.loc[:, "source_genome_id"] = "source-genome-fixture-1"
    derivative_features = features[
        features["accession_id"].isin(derivative_labels["assembly_accession"])
    ]
    result = build_labeled_datasets(derivative_features, derivative_labels)

    assert result.full_dataset["source_genome_id"].tolist() == [
        "source-genome-fixture-1",
        "source-genome-fixture-1",
    ]


def test_training_candidate_without_reference_is_reported_and_excluded(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    missing_reference = labels.iloc[[0]].copy()
    missing_reference.loc[:, "evidence_reference"] = ""
    result = build_labeled_datasets(features.iloc[[1]], missing_reference)

    assert result.training_dataset.empty
    assert (
        result.report["training_exclusion_reasons"]["missing_evidence_reference"] == 1
    )
    assert any(
        "evidence_reference is blank" in warning
        for warning in result.report["validation_warnings"]
    )


def test_cli_returns_nonzero_and_writes_report_on_fatal_validation(
    tmp_path: Path, features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    feature_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    output_path = tmp_path / "full.csv"
    training_path = tmp_path / "training.csv"
    report_path = tmp_path / "report.json"
    features.to_csv(feature_path, index=False)
    invalid = labels.copy()
    invalid.loc[0, "target_label"] = "feature_inferred"
    invalid.to_csv(label_path, index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "label_dataset.py"),
            "--features",
            str(feature_path),
            "--labels",
            str(label_path),
            "--output",
            str(output_path),
            "--training-output",
            str(training_path),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert report_path.is_file()
    assert json.loads(report_path.read_text())["validation_errors"]
    assert not output_path.exists()


def test_outputs_are_deterministic(
    features: pd.DataFrame, labels: pd.DataFrame
) -> None:
    first = build_labeled_datasets(features, labels)
    second = build_labeled_datasets(
        features.sample(frac=1, random_state=42),
        labels.sample(frac=1, random_state=24),
    )

    assert first.full_dataset.to_csv(index=False) == second.full_dataset.to_csv(
        index=False
    )
    assert first.training_dataset.to_csv(index=False) == second.training_dataset.to_csv(
        index=False
    )


def test_manifest_template_header_matches_contract() -> None:
    template = pd.read_csv(ROOT / "data" / "labels" / "label_manifest.csv")
    assert tuple(template.columns) == MANIFEST_COLUMNS
    assert template.empty


def test_empty_manifest_cannot_create_research_outputs(
    features: pd.DataFrame,
) -> None:
    template = pd.read_csv(ROOT / "data" / "labels" / "label_manifest.csv")
    _assert_error(features, template, "contains no curated rows")