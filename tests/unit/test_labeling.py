"""Unit tests for the independent-manifest labeling service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import nifpredict.data.labeling as legacy_labeling
import nifpredict.labeling as new_labeling
from nifpredict.labeling import (
    MANIFEST_COLUMNS,
    LabelingValidationError,
    build_labeled_datasets,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST_TEMPLATE = ROOT / "data" / "labels" / "label_manifest.csv"


@pytest.fixture
def features() -> pd.DataFrame:
    """Load the small offline feature fixture."""

    return pd.read_csv(
        FIXTURES / "feature_matrix.csv",
        dtype=str,
        keep_default_na=False,
    )


@pytest.fixture
def labels() -> pd.DataFrame:
    """Load the small offline curated-label fixture."""

    return pd.read_csv(
        FIXTURES / "label_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )


def _assert_error(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    expected: str,
    **kwargs: object,
) -> dict[str, object]:
    """Run labeling and assert that a fatal validation error is reported."""

    with pytest.raises(LabelingValidationError) as captured:
        build_labeled_datasets(
            features,
            labels,
            **kwargs,
        )

    report = captured.value.report

    assert any(
        expected in message
        for message in report["validation_errors"]
    )

    return report


def test_new_and_legacy_import_paths_export_same_api() -> None:
    """The old import path must remain a compatibility shim."""

    assert (
        new_labeling.build_labeled_datasets
        is legacy_labeling.build_labeled_datasets
    )
    assert (
        new_labeling.run_labeling_pipeline
        is legacy_labeling.run_labeling_pipeline
    )
    assert (
        new_labeling.LabelingValidationError
        is legacy_labeling.LabelingValidationError
    )


def test_valid_join_is_stable(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    result = build_labeled_datasets(features, labels)

    assert result.full_dataset["assembly_accession"].tolist() == [
        "GCF_000000001.1",
        "GCF_000000002.1",
        "GCF_000000003.1",
    ]
    assert result.report["matched_rows"] == 3
    assert result.report["validation_errors"] == []


def test_missing_required_manifest_column_is_fatal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    invalid_labels = labels.drop(columns=["target_label"])

    _assert_error(
        features,
        invalid_labels,
        "missing required column",
    )


def test_duplicate_manifest_accession_is_fatal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    invalid_labels = pd.concat(
        [labels, labels.iloc[[0]]],
        ignore_index=True,
    )

    report = _assert_error(
        features,
        invalid_labels,
        "Duplicate assembly_accession",
    )

    assert report["duplicate_label_key_count"] == 1
    assert report["duplicate_key_count"] == 1


def test_unversioned_manifest_accession_is_fatal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    invalid_labels = labels.copy()
    invalid_labels.loc[0, "assembly_accession"] = "GCF_000000001"

    _assert_error(
        features,
        invalid_labels,
        "versioned NCBI assembly accession",
    )


@pytest.mark.parametrize(
    ("column", "invalid_value", "expected"),
    [
        ("target_label", "yes", "Invalid target_label"),
        ("evidence_tier", "D", "Invalid evidence_tier"),
        (
            "evidence_source",
            "unsupported_source",
            "Invalid evidence_source",
        ),
    ],
)
def test_invalid_controlled_vocabulary_is_fatal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    column: str,
    invalid_value: str,
    expected: str,
) -> None:
    invalid_labels = labels.copy()
    invalid_labels.loc[0, column] = invalid_value

    _assert_error(
        features,
        invalid_labels,
        expected,
    )


def test_uncertain_and_edge_case_rows_are_not_training_ready(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    modified_labels = labels.copy()
    uncertain_accession = modified_labels.loc[0, "assembly_accession"]
    edge_case_accession = modified_labels.loc[1, "assembly_accession"]

    modified_labels.loc[0, "target_label"] = "uncertain"
    modified_labels.loc[1, "dataset_role"] = "edge_case"

    result = build_labeled_datasets(
        features,
        modified_labels,
    )

    training_accessions = set(
        result.training_dataset["assembly_accession"]
    )

    assert uncertain_accession not in training_accessions
    assert edge_case_accession not in training_accessions
    assert (
        result.report["training_exclusion_reasons"][
            "target_label_not_binary"
        ]
        >= 1
    )
    assert (
        result.report["training_exclusion_reasons"][
            "dataset_role_not_train_candidate"
        ]
        >= 1
    )


def test_mismatches_are_warnings_by_default(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    modified_labels = labels.copy()
    modified_labels.loc[
        0,
        "assembly_accession",
    ] = "GCF_999999999.1"

    result = build_labeled_datasets(
        features,
        modified_labels,
    )

    assert result.report["unmatched_feature_count"] == 1
    assert result.report["unmatched_label_count"] == 1
    assert len(result.report["validation_warnings"]) == 2
    assert result.report["validation_errors"] == []


def test_strict_mismatches_are_fatal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    modified_labels = labels.copy()
    modified_labels.loc[
        0,
        "assembly_accession",
    ] = "GCF_999999999.1"

    report = _assert_error(
        features,
        modified_labels,
        "feature accession(s) have no manifest label",
        strict_mismatches=True,
    )

    assert report["unmatched_feature_count"] == 1
    assert report["unmatched_label_count"] == 1


def test_duplicate_feature_key_is_fatal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    invalid_features = pd.concat(
        [features, features.iloc[[0]]],
        ignore_index=True,
    )

    report = _assert_error(
        invalid_features,
        labels,
        "many-to-many join",
    )

    assert report["duplicate_feature_key_count"] == 1
    assert report["duplicate_key_count"] == 1


def test_source_genome_id_may_be_shared(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    modified_labels = labels.copy()
    modified_labels.loc[
        [0, 1],
        "source_genome_id",
    ] = "source-genome-fixture-1"

    result = build_labeled_datasets(
        features,
        modified_labels,
    )

    assert result.full_dataset.loc[
        result.full_dataset["assembly_accession"].isin(
            [
                "GCF_000000001.1",
                "GCF_000000002.1",
            ]
        ),
        "source_genome_id",
    ].tolist() == [
        "source-genome-fixture-1",
        "source-genome-fixture-1",
    ]


def test_missing_evidence_reference_excludes_training_row(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    one_label = labels.iloc[[0]].copy()
    accession = one_label.iloc[0]["assembly_accession"]

    one_feature = features.loc[
        features["accession_id"].eq(accession)
    ].copy()

    one_label.loc[:, "target_label"] = "positive"
    one_label.loc[:, "evidence_tier"] = "A"
    one_label.loc[:, "dataset_role"] = "train_candidate"
    one_label.loc[:, "evidence_reference"] = ""

    result = build_labeled_datasets(
        one_feature,
        one_label,
    )

    assert result.training_dataset.empty
    assert result.report["excluded_from_training_count"] == 1
    assert any(
        "evidence_reference is blank" in warning
        for warning in result.report["validation_warnings"]
    )


def test_outputs_are_deterministic(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    first = build_labeled_datasets(features, labels)

    shuffled_features = features.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    shuffled_labels = labels.sample(
        frac=1,
        random_state=24,
    ).reset_index(drop=True)

    second = build_labeled_datasets(
        shuffled_features,
        shuffled_labels,
    )

    assert (
        first.full_dataset.to_csv(index=False)
        == second.full_dataset.to_csv(index=False)
    )
    assert (
        first.training_dataset.to_csv(index=False)
        == second.training_dataset.to_csv(index=False)
    )


def test_repository_manifest_template_is_header_only() -> None:
    template = pd.read_csv(
        MANIFEST_TEMPLATE,
        dtype=str,
        keep_default_na=False,
    )

    assert tuple(template.columns) == MANIFEST_COLUMNS
    assert template.empty


def test_empty_manifest_is_rejected(
    features: pd.DataFrame,
) -> None:
    template = pd.read_csv(
        MANIFEST_TEMPLATE,
        dtype=str,
        keep_default_na=False,
    )

    _assert_error(
        features,
        template,
        "contains no curated rows",
    )