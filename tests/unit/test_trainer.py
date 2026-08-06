"""Unit tests for the model training engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from nifpredict.models import (
    LABEL_MAPPING,
    TrainingDataError,
    build_classifier,
    prepare_training_data,
    train_classifier,
)


@pytest.fixture
def training_frame() -> pd.DataFrame:
    """Create a small balanced training dataset."""

    return pd.DataFrame(
        {
            "accession_id": [
                "GCF_000000001.1",
                "GCF_000000002.1",
                "GCF_000000003.1",
                "GCF_000000004.1",
            ],
            "status": ["success"] * 4,
            "assembly_accession": [
                "GCF_000000001.1",
                "GCF_000000002.1",
                "GCF_000000003.1",
                "GCF_000000004.1",
            ],
            "organism_name": [
                "Organism A",
                "Organism B",
                "Organism C",
                "Organism D",
            ],
            "evidence_tier": ["A", "A", "B", "B"],
            "dataset_role": ["train_candidate"] * 4,
            "feature_a": [0.1, 0.2, 1.1, 1.2],
            "feature_b": [0, 1, 1, 2],
            "target_label": [
                "negative",
                "negative",
                "positive",
                "positive",
            ],
        }
    )


def test_prepare_training_data_excludes_metadata_and_maps_labels(
    training_frame: pd.DataFrame,
) -> None:
    prepared = prepare_training_data(training_frame)

    assert prepared.feature_names == ("feature_a", "feature_b")
    assert prepared.features.columns.tolist() == [
        "feature_a",
        "feature_b",
    ]
    assert prepared.target.tolist() == [0, 0, 1, 1]
    assert prepared.class_distribution == {
        "negative": 2,
        "positive": 2,
    }


def test_prepare_training_data_does_not_mutate_input(
    training_frame: pd.DataFrame,
) -> None:
    original = training_frame.copy(deep=True)

    prepare_training_data(training_frame)

    pd.testing.assert_frame_equal(training_frame, original)


def test_missing_target_column_is_fatal(
    training_frame: pd.DataFrame,
) -> None:
    invalid_frame = training_frame.drop(columns=["target_label"])

    with pytest.raises(TrainingDataError, match="target_label"):
        prepare_training_data(invalid_frame)


def test_non_binary_target_label_is_fatal(
    training_frame: pd.DataFrame,
) -> None:
    invalid_frame = training_frame.copy()
    invalid_frame.loc[0, "target_label"] = "uncertain"

    with pytest.raises(TrainingDataError, match="target_label"):
        prepare_training_data(invalid_frame)


def test_training_requires_both_target_classes(
    training_frame: pd.DataFrame,
) -> None:
    invalid_frame = training_frame.copy()
    invalid_frame["target_label"] = "positive"

    with pytest.raises(TrainingDataError, match="both"):
        prepare_training_data(invalid_frame)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_feature_value_is_fatal(
    training_frame: pd.DataFrame,
    invalid_value: float,
) -> None:
    invalid_frame = training_frame.copy()
    invalid_frame.loc[0, "feature_a"] = invalid_value

    with pytest.raises(TrainingDataError, match="finite"):
        prepare_training_data(invalid_frame)


def test_unexpected_non_numeric_feature_is_fatal(
    training_frame: pd.DataFrame,
) -> None:
    invalid_frame = training_frame.copy()
    invalid_frame["unexpected_text"] = ["a", "b", "c", "d"]

    with pytest.raises(TrainingDataError, match="non-numeric"):
        prepare_training_data(invalid_frame)


@pytest.mark.parametrize(
    ("model_name", "classifier_type", "uses_pipeline"),
    [
        ("logistic_regression", LogisticRegression, True),
        ("random_forest", RandomForestClassifier, False),
        ("svm", SVC, True),
    ],
)
def test_build_classifier_returns_expected_estimator(
    model_name: str,
    classifier_type: type,
    uses_pipeline: bool,
) -> None:
    estimator = build_classifier(
        model_name,
        random_seed=42,
        imbalance_handling="ClassWeight",
    )

    if uses_pipeline:
        assert isinstance(estimator, Pipeline)
        assert "scaler" in estimator.named_steps
        classifier = estimator.named_steps["classifier"]
    else:
        classifier = estimator

    assert isinstance(classifier, classifier_type)
    assert classifier.class_weight == "balanced"


def test_build_classifier_supports_no_class_weight() -> None:
    estimator = build_classifier(
        "random_forest",
        imbalance_handling="None",
    )

    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.class_weight is None


def test_smote_is_rejected_by_training_engine() -> None:
    with pytest.raises(ValueError, match="SMOTE"):
        build_classifier(
            "random_forest",
            imbalance_handling="SMOTE",
        )


def test_unknown_model_name_is_fatal() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        build_classifier("unknown_model")


def test_train_classifier_returns_fitted_result(
    training_frame: pd.DataFrame,
) -> None:
    result = train_classifier(
        training_frame,
        model_name="logistic_regression",
        random_seed=42,
    )

    predictions = result.estimator.predict(
        training_frame[["feature_a", "feature_b"]]
    )

    assert result.model_name == "logistic_regression"
    assert result.feature_names == ("feature_a", "feature_b")
    assert result.label_mapping == LABEL_MAPPING
    assert result.class_distribution == {
        "negative": 2,
        "positive": 2,
    }
    assert len(predictions) == len(training_frame)