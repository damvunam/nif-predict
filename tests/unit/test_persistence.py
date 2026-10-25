"""Unit tests for versioned model artifact persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nifpredict.ml.models import train_classifier
from nifpredict.ml.models.persistence import (
    MODEL_ARTIFACT_VERSION,
    ModelArtifactError,
    load_model_artifact,
    save_model_artifact,
    verify_prediction_round_trip,
)


@pytest.fixture
def training_frame() -> pd.DataFrame:
    """Return a small deterministic binary training dataset."""
    return pd.DataFrame(
        {
            "assembly_accession": [
                "GCF_000001.1",
                "GCF_000002.1",
                "GCF_000003.1",
                "GCF_000004.1",
            ],
            "target_label": [
                "negative",
                "negative",
                "positive",
                "positive",
            ],
            "feature_a": [0.0, 0.2, 0.8, 1.0],
            "feature_b": [1.0, 0.0, 1.0, 0.0],
        }
    )


def _save_artifact(
    training_frame: pd.DataFrame,
    path: Path,
):
    result = train_classifier(
        training_frame,
        model_name="logistic_regression",
        random_seed=7,
    )
    save_model_artifact(
        result,
        path,
        target_column="target_label",
        positive_class="positive",
        dataset_fingerprint="dataset-sha256",
        feature_fingerprint="features-sha256",
        hyperparameters={"random_seed": 7},
    )
    return result


def test_save_load_and_prediction_round_trip(
    training_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.joblib"
    result = _save_artifact(training_frame, path)

    loaded = load_model_artifact(path)
    verify_prediction_round_trip(
        result,
        loaded,
        training_frame,
    )

    assert loaded.artifact_version == MODEL_ARTIFACT_VERSION
    assert loaded.model_name == "logistic_regression"
    assert loaded.feature_names == ("feature_a", "feature_b")
    assert loaded.target_column == "target_label"
    assert loaded.positive_class == "positive"
    assert loaded.dataset_fingerprint == "dataset-sha256"
    assert loaded.feature_fingerprint == "features-sha256"
    assert loaded.hyperparameters == {"random_seed": 7}


def test_existing_artifact_is_not_overwritten_by_default(
    training_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.joblib"
    result = _save_artifact(training_frame, path)

    with pytest.raises(
        FileExistsError,
        match="Use overwrite=True",
    ):
        save_model_artifact(
            result,
            path,
            target_column="target_label",
            positive_class="positive",
            dataset_fingerprint="dataset-sha256",
            feature_fingerprint="features-sha256",
        )


def test_loaded_artifact_preserves_feature_order_and_rejects_missing_feature(
    training_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.joblib"
    _save_artifact(training_frame, path)
    loaded = load_model_artifact(path)

    reordered = training_frame.loc[
        :,
        [
            "feature_b",
            "assembly_accession",
            "feature_a",
            "target_label",
        ],
    ]

    ordered = loaded.ordered_features(reordered)

    assert tuple(ordered.columns) == (
        "feature_a",
        "feature_b",
    )
    np.testing.assert_array_equal(
        loaded.predict(reordered),
        loaded.estimator.predict(ordered),
    )

    with pytest.raises(
        ModelArtifactError,
        match="missing feature column",
    ):
        loaded.ordered_features(
            reordered.drop(columns="feature_a")
        )


def test_unsupported_artifact_version_is_rejected(
    training_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.joblib"
    _save_artifact(training_frame, path)

    payload = joblib.load(path)
    payload["artifact_version"] = "999.0"
    joblib.dump(payload, path)

    with pytest.raises(
        ModelArtifactError,
        match="Unsupported model artifact version",
    ):
        load_model_artifact(path)


def test_positive_class_must_map_to_one(
    training_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    result = train_classifier(
        training_frame,
        model_name="logistic_regression",
    )

    with pytest.raises(
        ModelArtifactError,
        match="positive class must map",
    ):
        save_model_artifact(
            result,
            tmp_path / "candidate.joblib",
            target_column="target_label",
            positive_class="negative",
            dataset_fingerprint="dataset-sha256",
            feature_fingerprint="features-sha256",
        )