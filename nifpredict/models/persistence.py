"""Versioned joblib persistence for fitted NifPredict classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .trainer import TrainingResult

MODEL_ARTIFACT_VERSION = "1.0"


class ModelArtifactError(ValueError):
    """Raised when a model artifact is unsafe, incompatible, or incomplete."""


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Validated fitted estimator and its inference contract."""

    estimator: Any
    model_name: str
    feature_names: tuple[str, ...]
    target_column: str
    positive_class: str
    label_mapping: dict[str, int]
    class_distribution: dict[str, int]
    hyperparameters: dict[str, Any]
    dataset_fingerprint: str
    feature_fingerprint: str
    artifact_version: str

    def ordered_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select features in their training order and reject missing columns."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Inference input must be a pandas DataFrame")
        if frame.columns.duplicated().any():
            raise ModelArtifactError("Inference input has duplicate column names")
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ModelArtifactError(
                "Inference input is missing feature column(s): " + ", ".join(missing)
            )
        return frame.loc[:, self.feature_names]

    def predict(self, frame: pd.DataFrame) -> NDArray[Any]:
        """Predict encoded labels using the persisted feature order."""
        return np.asarray(self.estimator.predict(self.ordered_features(frame)))


def save_model_artifact(
    training_result: TrainingResult,
    artifact_path: str | Path,
    *,
    target_column: str,
    positive_class: str,
    dataset_fingerprint: str,
    feature_fingerprint: str,
    hyperparameters: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist a fitted model without overwriting an artifact by default."""
    path = Path(artifact_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Model artifact already exists: {path}. Use overwrite=True to replace it."
        )
    if not target_column:
        raise ModelArtifactError("target_column must not be empty")
    if positive_class not in training_result.label_mapping:
        raise ModelArtifactError(
            f"Positive class '{positive_class}' is absent from the label mapping"
        )
    if training_result.label_mapping[positive_class] != 1:
        raise ModelArtifactError("The positive class must map to encoded label 1")
    if not dataset_fingerprint or not feature_fingerprint:
        raise ModelArtifactError(
            "Dataset and feature fingerprints must both be provided"
        )

    resolved_hyperparameters = (
        dict(hyperparameters)
        if hyperparameters is not None
        else dict(training_result.estimator.get_params(deep=True))
    )
    payload = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "estimator": training_result.estimator,
        "model_name": training_result.model_name,
        "feature_names": list(training_result.feature_names),
        "target_column": target_column,
        "positive_class": positive_class,
        "label_mapping": dict(training_result.label_mapping),
        "class_distribution": dict(training_result.class_distribution),
        "hyperparameters": resolved_hyperparameters,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_fingerprint": feature_fingerprint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def load_model_artifact(artifact_path: str | Path) -> LoadedModelArtifact:
    """Load and validate a supported model artifact."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")

    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ModelArtifactError("Model artifact payload must be a dictionary")

    required_keys = {
        "artifact_version",
        "estimator",
        "model_name",
        "feature_names",
        "target_column",
        "positive_class",
        "label_mapping",
        "class_distribution",
        "hyperparameters",
        "dataset_fingerprint",
        "feature_fingerprint",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise ModelArtifactError(
            "Model artifact is missing field(s): " + ", ".join(missing_keys)
        )
    if payload["artifact_version"] != MODEL_ARTIFACT_VERSION:
        raise ModelArtifactError(
            "Unsupported model artifact version: "
            f"{payload['artifact_version']}; expected {MODEL_ARTIFACT_VERSION}"
        )

    feature_names = tuple(str(name) for name in payload["feature_names"])
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ModelArtifactError(
            "Model artifact feature names must be non-empty and unique"
        )

    return LoadedModelArtifact(
        estimator=payload["estimator"],
        model_name=str(payload["model_name"]),
        feature_names=feature_names,
        target_column=str(payload["target_column"]),
        positive_class=str(payload["positive_class"]),
        label_mapping=dict(payload["label_mapping"]),
        class_distribution=dict(payload["class_distribution"]),
        hyperparameters=dict(payload["hyperparameters"]),
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
        feature_fingerprint=str(payload["feature_fingerprint"]),
        artifact_version=str(payload["artifact_version"]),
    )


def verify_prediction_round_trip(
    training_result: TrainingResult,
    loaded_artifact: LoadedModelArtifact,
    frame: pd.DataFrame,
) -> None:
    """Raise if reload changes predictions or available continuous scores."""
    original_features = frame.loc[:, training_result.feature_names]
    loaded_features = loaded_artifact.ordered_features(frame)
    original_predictions = np.asarray(
        training_result.estimator.predict(original_features)
    )
    loaded_predictions = np.asarray(
        loaded_artifact.estimator.predict(loaded_features)
    )
    if not np.array_equal(original_predictions, loaded_predictions):
        raise ModelArtifactError("Predictions changed after model artifact reload")

    for method_name in ("predict_proba", "decision_function"):
        original_method = getattr(training_result.estimator, method_name, None)
        loaded_method = getattr(loaded_artifact.estimator, method_name, None)
        if callable(original_method) != callable(loaded_method):
            raise ModelArtifactError(
                f"Estimator score method '{method_name}' changed after reload"
            )
        if callable(original_method) and callable(loaded_method):
            original_scores = np.asarray(original_method(original_features))
            loaded_scores = np.asarray(loaded_method(loaded_features))
            if not np.allclose(original_scores, loaded_scores, equal_nan=True):
                raise ModelArtifactError(
                    f"Estimator scores from '{method_name}' changed after reload"
                )