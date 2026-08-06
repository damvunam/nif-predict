"""Core model training engine for binary nitrogen-fixation prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABEL_MAPPING = {
    "negative": 0,
    "positive": 1,
}

METADATA_COLUMNS = frozenset(
    {
        "accession_id",
        "status",
        "assembly_accession",
        "organism_name",
        "strain",
        "evidence_tier",
        "evidence_source",
        "evidence_reference",
        "taxonomy_group",
        "source_genome_id",
        "dataset_role",
        "label_curator",
        "label_date",
        "notes",
    }
)


class TrainingDataError(ValueError):
    """Raised when a training dataset violates the model data contract."""


@dataclass(frozen=True)
class PreparedTrainingData:
    """Validated feature matrix and encoded binary target."""

    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]
    class_distribution: dict[str, int]


@dataclass(frozen=True)
class TrainingResult:
    """A fitted estimator and the metadata required for later inference."""

    estimator: Any
    model_name: str
    feature_names: tuple[str, ...]
    label_mapping: dict[str, int]
    class_distribution: dict[str, int]


def prepare_training_data(
    frame: pd.DataFrame,
    *,
    target_column: str = "target_label",
) -> PreparedTrainingData:
    """Validate and separate model features from labels and metadata."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Training input must be a pandas DataFrame")

    if frame.empty:
        raise TrainingDataError("Training dataset contains no rows")

    if target_column not in frame.columns:
        raise TrainingDataError(
            f"Required target_label column '{target_column}' is missing"
        )

    if frame.columns.duplicated().any():
        duplicate_columns = sorted(
            set(frame.columns[frame.columns.duplicated()].astype(str))
        )
        raise TrainingDataError(
            "Training dataset contains duplicate column names: "
            + ", ".join(duplicate_columns)
        )

    target_labels = frame[target_column]

    if target_labels.isna().any():
        raise TrainingDataError(
            "target_label contains missing values"
        )

    invalid_labels = sorted(
        set(target_labels.astype(str)) - set(LABEL_MAPPING)
    )

    if invalid_labels:
        raise TrainingDataError(
            "Invalid target_label value(s): "
            + ", ".join(invalid_labels)
        )

    present_labels = set(target_labels.astype(str))

    if present_labels != set(LABEL_MAPPING):
        raise TrainingDataError(
            "Training requires both target classes: negative and positive"
        )

    excluded_columns = set(METADATA_COLUMNS) | {target_column}
    feature_names = tuple(
        column
        for column in frame.columns
        if column not in excluded_columns
    )

    if not feature_names:
        raise TrainingDataError(
            "Training dataset contains no model feature columns"
        )

    non_numeric_columns = [
        column
        for column in feature_names
        if not is_numeric_dtype(frame[column])
    ]

    if non_numeric_columns:
        raise TrainingDataError(
            "Unexpected non-numeric feature column(s): "
            + ", ".join(non_numeric_columns)
        )

    features = frame.loc[:, feature_names].copy()

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise TrainingDataError(
            "Model features must contain only finite numeric values"
        )

    encoded_target = (
        target_labels
        .map(LABEL_MAPPING)
        .astype("int64")
        .copy()
    )

    class_distribution = {
        label: int((target_labels == label).sum())
        for label in LABEL_MAPPING
    }

    return PreparedTrainingData(
        features=features,
        target=encoded_target,
        feature_names=feature_names,
        class_distribution=class_distribution,
    )


def _resolve_class_weight(
    imbalance_handling: str,
) -> str | None:
    """Translate project configuration into a scikit-learn class weight."""
    if imbalance_handling == "ClassWeight":
        return "balanced"

    if imbalance_handling == "None":
        return None

    if imbalance_handling == "SMOTE":
        raise ValueError(
            "SMOTE is not supported by the core training engine; "
            "apply it inside cross-validation folds"
        )

    raise ValueError(
        "Unsupported imbalance handling strategy: "
        f"{imbalance_handling}"
    )


def build_classifier(
    model_name: str,
    *,
    random_seed: int = 42,
    imbalance_handling: str = "ClassWeight",
) -> Any:
    """Build an unfitted classifier using project-safe defaults."""
    class_weight = _resolve_class_weight(imbalance_handling)

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_seed,
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            random_state=random_seed,
            n_jobs=-1,
        )

    if model_name == "svm":
        classifier = SVC(
            class_weight=class_weight,
            probability=True,
            random_state=random_seed,
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def train_classifier(
    frame: pd.DataFrame,
    *,
    model_name: str,
    random_seed: int = 42,
    imbalance_handling: str = "ClassWeight",
    target_column: str = "target_label",
) -> TrainingResult:
    """Validate training data, build a classifier, and fit it."""
    prepared = prepare_training_data(
        frame,
        target_column=target_column,
    )

    estimator = build_classifier(
        model_name,
        random_seed=random_seed,
        imbalance_handling=imbalance_handling,
    )
    estimator.fit(prepared.features, prepared.target)

    return TrainingResult(
        estimator=estimator,
        model_name=model_name,
        feature_names=prepared.feature_names,
        label_mapping=dict(LABEL_MAPPING),
        class_distribution=dict(prepared.class_distribution),
    )