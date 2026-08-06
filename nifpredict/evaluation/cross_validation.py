"""Repeated stratified grouped cross-validation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold

from nifpredict.models.metrics import (
    ALL_METRICS,
    aggregate_cross_validation_metrics,
    compute_binary_metrics,
)
from nifpredict.models.selection import (
    ModelSelectionResult,
    select_baseline_candidate,
)
from nifpredict.models.trainer import (
    LABEL_MAPPING,
    prepare_training_data,
    train_classifier,
)

from .leakage import (
    GroupedDatasetSummary,
    validate_fold_manifest,
    validate_fold_split,
    validate_grouped_cv_input,
)

DEFAULT_MODELS = ("logistic_regression", "random_forest", "svm")

FOLD_MANIFEST_COLUMNS = (
    "repeat",
    "fold",
    "split",
    "assembly_accession",
    "source_genome_id",
    "target_label",
)

OOF_COLUMNS = (
    "model_name",
    "repeat",
    "fold",
    "assembly_accession",
    "source_genome_id",
    "true_label",
    "predicted_label",
    "positive_probability",
    "decision_score",
)

FOLD_METADATA_COLUMNS = (
    "model_name",
    "repeat",
    "fold",
    "train_row_count",
    "validation_row_count",
    "train_group_count",
    "validation_group_count",
    "train_negative_count",
    "train_positive_count",
    "validation_negative_count",
    "validation_positive_count",
)

FOLD_METRIC_COLUMNS = FOLD_METADATA_COLUMNS + ALL_METRICS


@dataclass(frozen=True)
class GroupedCVConfig:
    """Deterministic policy for grouped cross-validation."""

    model_names: tuple[str, ...] = DEFAULT_MODELS
    n_splits: int = 5
    n_repeats: int = 3
    random_seed: int = 42
    target_column: str = "target_label"
    positive_label: str = "positive"
    accession_column: str = "assembly_accession"
    group_column: str = "source_genome_id"
    imbalance_handling: str = "ClassWeight"
    primary_metric: str = "average_precision"
    tie_breakers: tuple[str, ...] = ("balanced_accuracy", "f1")

    def __post_init__(self) -> None:
        if not self.model_names:
            raise ValueError("Grouped CV requires at least one model name")
        if len(set(self.model_names)) != len(self.model_names):
            raise ValueError("Grouped CV model names must be unique")


@dataclass(frozen=True)
class GroupedCVResult:
    """All reusable outputs from one grouped cross-validation comparison."""

    config: GroupedCVConfig
    dataset_summary: GroupedDatasetSummary
    fold_manifest: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    selection: ModelSelectionResult


def generate_fold_manifest(
    frame: pd.DataFrame,
    config: GroupedCVConfig,
) -> tuple[pd.DataFrame, GroupedDatasetSummary]:
    """Create validation-only assignments shared by every model."""
    summary = validate_grouped_cv_input(
        frame,
        target_column=config.target_column,
        positive_label=config.positive_label,
        accession_column=config.accession_column,
        group_column=config.group_column,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
    )
    prepare_training_data(frame, target_column=config.target_column)

    manifest_parts: list[pd.DataFrame] = []
    for repeat_offset in range(config.n_repeats):
        repeat = repeat_offset + 1
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.random_seed + repeat_offset,
        )
        for fold_offset, (_, validation_indices) in enumerate(
            splitter.split(
                frame,
                y=frame[config.target_column],
                groups=frame[config.group_column],
            )
        ):
            fold = fold_offset + 1
            validation_rows = frame.iloc[validation_indices][
                [
                    config.accession_column,
                    config.group_column,
                    config.target_column,
                ]
            ].copy()
            validation_rows.insert(0, "split", "validation")
            validation_rows.insert(0, "fold", fold)
            validation_rows.insert(0, "repeat", repeat)
            manifest_parts.append(validation_rows)

    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest = manifest.sort_values(
        ["repeat", "fold", config.accession_column],
        kind="stable",
        ignore_index=True,
    )
    validate_fold_manifest(
        frame,
        manifest,
        target_column=config.target_column,
        accession_column=config.accession_column,
        group_column=config.group_column,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
    )
    return manifest, summary


def _continuous_scores(
    estimator: Any,
    features: pd.DataFrame,
    *,
    positive_encoded_label: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Return probability or decision score in distinct output channels."""
    predict_proba = getattr(estimator, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(features), dtype=float)
        classes = np.asarray(estimator.classes_)
        matches = np.flatnonzero(classes == positive_encoded_label)
        if len(matches) != 1:
            raise ValueError(
                "Fitted estimator does not expose exactly one positive "
                "probability column"
            )
        return probabilities[:, int(matches[0])], None

    decision_function = getattr(estimator, "decision_function", None)
    if callable(decision_function):
        scores = np.asarray(decision_function(features), dtype=float)
        if scores.ndim == 2:
            classes = np.asarray(estimator.classes_)
            matches = np.flatnonzero(classes == positive_encoded_label)
            if len(matches) != 1:
                raise ValueError(
                    "Fitted estimator does not expose exactly one positive score column"
                )
            scores = scores[:, int(matches[0])]
        return None, scores.reshape(-1)

    return None, None


def run_grouped_cross_validation(
    frame: pd.DataFrame,
    config: GroupedCVConfig,
    *,
    fold_manifest: pd.DataFrame | None = None,
) -> GroupedCVResult:
    """Train each model on identical validated grouped folds."""
    if fold_manifest is None:
        shared_manifest, summary = generate_fold_manifest(frame, config)
    else:
        summary = validate_grouped_cv_input(
            frame,
            target_column=config.target_column,
            positive_label=config.positive_label,
            accession_column=config.accession_column,
            group_column=config.group_column,
            n_splits=config.n_splits,
            n_repeats=config.n_repeats,
        )
        prepare_training_data(frame, target_column=config.target_column)
        shared_manifest = fold_manifest.copy(deep=True)
        validate_fold_manifest(
            frame,
            shared_manifest,
            target_column=config.target_column,
            accession_column=config.accession_column,
            group_column=config.group_column,
            n_splits=config.n_splits,
            n_repeats=config.n_repeats,
        )

    inverse_labels = {value: key for key, value in LABEL_MAPPING.items()}
    positive_encoded_label = LABEL_MAPPING[config.positive_label]
    negative_label = next(
        label for label in LABEL_MAPPING if label != config.positive_label
    )
    fold_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []

    assignments = shared_manifest[["repeat", "fold"]].drop_duplicates()
    assignments = assignments.sort_values(["repeat", "fold"], kind="stable")
    for model_name in config.model_names:
        for assignment in assignments.itertuples(index=False):
            repeat = int(assignment.repeat)
            fold = int(assignment.fold)
            fold_assignment = shared_manifest.loc[
                (shared_manifest["repeat"] == repeat)
                & (shared_manifest["fold"] == fold)
            ]
            validation_accessions = set(fold_assignment[config.accession_column])
            validation_frame = frame.loc[
                frame[config.accession_column].isin(validation_accessions)
            ].copy()
            validation_groups = set(validation_frame[config.group_column])
            training_frame = frame.loc[
                ~frame[config.group_column].isin(validation_groups)
            ].copy()
            validate_fold_split(
                training_frame,
                validation_frame,
                target_column=config.target_column,
                group_column=config.group_column,
                repeat=repeat,
                fold=fold,
            )

            fold_seed = (
                config.random_seed
                + (repeat - 1) * config.n_splits
                + (fold - 1)
            )
            training_result = train_classifier(
                training_frame,
                model_name=model_name,
                random_seed=fold_seed,
                imbalance_handling=config.imbalance_handling,
                target_column=config.target_column,
            )
            validation_features = validation_frame.loc[
                :, training_result.feature_names
            ]
            encoded_predictions = np.asarray(
                training_result.estimator.predict(validation_features)
            )
            predicted_labels = np.asarray(
                [inverse_labels[int(value)] for value in encoded_predictions]
            )
            positive_probability, decision_score = _continuous_scores(
                training_result.estimator,
                validation_features,
                positive_encoded_label=positive_encoded_label,
            )

            metrics = compute_binary_metrics(
                validation_frame[config.target_column],
                predicted_labels,
                positive_label=config.positive_label,
                negative_label=negative_label,
                positive_probability=positive_probability,
                decision_score=decision_score,
            )
            fold_rows.append(
                {
                    "model_name": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "train_row_count": len(training_frame),
                    "validation_row_count": len(validation_frame),
                    "train_group_count": int(
                        training_frame[config.group_column].nunique()
                    ),
                    "validation_group_count": int(
                        validation_frame[config.group_column].nunique()
                    ),
                    "train_negative_count": int(
                        (training_frame[config.target_column] == negative_label).sum()
                    ),
                    "train_positive_count": int(
                        (
                            training_frame[config.target_column]
                            == config.positive_label
                        ).sum()
                    ),
                    "validation_negative_count": int(
                        (
                            validation_frame[config.target_column] == negative_label
                        ).sum()
                    ),
                    "validation_positive_count": int(
                        (
                            validation_frame[config.target_column]
                            == config.positive_label
                        ).sum()
                    ),
                    **metrics,
                }
            )

            oof_parts.append(
                pd.DataFrame(
                    {
                        "model_name": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        config.accession_column: validation_frame[
                            config.accession_column
                        ].to_numpy(),
                        config.group_column: validation_frame[
                            config.group_column
                        ].to_numpy(),
                        "true_label": validation_frame[
                            config.target_column
                        ].to_numpy(),
                        "predicted_label": predicted_labels,
                        "positive_probability": (
                            positive_probability
                            if positive_probability is not None
                            else np.full(len(validation_frame), np.nan)
                        ),
                        "decision_score": (
                            decision_score
                            if decision_score is not None
                            else np.full(len(validation_frame), np.nan)
                        ),
                    }
                )
            )

    fold_metrics = pd.DataFrame(fold_rows, columns=FOLD_METRIC_COLUMNS).sort_values(
        ["model_name", "repeat", "fold"],
        kind="stable",
        ignore_index=True,
    )
    oof_predictions = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["model_name", "repeat", "fold", config.accession_column],
        kind="stable",
        ignore_index=True,
    )

    expected_oof_rows = len(frame) * config.n_repeats
    for model_name in config.model_names:
        model_oof = oof_predictions.loc[
            oof_predictions["model_name"] == model_name
        ]
        if len(model_oof) != expected_oof_rows:
            raise RuntimeError(
                f"Model '{model_name}' produced {len(model_oof)} OOF rows; "
                f"expected {expected_oof_rows}"
            )
        coverage = model_oof.groupby(["repeat", config.accession_column]).size()
        if not (coverage == 1).all():
            raise RuntimeError(
                f"Model '{model_name}' does not have exactly one OOF prediction "
                "per accession and repeat"
            )

    aggregate_metrics = aggregate_cross_validation_metrics(
        fold_metrics,
        oof_predictions,
        positive_label=config.positive_label,
        negative_label=negative_label,
    )
    selection = select_baseline_candidate(
        aggregate_metrics,
        primary_metric=config.primary_metric,
        tie_breakers=config.tie_breakers,
    )
    return GroupedCVResult(
        config=config,
        dataset_summary=summary,
        fold_manifest=shared_manifest,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate_metrics,
        oof_predictions=oof_predictions,
        selection=selection,
    )