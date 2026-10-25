"""Binary classification metrics for cross-validation evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCORE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)

COUNT_METRICS = (
    "tn",
    "fp",
    "fn",
    "tp",
    "negative_support",
    "positive_support",
)

ALL_METRICS = SCORE_METRICS + COUNT_METRICS

AGGREGATE_METRIC_COLUMNS = (
    "model_name",
    "metric",
    "fold_mean",
    "fold_std",
    "fold_min",
    "fold_max",
    "pooled_value",
    "valid_fold_count",
    "total_fold_count",
    "pooled_scope",
)


class MetricError(ValueError):
    """Raised when metric inputs or result tables are inconsistent."""


def compute_binary_metrics(
    true_labels: Sequence[Any] | NDArray[Any] | pd.Series,
    predicted_labels: Sequence[Any] | NDArray[Any] | pd.Series,
    *,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
    positive_probability: Sequence[float] | NDArray[Any] | pd.Series | None = None,
    decision_score: Sequence[float] | NDArray[Any] | pd.Series | None = None,
) -> dict[str, float | int]:
    """Compute serializable binary metrics without conflating score types."""
    true_array = np.asarray(true_labels)
    predicted_array = np.asarray(predicted_labels)

    if true_array.ndim != 1 or predicted_array.ndim != 1:
        raise MetricError("Metric labels must be one-dimensional")

    if len(true_array) == 0:
        raise MetricError("Metric input contains no rows")

    if len(true_array) != len(predicted_array):
        raise MetricError("True and predicted labels must have equal length")

    if positive_probability is not None and decision_score is not None:
        raise MetricError(
            "Provide either positive_probability or decision_score, not both"
        )

    allowed_labels = {negative_label, positive_label}
    observed_labels = set(true_array.tolist()) | set(predicted_array.tolist())
    invalid_labels = sorted(str(value) for value in observed_labels - allowed_labels)
    if invalid_labels:
        raise MetricError(
            "Metric labels contain unsupported value(s): " + ", ".join(invalid_labels)
        )

    if set(true_array.tolist()) != allowed_labels:
        raise MetricError(
            "Metric computation requires both negative and positive true labels"
        )

    score_values: NDArray[np.float64] | None = None
    if positive_probability is not None:
        score_values = np.asarray(positive_probability, dtype=float)
        if ((score_values < 0) | (score_values > 1)).any():
            raise MetricError("Positive probabilities must be within [0, 1]")
    elif decision_score is not None:
        score_values = np.asarray(decision_score, dtype=float)

    if score_values is not None:
        if score_values.ndim != 1 or len(score_values) != len(true_array):
            raise MetricError("Continuous scores must align with label rows")
        if not np.isfinite(score_values).all():
            raise MetricError("Continuous scores must contain only finite values")

    true_binary = (true_array == positive_label).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        true_array,
        predicted_array,
        labels=[negative_label, positive_label],
    ).ravel()

    roc_auc = float("nan")
    average_precision = float("nan")
    if score_values is not None:
        roc_auc = float(roc_auc_score(true_binary, score_values))
        average_precision = float(
            average_precision_score(true_binary, score_values)
        )

    return {
        "accuracy": float(accuracy_score(true_array, predicted_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_array, predicted_array)
        ),
        "precision": float(
            precision_score(
                true_array,
                predicted_array,
                pos_label=positive_label,
            )
        ),
        "recall": float(
            recall_score(
                true_array,
                predicted_array,
                pos_label=positive_label,
            )
        ),
        "f1": float(
            f1_score(
                true_array,
                predicted_array,
                pos_label=positive_label,
            )
        ),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "negative_support": int((true_array == negative_label).sum()),
        "positive_support": int((true_array == positive_label).sum()),
    }


def aggregate_cross_validation_metrics(
    fold_metrics: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    *,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
) -> pd.DataFrame:
    """Aggregate fold statistics and pooled metrics for each model."""
    required_fold_columns = {"model_name", *ALL_METRICS}
    required_oof_columns = {
        "model_name",
        "true_label",
        "predicted_label",
        "positive_probability",
        "decision_score",
    }
    missing_fold = sorted(required_fold_columns - set(fold_metrics.columns))
    missing_oof = sorted(required_oof_columns - set(oof_predictions.columns))
    if missing_fold:
        raise MetricError(
            "Fold metrics table is missing column(s): " + ", ".join(missing_fold)
        )
    if missing_oof:
        raise MetricError(
            "OOF predictions table is missing column(s): " + ", ".join(missing_oof)
        )

    fold_models = set(fold_metrics["model_name"].astype(str))
    oof_models = set(oof_predictions["model_name"].astype(str))
    if fold_models != oof_models:
        raise MetricError("Fold metrics and OOF predictions contain different models")

    rows: list[dict[str, Any]] = []
    for model_name in sorted(fold_models):
        model_folds = fold_metrics.loc[fold_metrics["model_name"] == model_name]
        model_oof = oof_predictions.loc[
            oof_predictions["model_name"] == model_name
        ]

        probability = model_oof["positive_probability"]
        decision = model_oof["decision_score"]
        has_probability = probability.notna().any()
        has_decision = decision.notna().any()
        if has_probability and has_decision:
            raise MetricError(
                f"Model '{model_name}' mixes probability and decision scores"
            )

        pooled_metrics = compute_binary_metrics(
            model_oof["true_label"],
            model_oof["predicted_label"],
            positive_label=positive_label,
            negative_label=negative_label,
            positive_probability=probability if has_probability else None,
            decision_score=decision if has_decision else None,
        )
        total_fold_count = len(model_folds)

        for metric_name in ALL_METRICS:
            values = pd.to_numeric(model_folds[metric_name], errors="coerce")
            valid_values = values.dropna()
            is_count_metric = metric_name in COUNT_METRICS
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric_name,
                    "fold_mean": (
                        float("nan")
                        if is_count_metric or valid_values.empty
                        else float(valid_values.mean())
                    ),
                    "fold_std": (
                        float("nan")
                        if is_count_metric or valid_values.empty
                        else float(valid_values.std(ddof=0))
                    ),
                    "fold_min": (
                        float("nan")
                        if is_count_metric or valid_values.empty
                        else float(valid_values.min())
                    ),
                    "fold_max": (
                        float("nan")
                        if is_count_metric or valid_values.empty
                        else float(valid_values.max())
                    ),
                    "pooled_value": pooled_metrics[metric_name],
                    "valid_fold_count": (
                        0 if is_count_metric else int(valid_values.size)
                    ),
                    "total_fold_count": int(total_fold_count),
                    "pooled_scope": "all_oof_rows_across_repeats",
                }
            )

    return pd.DataFrame(rows, columns=AGGREGATE_METRIC_COLUMNS).sort_values(
        ["model_name", "metric"],
        kind="stable",
        ignore_index=True,
    )
