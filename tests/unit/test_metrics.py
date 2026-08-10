"""Unit tests for binary classification metric computation."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nifpredict.models.metrics import (
    AGGREGATE_METRIC_COLUMNS,
    MetricError,
    aggregate_cross_validation_metrics,
    compute_binary_metrics,
)


def test_compute_binary_metrics_includes_scores_counts_and_support() -> None:
    metrics = compute_binary_metrics(
        ["negative", "negative", "positive", "positive"],
        ["negative", "positive", "negative", "positive"],
        positive_probability=[0.1, 0.8, 0.4, 0.9],
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert {name: metrics[name] for name in ("tn", "fp", "fn", "tp")} == {
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
    }
    assert metrics["negative_support"] == 2
    assert metrics["positive_support"] == 2


def test_metrics_without_continuous_scores_leave_rank_metrics_undefined() -> None:
    metrics = compute_binary_metrics(
        ["negative", "positive"],
        ["negative", "positive"],
    )

    assert math.isnan(float(metrics["roc_auc"]))
    assert math.isnan(float(metrics["average_precision"]))


def test_probability_and_decision_score_cannot_be_mixed() -> None:
    with pytest.raises(MetricError, match="either positive_probability"):
        compute_binary_metrics(
            ["negative", "positive"],
            ["negative", "positive"],
            positive_probability=[0.1, 0.9],
            decision_score=[-1.0, 1.0],
        )


def test_aggregate_metrics_use_fold_statistics_and_pooled_counts() -> None:
    fold_metrics = pd.DataFrame(
        [
            {
                "model_name": "model_a",
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "roc_auc": 0.75,
                "average_precision": 0.8,
                "tn": 1,
                "fp": 1,
                "fn": 1,
                "tp": 1,
                "negative_support": 2,
                "positive_support": 2,
            },
            {
                "model_name": "model_a",
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "roc_auc": 1.0,
                "average_precision": 1.0,
                "tn": 2,
                "fp": 0,
                "fn": 0,
                "tp": 2,
                "negative_support": 2,
                "positive_support": 2,
            },
        ]
    )
    oof_predictions = pd.DataFrame(
        {
            "model_name": ["model_a"] * 4,
            "true_label": ["negative", "negative", "positive", "positive"],
            "predicted_label": [
                "negative",
                "positive",
                "negative",
                "positive",
            ],
            "positive_probability": [0.1, 0.8, 0.4, 0.9],
            "decision_score": [float("nan")] * 4,
        }
    )

    aggregate = aggregate_cross_validation_metrics(
        fold_metrics,
        oof_predictions,
    )

    assert tuple(aggregate.columns) == AGGREGATE_METRIC_COLUMNS
    accuracy = aggregate.loc[aggregate["metric"] == "accuracy"].iloc[0]
    assert accuracy["fold_mean"] == pytest.approx(0.75)
    assert accuracy["fold_std"] == pytest.approx(0.25)
    assert accuracy["fold_min"] == pytest.approx(0.5)
    assert accuracy["fold_max"] == pytest.approx(1.0)
    assert accuracy["pooled_value"] == pytest.approx(0.5)
    assert accuracy["valid_fold_count"] == 2
    assert accuracy["total_fold_count"] == 2

    true_positives = aggregate.loc[aggregate["metric"] == "tp"].iloc[0]
    assert math.isnan(float(true_positives["fold_mean"]))
    assert true_positives["pooled_value"] == 1
    assert true_positives["valid_fold_count"] == 0
    assert (
        true_positives["pooled_scope"]
        == "all_oof_rows_across_repeats"
    )


def test_aggregate_rejects_mixed_continuous_score_semantics() -> None:
    metrics = compute_binary_metrics(
        ["negative", "positive"],
        ["negative", "positive"],
        positive_probability=[0.1, 0.9],
    )
    fold_metrics = pd.DataFrame(
        [{"model_name": "model_a", **metrics}]
    )
    oof_predictions = pd.DataFrame(
        {
            "model_name": ["model_a", "model_a"],
            "true_label": ["negative", "positive"],
            "predicted_label": ["negative", "positive"],
            "positive_probability": [0.1, float("nan")],
            "decision_score": [float("nan"), 1.0],
        }
    )

    with pytest.raises(
        MetricError,
        match="mixes probability and decision scores",
    ):
        aggregate_cross_validation_metrics(
            fold_metrics,
            oof_predictions,
        )