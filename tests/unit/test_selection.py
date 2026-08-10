"""Unit tests for deterministic baseline model selection."""

from __future__ import annotations

import pandas as pd
import pytest

from nifpredict.models.selection import (
    ModelSelectionError,
    select_baseline_candidate,
)


def _aggregate_rows(
    values: dict[str, tuple[float, float, float]],
) -> pd.DataFrame:
    rows = []
    for model_name, (
        average_precision,
        balanced_accuracy,
        f1,
    ) in values.items():
        rows.extend(
            [
                {
                    "model_name": model_name,
                    "metric": "average_precision",
                    "fold_mean": average_precision,
                },
                {
                    "model_name": model_name,
                    "metric": "balanced_accuracy",
                    "fold_mean": balanced_accuracy,
                },
                {
                    "model_name": model_name,
                    "metric": "f1",
                    "fold_mean": f1,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_primary_metric_is_respected_before_tie_breakers() -> None:
    aggregate = _aggregate_rows(
        {
            "model_a": (0.80, 0.60, 0.60),
            "model_b": (0.70, 0.95, 0.95),
        }
    )

    result = select_baseline_candidate(aggregate)

    assert result.selected_model == "model_a"
    assert result.ranking["model_name"].tolist() == [
        "model_a",
        "model_b",
    ]


def test_secondary_metric_breaks_primary_metric_tie() -> None:
    aggregate = _aggregate_rows(
        {
            "model_a": (0.80, 0.70, 0.90),
            "model_b": (0.80, 0.80, 0.60),
        }
    )

    result = select_baseline_candidate(aggregate)

    assert result.selected_model == "model_b"


def test_model_name_breaks_complete_metric_tie_deterministically() -> None:
    aggregate = _aggregate_rows(
        {
            "zeta_model": (0.80, 0.80, 0.80),
            "alpha_model": (0.80, 0.80, 0.80),
        }
    )

    result = select_baseline_candidate(aggregate)
    report = result.to_report()

    assert result.selected_model == "alpha_model"
    assert result.tied_models == ("alpha_model", "zeta_model")
    assert result.ranking["tied_after_metrics"].all()
    assert report["selected_baseline_candidate"] == "alpha_model"
    assert (
        report["selection_scope"]
        == "internal_grouped_cross_validation"
    )
    assert (
        "not a scientifically best or production model"
        in report["warning"]
    )


def test_custom_primary_metric_changes_ranking_policy() -> None:
    aggregate = _aggregate_rows(
        {
            "model_a": (0.90, 0.60, 0.70),
            "model_b": (0.70, 0.80, 0.90),
        }
    )

    result = select_baseline_candidate(
        aggregate,
        primary_metric="balanced_accuracy",
        tie_breakers=("f1", "average_precision"),
    )

    assert result.selected_model == "model_b"
    assert result.primary_metric == "balanced_accuracy"
    assert result.tie_breakers == ("f1", "average_precision")


def test_undefined_ranking_metric_is_rejected() -> None:
    aggregate = _aggregate_rows(
        {"model_a": (float("nan"), 0.8, 0.8)}
    )

    with pytest.raises(
        ModelSelectionError,
        match="undefined for model",
    ):
        select_baseline_candidate(aggregate)