"""Deterministic selection of an internal-CV baseline candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class ModelSelectionError(ValueError):
    """Raised when aggregate metrics cannot support model ranking."""


@dataclass(frozen=True)
class ModelSelectionResult:
    """Ranked candidates and an explicit internal-validation warning."""

    selected_model: str
    ranking: pd.DataFrame
    primary_metric: str
    tie_breakers: tuple[str, ...]
    tied_models: tuple[str, ...]

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serializable selection report."""
        return {
            "model_order": self.ranking["model_name"].tolist(),
            "primary_metric": self.primary_metric,
            "tie_breakers": list(self.tie_breakers),
            "aggregate_field": "fold_mean",
            "selected_baseline_candidate": self.selected_model,
            "tied_models": list(self.tied_models),
            "selection_scope": "internal_grouped_cross_validation",
            "warning": (
                "The selected baseline candidate was chosen by internal "
                "cross-validation and is not a scientifically best or production model."
            ),
        }


def select_baseline_candidate(
    aggregate_metrics: pd.DataFrame,
    *,
    primary_metric: str = "average_precision",
    tie_breakers: tuple[str, ...] = ("balanced_accuracy", "f1"),
) -> ModelSelectionResult:
    """Rank models by mean fold metrics with a deterministic name fallback."""
    required_columns = {"model_name", "metric", "fold_mean"}
    missing_columns = sorted(required_columns - set(aggregate_metrics.columns))
    if missing_columns:
        raise ModelSelectionError(
            "Aggregate metrics table is missing column(s): "
            + ", ".join(missing_columns)
        )

    ranking_metrics = (primary_metric, *tie_breakers)
    if len(set(ranking_metrics)) != len(ranking_metrics):
        raise ModelSelectionError("Ranking metrics must be unique")

    relevant = aggregate_metrics.loc[
        aggregate_metrics["metric"].isin(ranking_metrics),
        ["model_name", "metric", "fold_mean"],
    ].copy()
    duplicate_pairs = relevant.duplicated(["model_name", "metric"], keep=False)
    if duplicate_pairs.any():
        raise ModelSelectionError(
            "Aggregate metrics contain duplicate model/metric rows"
        )

    pivot = relevant.pivot(
        index="model_name",
        columns="metric",
        values="fold_mean",
    )
    missing_metrics = [metric for metric in ranking_metrics if metric not in pivot]
    if missing_metrics:
        raise ModelSelectionError(
            "Aggregate metrics do not contain ranking metric(s): "
            + ", ".join(missing_metrics)
        )
    if pivot.empty:
        raise ModelSelectionError("Aggregate metrics contain no model candidates")

    incomplete_models = pivot.loc[:, list(ranking_metrics)].isna().any(axis=1)
    if incomplete_models.any():
        names = sorted(pivot.index[incomplete_models].astype(str))
        raise ModelSelectionError(
            "Ranking metrics are undefined for model(s): " + ", ".join(names)
        )

    metric_values = {
        str(model_name): tuple(
            float(pivot.loc[model_name, metric]) for metric in ranking_metrics
        )
        for model_name in pivot.index
    }
    model_order = sorted(
        metric_values,
        key=lambda model_name: (
            *(-value for value in metric_values[model_name]),
            model_name,
        ),
    )

    best_values = metric_values[model_order[0]]
    tied_models = tuple(
        model_name
        for model_name in model_order
        if all(
            np.isclose(value, best, rtol=1e-12, atol=1e-12)
            for value, best in zip(metric_values[model_name], best_values, strict=True)
        )
    )

    ranking_rows = []
    for rank, model_name in enumerate(model_order, start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "model_name": model_name,
            "selected_baseline_candidate": rank == 1,
            "tied_after_metrics": model_name in tied_models and len(tied_models) > 1,
        }
        row.update(
            {
                metric: metric_values[model_name][position]
                for position, metric in enumerate(ranking_metrics)
            }
        )
        ranking_rows.append(row)

    return ModelSelectionResult(
        selected_model=model_order[0],
        ranking=pd.DataFrame(ranking_rows),
        primary_metric=primary_metric,
        tie_breakers=tuple(tie_breakers),
        tied_models=tied_models if len(tied_models) > 1 else (),
    )
