"""Local MLflow tracking for completed NifPredict training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from nifpredict.ml.workflows.training.service import TrainingWorkflowConfig, TrainingWorkflowResult

DEFAULT_EXPERIMENT_NAME = "nifpredict-training"


@dataclass(frozen=True)
class MLflowTrackingConfig:
    """Configuration for local SQLite-backed MLflow tracking."""

    database_path: Path
    artifact_root: Path
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")

    @property
    def tracking_uri(self) -> str:
        """Return an absolute SQLite tracking URI."""
        return f"sqlite:///{self.database_path.resolve().as_posix()}"


@dataclass(frozen=True)
class MLflowTrackingResult:
    """Identity and location of a recorded MLflow run."""

    run_id: str
    experiment_name: str
    tracking_uri: str


def _load_mlflow() -> ModuleType:
    try:
        mlflow = import_module("mlflow")
    except ImportError as exc:
        raise ImportError(
            "MLflow tracking requires the 'mlflow' dependency. "
            "Install the project dependencies or run with --no-mlflow."
        ) from exc
    return mlflow


def _metric_payload(result: TrainingWorkflowResult) -> dict[str, float]:
    ranking = result.cv_result.selection.ranking
    selected = result.cv_result.selection.selected_model
    selected_rows = ranking.loc[ranking["model_name"] == selected]
    if len(selected_rows) != 1:
        raise ValueError("Model comparison must contain one selected model row")
    payload: dict[str, float] = {}
    for column, value in selected_rows.iloc[0].items():
        if column == "model_name" or pd.isna(value):
            continue
        try:
            payload[f"selected_{column}"] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


def _flatten_parameters(config: TrainingWorkflowConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "model_names": ",".join(config.cv.model_names),
        "n_splits": config.cv.n_splits,
        "n_repeats": config.cv.n_repeats,
        "random_seed": config.cv.random_seed,
        "primary_metric": config.cv.primary_metric,
        "tie_breakers": ",".join(config.cv.tie_breakers),
        "imbalance_handling": config.cv.imbalance_handling,
        "refit_selected": config.refit_selected,
    }


def track_completed_training_run(
    result: TrainingWorkflowResult,
    config: TrainingWorkflowConfig,
    tracking: MLflowTrackingConfig,
    *,
    mlflow_module: ModuleType | Any | None = None,
) -> MLflowTrackingResult:
    """Record a successfully published workflow using local MLflow storage."""
    mlflow = mlflow_module or _load_mlflow()
    tracking.database_path.parent.mkdir(parents=True, exist_ok=True)
    tracking.artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking.tracking_uri)
    experiment = mlflow.set_experiment(
        tracking.experiment_name,
        artifact_location=tracking.artifact_root.resolve().as_uri(),
    )
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=config.run_name,
        tags={
            "nifpredict.dataset_fingerprint": result.dataset_fingerprint,
            "nifpredict.feature_fingerprint": result.feature_fingerprint,
            "nifpredict.evaluation_scope": "internal_grouped_cross_validation",
        },
    ) as active_run:
        mlflow.log_params(_flatten_parameters(config))
        mlflow.log_metrics(_metric_payload(result))
        mlflow.log_artifacts(str(result.output_directory), artifact_path="training_run")
        if result.model_output_directory is not None:
            mlflow.log_artifacts(
                str(result.model_output_directory), artifact_path="refit_model"
            )
        mlflow.log_dict(
            {
                "selected_baseline_candidate": (
                    result.cv_result.selection.selected_model
                ),
                "dataset_fingerprint": result.dataset_fingerprint,
                "feature_fingerprint": result.feature_fingerprint,
                "warning": (
                    "Selected by internal cross-validation; not a scientifically "
                    "best or production model."
                ),
            },
            "provenance.json",
        )
    return MLflowTrackingResult(
        run_id=str(active_run.info.run_id),
        experiment_name=tracking.experiment_name,
        tracking_uri=tracking.tracking_uri,
    )


def write_tracking_receipt(
    output_directory: Path, tracking_result: MLflowTrackingResult
) -> Path:
    """Write a small receipt beside published workflow artifacts."""
    receipt_path = output_directory / "mlflow_tracking.json"
    receipt_path.write_text(
        json.dumps(
            {
                "experiment_name": tracking_result.experiment_name,
                "run_id": tracking_result.run_id,
                "tracking_uri": tracking_result.tracking_uri,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path