"""Package-level orchestration for reproducible grouped training runs."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from nifpredict.ml.evaluation import (
    GroupedCVConfig,
    GroupedCVResult,
    run_grouped_cross_validation,
)
from nifpredict.ml.models import prepare_training_data

from nifpredict.ml.workflows.training.artifacts import (
    AtomicTrainingOutputs,
    fingerprint_dataset,
    fingerprint_features,
    validate_run_name,
    write_csv,
    write_json,
)
from nifpredict.ml.workflows.training.refit import REFIT_ARTIFACT_FILENAMES, refit_selected_candidate

TRAINING_RUN_SCHEMA_VERSION = "1.0"
RUN_ARTIFACT_FILENAMES = (
    "run_config.json",
    "validation_report.json",
    "fold_manifest.csv",
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "oof_predictions.csv",
    "model_comparison.csv",
    "run_summary.json",
)


@dataclass(frozen=True)
class TrainingWorkflowConfig:
    """Configuration owned by the training workflow rather than the CLI."""

    run_name: str
    cv: GroupedCVConfig = GroupedCVConfig()
    input_source: str | None = None
    refit_selected: bool = False

    def __post_init__(self) -> None:
        validate_run_name(self.run_name)
        if self.input_source is not None and not self.input_source.strip():
            raise ValueError("input_source must be non-empty when provided")
        if not isinstance(self.refit_selected, bool):
            raise TypeError("refit_selected must be a boolean")


@dataclass(frozen=True)
class TrainingWorkflowResult:
    """In-memory result plus published run and optional model locations."""

    output_directory: Path
    dataset_fingerprint: str
    feature_fingerprint: str
    cv_result: GroupedCVResult
    model_output_directory: Path | None = None

    @property
    def artifact_paths(self) -> dict[str, Path]:
        """Return each expected artifact path keyed by filename."""
        return {
            filename: self.output_directory / filename
            for filename in RUN_ARTIFACT_FILENAMES
        }

    @property
    def model_artifact_paths(self) -> dict[str, Path]:
        """Return refit artifact paths, or an empty mapping when not requested."""
        if self.model_output_directory is None:
            return {}
        return {
            filename: self.model_output_directory / filename
            for filename in REFIT_ARTIFACT_FILENAMES
        }


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _software_versions() -> dict[str, str]:
    return {
        "nifpredict": _package_version("nifpredict"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "python": platform.python_version(),
        "scikit_learn": _package_version("scikit-learn"),
    }


def _cv_config_payload(config: GroupedCVConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["model_names"] = list(config.model_names)
    payload["tie_breakers"] = list(config.tie_breakers)
    return payload


def _validation_report(
    result: GroupedCVResult,
    *,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    summary = result.dataset_summary
    return {
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "status": "passed",
        "checks": {
            "accessions_are_unique": True,
            "both_target_classes_present": True,
            "folds_contain_both_classes": True,
            "groups_are_disjoint_between_splits": True,
            "manifest_covers_each_accession_once_per_repeat": True,
            "model_features_are_finite_numeric_values": True,
            "oof_coverage_is_complete": True,
        },
        "dataset": {
            "row_count": summary.row_count,
            "accession_count": summary.accession_count,
            "group_count": summary.group_count,
            "class_row_counts": dict(summary.class_row_counts),
            "class_group_counts": dict(summary.class_group_counts),
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
        },
    }


def _run_summary(
    result: GroupedCVResult,
    *,
    run_name: str,
    dataset_fingerprint: str,
    feature_fingerprint: str,
    refit_selected: bool,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "run_name": run_name,
        "status": "completed",
        "evaluation_scope": "internal_grouped_cross_validation",
        "dataset_fingerprint": dataset_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "selected_baseline_candidate": result.selection.selected_model,
        "selection": result.selection.to_report(),
        "artifacts": list(RUN_ARTIFACT_FILENAMES),
        "refit": {
            "enabled": refit_selected,
            "status": "completed" if refit_selected else "not_requested",
            "scope": ("full_training_ready_dataset" if refit_selected else None),
            "model_name": (result.selection.selected_model if refit_selected else None),
            "round_trip_verified": refit_selected,
            "artifacts": (list(REFIT_ARTIFACT_FILENAMES) if refit_selected else []),
        },
        "warning": (
            "The selected baseline candidate was chosen by internal "
            "cross-validation and is not a scientifically best or production model."
        ),
    }


def run_training_workflow(
    frame: pd.DataFrame,
    config: TrainingWorkflowConfig,
    *,
    output_root: str | Path,
    model_output_root: str | Path | None = None,
    fold_manifest: pd.DataFrame | None = None,
) -> TrainingWorkflowResult:
    """Compare models and optionally publish a verified full-dataset refit."""
    output_root_path = Path(output_root)
    final_path = output_root_path / config.run_name
    resolved_model_root: Path | None = None
    if config.refit_selected:
        if model_output_root is None:
            raise ValueError(
                "model_output_root is required when refit_selected is enabled"
            )
        if config.cv.positive_label != "positive":
            raise ValueError("Refit persistence requires positive_label='positive'")
        resolved_model_root = Path(model_output_root)

    transaction = AtomicTrainingOutputs(
        output_root_path,
        config.run_name,
        model_output_root=resolved_model_root,
    )
    with transaction as staging:
        prepared = prepare_training_data(
            frame,
            target_column=config.cv.target_column,
        )
        cv_result = run_grouped_cross_validation(
            frame,
            config.cv,
            fold_manifest=fold_manifest,
        )
        dataset_hash = fingerprint_dataset(
            frame,
            accession_column=config.cv.accession_column,
        )
        feature_hash = fingerprint_features(
            frame,
            accession_column=config.cv.accession_column,
            feature_names=prepared.feature_names,
        )
        created_at = datetime.now(timezone.utc).isoformat()

        run_config = {
            "schema_version": TRAINING_RUN_SCHEMA_VERSION,
            "run_name": config.run_name,
            "created_at_utc": created_at,
            "input_source": config.input_source,
            "fold_manifest_source": (
                "provided" if fold_manifest is not None else "generated"
            ),
            "dataset_fingerprint": dataset_hash,
            "feature_fingerprint": feature_hash,
            "refit_selected": config.refit_selected,
            "cross_validation": _cv_config_payload(config.cv),
            "software_versions": _software_versions(),
        }
        validation_report = _validation_report(
            cv_result,
            feature_names=prepared.feature_names,
        )

        if config.refit_selected:
            if staging.model_directory is None:
                raise RuntimeError("Refit model staging directory is unavailable")
            refit_selected_candidate(
                frame,
                selected_model=cv_result.selection.selected_model,
                output_directory=staging.model_directory,
                run_name=config.run_name,
                target_column=config.cv.target_column,
                positive_class=config.cv.positive_label,
                random_seed=config.cv.random_seed,
                imbalance_handling=config.cv.imbalance_handling,
                dataset_fingerprint=dataset_hash,
                feature_fingerprint=feature_hash,
                created_at_utc=created_at,
            )

        run_summary = _run_summary(
            cv_result,
            run_name=config.run_name,
            dataset_fingerprint=dataset_hash,
            feature_fingerprint=feature_hash,
            refit_selected=config.refit_selected,
        )
        run_staging = staging.run_directory
        write_json(run_staging / "run_config.json", run_config)
        write_json(run_staging / "validation_report.json", validation_report)
        write_csv(run_staging / "fold_manifest.csv", cv_result.fold_manifest)
        write_csv(run_staging / "fold_metrics.csv", cv_result.fold_metrics)
        write_csv(
            run_staging / "aggregate_metrics.csv",
            cv_result.aggregate_metrics,
        )
        write_csv(run_staging / "oof_predictions.csv", cv_result.oof_predictions)
        write_csv(
            run_staging / "model_comparison.csv",
            cv_result.selection.ranking,
        )
        write_json(run_staging / "run_summary.json", run_summary)

    final_model_path = (
        resolved_model_root / config.run_name
        if resolved_model_root is not None
        else None
    )

    return TrainingWorkflowResult(
        output_directory=final_path,
        dataset_fingerprint=dataset_hash,
        feature_fingerprint=feature_hash,
        cv_result=cv_result,
        model_output_directory=final_model_path,
    )
