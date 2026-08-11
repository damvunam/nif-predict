#!/usr/bin/env python3
"""Run reproducible grouped baseline training for NifPredict."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from nifpredict.evaluation import DEFAULT_MODELS, GroupedCVConfig
from nifpredict.workflows.training import (
    DEFAULT_EXPERIMENT_NAME,
    MLflowTrackingConfig,
    TrainingArtifactError,
    TrainingWorkflowConfig,
    run_training_workflow,
    track_completed_training_run,
    write_tracking_receipt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "training_dataset.parquet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "training"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_MLFLOW_DATABASE = PROJECT_ROOT / "mlruns" / "mlflow.db"
DEFAULT_MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlruns" / "artifacts"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin training command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compare NifPredict baselines with grouped cross-validation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-output-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument(
        "--model", action="append", choices=DEFAULT_MODELS, dest="models"
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--refit-selected", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-database", type=Path, default=DEFAULT_MLFLOW_DATABASE)
    parser.add_argument(
        "--mlflow-artifact-root", type=Path, default=DEFAULT_MLFLOW_ARTIFACTS
    )
    parser.add_argument("--mlflow-experiment", default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args(argv)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input tables must use .csv, .parquet, or .pq")


def run(args: argparse.Namespace) -> int:
    """Translate CLI arguments into package workflow configuration."""
    frame = _read_table(args.input)
    fold_manifest = (
        _read_table(args.fold_manifest) if args.fold_manifest is not None else None
    )
    workflow_config = TrainingWorkflowConfig(
        run_name=args.run_name,
        cv=GroupedCVConfig(
            model_names=tuple(args.models or DEFAULT_MODELS),
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_seed=args.random_seed,
        ),
        input_source=str(args.input),
        refit_selected=args.refit_selected,
    )
    result = run_training_workflow(
        frame,
        workflow_config,
        output_root=args.output_root,
        model_output_root=args.model_output_root if args.refit_selected else None,
        fold_manifest=fold_manifest,
    )
    print(f"Training artifacts: {result.output_directory}")
    print(f"Selected baseline candidate: {result.cv_result.selection.selected_model}")
    if result.model_output_directory is not None:
        print(f"Refit model artifacts: {result.model_output_directory}")
    if not args.no_mlflow:
        tracking_result = track_completed_training_run(
            result,
            workflow_config,
            MLflowTrackingConfig(
                database_path=args.mlflow_database,
                artifact_root=args.mlflow_artifact_root,
                experiment_name=args.mlflow_experiment,
            ),
        )
        write_tracking_receipt(result.output_directory, tracking_result)
        print(f"MLflow run ID: {tracking_result.run_id}")
        print(f"MLflow tracking URI: {tracking_result.tracking_uri}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Map actionable failures to a non-zero exit code."""
    try:
        return run(parse_args(argv))
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
        TrainingArtifactError,
    ) as exc:
        print(f"Training pipeline failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())