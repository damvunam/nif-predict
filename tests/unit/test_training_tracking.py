"""Unit tests for local MLflow training-run tracking."""

from __future__ import annotations

from nifpredict.common import paths
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nifpredict.ml.evaluation import GroupedCVConfig
from nifpredict.ml.workflows.training import (
    MLflowTrackingConfig,
    MLflowTrackingResult,
    TrainingWorkflowConfig,
    run_training_workflow,
    track_completed_training_run,
    write_tracking_receipt,
)

FIXTURE_PATH = paths.FIXTURES_DIR / "grouped_training_dataset.csv"


class FakeRun:
    """Minimal active-run context manager used by the tracking tests."""

    info = SimpleNamespace(run_id="fake-run-id")

    def __enter__(self) -> FakeRun:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeMLflow:
    """Capture MLflow calls without accessing a server or database."""

    def __init__(self) -> None:
        self.calls: dict[str, Any] = {}

    def set_tracking_uri(self, uri: str) -> None:
        self.calls["tracking_uri"] = uri

    def set_experiment(self, name: str, *, artifact_location: str) -> Any:
        self.calls["experiment"] = (name, artifact_location)
        return SimpleNamespace(experiment_id="experiment-1")

    def start_run(self, **kwargs: Any) -> FakeRun:
        self.calls["start_run"] = kwargs
        return FakeRun()

    def log_params(self, values: dict[str, Any]) -> None:
        self.calls["params"] = values

    def log_metrics(self, values: dict[str, float]) -> None:
        self.calls["metrics"] = values

    def log_artifacts(self, path: str, *, artifact_path: str) -> None:
        self.calls.setdefault("artifacts", []).append((path, artifact_path))

    def log_dict(self, value: dict[str, Any], path: str) -> None:
        self.calls["dict"] = (value, path)


@pytest.fixture
def completed_run(tmp_path: Path) -> tuple[Any, TrainingWorkflowConfig]:
    """Create a small completed workflow for tracking assertions."""
    frame = pd.read_csv(FIXTURE_PATH)
    config = TrainingWorkflowConfig(
        run_name="tracked-run",
        cv=GroupedCVConfig(
            model_names=("logistic_regression", "random_forest"),
            n_splits=2,
            n_repeats=1,
            random_seed=31,
        ),
    )
    result = run_training_workflow(frame, config, output_root=tmp_path / "results")
    return result, config


def test_tracking_config_builds_absolute_sqlite_uri(tmp_path: Path) -> None:
    config = MLflowTrackingConfig(
        database_path=tmp_path / "tracking" / "mlflow.db",
        artifact_root=tmp_path / "artifacts",
    )
    assert config.tracking_uri.startswith("sqlite:////")
    assert config.tracking_uri.endswith("/tracking/mlflow.db")


def test_completed_run_is_logged_with_provenance_and_artifacts(
    completed_run: tuple[Any, TrainingWorkflowConfig],
    tmp_path: Path,
) -> None:
    result, config = completed_run
    fake = FakeMLflow()
    tracking = MLflowTrackingConfig(
        database_path=tmp_path / "mlruns" / "mlflow.db",
        artifact_root=tmp_path / "mlruns" / "artifacts",
        experiment_name="offline-test",
    )

    tracked = track_completed_training_run(
        result, config, tracking, mlflow_module=fake
    )

    assert tracked.run_id == "fake-run-id"
    assert fake.calls["tracking_uri"] == tracking.tracking_uri
    assert fake.calls["params"]["random_seed"] == 31
    assert fake.calls["params"]["model_names"] == (
        "logistic_regression,random_forest"
    )
    assert fake.calls["metrics"]
    assert fake.calls["artifacts"] == [
        (str(result.output_directory), "training_run")
    ]
    provenance, name = fake.calls["dict"]
    assert name == "provenance.json"
    assert provenance["dataset_fingerprint"] == result.dataset_fingerprint
    assert "not a scientifically best or production model" in provenance["warning"]


def test_refit_model_directory_is_logged_as_separate_artifact(
    completed_run: tuple[Any, TrainingWorkflowConfig],
    tmp_path: Path,
) -> None:
    result, config = completed_run
    model_directory = tmp_path / "models" / config.run_name
    model_directory.mkdir(parents=True)
    result = replace(result, model_output_directory=model_directory)
    fake = FakeMLflow()

    track_completed_training_run(
        result,
        config,
        MLflowTrackingConfig(tmp_path / "mlflow.db", tmp_path / "artifacts"),
        mlflow_module=fake,
    )

    assert fake.calls["artifacts"][-1] == (str(model_directory), "refit_model")


def test_tracking_receipt_is_machine_readable(tmp_path: Path) -> None:
    result = MLflowTrackingResult(
        run_id="abc123",
        experiment_name="nifpredict-training",
        tracking_uri="sqlite:////tmp/mlflow.db",
    )
    path = write_tracking_receipt(tmp_path, result)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "experiment_name": "nifpredict-training",
        "run_id": "abc123",
        "tracking_uri": "sqlite:////tmp/mlflow.db",
    }
