"""Offline integration tests for the training command-line interface."""

from __future__ import annotations

from nifpredict.common import paths
import os
import subprocess
import sys
from pathlib import Path

ROOT = paths.REPO_ROOT
FIXTURE = paths.FIXTURES_DIR / "grouped_training_dataset.csv"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def test_cli_runs_offline_and_writes_training_artifacts(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(paths.ML_SCRIPTS_DIR / "train_pipeline.py"),
            "--input",
            str(FIXTURE),
            "--run-name",
            "cli-offline",
            "--output-root",
            str(tmp_path / "results"),
            "--model",
            "logistic_regression",
            "--model",
            "random_forest",
            "--n-splits",
            "2",
            "--n-repeats",
            "1",
            "--random-seed",
            "17",
            "--no-mlflow",
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Selected baseline candidate:" in completed.stdout
    assert (tmp_path / "results" / "cli-offline" / "run_summary.json").is_file()


def test_cli_returns_two_for_missing_input(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(paths.ML_SCRIPTS_DIR / "train_pipeline.py"),
            "--input",
            str(tmp_path / "missing.csv"),
            "--run-name",
            "missing-input",
            "--no-mlflow",
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "Input table does not exist" in completed.stderr
