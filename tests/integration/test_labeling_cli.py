"""Integration tests for the labeling command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def test_cli_returns_nonzero_and_writes_report_on_validation_error(
    tmp_path: Path,
) -> None:
    labels = pd.read_csv(
        FIXTURES / "label_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )

    invalid_labels_path = tmp_path / "invalid_labels.csv"
    labels.drop(columns=["target_label"]).to_csv(
        invalid_labels_path,
        index=False,
    )

    output_path = tmp_path / "labeled_dataset.csv"
    training_output_path = tmp_path / "training_dataset.csv"
    report_path = tmp_path / "labeling_report.json"

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "label_dataset.py"),
            "--features",
            str(FIXTURES / "feature_matrix.csv"),
            "--labels",
            str(invalid_labels_path),
            "--output",
            str(output_path),
            "--training-output",
            str(training_output_path),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert report_path.is_file()

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert report["validation_errors"]
    assert not output_path.exists()
    assert not training_output_path.exists()