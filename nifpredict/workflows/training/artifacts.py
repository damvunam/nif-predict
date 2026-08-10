"""Deterministic fingerprints and atomic artifacts for training runs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import pandas as pd

RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TrainingArtifactError(ValueError):
    """Raised when a training run artifact would be unsafe or ambiguous."""


def validate_run_name(run_name: str) -> str:
    """Validate a portable run directory name and return it unchanged."""
    if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
        raise TrainingArtifactError(
            "run_name must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-' (maximum 128 characters)"
        )
    if run_name in {".", ".."}:
        raise TrainingArtifactError("run_name must not be '.' or '..'")
    return run_name


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_frame_json(
    frame: pd.DataFrame,
    *,
    sort_column: str,
    columns: list[str],
) -> bytes:
    if sort_column not in frame.columns:
        raise TrainingArtifactError(
            f"Fingerprint sort column '{sort_column}' is missing"
        )
    ordered = frame.sort_values(sort_column, kind="stable").loc[:, columns]
    payload = str(
        ordered.to_json(
            orient="split",
            date_format="iso",
            date_unit="us",
            double_precision=15,
            force_ascii=True,
            index=False,
        )
    )
    return payload.encode("utf-8")


def fingerprint_dataset(
    frame: pd.DataFrame,
    *,
    accession_column: str,
) -> str:
    """Hash all dataset columns independent of input row and column order."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset fingerprint input must be a pandas DataFrame")
    columns = sorted(str(column) for column in frame.columns)
    payload = _canonical_frame_json(
        frame,
        sort_column=accession_column,
        columns=columns,
    )
    return _sha256(payload)


def fingerprint_features(
    frame: pd.DataFrame,
    *,
    accession_column: str,
    feature_names: tuple[str, ...],
) -> str:
    """Hash ordered feature names and values independent of input row order."""
    if not feature_names:
        raise TrainingArtifactError("Feature fingerprint requires feature columns")
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise TrainingArtifactError(
            "Feature fingerprint input is missing column(s): " + ", ".join(missing)
        )
    header = json.dumps(
        {"feature_names": list(feature_names)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    values = _canonical_frame_json(
        frame,
        sort_column=accession_column,
        columns=[accession_column, *feature_names],
    )
    return _sha256(header + b"\n" + values)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict, readable JSON with a stable key order."""
    path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame using one consistent artifact policy."""
    frame.to_csv(path, index=False, lineterminator="\n")


class AtomicRunDirectory(AbstractContextManager[Path]):
    """Stage a run directory and publish it only after every write succeeds."""

    def __init__(self, output_root: str | Path, run_name: str) -> None:
        self.output_root = Path(output_root)
        self.run_name = validate_run_name(run_name)
        self.final_path = self.output_root / self.run_name
        self.staging_path: Path | None = None

    def __enter__(self) -> Path:
        if self.final_path.exists():
            raise FileExistsError(
                f"Training run directory already exists: {self.final_path}. "
                "Choose a new run name."
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        staging = tempfile.mkdtemp(
            prefix=f".{self.run_name}.",
            suffix=".staging",
            dir=self.output_root,
        )
        self.staging_path = Path(staging)
        return self.staging_path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self.staging_path is None:
            return False
        if exc_type is not None:
            shutil.rmtree(self.staging_path, ignore_errors=True)
            return False
        if self.final_path.exists():
            shutil.rmtree(self.staging_path, ignore_errors=True)
            raise FileExistsError(
                f"Training run directory appeared during execution: {self.final_path}"
            )
        try:
            self.staging_path.replace(self.final_path)
        except OSError:
            shutil.rmtree(self.staging_path, ignore_errors=True)
            raise
        return False


@dataclass(frozen=True)
class StagedTrainingOutputs:
    """Temporary run and optional model directories for one workflow."""

    run_directory: Path
    model_directory: Path | None


class AtomicTrainingOutputs(AbstractContextManager[StagedTrainingOutputs]):
    """Publish run and optional model directories as one recoverable operation."""

    def __init__(
        self,
        run_output_root: str | Path,
        run_name: str,
        *,
        model_output_root: str | Path | None = None,
    ) -> None:
        self.run_output_root = Path(run_output_root)
        self.model_output_root = (
            Path(model_output_root) if model_output_root is not None else None
        )
        self.run_name = validate_run_name(run_name)
        self.final_run_path = self.run_output_root / self.run_name
        self.final_model_path = (
            self.model_output_root / self.run_name
            if self.model_output_root is not None
            else None
        )
        self.run_staging_path: Path | None = None
        self.model_staging_path: Path | None = None

    def _validate_destinations(self) -> None:
        if self.final_run_path.exists():
            raise FileExistsError(
                f"Training run directory already exists: {self.final_run_path}. "
                "Choose a new run name."
            )
        if self.final_model_path is not None and self.final_model_path.exists():
            raise FileExistsError(
                f"Refit model directory already exists: {self.final_model_path}. "
                "Choose a new run name."
            )
        if self.final_model_path is None:
            return

        resolved_run = self.final_run_path.resolve()
        resolved_model = self.final_model_path.resolve()
        if (
            resolved_run == resolved_model
            or resolved_run in resolved_model.parents
            or resolved_model in resolved_run.parents
        ):
            raise TrainingArtifactError(
                "Training run and refit model directories must not overlap"
            )

    def __enter__(self) -> StagedTrainingOutputs:
        self._validate_destinations()
        self.run_output_root.mkdir(parents=True, exist_ok=True)
        self.run_staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{self.run_name}.",
                suffix=".staging",
                dir=self.run_output_root,
            )
        )
        try:
            if self.model_output_root is not None:
                self.model_output_root.mkdir(parents=True, exist_ok=True)
                self.model_staging_path = Path(
                    tempfile.mkdtemp(
                        prefix=f".{self.run_name}.",
                        suffix=".staging",
                        dir=self.model_output_root,
                    )
                )
        except OSError:
            shutil.rmtree(self.run_staging_path, ignore_errors=True)
            self.run_staging_path = None
            raise

        return StagedTrainingOutputs(
            run_directory=self.run_staging_path,
            model_directory=self.model_staging_path,
        )

    def _remove_staging_directories(self) -> None:
        for path in (self.run_staging_path, self.model_staging_path):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is not None:
            self._remove_staging_directories()
            return False

        try:
            self._validate_destinations()
        except (FileExistsError, TrainingArtifactError):
            self._remove_staging_directories()
            raise
        if self.run_staging_path is None:
            self._remove_staging_directories()
            raise RuntimeError("Training run staging directory was not initialized")

        published: list[Path] = []
        try:
            if self.model_staging_path is not None:
                if self.final_model_path is None:
                    raise RuntimeError("Refit model destination was not initialized")
                self.model_staging_path.replace(self.final_model_path)
                published.append(self.final_model_path)
            self.run_staging_path.replace(self.final_run_path)
            published.append(self.final_run_path)
        except OSError:
            for path in reversed(published):
                shutil.rmtree(path, ignore_errors=True)
            self._remove_staging_directories()
            raise
        return False