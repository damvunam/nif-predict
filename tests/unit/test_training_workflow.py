"""Unit tests for training orchestration, run artifacts, and refit outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import nifpredict.workflows.training.refit as refit_module
import nifpredict.workflows.training.service as service_module
from nifpredict.evaluation import GroupedCVConfig, generate_fold_manifest
from nifpredict.models import load_model_artifact
from nifpredict.workflows.training import (
    REFIT_ARTIFACT_FILENAMES,
    RUN_ARTIFACT_FILENAMES,
    TrainingArtifactError,
    TrainingWorkflowConfig,
    fingerprint_dataset,
    fingerprint_features,
    run_training_workflow,
    validate_run_name,
)
from nifpredict.workflows.training.artifacts import AtomicRunDirectory

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "grouped_training_dataset.csv"


@pytest.fixture
def grouped_frame() -> pd.DataFrame:
    """Load the deterministic grouped training fixture."""
    return pd.read_csv(FIXTURE_PATH)


def _workflow_config(run_name: str = "unit-run") -> TrainingWorkflowConfig:
    return TrainingWorkflowConfig(
        run_name=run_name,
        input_source="tests/fixtures/grouped_training_dataset.csv",
        cv=GroupedCVConfig(
            model_names=("logistic_regression", "random_forest"),
            n_splits=2,
            n_repeats=1,
            random_seed=23,
        ),
    )


def test_dataset_fingerprint_is_independent_of_row_and_column_order(
    grouped_frame: pd.DataFrame,
) -> None:
    expected = fingerprint_dataset(
        grouped_frame,
        accession_column="assembly_accession",
    )
    reordered = grouped_frame.sample(frac=1, random_state=3).loc[
        :, list(reversed(grouped_frame.columns))
    ]

    observed = fingerprint_dataset(
        reordered,
        accession_column="assembly_accession",
    )

    assert observed == expected
    assert expected.startswith("sha256:")
    assert len(expected) == len("sha256:") + 64


def test_feature_fingerprint_preserves_feature_contract_order(
    grouped_frame: pd.DataFrame,
) -> None:
    original = fingerprint_features(
        grouped_frame,
        accession_column="assembly_accession",
        feature_names=("feature_a", "feature_b"),
    )
    reordered_rows = fingerprint_features(
        grouped_frame.iloc[::-1],
        accession_column="assembly_accession",
        feature_names=("feature_a", "feature_b"),
    )
    reversed_features = fingerprint_features(
        grouped_frame,
        accession_column="assembly_accession",
        feature_names=("feature_b", "feature_a"),
    )

    assert reordered_rows == original
    assert reversed_features != original


@pytest.mark.parametrize(
    "run_name",
    ["../escape", "/absolute", "contains space", "", "x" * 129],
)
def test_unsafe_run_names_are_rejected(run_name: str) -> None:
    with pytest.raises(TrainingArtifactError, match="run_name"):
        validate_run_name(run_name)


def test_training_workflow_writes_complete_consistent_artifact_set(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    result = run_training_workflow(
        grouped_frame,
        _workflow_config(),
        output_root=tmp_path,
    )

    assert result.output_directory == tmp_path / "unit-run"
    assert tuple(sorted(path.name for path in result.output_directory.iterdir())) == (
        "aggregate_metrics.csv",
        "fold_manifest.csv",
        "fold_metrics.csv",
        "model_comparison.csv",
        "oof_predictions.csv",
        "run_config.json",
        "run_summary.json",
        "validation_report.json",
    )
    assert set(result.artifact_paths) == set(RUN_ARTIFACT_FILENAMES)
    assert result.model_output_directory is None
    assert result.model_artifact_paths == {}

    run_config = json.loads(
        (result.output_directory / "run_config.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (result.output_directory / "validation_report.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (result.output_directory / "run_summary.json").read_text(encoding="utf-8")
    )

    assert run_config["fold_manifest_source"] == "generated"
    assert run_config["dataset_fingerprint"] == result.dataset_fingerprint
    assert run_config["feature_fingerprint"] == result.feature_fingerprint
    assert run_config["cross_validation"]["random_seed"] == 23
    assert validation["status"] == "passed"
    assert validation["dataset"]["row_count"] == len(grouped_frame)
    assert validation["dataset"]["feature_names"] == ["feature_a", "feature_b"]
    assert summary["status"] == "completed"
    assert summary["refit"]["status"] == "not_requested"
    assert (
        summary["selected_baseline_candidate"]
        == result.cv_result.selection.selected_model
    )
    assert "not a scientifically best or production model" in summary["warning"]

    pd.testing.assert_frame_equal(
        pd.read_csv(result.output_directory / "fold_manifest.csv"),
        result.cv_result.fold_manifest,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(result.output_directory / "model_comparison.csv"),
        result.cv_result.selection.ranking,
        check_dtype=False,
    )


def test_provided_manifest_is_preserved_and_recorded(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    config = _workflow_config("provided-folds")
    manifest, _ = generate_fold_manifest(grouped_frame, config.cv)

    result = run_training_workflow(
        grouped_frame,
        config,
        output_root=tmp_path,
        fold_manifest=manifest,
    )

    persisted = pd.read_csv(result.output_directory / "fold_manifest.csv")
    pd.testing.assert_frame_equal(persisted, manifest, check_dtype=False)
    run_config = json.loads(
        (result.output_directory / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["fold_manifest_source"] == "provided"


def test_existing_run_is_rejected_before_cross_validation(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "unit-run").mkdir()

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("Cross-validation must not run before overwrite rejection")

    monkeypatch.setattr(
        service_module,
        "run_grouped_cross_validation",
        fail_if_called,
    )

    with pytest.raises(FileExistsError, match="Choose a new run name"):
        run_training_workflow(
            grouped_frame,
            _workflow_config(),
            output_root=tmp_path,
        )


def test_atomic_directory_removes_partial_artifacts_after_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="simulated failure"):
        with AtomicRunDirectory(tmp_path, "failed-run") as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("simulated failure")

    assert not (tmp_path / "failed-run").exists()
    assert not list(tmp_path.glob(".failed-run.*.staging"))


def test_refit_writes_verified_model_artifact_contract(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    config = replace(
        _workflow_config("refit-run"),
        refit_selected=True,
    )
    result = run_training_workflow(
        grouped_frame,
        config,
        output_root=tmp_path / "results",
        model_output_root=tmp_path / "models",
    )

    expected_model_directory = tmp_path / "models" / "refit-run"
    assert result.model_output_directory == expected_model_directory
    assert set(result.model_artifact_paths) == set(REFIT_ARTIFACT_FILENAMES)
    assert tuple(
        sorted(path.name for path in expected_model_directory.iterdir())
    ) == tuple(sorted(REFIT_ARTIFACT_FILENAMES))

    loaded = load_model_artifact(expected_model_directory / "selected_model.joblib")
    metadata = json.loads(
        (expected_model_directory / "metadata.json").read_text(encoding="utf-8")
    )
    feature_contract = json.loads(
        (expected_model_directory / "feature_names.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (result.output_directory / "run_summary.json").read_text(encoding="utf-8")
    )

    assert loaded.model_name == result.cv_result.selection.selected_model
    assert loaded.dataset_fingerprint == result.dataset_fingerprint
    assert loaded.feature_fingerprint == result.feature_fingerprint
    assert loaded.hyperparameters == {
        "imbalance_handling": "ClassWeight",
        "random_seed": 23,
    }
    assert metadata["round_trip_verified"] is True
    assert metadata["refit_scope"] == "full_training_ready_dataset"
    assert metadata["training_row_count"] == len(grouped_frame)
    assert metadata["model_name"] == loaded.model_name
    assert "not a scientifically best or production model" in metadata["warning"]
    assert feature_contract["feature_names"] == ["feature_a", "feature_b"]
    assert feature_contract["feature_count"] == 2
    assert summary["refit"] == {
        "artifacts": list(REFIT_ARTIFACT_FILENAMES),
        "enabled": True,
        "model_name": loaded.model_name,
        "round_trip_verified": True,
        "scope": "full_training_ready_dataset",
        "status": "completed",
    }


def test_refit_requires_model_output_root_before_cross_validation(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("Cross-validation must not run before path validation")

    monkeypatch.setattr(
        service_module,
        "run_grouped_cross_validation",
        fail_if_called,
    )
    config = replace(_workflow_config("missing-model-root"), refit_selected=True)

    with pytest.raises(ValueError, match="model_output_root is required"):
        run_training_workflow(
            grouped_frame,
            config,
            output_root=tmp_path / "results",
        )

    assert not (tmp_path / "results" / "missing-model-root").exists()


def test_existing_refit_directory_is_rejected_before_cross_validation(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = tmp_path / "models" / "existing-model"
    model_directory.mkdir(parents=True)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("Cross-validation must not run before overwrite rejection")

    monkeypatch.setattr(
        service_module,
        "run_grouped_cross_validation",
        fail_if_called,
    )
    config = replace(_workflow_config("existing-model"), refit_selected=True)

    with pytest.raises(FileExistsError, match="Refit model directory already exists"):
        run_training_workflow(
            grouped_frame,
            config,
            output_root=tmp_path / "results",
            model_output_root=tmp_path / "models",
        )

    assert not (tmp_path / "results" / "existing-model").exists()


def test_refit_failure_removes_partial_run_and_model_outputs(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_round_trip(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated round-trip failure")

    monkeypatch.setattr(
        refit_module,
        "verify_prediction_round_trip",
        fail_round_trip,
    )
    config = replace(_workflow_config("failed-refit"), refit_selected=True)

    with pytest.raises(RuntimeError, match="simulated round-trip failure"):
        run_training_workflow(
            grouped_frame,
            config,
            output_root=tmp_path / "results",
            model_output_root=tmp_path / "models",
        )

    assert not (tmp_path / "results" / "failed-refit").exists()
    assert not (tmp_path / "models" / "failed-refit").exists()
    assert not list((tmp_path / "results").glob(".failed-refit.*.staging"))
    assert not list((tmp_path / "models").glob(".failed-refit.*.staging"))


def test_run_and_model_destinations_must_not_overlap(
    grouped_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("Cross-validation must not run before path validation")

    monkeypatch.setattr(
        service_module,
        "run_grouped_cross_validation",
        fail_if_called,
    )
    config = replace(_workflow_config("overlap"), refit_selected=True)

    with pytest.raises(TrainingArtifactError, match="must not overlap"):
        run_training_workflow(
            grouped_frame,
            config,
            output_root=tmp_path / "shared",
            model_output_root=tmp_path / "shared",
        )