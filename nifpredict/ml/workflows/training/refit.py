"""Full-dataset refit and verified model artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nifpredict.ml.models import (
    MODEL_ARTIFACT_VERSION,
    TrainingResult,
    load_model_artifact,
    save_model_artifact,
    train_classifier,
    verify_prediction_round_trip,
)
from nifpredict.ml.models.persistence import LoadedModelArtifact

from nifpredict.ml.workflows.training.artifacts import write_json

MODEL_METADATA_SCHEMA_VERSION = "1.0"
REFIT_ARTIFACT_FILENAMES = (
    "selected_model.joblib",
    "metadata.json",
    "feature_names.json",
)


@dataclass(frozen=True)
class RefitArtifactResult:
    """Verified full-dataset refit and its staged output files."""

    output_directory: Path
    training_result: TrainingResult
    loaded_artifact: LoadedModelArtifact

    @property
    def artifact_paths(self) -> dict[str, Path]:
        """Return each expected refit artifact path keyed by filename."""
        return {
            filename: self.output_directory / filename
            for filename in REFIT_ARTIFACT_FILENAMES
        }


def refit_selected_candidate(
    frame: pd.DataFrame,
    *,
    selected_model: str,
    output_directory: str | Path,
    run_name: str,
    target_column: str,
    positive_class: str,
    random_seed: int,
    imbalance_handling: str,
    dataset_fingerprint: str,
    feature_fingerprint: str,
    created_at_utc: str,
) -> RefitArtifactResult:
    """Refit the selected candidate and write verified staged artifacts."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    existing = [
        filename
        for filename in REFIT_ARTIFACT_FILENAMES
        if (output_path / filename).exists()
    ]
    if existing:
        raise FileExistsError(
            "Refit output directory already contains artifact(s): "
            + ", ".join(existing)
        )

    training_result = train_classifier(
        frame,
        model_name=selected_model,
        random_seed=random_seed,
        imbalance_handling=imbalance_handling,
        target_column=target_column,
    )
    hyperparameters = {
        "imbalance_handling": imbalance_handling,
        "random_seed": random_seed,
    }
    model_path = output_path / "selected_model.joblib"
    save_model_artifact(
        training_result,
        model_path,
        target_column=target_column,
        positive_class=positive_class,
        dataset_fingerprint=dataset_fingerprint,
        feature_fingerprint=feature_fingerprint,
        hyperparameters=hyperparameters,
    )
    loaded_artifact = load_model_artifact(model_path)
    verify_prediction_round_trip(training_result, loaded_artifact, frame)

    metadata = {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "created_at_utc": created_at_utc,
        "run_name": run_name,
        "model_name": selected_model,
        "selection_scope": "internal_grouped_cross_validation",
        "refit_scope": "full_training_ready_dataset",
        "target_column": target_column,
        "positive_class": positive_class,
        "label_mapping": loaded_artifact.label_mapping,
        "class_distribution": loaded_artifact.class_distribution,
        "hyperparameters": hyperparameters,
        "training_row_count": len(frame),
        "dataset_fingerprint": dataset_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "round_trip_verified": True,
        "warning": (
            "This refitted baseline candidate was selected by internal "
            "cross-validation and is not a scientifically best or production model."
        ),
    }
    feature_contract = {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "feature_count": len(loaded_artifact.feature_names),
        "feature_names": list(loaded_artifact.feature_names),
        "feature_fingerprint": feature_fingerprint,
    }
    write_json(output_path / "metadata.json", metadata)
    write_json(output_path / "feature_names.json", feature_contract)

    return RefitArtifactResult(
        output_directory=output_path,
        training_result=training_result,
        loaded_artifact=loaded_artifact,
    )