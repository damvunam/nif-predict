"""Public API for NifPredict model training and evaluation."""

from .metrics import (
    AGGREGATE_METRIC_COLUMNS,
    MetricError,
    aggregate_cross_validation_metrics,
    compute_binary_metrics,
)
from .persistence import (
    MODEL_ARTIFACT_VERSION,
    ModelArtifactError,
    load_model_artifact,
    save_model_artifact,
    verify_prediction_round_trip,
)
from .selection import (
    ModelSelectionError,
    select_baseline_candidate,
)
from .trainer import (
    LABEL_MAPPING,
    PreparedTrainingData,
    TrainingDataError,
    TrainingResult,
    build_classifier,
    prepare_training_data,
    train_classifier,
)

__all__ = [
    "AGGREGATE_METRIC_COLUMNS",
    "LABEL_MAPPING",
    "MODEL_ARTIFACT_VERSION",
    "MetricError",
    "ModelArtifactError",
    "ModelSelectionError",
    "PreparedTrainingData",
    "TrainingDataError",
    "TrainingResult",
    "aggregate_cross_validation_metrics",
    "build_classifier",
    "compute_binary_metrics",
    "load_model_artifact",
    "prepare_training_data",
    "save_model_artifact",
    "select_baseline_candidate",
    "train_classifier",
    "verify_prediction_round_trip",
]