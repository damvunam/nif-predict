"""Public API for NifPredict model training."""

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
    "LABEL_MAPPING",
    "PreparedTrainingData",
    "TrainingDataError",
    "TrainingResult",
    "build_classifier",
    "prepare_training_data",
    "train_classifier",
]