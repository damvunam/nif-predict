"""Public grouped evaluation API for NifPredict."""

from .cross_validation import (
    DEFAULT_MODELS,
    FOLD_MANIFEST_COLUMNS,
    FOLD_METADATA_COLUMNS,
    FOLD_METRIC_COLUMNS,
    OOF_COLUMNS,
    GroupedCVConfig,
    GroupedCVResult,
    generate_fold_manifest,
    run_grouped_cross_validation,
)
from .leakage import (
    GroupedCVValidationError,
    GroupedDatasetSummary,
    validate_fold_manifest,
    validate_fold_split,
    validate_grouped_cv_input,
)

__all__ = [
    "DEFAULT_MODELS",
    "FOLD_MANIFEST_COLUMNS",
    "FOLD_METRIC_COLUMNS",
    "FOLD_METADATA_COLUMNS",
    "OOF_COLUMNS",
    "GroupedDatasetSummary",
    "GroupedCVConfig",
    "GroupedCVResult",
    "GroupedCVValidationError",
    "generate_fold_manifest",
    "run_grouped_cross_validation",
    "validate_fold_manifest",
    "validate_fold_split",
    "validate_grouped_cv_input",
]
