"""Public API for independent-manifest labeling."""

from .schema import (
    ACCESSION_PATTERN,
    CONTRACT_VERSION,
    DATASET_ROLES,
    DEFAULT_EVIDENCE_SOURCES,
    DEFAULT_TRAINING_EVIDENCE_TIERS,
    EVIDENCE_TIERS,
    FEATURE_JOIN_KEY,
    MANIFEST_COLUMNS,
    MANIFEST_JOIN_KEY,
    REQUIRED_VALUE_COLUMNS,
    TARGET_LABELS,
    LabelingResult,
)
from .service import (
    build_labeled_datasets,
    read_table,
    run_labeling_pipeline,
    write_report,
    write_table,
)
from .validation import LabelingValidationError

__all__ = [
    "ACCESSION_PATTERN",
    "CONTRACT_VERSION",
    "DATASET_ROLES",
    "DEFAULT_EVIDENCE_SOURCES",
    "DEFAULT_TRAINING_EVIDENCE_TIERS",
    "EVIDENCE_TIERS",
    "FEATURE_JOIN_KEY",
    "LabelingResult",
    "LabelingValidationError",
    "MANIFEST_COLUMNS",
    "MANIFEST_JOIN_KEY",
    "REQUIRED_VALUE_COLUMNS",
    "TARGET_LABELS",
    "build_labeled_datasets",
    "read_table",
    "run_labeling_pipeline",
    "write_report",
    "write_table",
]