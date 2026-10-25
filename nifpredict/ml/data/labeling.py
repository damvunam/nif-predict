"""Backward-compatible imports for the labeling API.

New code should import from ``nifpredict.ml.labeling``.
This module preserves the previous public import path.
"""

from nifpredict.ml.labeling import (
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
    LabelingValidationError,
    build_labeled_datasets,
    read_table,
    run_labeling_pipeline,
    write_report,
    write_table,
)

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