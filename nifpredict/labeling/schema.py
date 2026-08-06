"""Schema, controlled vocabularies, and result types for labeling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

CONTRACT_VERSION = "1.0.0"
FEATURE_JOIN_KEY = "accession_id"
MANIFEST_JOIN_KEY = "assembly_accession"
ACCESSION_PATTERN = re.compile(r"^GC[AF]_\d{9}\.\d+$")

MANIFEST_COLUMNS = (
    "assembly_accession",
    "organism_name",
    "strain",
    "target_label",
    "evidence_tier",
    "evidence_source",
    "evidence_reference",
    "taxonomy_group",
    "source_genome_id",
    "dataset_role",
    "label_curator",
    "label_date",
    "notes",
)

REQUIRED_VALUE_COLUMNS = (
    "assembly_accession",
    "organism_name",
    "target_label",
    "evidence_tier",
    "evidence_source",
    "source_genome_id",
    "dataset_role",
    "label_curator",
    "label_date",
)

TARGET_LABELS = frozenset({"positive", "negative", "uncertain"})
EVIDENCE_TIERS = frozenset({"A", "B", "C"})

DEFAULT_EVIDENCE_SOURCES = frozenset(
    {
        "literature",
        "curated_database",
        "experimental_record",
        "gene_inference",
    }
)

DATASET_ROLES = frozenset(
    {
        "train_candidate",
        "external_test",
        "acceptance_test",
        "edge_case",
    }
)

DEFAULT_TRAINING_EVIDENCE_TIERS = frozenset({"A", "B"})


@dataclass(frozen=True)
class LabelingResult:
    """Validated outputs and their machine-readable validation report."""

    full_dataset: pd.DataFrame
    training_dataset: pd.DataFrame
    report: dict[str, Any]