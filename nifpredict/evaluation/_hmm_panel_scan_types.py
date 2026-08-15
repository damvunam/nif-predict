"""Immutable records and stable columns for HMM-panel scans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HIT_COLUMNS = (
    "accession",
    "protein_id",
    "protein_source",
    "protein_length",
    "protein_sha256",
    "model_role",
    "model_accession",
    "model_name",
    "sequence_score",
    "sequence_evalue",
    "best_domain_score",
    "best_domain_i_evalue",
    "accepted_by",
)


@dataclass(frozen=True)
class PanelScanHit:
    """One protein-model assignment accepted by the model's trusted cutoff."""

    accession: str
    protein_id: str
    protein_source: str
    protein_length: int
    protein_sha256: str
    model_role: str
    model_accession: str
    model_name: str
    sequence_score: float
    sequence_evalue: float
    best_domain_score: float
    best_domain_i_evalue: float
    accepted_by: str = "trusted"


@dataclass(frozen=True)
class FastaScanResult:
    """Accepted hits and provenance for one protein FASTA."""

    accession: str
    protein_path: Path
    protein_sha256: str
    protein_count: int
    hits: tuple[PanelScanHit, ...]


@dataclass(frozen=True)
class PanelScanResult:
    """Persisted result of one complete panel scan."""

    output_directory: Path
    manifest_path: Path
    accepted_hits_path: Path
    genome_count: int
    protein_count: int
    accepted_hit_count: int
    unique_hit_protein_count: int
    multi_model_protein_count: int
