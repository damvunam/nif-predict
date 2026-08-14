"""Shared immutable records for the nitrogenase HMM-panel audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


class HMMPanelAuditError(ValueError):
    """Raised when panel provenance or an audited biological file is invalid."""


class ExternalHMMToolError(RuntimeError):
    """Raised when a required HMMER executable cannot validate a model."""


@dataclass(frozen=True)
class ModelChecksums:
    """Expected SHA-256 digests for one model and its seed alignment."""

    hmm_raw_sha256: str
    hmmconvert_sha256: str
    seed_raw_sha256: str


@dataclass(frozen=True)
class HMMModelSpec:
    """Validated model metadata loaded from the versioned panel manifest."""

    role: str
    accession: str
    source_identifier: str
    hmm_name: str
    gene_symbols: tuple[str, ...]
    hmm_length: int
    sequence_cutoff: float
    domain_cutoff: float
    noise_cutoff: float
    family_type: str
    product: str
    seed_sequence_count: int
    checksums: ModelChecksums


@dataclass(frozen=True)
class HMMPanelSpec:
    """Panel-level provenance and its ordered collection of models."""

    schema_version: int
    panel_id: str
    provider: str
    release_name: str
    release_date: str
    registry_url: str
    model_base_url: str
    seed_base_url: str
    cutoff_policy: str
    normalized_with: str
    models: tuple[HMMModelSpec, ...]


@dataclass(frozen=True)
class StockholmRecord:
    """One aligned and ungapped sequence parsed from a Stockholm file."""

    identifier: str
    aligned_sequence: str
    sequence: str


@dataclass(frozen=True)
class HMMHeader:
    """Relevant metadata parsed from the text header of an HMMER model."""

    name: str
    accession: str
    description: str
    length: int
    sequence_count: int
    gathering_cutoff: tuple[float, float]
    trusted_cutoff: tuple[float, float]
    noise_cutoff: tuple[float, float]


@dataclass(frozen=True)
class ModelInputAudit:
    """Observed validation result for one HMM and its seed alignment."""

    role: str
    accession: str
    hmm_path: Path
    seed_path: Path
    raw_hmm_sha256: str
    normalized_hmm_sha256: str | None
    raw_seed_sha256: str
    seed_sequence_count: int
    hmmstat_valid: bool | None
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Return whether every enabled validation check passed."""
        return not self.errors


@dataclass(frozen=True)
class FastaRecord:
    """A protein FASTA record with its source file."""

    source: Path
    identifier: str
    sequence: str


@dataclass(frozen=True)
class SeedReference:
    """Provenance for one ungapped seed sequence."""

    role: str
    model_accession: str
    seed_identifier: str
    sequence_sha256: str


@dataclass(frozen=True)
class ExactSeedMatch:
    """An exact local-protein match to one or more panel seed sequences."""

    accession: str
    protein_id: str
    protein_source: Path
    sequence_sha256: str
    seed_references: tuple[SeedReference, ...]


@dataclass(frozen=True)
class ExactLeakageAudit:
    """Summary of exact sequence overlap between evaluation proteins and seeds."""

    total_seed_sequences: int
    unique_seed_hashes: int
    cross_model_duplicate_hashes: int
    local_proteins_scanned: int
    matches: tuple[ExactSeedMatch, ...]


@dataclass(frozen=True)
class DownloadedModelInputs:
    """Verified local HMM and SEED paths for one panel model."""

    role: str
    accession: str
    hmm_path: Path
    seed_path: Path
    hmm_downloaded: bool
    seed_downloaded: bool


@dataclass(frozen=True)
class NearDuplicatePolicy:
    """Explicit operational policy for flagging highly similar protein pairs."""

    min_sequence_identity: float
    min_query_coverage: float
    min_seed_coverage: float
    substitution_matrix: str = "BLOSUM62"
    open_gap_score: float = -11.0
    extend_gap_score: float = -1.0

    def __post_init__(self) -> None:
        """Reject implicit or invalid screening parameters."""
        for field_name in (
            "min_sequence_identity",
            "min_query_coverage",
            "min_seed_coverage",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 < value <= 100.0
            ):
                raise HMMPanelAuditError(
                    f"{field_name} must be a finite percentage in (0, 100]"
                )
        if not isinstance(self.substitution_matrix, str) or not (
            self.substitution_matrix.strip()
        ):
            raise HMMPanelAuditError("substitution_matrix must be a non-empty string")
        for field_name in ("open_gap_score", "extend_gap_score"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise HMMPanelAuditError(f"{field_name} must be finite")
        if self.open_gap_score >= 0.0 or self.extend_gap_score > 0.0:
            raise HMMPanelAuditError(
                "open_gap_score must be negative and extend_gap_score non-positive"
            )


@dataclass(frozen=True)
class AcceptedProteinHit:
    """One accepted HMM hit with the protein sequence needed for B4."""

    accession: str
    protein_id: str
    model_role: str
    sequence: str

    def __post_init__(self) -> None:
        """Normalize a valid protein sequence while preserving hit provenance."""
        for field_name in ("accession", "protein_id", "model_role"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise HMMPanelAuditError(
                    f"AcceptedProteinHit.{field_name} must be a non-empty string"
                )
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.sequence, str):
            raise HMMPanelAuditError(
                "AcceptedProteinHit.sequence must be a protein string"
            )
        normalized_sequence = self.sequence.strip().upper()
        if not normalized_sequence or not normalized_sequence.isalpha():
            raise HMMPanelAuditError(
                "AcceptedProteinHit.sequence must contain alphabetic residues only"
            )
        object.__setattr__(self, "sequence", normalized_sequence)


@dataclass(frozen=True)
class PairwiseSimilarity:
    """Metrics from one locally optimal protein alignment."""

    alignment_score: float
    matches: int
    alignment_columns: int
    sequence_identity: float
    query_coverage: float
    seed_coverage: float


@dataclass(frozen=True)
class NearDuplicateComparison:
    """One accepted-hit to same-model SEED comparison."""

    accession: str
    protein_id: str
    model_role: str
    model_accession: str
    seed_identifier: str
    protein_length: int
    seed_length: int
    alignment_score: float
    matches: int
    alignment_columns: int
    sequence_identity: float
    query_coverage: float
    seed_coverage: float
    exact_sequence_match: bool
    flagged: bool


@dataclass(frozen=True)
class MultiModelAcceptedProtein:
    """One accepted protein assigned to more than one panel role."""

    accession: str
    protein_id: str
    model_roles: tuple[str, ...]


@dataclass(frozen=True)
class NearDuplicateLeakageAudit:
    """B4 result without changing samples, labels, or accepted-hit decisions."""

    policy: NearDuplicatePolicy
    accepted_hit_count: int
    unique_protein_count: int
    comparisons: tuple[NearDuplicateComparison, ...]
    multi_model_proteins: tuple[MultiModelAcceptedProtein, ...]

    @property
    def flagged_comparisons(self) -> tuple[NearDuplicateComparison, ...]:
        """Return comparisons meeting every explicit screening threshold."""
        return tuple(
            comparison for comparison in self.comparisons if comparison.flagged
        )
