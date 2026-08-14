"""Stable public API for reproducible nitrogenase HMM-panel audits.

Implementation is separated by responsibility; existing imports from this module
remain supported.
"""

from __future__ import annotations

from . import _hmm_panel_inputs as _inputs
from ._hmm_panel_files import (
    accession_from_fasta_path,
    hmmstat_valid,
    iter_fasta_records,
    normalized_hmm_sha256,
    parse_hmm_header,
    parse_stockholm,
    sequence_sha256,
    sha256_file,
    ungap_stockholm_sequence,
)
from ._hmm_panel_inputs import audit_panel_inputs, download_panel_inputs
from ._hmm_panel_leakage import (
    audit_exact_seed_leakage,
    build_seed_hash_index,
)
from ._hmm_panel_manifest import load_panel_spec
from ._hmm_panel_similarity import (
    audit_near_duplicate_seed_leakage,
    calculate_pairwise_similarity,
)
from ._hmm_panel_types import (
    AcceptedProteinHit,
    DownloadedModelInputs,
    ExactLeakageAudit,
    ExactSeedMatch,
    ExternalHMMToolError,
    FastaRecord,
    HMMHeader,
    HMMModelSpec,
    HMMPanelAuditError,
    HMMPanelSpec,
    ModelChecksums,
    ModelInputAudit,
    MultiModelAcceptedProtein,
    NearDuplicateComparison,
    NearDuplicateLeakageAudit,
    NearDuplicatePolicy,
    PairwiseSimilarity,
    SeedReference,
    StockholmRecord,
)

# Compatibility hook for existing tests that patch the former module path.
urllib = _inputs.urllib

__all__ = [
    "AcceptedProteinHit",
    "DownloadedModelInputs",
    "ExactLeakageAudit",
    "ExactSeedMatch",
    "ExternalHMMToolError",
    "FastaRecord",
    "HMMHeader",
    "HMMModelSpec",
    "HMMPanelAuditError",
    "HMMPanelSpec",
    "ModelChecksums",
    "ModelInputAudit",
    "MultiModelAcceptedProtein",
    "NearDuplicateComparison",
    "NearDuplicateLeakageAudit",
    "NearDuplicatePolicy",
    "PairwiseSimilarity",
    "SeedReference",
    "StockholmRecord",
    "accession_from_fasta_path",
    "audit_exact_seed_leakage",
    "audit_near_duplicate_seed_leakage",
    "audit_panel_inputs",
    "build_seed_hash_index",
    "calculate_pairwise_similarity",
    "download_panel_inputs",
    "hmmstat_valid",
    "iter_fasta_records",
    "load_panel_spec",
    "normalized_hmm_sha256",
    "parse_hmm_header",
    "parse_stockholm",
    "sequence_sha256",
    "sha256_file",
    "ungap_stockholm_sequence",
]
