"""Near-duplicate protein-to-SEED screening for accepted HMM hits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from Bio.Align import Alignment, PairwiseAligner, substitution_matrices

from nifpredict.ml.evaluation._hmm_panel_files import (
    _coerce_roots,
    _resolve_unique_model_file,
    parse_stockholm,
)
from nifpredict.ml.evaluation._hmm_panel_types import (
    AcceptedProteinHit,
    HMMPanelAuditError,
    HMMPanelSpec,
    MultiModelAcceptedProtein,
    NearDuplicateComparison,
    NearDuplicateLeakageAudit,
    NearDuplicatePolicy,
    PairwiseSimilarity,
    StockholmRecord,
)


def _build_local_protein_aligner(
    policy: NearDuplicatePolicy,
) -> tuple[PairwiseAligner, frozenset[str]]:
    """Build the declared Smith-Waterman protein alignment configuration."""
    try:
        matrix = substitution_matrices.load(policy.substitution_matrix)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HMMPanelAuditError(
            f"Cannot load substitution matrix '{policy.substitution_matrix}'"
        ) from error

    matrix_alphabet = getattr(matrix, "alphabet", None)
    if not matrix_alphabet:
        raise HMMPanelAuditError(
            f"Substitution matrix '{policy.substitution_matrix}' has no alphabet"
        )

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = policy.open_gap_score
    aligner.extend_gap_score = policy.extend_gap_score
    return aligner, frozenset(matrix_alphabet)


def _normalize_alignment_sequence(
    sequence: str,
    *,
    label: str,
    matrix_alphabet: frozenset[str],
) -> str:
    """Normalize a protein sequence and reject residues absent from the matrix."""
    if not isinstance(sequence, str):
        raise HMMPanelAuditError(f"{label} must be a protein string")
    normalized = sequence.strip().upper()
    if not normalized or not normalized.isalpha():
        raise HMMPanelAuditError(f"{label} must contain alphabetic residues only")
    unsupported = sorted(set(normalized) - matrix_alphabet)
    if unsupported:
        raise HMMPanelAuditError(
            f"{label} contains residue(s) unsupported by the substitution matrix: "
            + ", ".join(unsupported)
        )
    return normalized


def _alignment_metrics(
    alignment: Alignment,
    query_sequence: str,
    seed_sequence: str,
) -> PairwiseSimilarity:
    """Calculate identity and bidirectional coverage from alignment coordinates."""
    coordinates = alignment.coordinates
    alignment_columns = sum(
        max(
            int(coordinates[0, index + 1] - coordinates[0, index]),
            int(coordinates[1, index + 1] - coordinates[1, index]),
        )
        for index in range(coordinates.shape[1] - 1)
    )
    if alignment_columns < 1:
        raise HMMPanelAuditError("Local alignment produced no alignment columns")

    query_blocks, seed_blocks = alignment.aligned
    matches = 0
    for query_block, seed_block in zip(query_blocks, seed_blocks, strict=True):
        query_start, query_end = (int(value) for value in query_block)
        seed_start, seed_end = (int(value) for value in seed_block)
        query_fragment = query_sequence[query_start:query_end]
        seed_fragment = seed_sequence[seed_start:seed_end]
        if len(query_fragment) != len(seed_fragment):
            raise HMMPanelAuditError(
                "Pairwise alignment returned unequal residue-residue blocks"
            )
        matches += sum(
            query_residue == seed_residue
            for query_residue, seed_residue in zip(
                query_fragment, seed_fragment, strict=True
            )
        )

    query_span = int(coordinates[0, -1] - coordinates[0, 0])
    seed_span = int(coordinates[1, -1] - coordinates[1, 0])
    return PairwiseSimilarity(
        alignment_score=float(alignment.score),
        matches=matches,
        alignment_columns=alignment_columns,
        sequence_identity=100.0 * matches / alignment_columns,
        query_coverage=100.0 * query_span / len(query_sequence),
        seed_coverage=100.0 * seed_span / len(seed_sequence),
    )


def _align_protein_pair(
    query_sequence: str,
    seed_sequence: str,
    *,
    aligner: PairwiseAligner,
    matrix_alphabet: frozenset[str],
) -> PairwiseSimilarity:
    query = _normalize_alignment_sequence(
        query_sequence,
        label="query_sequence",
        matrix_alphabet=matrix_alphabet,
    )
    seed = _normalize_alignment_sequence(
        seed_sequence,
        label="seed_sequence",
        matrix_alphabet=matrix_alphabet,
    )
    alignments = aligner.align(query, seed)
    if len(alignments) == 0:
        return PairwiseSimilarity(
            alignment_score=0.0,
            matches=0,
            alignment_columns=0,
            sequence_identity=0.0,
            query_coverage=0.0,
            seed_coverage=0.0,
        )
    return _alignment_metrics(alignments[0], query, seed)


def calculate_pairwise_similarity(
    query_sequence: str,
    seed_sequence: str,
    *,
    policy: NearDuplicatePolicy,
) -> PairwiseSimilarity:
    """Align one protein pair using the fully declared B4 scoring policy."""
    aligner, matrix_alphabet = _build_local_protein_aligner(policy)
    return _align_protein_pair(
        query_sequence,
        seed_sequence,
        aligner=aligner,
        matrix_alphabet=matrix_alphabet,
    )


def audit_near_duplicate_seed_leakage(
    panel: HMMPanelSpec,
    *,
    seed_roots: Path | str | Iterable[Path | str],
    accepted_hits: Iterable[AcceptedProteinHit],
    policy: NearDuplicatePolicy,
) -> NearDuplicateLeakageAudit:
    """Compare accepted hits only with SEED sequences from the matched model."""
    hits = tuple(accepted_hits)
    if not hits:
        raise HMMPanelAuditError("B4 requires at least one accepted protein hit")
    if any(not isinstance(hit, AcceptedProteinHit) for hit in hits):
        raise TypeError("accepted_hits must contain AcceptedProteinHit records")

    models_by_role = {model.role: model for model in panel.models}
    seed_directories = _coerce_roots(seed_roots)
    seeds_by_role: dict[str, tuple[StockholmRecord, ...]] = {}
    for model in panel.models:
        seed_path = _resolve_unique_model_file(
            seed_directories, model.accession, "SEED"
        )
        records = parse_stockholm(seed_path)
        if len(records) != model.seed_sequence_count:
            raise HMMPanelAuditError(
                f"SEED count mismatch for {model.role}: expected "
                f"{model.seed_sequence_count}, observed {len(records)}"
            )
        seeds_by_role[model.role] = records

    seen_assignments: set[tuple[str, str, str]] = set()
    sequences_by_protein: dict[tuple[str, str], str] = {}
    roles_by_protein: dict[tuple[str, str], set[str]] = defaultdict(set)
    for hit in hits:
        if hit.model_role not in models_by_role:
            raise HMMPanelAuditError(
                f"Accepted hit references unknown model role '{hit.model_role}'"
            )
        assignment = (hit.accession, hit.protein_id, hit.model_role)
        if assignment in seen_assignments:
            raise HMMPanelAuditError(
                "Duplicate accepted-hit assignment: " + "/".join(assignment)
            )
        seen_assignments.add(assignment)

        protein_key = (hit.accession, hit.protein_id)
        previous_sequence = sequences_by_protein.setdefault(protein_key, hit.sequence)
        if previous_sequence != hit.sequence:
            raise HMMPanelAuditError(
                "Conflicting sequences for accepted protein "
                f"{hit.accession}/{hit.protein_id}"
            )
        roles_by_protein[protein_key].add(hit.model_role)

    aligner, matrix_alphabet = _build_local_protein_aligner(policy)
    comparisons: list[NearDuplicateComparison] = []
    for hit in hits:
        model = models_by_role[hit.model_role]
        for seed in seeds_by_role[hit.model_role]:
            similarity = _align_protein_pair(
                hit.sequence,
                seed.sequence,
                aligner=aligner,
                matrix_alphabet=matrix_alphabet,
            )
            flagged = (
                similarity.sequence_identity >= policy.min_sequence_identity
                and similarity.query_coverage >= policy.min_query_coverage
                and similarity.seed_coverage >= policy.min_seed_coverage
            )
            comparisons.append(
                NearDuplicateComparison(
                    accession=hit.accession,
                    protein_id=hit.protein_id,
                    model_role=hit.model_role,
                    model_accession=model.accession,
                    seed_identifier=seed.identifier,
                    protein_length=len(hit.sequence),
                    seed_length=len(seed.sequence),
                    alignment_score=similarity.alignment_score,
                    matches=similarity.matches,
                    alignment_columns=similarity.alignment_columns,
                    sequence_identity=similarity.sequence_identity,
                    query_coverage=similarity.query_coverage,
                    seed_coverage=similarity.seed_coverage,
                    exact_sequence_match=hit.sequence == seed.sequence,
                    flagged=flagged,
                )
            )

    multi_model_proteins = tuple(
        MultiModelAcceptedProtein(
            accession=accession,
            protein_id=protein_id,
            model_roles=tuple(sorted(roles)),
        )
        for (accession, protein_id), roles in sorted(roles_by_protein.items())
        if len(roles) > 1
    )
    return NearDuplicateLeakageAudit(
        policy=policy,
        accepted_hit_count=len(hits),
        unique_protein_count=len(sequences_by_protein),
        comparisons=tuple(comparisons),
        multi_model_proteins=multi_model_proteins,
    )
