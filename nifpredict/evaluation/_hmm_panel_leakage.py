"""Exact protein-to-SEED leakage checks for the HMM panel."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from ._hmm_panel_files import (
    _coerce_roots,
    _resolve_unique_model_file,
    accession_from_fasta_path,
    iter_fasta_records,
    parse_stockholm,
    sequence_sha256,
)
from ._hmm_panel_types import (
    ExactLeakageAudit,
    ExactSeedMatch,
    HMMPanelSpec,
    SeedReference,
)


def build_seed_hash_index(
    panel: HMMPanelSpec,
    seed_roots: Path | str | Iterable[Path | str],
) -> dict[str, tuple[SeedReference, ...]]:
    """Index every ungapped panel seed by its exact sequence digest."""
    roots = _coerce_roots(seed_roots)
    index: dict[str, list[SeedReference]] = defaultdict(list)
    for model in panel.models:
        seed_path = _resolve_unique_model_file(roots, model.accession, "SEED")
        for record in parse_stockholm(seed_path):
            digest = sequence_sha256(record.sequence)
            index[digest].append(
                SeedReference(
                    role=model.role,
                    model_accession=model.accession,
                    seed_identifier=record.identifier,
                    sequence_sha256=digest,
                )
            )
    return {digest: tuple(references) for digest, references in index.items()}


def audit_exact_seed_leakage(
    panel: HMMPanelSpec,
    *,
    seed_roots: Path | str | Iterable[Path | str],
    protein_fastas: Iterable[Path | str],
) -> ExactLeakageAudit:
    """Audit exact sequence overlap without retaining all local proteins in memory."""
    seed_index = build_seed_hash_index(panel, seed_roots)
    matches: list[ExactSeedMatch] = []
    local_count = 0

    for protein in iter_fasta_records(protein_fastas):
        local_count += 1
        digest = sequence_sha256(protein.sequence)
        references = seed_index.get(digest)
        if references:
            matches.append(
                ExactSeedMatch(
                    accession=accession_from_fasta_path(protein.source),
                    protein_id=protein.identifier,
                    protein_source=protein.source,
                    sequence_sha256=digest,
                    seed_references=references,
                )
            )

    total_seed_sequences = sum(len(references) for references in seed_index.values())
    cross_model_duplicates = sum(
        len({reference.role for reference in references}) > 1
        for references in seed_index.values()
    )
    return ExactLeakageAudit(
        total_seed_sequences=total_seed_sequences,
        unique_seed_hashes=len(seed_index),
        cross_model_duplicate_hashes=cross_model_duplicates,
        local_proteins_scanned=local_count,
        matches=tuple(matches),
    )
