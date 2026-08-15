"""Trusted-cutoff protein scanning for a validated HMM panel."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ._hmm_panel_files import iter_fasta_records, sequence_sha256, sha256_file
from ._hmm_panel_inputs import audit_panel_inputs
from ._hmm_panel_manifest import load_panel_spec
from ._hmm_panel_scan_report import build_run_manifest, write_hits
from ._hmm_panel_scan_types import (
    HIT_COLUMNS,
    FastaScanResult,
    PanelScanHit,
    PanelScanResult,
)
from ._hmm_panel_types import HMMPanelAuditError, HMMPanelSpec

ACCESSION_PATTERN = re.compile(r"^(?:GCA|GCF)_\d+\.\d+$")


def _validate_accession(accession: str) -> str:
    if ACCESSION_PATTERN.fullmatch(accession) is None:
        raise HMMPanelAuditError(f"Invalid assembly accession: {accession!r}")
    return accession


def load_scan_accessions(path: Path | str) -> tuple[str, ...]:
    """Load a non-empty, duplicate-free accession list."""
    accession_path = Path(path)
    try:
        accessions = tuple(
            line.strip()
            for line in accession_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as error:
        raise HMMPanelAuditError(
            f"Cannot read accession list '{accession_path}': {error}"
        ) from error
    if not accessions:
        raise HMMPanelAuditError("Accession list contains no accessions")
    for accession in accessions:
        _validate_accession(accession)
    duplicates = sorted(
        accession for accession in set(accessions) if accessions.count(accession) > 1
    )
    if duplicates:
        raise HMMPanelAuditError(
            "Accession list contains duplicate values: " + ", ".join(duplicates)
        )
    return accessions


def resolve_protein_fastas(
    accessions: Sequence[str], protein_directory: Path | str
) -> tuple[tuple[str, Path], ...]:
    """Resolve one standardized uncompressed protein FASTA per accession."""
    directory = Path(protein_directory)
    if not directory.is_dir():
        raise HMMPanelAuditError(f"Protein FASTA directory does not exist: {directory}")
    resolved: list[tuple[str, Path]] = []
    for accession in accessions:
        _validate_accession(accession)
        path = directory / f"{accession}_protein.faa"
        if not path.is_file() or path.stat().st_size == 0:
            raise HMMPanelAuditError(
                f"Protein FASTA is missing or empty for {accession}: {path}"
            )
        resolved.append((accession, path.resolve()))
    return tuple(resolved)


def _require_verified_input_audit(
    path: Path,
    *,
    panel: HMMPanelSpec,
    manifest_sha256: str,
) -> None:
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HMMPanelAuditError(
            f"Cannot read panel input audit '{path}': {error}"
        ) from error
    required = {
        "panel_id": panel.panel_id,
        "manifest_sha256": manifest_sha256,
        "model_count": len(panel.models),
        "valid_model_count": len(panel.models),
        "invalid_model_count": 0,
        "normalized_check_enabled": True,
        "hmmstat_check_enabled": True,
    }
    mismatches = [
        f"{key}: expected {expected!r}, observed {audit.get(key)!r}"
        for key, expected in required.items()
        if audit.get(key) != expected
    ]
    if mismatches:
        raise HMMPanelAuditError(
            "Panel input audit is not valid for this scan: " + "; ".join(mismatches)
        )


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_single_query(path: Path) -> Any:
    try:
        from pyhmmer.plan7 import HMMFile
    except ImportError as error:
        raise HMMPanelAuditError("PyHMMER is required for panel scanning") from error
    with HMMFile(path) as hmm_file:
        queries = list(hmm_file)
    if len(queries) != 1:
        raise HMMPanelAuditError(
            f"Expected exactly one model in '{path}', observed {len(queries)}"
        )
    return queries[0]


def _scan_one_fasta(
    panel: HMMPanelSpec,
    model_paths: Sequence[Path],
    accession: str,
    protein_path: Path,
    cpus: int,
) -> tuple[FastaScanResult, str]:
    try:
        import pyhmmer
        from pyhmmer.easel import Alphabet, SequenceFile
    except ImportError as error:
        raise HMMPanelAuditError("PyHMMER is required for panel scanning") from error

    proteins = tuple(iter_fasta_records((protein_path,)))
    protein_index: dict[str, tuple[int, str]] = {}
    for protein in proteins:
        if protein.identifier in protein_index:
            raise HMMPanelAuditError(
                f"Duplicate protein identifier in '{protein_path}': "
                f"{protein.identifier}"
            )
        protein_index[protein.identifier] = (
            len(protein.sequence),
            sequence_sha256(protein.sequence),
        )

    alphabet = Alphabet.amino()
    with SequenceFile(protein_path, digital=True, alphabet=alphabet) as sequence_file:
        sequences = sequence_file.read_block()
    queries = [_load_single_query(path) for path in model_paths]
    top_hits_collection = list(
        pyhmmer.hmmer.hmmsearch(
            queries=queries,
            sequences=sequences,
            cpus=cpus,
            bit_cutoffs="trusted",
        )
    )

    hits: list[PanelScanHit] = []
    for model, top_hits in zip(panel.models, top_hits_collection, strict=True):
        for hit in top_hits:
            protein_id = _as_text(hit.name)
            if protein_id not in protein_index:
                raise HMMPanelAuditError(
                    f"PyHMMER returned an unknown protein identifier: {protein_id}"
                )
            protein_length, protein_digest = protein_index[protein_id]
            best_domain = hit.best_domain
            hits.append(
                PanelScanHit(
                    accession=accession,
                    protein_id=protein_id,
                    protein_source=str(protein_path),
                    protein_length=protein_length,
                    protein_sha256=protein_digest,
                    model_role=model.role,
                    model_accession=model.accession,
                    model_name=model.hmm_name,
                    sequence_score=float(hit.score),
                    sequence_evalue=float(hit.evalue),
                    best_domain_score=float(best_domain.score),
                    best_domain_i_evalue=float(best_domain.i_evalue),
                )
            )
    model_order = {model.role: index for index, model in enumerate(panel.models)}
    ordered_hits = tuple(
        sorted(hits, key=lambda hit: (model_order[hit.model_role], hit.protein_id))
    )
    return (
        FastaScanResult(
            accession=accession,
            protein_path=protein_path,
            protein_sha256=sha256_file(protein_path),
            protein_count=len(proteins),
            hits=ordered_hits,
        ),
        pyhmmer.__version__,
    )


def run_panel_scan(
    *,
    panel_manifest_path: Path | str,
    panel_root: Path | str,
    panel_input_audit_path: Path | str,
    protein_fastas: Sequence[tuple[str, Path]],
    output_directory: Path | str,
    cpus: int = 1,
    progress: Callable[[str, int, int, int], None] | None = None,
) -> PanelScanResult:
    """Scan all FASTAs atomically using only panel trusted cutoffs."""
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise HMMPanelAuditError("cpus must be a positive integer")
    if not protein_fastas:
        raise HMMPanelAuditError("At least one protein FASTA is required")
    accessions = tuple(accession for accession, _ in protein_fastas)
    for accession in accessions:
        _validate_accession(accession)
    if len(set(accessions)) != len(accessions):
        raise HMMPanelAuditError("Protein FASTA inputs contain duplicate accessions")

    manifest_path = Path(panel_manifest_path).resolve()
    root = Path(panel_root).resolve()
    input_audit_path = Path(panel_input_audit_path).resolve()
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise FileExistsError(f"Panel scan output already exists: {output_path}")

    panel = load_panel_spec(manifest_path)
    manifest_digest = sha256_file(manifest_path)
    _require_verified_input_audit(
        input_audit_path,
        panel=panel,
        manifest_sha256=manifest_digest,
    )
    audits = audit_panel_inputs(
        panel,
        hmm_roots=root / "models",
        seed_roots=root / "seeds",
    )
    if len(audits) != len(panel.models):
        raise HMMPanelAuditError(
            "Panel input audit returned an unexpected number of model results"
        )
    invalid = tuple(audit for audit in audits if not audit.valid)
    if invalid:
        raise HMMPanelAuditError(
            f"{len(invalid)} panel model input(s) failed validation"
        )
    model_paths = tuple(
        root / "models" / f"{model.accession}.HMM" for model in panel.models
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        hits_directory = temporary_path / "hits"
        hits_directory.mkdir()
        scans: list[FastaScanResult] = []
        pyhmmer_version: str | None = None
        total = len(protein_fastas)
        for index, (accession, protein_path) in enumerate(protein_fastas, start=1):
            scan, observed_version = _scan_one_fasta(
                panel,
                model_paths,
                accession,
                Path(protein_path).resolve(),
                cpus,
            )
            if pyhmmer_version is None:
                pyhmmer_version = observed_version
            elif pyhmmer_version != observed_version:
                raise HMMPanelAuditError("PyHMMER version changed during the scan")
            scans.append(scan)
            write_hits(hits_directory / f"{accession}_hmm_hits.tsv", scan.hits)
            if progress is not None:
                progress(accession, index, total, len(scan.hits))

        all_hits = tuple(hit for scan in scans for hit in scan.hits)
        write_hits(temporary_path / "accepted_hits.tsv", all_hits)
        run_manifest = build_run_manifest(
            panel=panel,
            panel_manifest_path=manifest_path,
            panel_input_audit_path=input_audit_path,
            model_paths=model_paths,
            scans=scans,
            pyhmmer_version=pyhmmer_version or "unknown",
            cpus=cpus,
        )
        (temporary_path / "scan_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_path.exists():
            raise FileExistsError(f"Panel scan output already exists: {output_path}")
        temporary_path.replace(output_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise

    return PanelScanResult(
        output_directory=output_path,
        manifest_path=output_path / "scan_manifest.json",
        accepted_hits_path=output_path / "accepted_hits.tsv",
        genome_count=int(run_manifest["genome_count"]),
        protein_count=int(run_manifest["protein_count"]),
        accepted_hit_count=int(run_manifest["accepted_hit_count"]),
        unique_hit_protein_count=int(run_manifest["unique_hit_protein_count"]),
        multi_model_protein_count=int(run_manifest["multi_model_protein_count"]),
    )


__all__ = [
    "HIT_COLUMNS",
    "FastaScanResult",
    "PanelScanHit",
    "PanelScanResult",
    "load_scan_accessions",
    "resolve_protein_fastas",
    "run_panel_scan",
]
