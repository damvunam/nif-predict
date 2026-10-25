"""Stable TSV and JSON serialization for HMM-panel scan results."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from nifpredict.ml.evaluation._hmm_panel_files import sha256_file
from nifpredict.ml.evaluation._hmm_panel_scan_types import HIT_COLUMNS, FastaScanResult, PanelScanHit
from nifpredict.ml.evaluation._hmm_panel_types import HMMPanelSpec


def write_hits(path: Path, hits: Sequence[PanelScanHit]) -> None:
    """Write hit rows with the same schema for empty and non-empty results."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HIT_COLUMNS, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(asdict(hit) for hit in hits)


def build_run_manifest(
    *,
    panel: HMMPanelSpec,
    panel_manifest_path: Path,
    panel_input_audit_path: Path,
    model_paths: Sequence[Path],
    scans: Sequence[FastaScanResult],
    pyhmmer_version: str,
    cpus: int,
) -> dict[str, object]:
    """Build deterministic run provenance and summary counts."""
    all_hits = tuple(hit for scan in scans for hit in scan.hits)
    roles_by_protein: dict[tuple[str, str], set[str]] = defaultdict(set)
    for hit in all_hits:
        roles_by_protein[(hit.accession, hit.protein_id)].add(hit.model_role)
    return {
        "schema_version": 1,
        "panel_id": panel.panel_id,
        "panel_release": panel.release_name,
        "cutoff_policy": "trusted",
        "additional_hit_thresholds": [],
        "panel_manifest_path": str(panel_manifest_path),
        "panel_manifest_sha256": sha256_file(panel_manifest_path),
        "panel_input_audit_path": str(panel_input_audit_path),
        "panel_input_audit_sha256": sha256_file(panel_input_audit_path),
        "pyhmmer_version": pyhmmer_version,
        "cpus": cpus,
        "genome_count": len(scans),
        "protein_count": sum(scan.protein_count for scan in scans),
        "accepted_hit_count": len(all_hits),
        "unique_hit_protein_count": len(roles_by_protein),
        "multi_model_protein_count": sum(
            len(roles) > 1 for roles in roles_by_protein.values()
        ),
        "models": [
            {
                "role": model.role,
                "accession": model.accession,
                "name": model.hmm_name,
                "hmm_path": str(path),
                "hmm_sha256": sha256_file(path),
            }
            for model, path in zip(panel.models, model_paths, strict=True)
        ],
        "protein_inputs": [
            {
                "accession": scan.accession,
                "path": str(scan.protein_path),
                "sha256": scan.protein_sha256,
                "protein_count": scan.protein_count,
                "accepted_hit_count": len(scan.hits),
            }
            for scan in scans
        ],
    }
