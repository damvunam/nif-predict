#!/usr/bin/env python3
"""Scan protein FASTAs with a validated, versioned HMM panel."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nifpredict.common import paths
from nifpredict.ml.evaluation.hmm_panel_scan import (
    load_scan_accessions,
    resolve_protein_fastas,
    run_panel_scan,
)

PROJECT_ROOT = paths.REPO_ROOT
DEFAULT_CONFIG = paths.HMM_PANEL_CONFIG
DEFAULT_PANEL_ROOT = paths.ML_HMM_PANELS / "pgap_19_0"
DEFAULT_PROTEIN_DIRECTORY = paths.ML_RAW_GENOMES
DEFAULT_OUTPUT_DIRECTORY = (
    paths.ML_INTERIM / "hmm_panel_scans" / "pgap_19_0_b2"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin trusted-cutoff panel scan CLI."""
    parser = argparse.ArgumentParser(
        description="Scan protein FASTAs using only panel trusted cutoffs."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEIN_DIRECTORY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument(
        "--panel-input-audit",
        type=Path,
        help="Defaults to PANEL_ROOT/input_audit.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--cpus", type=int, default=4)
    return parser.parse_args(argv)


def _report_progress(accession: str, index: int, total: int, hit_count: int) -> None:
    print(f"[{index}/{total}] {accession}: {hit_count} accepted hit(s)")


def run(args: argparse.Namespace) -> int:
    """Resolve CLI inputs and delegate the scientific scan to the package."""
    accessions = load_scan_accessions(args.input_file)
    protein_fastas = resolve_protein_fastas(accessions, args.protein_dir)
    panel_input_audit = (
        args.panel_input_audit
        if args.panel_input_audit is not None
        else args.panel_root / "input_audit.json"
    )
    result = run_panel_scan(
        panel_manifest_path=args.config,
        panel_root=args.panel_root,
        panel_input_audit_path=panel_input_audit,
        protein_fastas=protein_fastas,
        output_directory=args.output_dir,
        cpus=args.cpus,
        progress=_report_progress,
    )
    print("=== HMM PANEL SCAN SUMMARY ===")
    print(f"Genomes: {result.genome_count}")
    print(f"Proteins: {result.protein_count}")
    print(f"Accepted hits: {result.accepted_hit_count}")
    print(f"Unique hit proteins: {result.unique_hit_protein_count}")
    print(f"Multi-model proteins: {result.multi_model_protein_count}")
    print(f"Output directory: {result.output_directory}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Map actionable scan failures to exit code two."""
    try:
        return run(parse_args(argv))
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"HMM panel scan failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
