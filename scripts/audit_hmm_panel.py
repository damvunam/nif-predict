#!/usr/bin/env python3
"""Prepare and validate versioned nitrogenase HMM-panel inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from nifpredict.evaluation.hmm_panel_audit import (
    ExternalHMMToolError,
    HMMPanelAuditError,
    HMMPanelSpec,
    ModelInputAudit,
    audit_panel_inputs,
    download_panel_inputs,
    load_panel_spec,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "hmm_panel_pgap_19.yaml"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "hmm_panels" / "pgap_19_0"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--report",
        type=Path,
        help="Audit JSON path; defaults to DESTINATION/input_audit.json.",
    )
    parser.add_argument(
        "--no-normalized-check",
        action="store_true",
        help="Skip the hmmconvert-based normalized checksum check.",
    )
    parser.add_argument(
        "--no-hmmstat-check",
        action="store_true",
        help="Skip validation with the external hmmstat executable.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin HMM-panel audit command-line interface."""
    parser = argparse.ArgumentParser(
        description="Prepare or validate the versioned nitrogenase HMM panel."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Download manifest-declared inputs and validate them.",
    )
    _add_common_arguments(prepare_parser)
    prepare_parser.add_argument("--timeout", type=float, default=60.0)
    prepare_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate inputs already present in the destination.",
    )
    _add_common_arguments(validate_parser)
    return parser.parse_args(argv)


def _audit_to_mapping(audit: ModelInputAudit) -> dict[str, object]:
    return {
        "role": audit.role,
        "accession": audit.accession,
        "hmm_path": str(audit.hmm_path),
        "seed_path": str(audit.seed_path),
        "raw_hmm_sha256": audit.raw_hmm_sha256,
        "normalized_hmm_sha256": audit.normalized_hmm_sha256,
        "raw_seed_sha256": audit.raw_seed_sha256,
        "seed_sequence_count": audit.seed_sequence_count,
        "hmmstat_valid": audit.hmmstat_valid,
        "valid": audit.valid,
        "errors": list(audit.errors),
    }


def _write_report(
    path: Path,
    *,
    command: str,
    config_path: Path,
    destination: Path,
    panel: HMMPanelSpec,
    audits: tuple[ModelInputAudit, ...],
    downloaded_file_count: int,
    reused_file_count: int,
    normalized_check_enabled: bool,
    hmmstat_check_enabled: bool,
) -> None:
    invalid_count = sum(not audit.valid for audit in audits)
    report = {
        "schema_version": 1,
        "operation": command,
        "manifest_path": str(config_path),
        "manifest_sha256": sha256_file(config_path),
        "destination": str(destination),
        "panel_id": panel.panel_id,
        "provider": panel.provider,
        "release_name": panel.release_name,
        "model_count": len(panel.models),
        "downloaded_file_count": downloaded_file_count,
        "reused_file_count": reused_file_count,
        "normalized_check_enabled": normalized_check_enabled,
        "hmmstat_check_enabled": hmmstat_check_enabled,
        "valid_model_count": len(audits) - invalid_count,
        "invalid_model_count": invalid_count,
        "models": [_audit_to_mapping(audit) for audit in audits],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def run(args: argparse.Namespace) -> int:
    """Translate CLI arguments into package audit operations."""
    config_path = args.config.resolve()
    destination = args.destination.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else destination / "input_audit.json"
    )
    panel = load_panel_spec(config_path)

    downloaded_file_count = 0
    reused_file_count = 0
    if args.command == "prepare":
        downloads = download_panel_inputs(
            panel,
            destination,
            timeout=args.timeout,
            overwrite=args.overwrite,
        )
        downloaded_file_count = sum(
            item.hmm_downloaded + item.seed_downloaded for item in downloads
        )
        reused_file_count = 2 * len(downloads) - downloaded_file_count

    audits = audit_panel_inputs(
        panel,
        hmm_roots=destination / "models",
        seed_roots=destination / "seeds",
        check_normalized=not args.no_normalized_check,
        check_hmmstat=not args.no_hmmstat_check,
    )
    _write_report(
        report_path,
        command=args.command,
        config_path=config_path,
        destination=destination,
        panel=panel,
        audits=audits,
        downloaded_file_count=downloaded_file_count,
        reused_file_count=reused_file_count,
        normalized_check_enabled=not args.no_normalized_check,
        hmmstat_check_enabled=not args.no_hmmstat_check,
    )

    invalid = tuple(audit for audit in audits if not audit.valid)
    if invalid:
        for audit in invalid:
            print(
                f"{audit.role} ({audit.accession}): " + "; ".join(audit.errors),
                file=sys.stderr,
            )
        raise HMMPanelAuditError(
            f"{len(invalid)} of {len(audits)} panel models failed validation"
        )

    print(f"Panel: {panel.panel_id}")
    print(f"Validated models: {len(audits)}")
    if args.command == "prepare":
        print(f"Downloaded files: {downloaded_file_count}")
        print(f"Reused files: {reused_file_count}")
    print(f"Audit report: {report_path}")
    print("Panel input validation: PASS")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Map actionable panel preparation failures to exit code two."""
    try:
        return run(parse_args(argv))
    except (
        ExternalHMMToolError,
        HMMPanelAuditError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"HMM-panel audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
