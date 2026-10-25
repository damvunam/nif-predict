#!/usr/bin/env python3
"""Command-line entry point for the NifPredict feature matrix workflow."""

import argparse
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from nifpredict.common import paths
from nifpredict.ml.features.cluster_filter import ClusterFilter
from nifpredict.common.config import load_config
from nifpredict.common.logger import get_logger
from nifpredict.ml.workflows.feature_matrix import (
    PanelFeatureInput,
    build_feature_matrix,
    build_panel_raw_records,
    extract_raw_records,
    index_genome_paths,
    load_accessions,
    load_panel_feature_input,
    prepare_input_files,
    write_feature_matrix,
    write_run_metadata,
)
from nifpredict.ml.workflows.feature_matrix.output import get_feature_names


LOGGER = get_logger("nifpredict.scripts.build_feature_matrix")
PROJECT_ROOT = paths.REPO_ROOT
DEFAULT_INPUT_FILE = paths.ML_BATCH_ACCESSIONS
DEFAULT_GENOMES_DIR = paths.ML_RAW_GENOMES
DEFAULT_OUTPUT_PATH = paths.ML_PROCESSED / "feature_matrix.parquet"
DEFAULT_CONFIG_PATH = paths.CONFIG_YAML


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the NifPredict genome feature matrix"
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Accession batch file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--genomes-dir",
        "-g",
        type=Path,
        default=DEFAULT_GENOMES_DIR,
        help=f"Raw genome directory (default: {DEFAULT_GENOMES_DIR})",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output .parquet or .csv file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Run metadata path; panel mode defaults beside the output matrix.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Application configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--panel-scan-dir",
        type=Path,
        help=(
            "Use a validated trusted-cutoff panel scan instead of rerunning "
            "the legacy HMM annotator."
        ),
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Number of genomes submitted to the worker pool at a time",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--no-annotate-missing",
        action="store_true",
        help="Do not generate missing FAA/GFF files with Prodigal",
    )
    parser.add_argument(
        "--prodigal-mode",
        choices=("single", "meta"),
        default="single",
        help="Prodigal training mode used for missing annotations",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments."""
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_metadata_path(args: argparse.Namespace) -> Path:
    if args.metadata_path is not None:
        return args.metadata_path
    if args.panel_scan_dir is not None:
        return args.output_path.with_name(f"{args.output_path.stem}_metadata.json")
    return args.output_path.parent / "feature_names.json"


def _require_new_panel_outputs(output_path: Path, metadata_path: Path) -> None:
    if output_path.resolve() == metadata_path.resolve():
        raise ValueError("Feature matrix and metadata paths must be different")
    existing = [path for path in (output_path, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Panel feature output already exists; choose new output paths: "
            + ", ".join(str(path) for path in existing)
        )


def _validate_panel_cohort(
    requested_accessions: Sequence[str],
    scanned_accessions: Sequence[str],
) -> None:
    requested = tuple(requested_accessions)
    scanned = tuple(scanned_accessions)
    if requested == scanned:
        return

    missing = sorted(set(scanned).difference(requested))
    unexpected = sorted(set(requested).difference(scanned))
    order_only = not missing and not unexpected and len(requested) == len(scanned)
    raise ValueError(
        "Requested accession cohort does not exactly match the panel scan; "
        f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
        f"order_only_mismatch={order_only}"
    )


def _panel_source_provenance(
    panel_input: PanelFeatureInput,
    gff_map: Mapping[str, Path],
    config_path: Path,
    accession_input_path: Path,
) -> dict[str, object]:
    scan_directory = panel_input.scan_directory
    manifest = panel_input.manifest
    manifest_path = scan_directory / "scan_manifest.json"
    hits_path = scan_directory / "accepted_hits.tsv"
    return {
        "input_mode": "trusted_panel_scan",
        "panel_scan_directory": str(scan_directory),
        "scan_manifest_sha256": _sha256_file(manifest_path),
        "accepted_hits_sha256": _sha256_file(hits_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "accession_input_path": str(accession_input_path.resolve()),
        "accession_input_sha256": _sha256_file(accession_input_path),
        "panel_id": manifest["panel_id"],
        "panel_release": manifest["panel_release"],
        "cutoff_policy": manifest["cutoff_policy"],
        "accepted_hit_count": manifest["accepted_hit_count"],
        "unique_hit_protein_count": manifest["unique_hit_protein_count"],
        "multi_model_protein_count": manifest["multi_model_protein_count"],
        "gff_inputs": [
            {
                "accession": accession,
                "path": str(gff_map[accession].resolve()),
                "sha256": _sha256_file(gff_map[accession]),
            }
            for accession in panel_input.accessions
        ],
        "role_compatibility": {
            "nitrogenase_reductase": {
                "feature_group": "marker_core_nifH",
                "interpretation": (
                    "Nitrogenase reductase family (NifH/VnfH/AnfH); "
                    "not an exact nifH gene-symbol assignment."
                ),
            }
        },
    }


def run(args: argparse.Namespace) -> int:
    """Execute the complete feature matrix workflow."""
    validate_args(args)
    metadata_path = _panel_metadata_path(args)
    if args.panel_scan_dir is not None:
        _require_new_panel_outputs(args.output_path, metadata_path)

    config = load_config(args.config, auto_create_dirs=True)
    accessions = load_accessions(args.input_file)
    annotation_dir = Path(config.paths.annotation_dir)

    duplicate_count = len(accessions) - len(set(accessions))
    if duplicate_count:
        LOGGER.warning(
            "The batch contains %d duplicate accession line(s). Each unique "
            "genome will be processed once, then repeated rows will be restored "
            "in input order. Repeated source accessions are not equivalent to "
            "generated synthetic edge-case genomes.",
            duplicate_count,
        )

    faa_map, gff_map, fna_map = index_genome_paths(
        genomes_dir=args.genomes_dir,
        annotation_dir=annotation_dir,
    )
    LOGGER.info(
        "Indexed inputs: FAA=%d, GFF=%d, genomic FASTA=%d",
        len(faa_map),
        len(gff_map),
        len(fna_map),
    )

    source_provenance = None
    if args.panel_scan_dir is not None:
        panel_input = load_panel_feature_input(args.panel_scan_dir)
        _validate_panel_cohort(accessions, panel_input.accessions)
        raw_records = build_panel_raw_records(
            panel_input,
            gff_map,
            cluster_filter=ClusterFilter(config=config),
        )
        failures: dict[str, str] = {}
        source_provenance = _panel_source_provenance(
            panel_input,
            gff_map,
            args.config,
            args.input_file,
        )
    else:
        faa_map, gff_map, preparation_failures = prepare_input_files(
            accessions=accessions,
            faa_map=faa_map,
            gff_map=gff_map,
            fna_map=fna_map,
            annotation_dir=annotation_dir,
            annotate_missing=not args.no_annotate_missing,
            prodigal_mode=args.prodigal_mode,
        )
        raw_records, failures = extract_raw_records(
            accessions=accessions,
            faa_map=faa_map,
            gff_map=gff_map,
            initial_failures=preparation_failures,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )

    feature_frame = build_feature_matrix(raw_records, config)
    write_feature_matrix(feature_frame, args.output_path)
    write_run_metadata(
        metadata_path=metadata_path,
        feature_frame=feature_frame,
        requested_accessions=accessions,
        failures=failures,
        source_provenance=source_provenance,
    )

    LOGGER.info("=== FEATURE MATRIX BUILD SUMMARY ===")
    LOGGER.info("Requested genomes: %d", len(accessions))
    LOGGER.info("Successful genomes: %d", len(feature_frame))
    LOGGER.info("Failed genomes: %d", len(failures))
    LOGGER.info(
        "Feature input mode: %s",
        "trusted_panel_scan" if args.panel_scan_dir is not None else "legacy_hmm",
    )
    LOGGER.info("Feature count: %d", len(get_feature_names(feature_frame)))
    LOGGER.info("Feature matrix: %s", args.output_path)
    LOGGER.info("Run metadata: %s", metadata_path)

    for accession, reason in failures.items():
        LOGGER.error("[%s] %s", accession, reason)
    return 1 if failures else 0


def main() -> None:
    """Parse arguments, execute the workflow, and map errors to exit codes."""
    try:
        exit_code = run(parse_args())
    except Exception as exc:
        LOGGER.exception("Feature matrix build failed: %s", exc)
        exit_code = 2
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
