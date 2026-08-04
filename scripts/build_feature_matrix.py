#!/usr/bin/env python3
"""Command-line entry point for the NifPredict feature matrix workflow."""

import argparse
import os
import sys
from pathlib import Path

from nifpredict.utils.config import load_config
from nifpredict.utils.logger import get_logger
from nifpredict.workflows.feature_matrix import (
    build_feature_matrix,
    extract_raw_records,
    index_genome_paths,
    load_accessions,
    prepare_input_files,
    write_feature_matrix,
    write_run_metadata,
)
from nifpredict.workflows.feature_matrix.output import get_feature_names


LOGGER = get_logger("nifpredict.scripts.build_feature_matrix")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "batch_accessions.txt"
DEFAULT_GENOMES_DIR = PROJECT_ROOT / "data" / "raw" / "genomes"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_matrix.parquet"


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments."""
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")


def run(args: argparse.Namespace) -> int:
    """Execute the complete feature matrix workflow."""
    validate_args(args)
    config = load_config(auto_create_dirs=True)
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
    metadata_path = args.output_path.parent / "feature_names.json"
    write_run_metadata(
        metadata_path=metadata_path,
        feature_frame=feature_frame,
        requested_accessions=accessions,
        failures=failures,
    )

    LOGGER.info("=== FEATURE MATRIX BUILD SUMMARY ===")
    LOGGER.info("Requested genomes: %d", len(accessions))
    LOGGER.info("Successful genomes: %d", len(feature_frame))
    LOGGER.info("Failed genomes: %d", len(failures))
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