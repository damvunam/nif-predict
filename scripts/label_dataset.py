#!/usr/bin/env python3
"""Create full and training-ready datasets from an independent label manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nifpredict.labeling import (
    DEFAULT_EVIDENCE_SOURCES,
    LabelingValidationError,
    run_labeling_pipeline,
)
from nifpredict.utils.logger import setup_logger_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "feature_matrix.parquet"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "label_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "labeled_dataset.parquet"
DEFAULT_TRAINING_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "training_dataset.parquet"
)
DEFAULT_REPORT = PROJECT_ROOT / "results" / "labeling" / "label_validation_report.json"
LOGGER = setup_logger_from_config("nifpredict.scripts.label_dataset")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse labeling CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Join a NifPredict feature matrix with independently curated labels. "
            "This command never infers targets from feature values."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-output", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--strict-mismatches",
        action="store_true",
        help="Fail if either input contains an accession absent from the other input",
    )
    parser.add_argument(
        "--training-evidence-tier",
        action="append",
        choices=("A", "B", "C"),
        dest="training_evidence_tiers",
        help="Allowed training tier; repeat to set multiple tiers (default: A and B)",
    )
    parser.add_argument(
        "--additional-evidence-source",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Add a controlled evidence_source value beyond the defaults: "
            + ", ".join(sorted(DEFAULT_EVIDENCE_SOURCES))
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Run the labeling workflow and return a process exit code."""
    training_tiers = args.training_evidence_tiers or ["A", "B"]
    result = run_labeling_pipeline(
        args.features,
        args.labels,
        args.output,
        args.training_output,
        args.report,
        strict_mismatches=args.strict_mismatches,
        training_evidence_tiers=training_tiers,
        additional_evidence_sources=args.additional_evidence_source,
    )
    LOGGER.info("Matched labeled rows: %d", result.report["matched_rows"])
    LOGGER.info("Training-ready rows: %d", result.report["training_ready_rows"])
    LOGGER.info("Full labeled dataset: %s", args.output)
    LOGGER.info("Training-ready dataset: %s", args.training_output)
    LOGGER.info("Validation report: %s", args.report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Map actionable validation and I/O failures to non-zero exit codes."""
    try:
        return run(parse_args(argv))
    except LabelingValidationError as exc:
        LOGGER.error("Label validation failed: %s", exc)
        return 2
    except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
        LOGGER.error("Labeling pipeline failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())