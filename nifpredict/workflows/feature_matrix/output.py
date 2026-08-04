"""Matrix transformation and output persistence for feature matrix builds."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from nifpredict.features.feature_extractor import GenomeFeatureExtractor
from nifpredict.utils.config import AppConfig


IDENTIFIER_COLUMNS = {"accession_id", "status"}


def build_feature_matrix(
    raw_records: List[Dict[str, Any]],
    config: AppConfig,
) -> pd.DataFrame:
    """Fit once on all records and create one consistent feature matrix."""
    if not raw_records:
        raise RuntimeError("No successful genome records are available for transformation")

    extractor = GenomeFeatureExtractor(config=config)
    extractor.fit(raw_records)
    feature_frame = extractor.transform(raw_records, return_sparse=False)

    if not isinstance(feature_frame, pd.DataFrame):
        feature_frame = pd.DataFrame(feature_frame)
    if feature_frame.empty or len(feature_frame) != len(raw_records):
        raise RuntimeError(
            "Feature extractor returned an empty matrix or an unexpected row count: "
            f"expected={len(raw_records)}, actual={len(feature_frame)}"
        )

    accessions = [str(record["accession"]) for record in raw_records]
    if "accession_id" in feature_frame.columns:
        feature_frame["accession_id"] = accessions
        if "accession" in feature_frame.columns:
            feature_frame = feature_frame.drop(columns=["accession"])
    elif "accession" in feature_frame.columns:
        feature_frame = feature_frame.rename(columns={"accession": "accession_id"})
        feature_frame["accession_id"] = accessions
    else:
        feature_frame.insert(0, "accession_id", accessions)

    if "status" in feature_frame.columns:
        feature_frame["status"] = "SUCCESS"
    else:
        feature_frame.insert(1, "status", "SUCCESS")

    priority_columns = ["accession_id", "status"]
    feature_columns = sorted(
        column for column in feature_frame.columns if column not in priority_columns
    )
    return feature_frame[priority_columns + feature_columns]


def write_feature_matrix(feature_frame: pd.DataFrame, output_path: Path) -> None:
    """Atomically write a feature matrix as Parquet or CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        if output_path.suffix.lower() == ".parquet":
            feature_frame.to_parquet(temporary_path, engine="pyarrow", index=False)
        elif output_path.suffix.lower() == ".csv":
            feature_frame.to_csv(temporary_path, index=False)
        else:
            raise ValueError(
                f"Unsupported output format '{output_path.suffix}'. Use .parquet or .csv"
            )
        os.replace(temporary_path, output_path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def get_feature_names(feature_frame: pd.DataFrame) -> List[str]:
    """Return matrix columns that contain model features."""
    return [
        column for column in feature_frame.columns if column not in IDENTIFIER_COLUMNS
    ]


def write_run_metadata(
    metadata_path: Path,
    feature_frame: pd.DataFrame,
    requested_accessions: Sequence[str],
    failures: Dict[str, str],
) -> None:
    """Write feature names, counts, and failure details as JSON."""
    feature_names = get_feature_names(feature_frame)
    metadata = {
        "requested_samples": len(requested_accessions),
        "successful_samples": len(feature_frame),
        "failed_samples": len(failures),
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "failed_accessions": failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="ascii") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
