"""Feature matrix workflow components."""

from .extraction import extract_raw_records
from .inputs import index_genome_paths, load_accessions, prepare_input_files
from .output import build_feature_matrix, write_feature_matrix, write_run_metadata

__all__ = [
    "build_feature_matrix",
    "extract_raw_records",
    "index_genome_paths",
    "load_accessions",
    "prepare_input_files",
    "write_feature_matrix",
    "write_run_metadata",
]
