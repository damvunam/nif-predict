"""Feature matrix workflow components."""

from nifpredict.ml.workflows.feature_matrix.extraction import extract_raw_records
from nifpredict.ml.workflows.feature_matrix.inputs import index_genome_paths, load_accessions, prepare_input_files
from nifpredict.ml.workflows.feature_matrix.output import build_feature_matrix, write_feature_matrix, write_run_metadata
from nifpredict.ml.workflows.feature_matrix.panel_hits import (
    PanelFeatureInput,
    PanelFeatureInputError,
    adapt_panel_hits,
    build_panel_raw_records,
    join_panel_hits_with_gff,
    load_panel_feature_input,
)

__all__ = [
    "build_feature_matrix",
    "extract_raw_records",
    "index_genome_paths",
    "load_accessions",
    "load_panel_feature_input",
    "PanelFeatureInput",
    "PanelFeatureInputError",
    "prepare_input_files",
    "adapt_panel_hits",
    "build_panel_raw_records",
    "join_panel_hits_with_gff",
    "write_feature_matrix",
    "write_run_metadata",
]
