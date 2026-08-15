"""Tests for explicit panel-role compatibility in legacy feature fields."""

from __future__ import annotations

import pandas as pd

from nifpredict.features.cluster_filter import ClusterFilter
from nifpredict.features.feature_extractor import MarkerClusterExtractor


def test_reductase_role_contributes_to_legacy_nifh_feature_group() -> None:
    hits = pd.DataFrame(
        {
            "target_protein": ["P_H"],
            "gene_family": ["nitrogenase_reductase"],
            "effective_score": [500.0],
            "seq_evalue": [1e-100],
        }
    )

    features = MarkerClusterExtractor().extract_single_record(
        "GCF_000000001.1",
        hits,
        [],
    )

    assert features["marker_core_nifH_count"] == 1
    assert features["marker_core_nifH_max_bitscore"] == 500.0
    assert features["PF00142_min_evalue"] == 1e-100
    assert set(hits["gene_family"]) == {"nitrogenase_reductase"}


def test_reductase_role_is_recognized_as_h_family_for_synteny() -> None:
    cluster_filter = ClusterFilter.__new__(ClusterFilter)

    cluster_type, has_catalytic_core, has_any_core = (
        cluster_filter._resolve_cluster_type(
            {"nitrogenase_reductase", "nifD"}
        )
    )

    assert cluster_type == "nif"
    assert has_catalytic_core is True
    assert has_any_core is True
