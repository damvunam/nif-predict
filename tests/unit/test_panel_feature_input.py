"""Unit tests for the trusted panel-to-feature bridge."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from nifpredict.ml.features.gff_mapping import parse_gff3
from nifpredict.ml.workflows.feature_matrix.panel_hits import (
    FEATURE_HIT_COLUMNS,
    PANEL_HIT_COLUMNS,
    PanelFeatureInputError,
    adapt_panel_hits,
    build_panel_raw_records,
    join_panel_hits_with_gff,
    load_panel_feature_input,
)


GCF_ACCESSION = "GCF_000000001.1"
GCA_ACCESSION = "GCA_000000002.1"


def _model(role: str, accession: str, name: str) -> dict[str, str]:
    return {
        "role": role,
        "accession": accession,
        "name": name,
        "hmm_path": f"models/{accession}.HMM",
        "hmm_sha256": "a" * 64,
    }


def _hit(
    role: str = "nifD",
    *,
    accession: str = GCF_ACCESSION,
    protein_id: str = "P1",
) -> dict[str, object]:
    model_accession, model_name = {
        "nifD": ("TIGR01282.1", "nifD"),
        "nitrogenase_reductase": ("TIGR01287.1", "nifH"),
    }[role]
    return {
        "accession": accession,
        "protein_id": protein_id,
        "protein_source": f"{accession}_protein.faa",
        "protein_length": 400,
        "protein_sha256": "b" * 64,
        "model_role": role,
        "model_accession": model_accession,
        "model_name": model_name,
        "sequence_score": 500.0,
        "sequence_evalue": 1e-100,
        "best_domain_score": 490.0,
        "best_domain_i_evalue": 1e-95,
        "accepted_by": "trusted",
    }


def _write_scan(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    manifest_updates: dict[str, object] | None = None,
) -> Path:
    rows = rows if rows is not None else [
        _hit("nitrogenase_reductase"),
        _hit("nifD"),
    ]
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    roles_by_protein: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (str(row["accession"]), str(row["protein_id"]))
        roles_by_protein.setdefault(key, set()).add(str(row["model_role"]))

    counts = {
        accession: sum(row["accession"] == accession for row in rows)
        for accession in (GCF_ACCESSION, GCA_ACCESSION)
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "panel_id": "test_panel",
        "panel_release": "test/1.0",
        "cutoff_policy": "trusted",
        "additional_hit_thresholds": [],
        "panel_manifest_path": "panel.yaml",
        "panel_manifest_sha256": "c" * 64,
        "panel_input_audit_path": "input_audit.json",
        "panel_input_audit_sha256": "d" * 64,
        "pyhmmer_version": "test",
        "cpus": 1,
        "genome_count": 2,
        "protein_count": 2,
        "accepted_hit_count": len(rows),
        "unique_hit_protein_count": len(roles_by_protein),
        "multi_model_protein_count": sum(
            len(roles) > 1 for roles in roles_by_protein.values()
        ),
        "models": [
            _model("nifD", "TIGR01282.1", "nifD"),
            _model("nitrogenase_reductase", "TIGR01287.1", "nifH"),
        ],
        "protein_inputs": [
            {
                "accession": GCF_ACCESSION,
                "path": f"{GCF_ACCESSION}_protein.faa",
                "sha256": "e" * 64,
                "protein_count": 1,
                "accepted_hit_count": counts[GCF_ACCESSION],
            },
            {
                "accession": GCA_ACCESSION,
                "path": f"{GCA_ACCESSION}_protein.faa",
                "sha256": "f" * 64,
                "protein_count": 1,
                "accepted_hit_count": counts[GCA_ACCESSION],
            },
        ],
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    (scan_dir / "scan_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with (scan_dir / "accepted_hits.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PANEL_HIT_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return scan_dir


def _feature_hit(protein_id: str, role: str = "nifD") -> pd.DataFrame:
    return pd.DataFrame(
        [[protein_id, role, 500.0, 1e-100]], columns=FEATURE_HIT_COLUMNS
    )


def test_loads_valid_scan_and_preserves_multi_model_assignment(tmp_path: Path) -> None:
    panel_input = load_panel_feature_input(_write_scan(tmp_path))

    assert panel_input.accessions == (GCF_ACCESSION, GCA_ACCESSION)
    assert panel_input.hits["model_role"].tolist() == [
        "nifD",
        "nitrogenase_reductase",
    ]
    assert panel_input.manifest["multi_model_protein_count"] == 1


@pytest.mark.parametrize("missing_name", ["scan_manifest.json", "accepted_hits.tsv"])
def test_requires_both_scan_artifacts(tmp_path: Path, missing_name: str) -> None:
    scan_dir = _write_scan(tmp_path)
    (scan_dir / missing_name).unlink()

    with pytest.raises(PanelFeatureInputError, match="Missing"):
        load_panel_feature_input(scan_dir)


def test_rejects_unsupported_scan_schema(tmp_path: Path) -> None:
    scan_dir = _write_scan(tmp_path, manifest_updates={"schema_version": 2})

    with pytest.raises(PanelFeatureInputError, match="Unsupported scan schema"):
        load_panel_feature_input(scan_dir)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"cutoff_policy": "gathering"}, "cutoff_policy"),
        ({"additional_hit_thresholds": ["E=1e-5"]}, "additional_hit_thresholds"),
    ],
)
def test_requires_unmodified_trusted_cutoff_policy(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    scan_dir = _write_scan(tmp_path, manifest_updates=updates)

    with pytest.raises(PanelFeatureInputError, match=message):
        load_panel_feature_input(scan_dir)


def test_rejects_wrong_tsv_header(tmp_path: Path) -> None:
    scan_dir = _write_scan(tmp_path)
    (scan_dir / "accepted_hits.tsv").write_text(
        "accession\tprotein_id\n", encoding="utf-8"
    )

    with pytest.raises(PanelFeatureInputError, match="header"):
        load_panel_feature_input(scan_dir)


def test_rejects_manifest_hit_count_mismatch(tmp_path: Path) -> None:
    scan_dir = _write_scan(tmp_path, manifest_updates={"accepted_hit_count": 3})

    with pytest.raises(PanelFeatureInputError, match="accepted_hit_count"):
        load_panel_feature_input(scan_dir)


def test_rejects_unknown_model_role(tmp_path: Path) -> None:
    row = _hit()
    row["model_role"] = "unknown"
    scan_dir = _write_scan(tmp_path, rows=[row])

    with pytest.raises(PanelFeatureInputError, match="unknown model role"):
        load_panel_feature_input(scan_dir)


def test_rejects_duplicate_exact_assignment(tmp_path: Path) -> None:
    row = _hit()
    scan_dir = _write_scan(tmp_path, rows=[row, row.copy()])

    with pytest.raises(PanelFeatureInputError, match="Duplicate accepted-hit"):
        load_panel_feature_input(scan_dir)


def test_zero_hit_accession_has_stable_empty_feature_schema(tmp_path: Path) -> None:
    panel_input = load_panel_feature_input(_write_scan(tmp_path))

    empty = panel_input.feature_hits_for_accession(GCA_ACCESSION)

    assert empty.empty
    assert tuple(empty.columns) == FEATURE_HIT_COLUMNS


def test_adapter_maps_only_feature_columns_and_preserves_role(tmp_path: Path) -> None:
    panel_input = load_panel_feature_input(_write_scan(tmp_path))

    adapted = adapt_panel_hits(panel_input.hits)

    assert tuple(adapted.columns) == FEATURE_HIT_COLUMNS
    assert "accession" not in adapted
    assert "protein_source" not in adapted
    assert "nitrogenase_reductase" in set(adapted["gene_family"])
    assert "nifH" not in set(adapted["gene_family"])


def test_loader_output_order_is_deterministic(tmp_path: Path) -> None:
    rows = [
        _hit("nitrogenase_reductase", protein_id="P2"),
        _hit("nifD", protein_id="P2"),
        _hit("nifD", protein_id="P1"),
    ]

    panel_input = load_panel_feature_input(_write_scan(tmp_path, rows=rows))

    assert list(
        panel_input.hits[["model_role", "protein_id"]].itertuples(
            index=False, name=None
        )
    ) == [
        ("nifD", "P1"),
        ("nifD", "P2"),
        ("nitrogenase_reductase", "P2"),
    ]


def test_ncbi_protein_id_maps_to_gff(tmp_path: Path) -> None:
    gff_path = tmp_path / "ncbi.gff"
    gff_path.write_text(
        "contig\tRefSeq\tCDS\t1\t300\t.\t+\t0\t"
        "ID=cds-WP_000000001.1;Name=WP_000000001.1;"
        "protein_id=WP_000000001.1;locus_tag=LOC1\n",
        encoding="utf-8",
    )

    merged = join_panel_hits_with_gff(
        _feature_hit("WP_000000001.1"), parse_gff3(gff_path)
    )

    assert len(merged) == 1
    assert merged.iloc[0]["seqid"] == "contig"


def test_prodigal_identifier_maps_to_gff(tmp_path: Path) -> None:
    gff_path = tmp_path / "prodigal.gff"
    gff_path.write_text(
        "BCTD01000027.1\tProdigal\tCDS\t1\t300\t.\t+\t0\t"
        "ID=1_45;partial=00;start_type=ATG\n",
        encoding="utf-8",
    )

    merged = join_panel_hits_with_gff(
        _feature_hit("BCTD01000027.1_45"), parse_gff3(gff_path)
    )

    assert len(merged) == 1
    assert merged.iloc[0]["gff_protein_id"] == "BCTD01000027.1_45"


def test_unmatched_gff_mapping_is_not_silently_dropped(tmp_path: Path) -> None:
    gff_path = tmp_path / "unmatched.gff"
    gff_path.write_text(
        "contig\tRefSeq\tCDS\t1\t300\t.\t+\t0\t"
        "ID=cds-PRESENT;protein_id=PRESENT\n",
        encoding="utf-8",
    )

    with pytest.raises(PanelFeatureInputError, match="unmatched"):
        join_panel_hits_with_gff(_feature_hit("ABSENT"), parse_gff3(gff_path))


def test_ambiguous_inferred_gff_mapping_is_reported(tmp_path: Path) -> None:
    gff_path = tmp_path / "ambiguous.gff"
    gff_path.write_text(
        "contig\tRefSeq\tCDS\t1\t300\t.\t+\t0\tID=cds-A;locus_tag=AMB\n"
        "contig\tRefSeq\tCDS\t401\t700\t.\t+\t0\tID=cds-B;locus_tag=AMB\n",
        encoding="utf-8",
    )

    with pytest.raises(PanelFeatureInputError, match=r"ambiguous=\['AMB'\]"):
        join_panel_hits_with_gff(_feature_hit("AMB"), parse_gff3(gff_path))


def test_build_raw_records_keeps_zero_hit_genome(tmp_path: Path) -> None:
    panel_input = load_panel_feature_input(
        _write_scan(tmp_path, rows=[_hit("nifD", protein_id="P1")])
    )
    gcf_gff = tmp_path / "gcf.gff"
    gcf_gff.write_text(
        "contig\tRefSeq\tCDS\t1\t300\t.\t+\t0\tID=cds-P1;protein_id=P1\n",
        encoding="utf-8",
    )
    gca_gff = tmp_path / "gca.gff"
    gca_gff.write_text(
        "contig\tRefSeq\tCDS\t1\t300\t.\t+\t0\tID=cds-Q1;protein_id=Q1\n",
        encoding="utf-8",
    )

    class FakeClusterFilter:
        def group_into_clusters(
            self, hits: pd.DataFrame, gff: pd.DataFrame
        ) -> list[dict[str, object]]:
            assert not hits.empty
            assert not gff.empty
            return []

    records = build_panel_raw_records(
        panel_input,
        {GCF_ACCESSION: gcf_gff, GCA_ACCESSION: gca_gff},
        cluster_filter=FakeClusterFilter(),  # type: ignore[arg-type]
    )

    assert [record["accession"] for record in records] == [
        GCF_ACCESSION,
        GCA_ACCESSION,
    ]
    assert records[0]["pfam_domains"] == ["nifD"]
    assert records[1]["df_all_hits"].empty
    assert records[1]["clusters"] == []
    assert records[1]["status"] == "SUCCESS"
    assert records[1]["metadata"] == {}
