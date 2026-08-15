"""Offline integration tests for trusted panel feature-matrix builds."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from nifpredict.workflows.feature_matrix.panel_hits import PANEL_HIT_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
POSITIVE_ACCESSION = "GCF_000000001.1"
ZERO_HIT_ACCESSION = "GCA_000000002.1"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def _write_config(tmp_path: Path) -> Path:
    config = yaml.safe_load(
        (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    )
    config["paths"]["base_dir"] = str(tmp_path)
    config["computing"]["max_threads"] = 1
    config["logging"]["file_path"] = str(tmp_path / "logs" / "test.log")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _model(role: str, accession: str, name: str) -> dict[str, object]:
    return {
        "role": role,
        "accession": accession,
        "name": name,
        "hmm_path": f"models/{accession}.HMM",
        "hmm_sha256": "a" * 64,
    }


def _hit(
    protein_id: str,
    role: str,
    model_accession: str,
    model_name: str,
) -> dict[str, object]:
    return {
        "accession": POSITIVE_ACCESSION,
        "protein_id": protein_id,
        "protein_source": f"{POSITIVE_ACCESSION}_protein.faa",
        "protein_length": 300,
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


def _write_scan(tmp_path: Path) -> Path:
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()
    models = [
        _model("nifD", "TIGR01282.1", "nifD"),
        _model("nifK", "TIGR01286.1", "nifK"),
        _model("nitrogenase_reductase", "TIGR01287.1", "nifH"),
    ]
    rows = [
        _hit("P_D", "nifD", "TIGR01282.1", "nifD"),
        _hit("P_K", "nifK", "TIGR01286.1", "nifK"),
        _hit(
            "P_H",
            "nitrogenase_reductase",
            "TIGR01287.1",
            "nifH",
        ),
    ]
    manifest = {
        "schema_version": 1,
        "panel_id": "integration_panel",
        "panel_release": "integration/1.0",
        "cutoff_policy": "trusted",
        "additional_hit_thresholds": [],
        "panel_manifest_path": "panel.yaml",
        "panel_manifest_sha256": "c" * 64,
        "panel_input_audit_path": "input_audit.json",
        "panel_input_audit_sha256": "d" * 64,
        "pyhmmer_version": "test",
        "cpus": 1,
        "genome_count": 2,
        "protein_count": 4,
        "accepted_hit_count": 3,
        "unique_hit_protein_count": 3,
        "multi_model_protein_count": 0,
        "models": models,
        "protein_inputs": [
            {
                "accession": POSITIVE_ACCESSION,
                "path": f"{POSITIVE_ACCESSION}_protein.faa",
                "sha256": "e" * 64,
                "protein_count": 3,
                "accepted_hit_count": 3,
            },
            {
                "accession": ZERO_HIT_ACCESSION,
                "path": f"{ZERO_HIT_ACCESSION}_protein.faa",
                "sha256": "f" * 64,
                "protein_count": 1,
                "accepted_hit_count": 0,
            },
        ],
    }
    (scan_directory / "scan_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with (scan_directory / "accepted_hits.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PANEL_HIT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return scan_directory


def _write_genome_inputs(tmp_path: Path) -> Path:
    genomes_directory = tmp_path / "genomes"
    genomes_directory.mkdir()
    (genomes_directory / f"{POSITIVE_ACCESSION}_genomic.gff").write_text(
        "contig\tRefSeq\tCDS\t100\t300\t.\t+\t0\tID=cds-P_H;protein_id=P_H\n"
        "contig\tRefSeq\tCDS\t600\t900\t.\t+\t0\tID=cds-P_D;protein_id=P_D\n"
        "contig\tRefSeq\tCDS\t1200\t1500\t.\t+\t0\tID=cds-P_K;protein_id=P_K\n",
        encoding="utf-8",
    )
    (genomes_directory / f"{ZERO_HIT_ACCESSION}_genomic.gff").write_text(
        "contig\tRefSeq\tCDS\t1\t300\t.\t+\t0\tID=cds-Q1;protein_id=Q1\n",
        encoding="utf-8",
    )
    return genomes_directory


def _write_accessions(tmp_path: Path, *, reverse: bool = False) -> Path:
    accessions = [POSITIVE_ACCESSION, ZERO_HIT_ACCESSION]
    if reverse:
        accessions.reverse()
    path = tmp_path / ("reversed.txt" if reverse else "accessions.txt")
    path.write_text("\n".join(accessions) + "\n", encoding="utf-8")
    return path


def _command(
    *,
    config_path: Path,
    input_path: Path,
    genomes_directory: Path,
    scan_directory: Path,
    output_path: Path,
    metadata_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "build_feature_matrix.py"),
        "--config",
        str(config_path),
        "--input-file",
        str(input_path),
        "--genomes-dir",
        str(genomes_directory),
        "--panel-scan-dir",
        str(scan_directory),
        "--output-path",
        str(output_path),
        "--metadata-path",
        str(metadata_path),
    ]


def test_panel_cli_builds_nonzero_features_and_provenance(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    scan_directory = _write_scan(tmp_path)
    genomes_directory = _write_genome_inputs(tmp_path)
    input_path = _write_accessions(tmp_path)
    output_path = tmp_path / "panel_features.csv"
    metadata_path = tmp_path / "panel_features_metadata.json"

    completed = subprocess.run(
        _command(
            config_path=config_path,
            input_path=input_path,
            genomes_directory=genomes_directory,
            scan_directory=scan_directory,
            output_path=output_path,
            metadata_path=metadata_path,
        ),
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    feature_frame = pd.read_csv(output_path)
    positive = feature_frame.loc[
        feature_frame["accession_id"] == POSITIVE_ACCESSION
    ].iloc[0]
    zero_hit = feature_frame.loc[
        feature_frame["accession_id"] == ZERO_HIT_ACCESSION
    ].iloc[0]
    assert positive["marker_core_nifH_count"] == 1
    assert positive["marker_core_nifD_count"] == 1
    assert positive["marker_core_nifK_count"] == 1
    assert positive["operon_core_completeness_score"] == 1.0
    assert positive["clusters_found_count"] == 1
    assert zero_hit["marker_core_nifH_count"] == 0
    assert zero_hit["marker_core_nifD_count"] == 0
    assert zero_hit["marker_core_nifK_count"] == 0

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = metadata["source_provenance"]
    assert provenance["input_mode"] == "trusted_panel_scan"
    assert provenance["cutoff_policy"] == "trusted"
    assert provenance["accepted_hit_count"] == 3
    assert len(provenance["scan_manifest_sha256"]) == 64
    assert len(provenance["accepted_hits_sha256"]) == 64
    assert len(provenance["config_sha256"]) == 64
    assert len(provenance["accession_input_sha256"]) == 64
    assert [item["accession"] for item in provenance["gff_inputs"]] == [
        POSITIVE_ACCESSION,
        ZERO_HIT_ACCESSION,
    ]
    assert all(len(item["sha256"]) == 64 for item in provenance["gff_inputs"])
    interpretation = provenance["role_compatibility"][
        "nitrogenase_reductase"
    ]["interpretation"]
    assert "not an exact nifH" in interpretation
    hmmer_directory = tmp_path / "data" / "interim" / "hmm_outputs"
    assert not any(path.is_file() for path in hmmer_directory.rglob("*"))

    second_run = subprocess.run(
        _command(
            config_path=config_path,
            input_path=input_path,
            genomes_directory=genomes_directory,
            scan_directory=scan_directory,
            output_path=output_path,
            metadata_path=metadata_path,
        ),
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert second_run.returncode == 2
    assert "already exists" in second_run.stderr


def test_panel_cli_rejects_cohort_order_mismatch(tmp_path: Path) -> None:
    output_path = tmp_path / "panel_features.csv"
    metadata_path = tmp_path / "panel_features_metadata.json"
    completed = subprocess.run(
        _command(
            config_path=_write_config(tmp_path),
            input_path=_write_accessions(tmp_path, reverse=True),
            genomes_directory=_write_genome_inputs(tmp_path),
            scan_directory=_write_scan(tmp_path),
            output_path=output_path,
            metadata_path=metadata_path,
        ),
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "does not exactly match" in completed.stderr
    assert "order_only_mismatch=True" in completed.stderr
    assert not output_path.exists()
    assert not metadata_path.exists()
