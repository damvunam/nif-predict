"""Offline integration tests for the trusted-cutoff panel scan CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from nifpredict.evaluation.hmm_panel_scan import PanelScanResult

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "scan_hmm_panel.py"


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scan_hmm_panel_cli", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CLI module: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_resolves_inputs_and_reports_scan_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    accession = "GCF_000000001.1"
    input_file = tmp_path / "accessions.txt"
    input_file.write_text(f"{accession}\n", encoding="utf-8")
    protein_directory = tmp_path / "proteins"
    protein_directory.mkdir()
    (protein_directory / f"{accession}_protein.faa").write_text(
        ">P1\nACDE\n", encoding="utf-8"
    )
    output_directory = tmp_path / "scan"
    observed: dict[str, object] = {}

    def fake_run_panel_scan(**kwargs: object) -> PanelScanResult:
        observed.update(kwargs)
        return PanelScanResult(
            output_directory=output_directory,
            manifest_path=output_directory / "scan_manifest.json",
            accepted_hits_path=output_directory / "accepted_hits.tsv",
            genome_count=1,
            protein_count=1,
            accepted_hit_count=2,
            unique_hit_protein_count=1,
            multi_model_protein_count=1,
        )

    monkeypatch.setattr(module, "run_panel_scan", fake_run_panel_scan)
    exit_code = module.main(
        [
            "--input-file",
            str(input_file),
            "--protein-dir",
            str(protein_directory),
            "--config",
            str(tmp_path / "panel.yaml"),
            "--panel-root",
            str(tmp_path / "panel"),
            "--output-dir",
            str(output_directory),
            "--cpus",
            "3",
        ]
    )

    assert exit_code == 0
    assert observed["cpus"] == 3
    assert observed["protein_fastas"] == (
        (accession, (protein_directory / f"{accession}_protein.faa").resolve()),
    )
    assert "Accepted hits: 2" in capsys.readouterr().out


def test_cli_returns_two_for_missing_protein_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_cli_module()
    input_file = tmp_path / "accessions.txt"
    input_file.write_text("GCF_000000001.1\n", encoding="utf-8")

    exit_code = module.main(
        [
            "--input-file",
            str(input_file),
            "--protein-dir",
            str(tmp_path / "missing"),
        ]
    )

    assert exit_code == 2
    assert "does not exist" in capsys.readouterr().err
