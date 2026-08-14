"""Offline integration tests for the HMM-panel audit command-line interface."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_hmm_panel.py"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _valid_hmm_text() -> str:
    return """HMMER3/f [3.4 | Aug 2023]
NAME  nifD
ACC   TIGR00001.1
DESC  Test HMM
LENG  4
NSEQ  2
GA    10.0 11.0
TC    10.0 11.0
NC    5.0 5.0
HMM          A        C
//
"""


def _valid_seed_text() -> str:
    return """# STOCKHOLM 1.0
seed_one    AC-D
seed_two    A.ED
//
"""


def _write_offline_panel(tmp_path: Path) -> tuple[Path, Path]:
    hmm_text = _valid_hmm_text()
    seed_text = _valid_seed_text()
    config = {
        "schema_version": 1,
        "panel": {
            "id": "offline_test_panel",
            "provider": "Test provider",
            "release_name": "test/1.0",
            "release_date": "2026-01-01",
            "registry_url": "https://example.test/registry.tsv",
            "model_base_url": "https://example.test/models",
            "seed_base_url": "https://example.test/seeds",
            "cutoff_policy": "trusted",
            "normalized_with": "HMMER test",
        },
        "models": [
            {
                "role": "nifD",
                "accession": "TIGR00001.1",
                "source_identifier": "TIGR00001",
                "hmm_name": "nifD",
                "gene_symbols": ["nifD"],
                "hmm_length": 4,
                "sequence_cutoff": 10.0,
                "domain_cutoff": 11.0,
                "noise_cutoff": 5.0,
                "family_type": "equivalog",
                "product": "Test nitrogenase protein",
                "seed_sequence_count": 2,
                "checksums": {
                    "hmm_raw_sha256": _sha256(hmm_text),
                    "hmmconvert_sha256": "0" * 64,
                    "seed_raw_sha256": _sha256(seed_text),
                },
            }
        ],
    }
    config_path = tmp_path / "panel.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    destination = tmp_path / "panel_inputs"
    (destination / "models").mkdir(parents=True)
    (destination / "seeds").mkdir(parents=True)
    (destination / "models" / "TIGR00001.1.HMM").write_text(
        hmm_text,
        encoding="utf-8",
    )
    (destination / "seeds" / "TIGR00001.1.SEED").write_text(
        seed_text,
        encoding="utf-8",
    )
    return config_path, destination


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def test_prepare_reuses_verified_inputs_and_writes_report(tmp_path: Path) -> None:
    config_path, destination = _write_offline_panel(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            "--config",
            str(config_path),
            "--destination",
            str(destination),
            "--no-normalized-check",
            "--no-hmmstat-check",
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Downloaded files: 0" in completed.stdout
    assert "Reused files: 2" in completed.stdout
    assert "Panel input validation: PASS" in completed.stdout
    report = json.loads((destination / "input_audit.json").read_text(encoding="utf-8"))
    assert report["panel_id"] == "offline_test_panel"
    assert report["valid_model_count"] == 1
    assert report["invalid_model_count"] == 0
    assert report["downloaded_file_count"] == 0
    assert report["reused_file_count"] == 2


def test_validate_returns_two_for_missing_destination(tmp_path: Path) -> None:
    config_path, _ = _write_offline_panel(tmp_path)
    missing_destination = tmp_path / "missing"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate",
            "--config",
            str(config_path),
            "--destination",
            str(missing_destination),
            "--no-normalized-check",
            "--no-hmmstat-check",
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Input root(s) do not exist" in completed.stderr
