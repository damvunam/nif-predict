"""Unit tests for trusted-cutoff HMM-panel scanning."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Self

import pytest
import yaml

import nifpredict.ml.evaluation.hmm_panel_scan as scan_module
from nifpredict.ml.evaluation._hmm_panel_manifest import load_panel_spec
from nifpredict.ml.evaluation._hmm_panel_scan_types import (
    FastaScanResult,
    PanelScanHit,
)
from nifpredict.ml.evaluation._hmm_panel_types import ModelInputAudit
from nifpredict.ml.evaluation.hmm_panel_scan import (
    HIT_COLUMNS,
    HMMPanelAuditError,
    load_scan_accessions,
    resolve_protein_fastas,
    run_panel_scan,
)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _model(role: str, accession: str, name: str) -> dict[str, object]:
    return {
        "role": role,
        "accession": accession,
        "source_identifier": accession.removesuffix(".1"),
        "hmm_name": name,
        "gene_symbols": [role],
        "hmm_length": 4,
        "sequence_cutoff": 10.0,
        "domain_cutoff": 10.0,
        "noise_cutoff": 5.0,
        "family_type": "equivalog",
        "product": f"Test product for {role}",
        "seed_sequence_count": 1,
        "checksums": {
            "hmm_raw_sha256": "0" * 64,
            "hmmconvert_sha256": "1" * 64,
            "seed_raw_sha256": "2" * 64,
        },
    }


def _write_environment(
    tmp_path: Path,
    *,
    roles: tuple[tuple[str, str, str], ...] = (
        ("nifD", "TIGR00001.1", "nifD"),
        ("vnfD", "TIGR00002.1", "VNFD"),
    ),
) -> tuple[Path, Path, Path, Path]:
    config = {
        "schema_version": 1,
        "panel": {
            "id": "scan_test_panel",
            "provider": "Test provider",
            "release_name": "test/1.0",
            "release_date": "2026-01-01",
            "registry_url": "https://example.test/registry.tsv",
            "model_base_url": "https://example.test/models",
            "seed_base_url": "https://example.test/seeds",
            "cutoff_policy": "trusted",
            "normalized_with": "HMMER test",
        },
        "models": [_model(*role) for role in roles],
    }
    manifest_path = tmp_path / "panel.yaml"
    manifest_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    panel_root = tmp_path / "panel"
    (panel_root / "models").mkdir(parents=True)
    (panel_root / "seeds").mkdir()
    for _, accession, _ in roles:
        (panel_root / "models" / f"{accession}.HMM").write_text(
            f"placeholder {accession}\n", encoding="utf-8"
        )
        (panel_root / "seeds" / f"{accession}.SEED").write_text(
            "# STOCKHOLM 1.0\nseed ACDE\n//\n", encoding="utf-8"
        )

    input_audit_path = panel_root / "input_audit.json"
    input_audit_path.write_text(
        json.dumps(
            {
                "panel_id": "scan_test_panel",
                "manifest_sha256": _sha256_text(
                    manifest_path.read_text(encoding="utf-8")
                ),
                "model_count": len(roles),
                "valid_model_count": len(roles),
                "invalid_model_count": 0,
                "normalized_check_enabled": True,
                "hmmstat_check_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    protein_path = tmp_path / "GCF_000000001.1_protein.faa"
    protein_path.write_text(">P1\nACDE\n>P2\nFGHI\n", encoding="utf-8")
    return manifest_path, panel_root, input_audit_path, protein_path


def _valid_audits(panel_root: Path, count: int) -> tuple[ModelInputAudit, ...]:
    return tuple(
        ModelInputAudit(
            role=f"role_{index}",
            accession=f"model_{index}",
            hmm_path=panel_root / "models" / "placeholder.HMM",
            seed_path=panel_root / "seeds" / "placeholder.SEED",
            raw_hmm_sha256="0" * 64,
            normalized_hmm_sha256="1" * 64,
            raw_seed_sha256="2" * 64,
            seed_sequence_count=1,
            hmmstat_valid=True,
            errors=(),
        )
        for index in range(count)
    )


def test_load_scan_accessions_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "accessions.txt"
    path.write_text("GCF_000000001.1\nGCF_000000001.1\n", encoding="utf-8")

    with pytest.raises(HMMPanelAuditError, match="duplicate"):
        load_scan_accessions(path)


def test_load_scan_accessions_rejects_unsafe_values(tmp_path: Path) -> None:
    path = tmp_path / "accessions.txt"
    path.write_text("../outside\n", encoding="utf-8")

    with pytest.raises(HMMPanelAuditError, match="Invalid assembly accession"):
        load_scan_accessions(path)


def test_resolve_protein_fastas_requires_every_accession(tmp_path: Path) -> None:
    (tmp_path / "GCF_000000001.1_protein.faa").write_text(
        ">P1\nACDE\n", encoding="utf-8"
    )

    with pytest.raises(HMMPanelAuditError, match="GCF_000000002.1"):
        resolve_protein_fastas(("GCF_000000001.1", "GCF_000000002.1"), tmp_path)


def test_scan_backend_uses_only_trusted_cutoffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, panel_root, _, protein_path = _write_environment(tmp_path)
    panel = load_panel_spec(manifest_path)
    observed_options: dict[str, object] = {}

    class FakeSequenceFile:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read_block(self) -> object:
            return object()

    class FakeAlphabet:
        @staticmethod
        def amino() -> object:
            return object()

    domain = SimpleNamespace(score=9.0, i_evalue=1e-20)
    hit = SimpleNamespace(
        name=b"P1",
        score=11.0,
        evalue=1e-25,
        best_domain=domain,
    )

    def fake_hmmsearch(**options: object) -> list[list[object]]:
        observed_options.update(options)
        return [[hit], [hit]]

    pyhmmer_module = ModuleType("pyhmmer")
    pyhmmer_module.__version__ = "test-version"
    pyhmmer_module.hmmer = SimpleNamespace(hmmsearch=fake_hmmsearch)
    easel_module = ModuleType("pyhmmer.easel")
    easel_module.Alphabet = FakeAlphabet
    easel_module.SequenceFile = FakeSequenceFile
    monkeypatch.setitem(sys.modules, "pyhmmer", pyhmmer_module)
    monkeypatch.setitem(sys.modules, "pyhmmer.easel", easel_module)
    monkeypatch.setattr(scan_module, "_load_single_query", lambda path: path)

    result, version = scan_module._scan_one_fasta(
        panel,
        tuple(
            panel_root / "models" / f"{model.accession}.HMM" for model in panel.models
        ),
        "GCF_000000001.1",
        protein_path,
        3,
    )

    assert observed_options["bit_cutoffs"] == "trusted"
    assert "E" not in observed_options
    assert "domE" not in observed_options
    assert "T" not in observed_options
    assert observed_options["cpus"] == 3
    assert [hit.model_role for hit in result.hits] == ["nifD", "vnfD"]
    assert version == "test-version"


def test_run_panel_scan_writes_stable_artifacts_and_multi_model_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, panel_root, input_audit_path, protein_path = _write_environment(
        tmp_path
    )
    monkeypatch.setattr(
        scan_module,
        "audit_panel_inputs",
        lambda *args, **kwargs: _valid_audits(panel_root, 2),
    )
    digest = _sha256_text("ACDE")
    hits = tuple(
        PanelScanHit(
            accession="GCF_000000001.1",
            protein_id="P1",
            protein_source=str(protein_path.resolve()),
            protein_length=4,
            protein_sha256=digest,
            model_role=role,
            model_accession=accession,
            model_name=name,
            sequence_score=100.0,
            sequence_evalue=1e-30,
            best_domain_score=95.0,
            best_domain_i_evalue=1e-28,
        )
        for role, accession, name in (
            ("nifD", "TIGR00001.1", "nifD"),
            ("vnfD", "TIGR00002.1", "VNFD"),
        )
    )
    monkeypatch.setattr(
        scan_module,
        "_scan_one_fasta",
        lambda *args, **kwargs: (
            FastaScanResult(
                accession="GCF_000000001.1",
                protein_path=protein_path.resolve(),
                protein_sha256=_sha256_text(protein_path.read_text(encoding="utf-8")),
                protein_count=2,
                hits=hits,
            ),
            "test-version",
        ),
    )
    output_directory = tmp_path / "scan"

    result = run_panel_scan(
        panel_manifest_path=manifest_path,
        panel_root=panel_root,
        panel_input_audit_path=input_audit_path,
        protein_fastas=(("GCF_000000001.1", protein_path),),
        output_directory=output_directory,
        cpus=2,
    )

    assert result.accepted_hit_count == 2
    assert result.unique_hit_protein_count == 1
    assert result.multi_model_protein_count == 1
    with result.accepted_hits_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    assert tuple(rows[0]) == HIT_COLUMNS
    assert [row["model_role"] for row in rows] == ["nifD", "vnfD"]
    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["cutoff_policy"] == "trusted"
    assert run_manifest["additional_hit_thresholds"] == []
    assert run_manifest["protein_count"] == 2


def test_run_panel_scan_refuses_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, panel_root, input_audit_path, protein_path = _write_environment(
        tmp_path
    )
    output_directory = tmp_path / "scan"
    output_directory.mkdir()
    monkeypatch.setattr(
        scan_module,
        "audit_panel_inputs",
        lambda *args, **kwargs: pytest.fail("audit should not run"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_panel_scan(
            panel_manifest_path=manifest_path,
            panel_root=panel_root,
            panel_input_audit_path=input_audit_path,
            protein_fastas=(("GCF_000000001.1", protein_path),),
            output_directory=output_directory,
        )


def test_run_panel_scan_rejects_mismatched_input_audit(tmp_path: Path) -> None:
    manifest_path, panel_root, input_audit_path, protein_path = _write_environment(
        tmp_path
    )
    audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    audit["manifest_sha256"] = "f" * 64
    input_audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(HMMPanelAuditError, match="not valid for this scan"):
        run_panel_scan(
            panel_manifest_path=manifest_path,
            panel_root=panel_root,
            panel_input_audit_path=input_audit_path,
            protein_fastas=(("GCF_000000001.1", protein_path),),
            output_directory=tmp_path / "scan",
        )
