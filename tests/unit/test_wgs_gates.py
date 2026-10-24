"""Unit tests for WGS decision functions (W9): gate 1, gate 2, identity, read coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# wgs_qc.py / run_wgs_sample.py import sibling modules by bare name (cwd = nifpredict/)
NIFPREDICT_DIR = Path(__file__).resolve().parents[2] / "nifpredict"
sys.path.insert(0, str(NIFPREDICT_DIR))

import run_wgs_sample  # noqa: E402
import wgs_qc  # noqa: E402

# Thresholds used by run_wgs_sample.py for gate 2
GATE2 = {"min_n50": 50000, "min_busco_completeness": 95.0}

SKANI_HEADER = (
    "Ref_file\tQuery_file\tANI\tAlign_fraction_ref\t"
    "Align_fraction_query\tRef_name\tQuery_name\n"
)


def _write_fastp_json(path: Path, q30: float = 0.85, dup: float = 0.05, total_bases: int = 1000000) -> Path:
    data = {
        "summary": {"after_filtering": {"q30_rate": q30, "total_bases": total_bases}},
        "duplication": {"rate": dup},
    }
    path.write_text(json.dumps(data))
    return path


def _write_quast_tsv(path: Path, n50: int | None = 100000, n90: int | None = None) -> Path:
    lines = ["Assembly\tassembly", "# contigs\t108", "Total length\t7298197"]
    if n90 is not None:
        lines.append(f"N90\t{n90}")  # placed before N50 on purpose: parser must pick the N50 line
    if n50 is not None:
        lines.append(f"N50\t{n50}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_busco_json(path: Path, complete: float = 98.0) -> Path:
    path.write_text(json.dumps({"results": {"Complete percentage": complete}}))
    return path


def _gate2_files(tmp_path: Path, n50: int | None, busco: float, n90: int | None = None) -> tuple[str, str]:
    quast = _write_quast_tsv(tmp_path / "report.tsv", n50=n50, n90=n90)
    busco_json = _write_busco_json(tmp_path / "short_summary.json", complete=busco)
    return str(quast), str(busco_json)


def _identity_inputs(tmp_path: Path, ani: float | None) -> tuple[str, str, str]:
    asm = tmp_path / "asm.fasta"
    ref = tmp_path / "ref.fna"
    asm.write_text(">c1\nATGC\n")
    ref.write_text(">r1\nATGC\n")
    tsv = tmp_path / "skani_vs_ref.tsv"
    body = "" if ani is None else f"{ref}\t{asm}\t{ani}\t99.11\t99.43\tref_org\tquery_org\n"
    tsv.write_text(SKANI_HEADER + body)
    return str(asm), str(ref), str(tsv)


# --- Gate 1 -------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "q30,dup,expected",
    [
        (0.826, 0.0007, True),  # PSON
        (0.8, 0.2, True),       # boundary passes (>= / <=)
        (0.799, 0.05, False),
        (0.85, 0.201, False),
    ],
)
def test_check_qc_report_thresholds(tmp_path: Path, q30: float, dup: float, expected: bool) -> None:
    json_path = _write_fastp_json(tmp_path / "fastp.json", q30=q30, dup=dup)
    assert wgs_qc.check_qc_report(str(json_path)) is expected


def test_check_qc_report_bad_input(tmp_path: Path) -> None:
    assert wgs_qc.check_qc_report(str(tmp_path / "missing.json")) is False

    bad = tmp_path / "bad.json"
    bad.write_text("{not_valid_json")
    assert wgs_qc.check_qc_report(str(bad)) is False

    no_dup = tmp_path / "no_dup.json"
    no_dup.write_text(json.dumps({"summary": {"after_filtering": {"q30_rate": 0.9}}}))
    assert wgs_qc.check_qc_report(str(no_dup)) is False


def test_is_sample_qc_gate_passed(tmp_path: Path) -> None:
    ok = _write_fastp_json(tmp_path / "ok.fastp.json", q30=0.85, dup=0.01)
    assert run_wgs_sample.is_sample_qc_gate_passed(str(ok), "T1") is True
    assert run_wgs_sample.is_sample_qc_gate_passed(str(tmp_path / "none.json"), "T1") is False


# --- Gate 2 -------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n50,busco,expected",
    [
        (156988, 99.6, True),   # PSON
        (50000, 95.0, True),    # boundary passes
        (49999, 99.0, False),
        (100000, 94.9, False),
    ],
)
def test_check_assembly_quality_thresholds(tmp_path: Path, n50: int, busco: float, expected: bool) -> None:
    quast, busco_json = _gate2_files(tmp_path, n50, busco)
    assert wgs_qc.check_assembly_quality(quast, busco_json, **GATE2) is expected


def test_check_assembly_quality_bad_input(tmp_path: Path) -> None:
    quast, busco_json = _gate2_files(tmp_path, None, 99.0)
    assert wgs_qc.check_assembly_quality(quast, busco_json, **GATE2) is False

    no_results = tmp_path / "busco_no_results.json"
    no_results.write_text(json.dumps({"other": {}}))
    quast_ok = _write_quast_tsv(tmp_path / "report_ok.tsv", n50=100000)
    assert wgs_qc.check_assembly_quality(str(quast_ok), str(no_results), **GATE2) is False


def test_check_assembly_quality_reads_n50_not_n90(tmp_path: Path) -> None:
    quast, busco_json = _gate2_files(tmp_path, n50=60000, busco=98.0, n90=1000)
    assert wgs_qc.check_assembly_quality(quast, busco_json, **GATE2) is True


@pytest.mark.parametrize("n50,expected", [(120000, True), (49999, False)])
def test_is_sample_gate2_passed_reuses_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n50: int, expected: bool
) -> None:
    qc_dir = tmp_path / "assembly_qc"
    (qc_dir / "busco_result").mkdir(parents=True)
    _write_quast_tsv(qc_dir / "report.tsv", n50=n50)
    _write_busco_json(qc_dir / "busco_result" / "short_summary.x.json", complete=97.5)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("QUAST/BUSCO must not run when reports exist")

    monkeypatch.setattr(run_wgs_sample, "run_assembly_qc_gate", _must_not_run)
    asm = tmp_path / "assembly.fasta"
    asm.write_text(">c1\nATGC\n")
    assert run_wgs_sample.is_sample_gate2_passed(str(asm), str(qc_dir), 4) is expected


# --- Identity -----------------------------------------------------------------------------

def test_check_sample_identity_missing_ref(tmp_path: Path) -> None:
    asm, _, tsv = _identity_inputs(tmp_path, 99.0)
    res = run_wgs_sample.check_sample_identity(asm, str(tmp_path / "missing.fna"), tsv)
    assert res["status"] == "ERROR"


def test_check_sample_identity_header_only(tmp_path: Path) -> None:
    asm, ref, tsv = _identity_inputs(tmp_path, None)
    res = run_wgs_sample.check_sample_identity(asm, ref, tsv)
    assert res["status"] == "FAIL"
    assert res["ani"] is None


@pytest.mark.parametrize("ani,expected", [(94.99, "FAIL"), (95.0, "PASS"), (99.99, "PASS")])
def test_check_sample_identity_ani_threshold(tmp_path: Path, ani: float, expected: str) -> None:
    asm, ref, tsv = _identity_inputs(tmp_path, ani)
    res = run_wgs_sample.check_sample_identity(asm, ref, tsv)
    assert res["status"] == expected
    assert res["ani"] == pytest.approx(ani)


# --- Coverage -----------------------------------------------------------------------------

def test_get_fasta_total_length(tmp_path: Path) -> None:
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">c1\nATGCATGCAT\n\n>c2\nATGCATGCAT\nGCATGCATGC\n\n>c3\nATGCA\n")
    assert wgs_qc.get_fasta_total_length(str(fasta)) == 35

    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    assert wgs_qc.get_fasta_total_length(str(empty)) is None
    assert wgs_qc.get_fasta_total_length(str(tmp_path / "no_such.fasta")) is None


def test_get_read_coverage(tmp_path: Path) -> None:
    fastp_json = _write_fastp_json(tmp_path / "pson.fastp.json", total_bases=721359808)
    res = wgs_qc.get_read_coverage(str(fastp_json), 7316671)
    assert res == {"total_bases": 721359808, "genome_size": 7316671, "coverage": pytest.approx(98.59, abs=0.01)}

    for bad_size in (0, None, -100):
        assert wgs_qc.get_read_coverage(str(fastp_json), bad_size) is None

    no_bases = tmp_path / "no_bases.json"
    no_bases.write_text(json.dumps({"summary": {"after_filtering": {}}}))
    assert wgs_qc.get_read_coverage(str(no_bases), 5000000) is None


def test_print_sample_coverage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fastp_json = _write_fastp_json(tmp_path / "fastp.json", total_bases=1000)
    asm = tmp_path / "assembly.fasta"
    asm.write_text(">c1\n" + "A" * 100 + "\n")
    ref = tmp_path / "ref.fna"
    ref.write_text(">r1\n" + "A" * 50 + "\n")

    assert run_wgs_sample.print_sample_coverage(str(fastp_json), str(asm), str(tmp_path / "missing.fna")) is None
    out = capsys.readouterr().out.splitlines()
    assert "COVERAGE: 10.0x vs assembly (1,000 bases after fastp / 100 bp)" in out
    assert any(line.startswith("Warning: COVERAGE vs reference skipped") for line in out)

    run_wgs_sample.print_sample_coverage(str(fastp_json), str(asm), str(ref))
    out = capsys.readouterr().out
    assert f"COVERAGE: 20.0x vs reference (1,000 bases after fastp / 50 bp, {ref})" in out
