"""Unit tests for W16: separating ERROR from FAIL in WGS QC gates, BUSCO offline mode, and run_sample main()."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nifpredict.wgs import qc, run_sample


def _write_fastp_json(path: Path, q30: float = 0.85, dup: float = 0.05, total_bases: int = 1000000) -> Path:
    data = {
        "summary": {"after_filtering": {"q30_rate": q30, "total_bases": total_bases}},
        "duplication": {"rate": dup},
    }
    path.write_text(json.dumps(data))
    return path


def _write_quast_tsv(path: Path, n50: int | None = 100000) -> Path:
    lines = ["Assembly\tassembly", "# contigs\t10", "Total length\t4000000"]
    if n50 is not None:
        lines.append(f"N50\t{n50}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_busco_json(path: Path, complete: float = 98.5) -> Path:
    path.write_text(json.dumps({"results": {"Complete percentage": complete}}))
    return path


def _last_nonempty_line(captured_out: str) -> str:
    lines = [line.strip() for line in captured_out.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ==============================================================================
# 1. Gate 1: evaluate_qc_report
# ==============================================================================


def test_evaluate_qc_report_pass(tmp_path: Path) -> None:
    report = _write_fastp_json(tmp_path / "pass.json", q30=0.90, dup=0.05)
    res = qc.evaluate_qc_report(str(report))
    assert res["status"] == "PASS"
    assert res["q30"] == pytest.approx(0.90)
    assert res["dup"] == pytest.approx(0.05)
    assert isinstance(res["reason"], str) and res["reason"]


def test_evaluate_qc_report_fail_low_q30(tmp_path: Path) -> None:
    report = _write_fastp_json(tmp_path / "low_q30.json", q30=0.75, dup=0.05)
    res = qc.evaluate_qc_report(str(report))
    assert res["status"] == "FAIL"
    assert res["q30"] == pytest.approx(0.75)
    assert res["dup"] == pytest.approx(0.05)
    assert "Q30" in res["reason"]


def test_evaluate_qc_report_fail_high_dup(tmp_path: Path) -> None:
    report = _write_fastp_json(tmp_path / "high_dup.json", q30=0.88, dup=0.25)
    res = qc.evaluate_qc_report(str(report))
    assert res["status"] == "FAIL"
    assert res["q30"] == pytest.approx(0.88)
    assert res["dup"] == pytest.approx(0.25)
    assert "Duplication" in res["reason"]


def test_evaluate_qc_report_error_corrupt_json(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{corrupt_json:")
    res = qc.evaluate_qc_report(str(corrupt))
    assert res["status"] == "ERROR"
    assert res["q30"] is None
    assert res["dup"] is None
    assert res["reason"]


def test_evaluate_qc_report_error_missing_key(tmp_path: Path) -> None:
    missing_key = tmp_path / "missing_key.json"
    missing_key.write_text(json.dumps({"summary": {"after_filtering": {"q30_rate": 0.9}}}))
    res = qc.evaluate_qc_report(str(missing_key))
    assert res["status"] == "ERROR"
    assert res["q30"] is None
    assert res["dup"] is None


def test_evaluate_qc_report_error_file_not_found(tmp_path: Path) -> None:
    res = qc.evaluate_qc_report(str(tmp_path / "does_not_exist.json"))
    assert res["status"] == "ERROR"
    assert res["q30"] is None
    assert res["dup"] is None


# ==============================================================================
# 2. Gate 2: evaluate_assembly_quality & run_assembly_qc_status
# ==============================================================================


def test_evaluate_assembly_quality_pass(tmp_path: Path) -> None:
    quast_tsv = _write_quast_tsv(tmp_path / "report.tsv", n50=120000)
    busco_json = _write_busco_json(tmp_path / "short_summary.json", complete=98.0)
    res = qc.evaluate_assembly_quality(str(quast_tsv), str(busco_json))
    assert res["status"] == "PASS"
    assert res["n50"] == 120000
    assert res["busco_complete"] == pytest.approx(98.0)


def test_evaluate_assembly_quality_fail_low_n50(tmp_path: Path) -> None:
    quast_tsv = _write_quast_tsv(tmp_path / "report.tsv", n50=30000)
    busco_json = _write_busco_json(tmp_path / "short_summary.json", complete=98.0)
    res = qc.evaluate_assembly_quality(str(quast_tsv), str(busco_json))
    assert res["status"] == "FAIL"
    assert res["n50"] == 30000
    assert res["busco_complete"] == pytest.approx(98.0)
    assert "N50" in res["reason"]


def test_evaluate_assembly_quality_fail_low_busco(tmp_path: Path) -> None:
    quast_tsv = _write_quast_tsv(tmp_path / "report.tsv", n50=100000)
    busco_json = _write_busco_json(tmp_path / "short_summary.json", complete=85.0)
    res = qc.evaluate_assembly_quality(str(quast_tsv), str(busco_json))
    assert res["status"] == "FAIL"
    assert res["n50"] == 100000
    assert res["busco_complete"] == pytest.approx(85.0)
    assert "BUSCO" in res["reason"]


def test_evaluate_assembly_quality_error_missing_n50(tmp_path: Path) -> None:
    quast_tsv = _write_quast_tsv(tmp_path / "report.tsv", n50=None)
    busco_json = _write_busco_json(tmp_path / "short_summary.json", complete=98.0)
    res = qc.evaluate_assembly_quality(str(quast_tsv), str(busco_json))
    assert res["status"] == "ERROR"
    assert res["n50"] is None
    assert "N50" in res["reason"]


def test_evaluate_assembly_quality_error_corrupt_busco_json(tmp_path: Path) -> None:
    quast_tsv = _write_quast_tsv(tmp_path / "report.tsv", n50=100000)
    busco_json = tmp_path / "short_summary.json"
    busco_json.write_text("{corrupt_busco_json")
    res = qc.evaluate_assembly_quality(str(quast_tsv), str(busco_json))
    assert res["status"] == "ERROR"
    assert res["n50"] == 100000
    assert res["busco_complete"] is None


def test_run_assembly_qc_status_error_when_quast_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qc, "run_quast", lambda *a, **kw: None)
    monkeypatch.setattr(
        qc,
        "run_busco",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("run_busco should not be called if QUAST failed")),
    )
    res = qc.run_assembly_qc_status(str(tmp_path / "asm.fasta"), str(tmp_path / "out"))
    assert res["status"] == "ERROR"
    assert "QUAST" in res["reason"]


def test_run_assembly_qc_status_error_when_busco_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quast_tsv = _write_quast_tsv(tmp_path / "report.tsv", n50=120000)
    monkeypatch.setattr(qc, "run_quast", lambda *a, **kw: str(quast_tsv))
    monkeypatch.setattr(qc, "run_busco", lambda *a, **kw: None)
    res = qc.run_assembly_qc_status(str(tmp_path / "asm.fasta"), str(tmp_path / "out"))
    assert res["status"] == "ERROR"
    assert "BUSCO" in res["reason"]


# ==============================================================================
# 3. run_busco (offline lineage check & FileNotFoundError) and run_quast
# ==============================================================================


def test_run_busco_missing_lineage_does_not_call_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    busco_dir = tmp_path / "busco"
    monkeypatch.setattr(qc.paths, "BUSCO_DIR", busco_dir)

    def _forbid_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called when lineage dataset.cfg is missing")

    monkeypatch.setattr(qc.subprocess, "run", _forbid_subprocess)

    res = qc.run_busco(str(tmp_path / "asm.fasta"), str(tmp_path / "out"), lineage="bacillales_odb10")
    assert res is None
    out = capsys.readouterr().out
    expected_dir = busco_dir / "lineages" / "bacillales_odb10"
    assert (
        f"Error: BUSCO lineage bacillales_odb10 not found in {expected_dir} - "
        "fetch with: python scripts/fetch_public_data.py --id BUSCO_BACILLALES"
    ) in out


def test_run_busco_with_local_lineage_passes_offline_and_download_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    busco_dir = tmp_path / "busco"
    lineage_dir = busco_dir / "lineages" / "bacillales_odb10"
    lineage_dir.mkdir(parents=True)
    (lineage_dir / "dataset.cfg").write_text("name=bacillales_odb10\n")
    monkeypatch.setattr(qc.paths, "BUSCO_DIR", busco_dir)

    out_dir = tmp_path / "out"
    summary_dir = out_dir / "busco_result"
    summary_dir.mkdir(parents=True)
    summary_file = _write_busco_json(summary_dir / "short_summary.specific.bacillales_odb10.busco_result.json", 99.0)

    captured_cmd: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        captured_cmd.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(qc.subprocess, "run", _fake_run)

    res = qc.run_busco(str(tmp_path / "asm.fasta"), str(out_dir), lineage="bacillales_odb10", threads=4)
    assert res == str(summary_file)
    assert "--offline" in captured_cmd
    assert "--download_path" in captured_cmd
    idx = captured_cmd.index("--download_path")
    assert captured_cmd[idx + 1] == str(busco_dir)


def test_run_busco_filenotfounderror_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    busco_dir = tmp_path / "busco"
    lineage_dir = busco_dir / "lineages" / "bacillales_odb10"
    lineage_dir.mkdir(parents=True)
    (lineage_dir / "dataset.cfg").write_text("name=bacillales_odb10\n")
    monkeypatch.setattr(qc.paths, "BUSCO_DIR", busco_dir)

    def _raise_fnf(*args, **kwargs):
        raise FileNotFoundError("No such file or directory: 'busco'")

    monkeypatch.setattr(qc.subprocess, "run", _raise_fnf)

    res = qc.run_busco(str(tmp_path / "asm.fasta"), str(tmp_path / "out"))
    assert res is None
    assert "Error: BUSCO failed" in capsys.readouterr().out


def test_run_quast_filenotfounderror_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise_fnf(*args, **kwargs):
        raise FileNotFoundError("No such file or directory: 'quast.py'")

    monkeypatch.setattr(qc.subprocess, "run", _raise_fnf)

    res = qc.run_quast(str(tmp_path / "asm.fasta"), str(tmp_path / "out"))
    assert res is None
    assert "Error: QUAST failed" in capsys.readouterr().out


# ==============================================================================
# 4. Runner: run_sample.main([...]) exit codes and final STATUS: / DONE: lines
# ==============================================================================


def _setup_happy_mocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Configures baseline mocks on run_sample for a complete PASS run, returning default CLI argv."""
    r1 = str(tmp_path / "S1_R1.fastq.gz")
    r2 = str(tmp_path / "S1_R2.fastq.gz")
    trim_r1 = str(tmp_path / "out" / "S1" / "qc" / "S1_R1.trim.fastq.gz")
    trim_r2 = str(tmp_path / "out" / "S1" / "qc" / "S1_R2.trim.fastq.gz")
    fastp_json = str(tmp_path / "out" / "S1" / "qc" / "S1.fastp.json")
    asm_fasta = str(tmp_path / "out" / "S1" / "assembly" / "assembly.fasta")
    ref_fna = str(tmp_path / "ref.fna")
    Path(ref_fna).write_text(">ref\nATGC\n")

    monkeypatch.setattr(run_sample, "get_sample_raw_reads", lambda d, s: (r1, r2))
    monkeypatch.setattr(run_sample, "determine_platform", lambda p, r: "short")
    monkeypatch.setattr(run_sample, "run_sample_qc", lambda r1_p, r2_p, q_d, s: (trim_r1, trim_r2, fastp_json))
    monkeypatch.setattr(
        run_sample,
        "sample_qc_gate_status",
        lambda j, s: {"status": "PASS", "q30": 0.9, "dup": 0.01, "reason": "ok"},
    )
    monkeypatch.setattr(run_sample, "run_sample_assembly", lambda p, a_d, tr1, tr2, t: asm_fasta)
    monkeypatch.setattr(
        run_sample,
        "sample_gate2_status",
        lambda a, q_d, t: {"status": "PASS", "n50": 150000, "busco_complete": 99.0, "reason": "ok"},
    )
    monkeypatch.setattr(run_sample, "run_sample_contig_stats", lambda a, c: True)
    monkeypatch.setattr(run_sample, "print_sample_coverage", lambda j, a, r: None)
    monkeypatch.setattr(
        run_sample,
        "check_sample_identity",
        lambda a, r, i, t=10: {"status": "PASS", "ani": 98.5, "af_ref": 95.0, "af_query": 95.0},
    )

    return ["-s", "S1", "-d", str(tmp_path), "-o", str(tmp_path / "out"), "-r", ref_fna]


def test_main_reads_not_found_returns_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(run_sample, "get_sample_raw_reads", lambda d, s: (None, None))

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - reads - ")


def test_main_platform_none_returns_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(run_sample, "determine_platform", lambda p, r: None)

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - reads - ")


def test_main_fastp_none_returns_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(run_sample, "run_sample_qc", lambda r1, r2, q_d, s: (None, None, None))

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - fastp - ")


def test_main_gate1_fail_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_sample,
        "sample_qc_gate_status",
        lambda j, s: {"status": "FAIL", "q30": 0.70, "dup": 0.05, "reason": "Q30 rate too low (0.7 < 0.8)"},
    )

    rc = run_sample.main(argv)
    assert rc == 1
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: FAIL - gate1 - ")


def test_main_gate1_error_returns_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_sample,
        "sample_qc_gate_status",
        lambda j, s: {"status": "ERROR", "q30": None, "dup": None, "reason": "could not read fastp report"},
    )

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - gate1 - ")


def test_main_assembly_none_returns_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(run_sample, "run_sample_assembly", lambda p, a_d, tr1, tr2, t: None)

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - assembly - ")


def test_main_gate2_fail_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_sample,
        "sample_gate2_status",
        lambda a, q_d, t: {"status": "FAIL", "n50": 25000, "busco_complete": 98.0, "reason": "N50 too low (25000 < 50000)"},
    )

    rc = run_sample.main(argv)
    assert rc == 1
    out = capsys.readouterr().out
    assert "Error: Gate 2 FAIL for" not in out
    last_line = _last_nonempty_line(out)
    assert last_line.startswith("STATUS: FAIL - gate2 - ")


def test_main_gate2_error_returns_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_sample,
        "sample_gate2_status",
        lambda a, q_d, t: {"status": "ERROR", "n50": None, "busco_complete": None, "reason": "BUSCO failed for lineage bacillales_odb10"},
    )

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - gate2 - ")


def test_main_identity_fail_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_sample,
        "check_sample_identity",
        lambda a, r, i, t=10: {"status": "FAIL", "ani": 91.2, "af_ref": 80.0, "af_query": 80.0},
    )

    rc = run_sample.main(argv)
    assert rc == 1
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: FAIL - identity - ")


def test_main_identity_error_returns_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_sample,
        "check_sample_identity",
        lambda a, r, i, t=10: {"status": "ERROR", "ani": None, "af_ref": None, "af_query": None},
    )

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - identity - ")


def test_main_all_pass_returns_0_and_prints_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)

    rc = run_sample.main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    assert "STATUS:" not in out
    last_line = _last_nonempty_line(out)
    assert last_line == "DONE: S1 passed gate 1, gate 2 and identity; stopped before annotation"


def test_main_dry_run_returns_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch) + ["--dry_run"]

    rc = run_sample.main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[DRY-RUN] Sample: S1" in out
    assert "STATUS:" not in out


# Added by PM (W16 review): missing default reference must stop before any work, exit 3
def test_main_missing_default_ref_returns_3_before_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    argv = argv[: argv.index("-r")]  # no -r: use the default reference
    monkeypatch.setattr(run_sample.paths, "GENOMES_DIR", tmp_path / "no_genomes")
    monkeypatch.setattr(
        run_sample,
        "get_sample_raw_reads",
        lambda d, s: (_ for _ in ()).throw(AssertionError("reads must not be searched when the default ref is missing")),
    )

    rc = run_sample.main(argv)
    assert rc == 3
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - identity - default reference not available")


# Added by PM (W16 review): the two runner status helpers were only ever mocked
def test_sample_qc_gate_status_missing_fail_pass(tmp_path: Path) -> None:
    assert run_sample.sample_qc_gate_status(str(tmp_path / "none.json"), "S1")["status"] == "ERROR"
    bad = _write_fastp_json(tmp_path / "bad.json", q30=0.70, dup=0.05)
    assert run_sample.sample_qc_gate_status(str(bad), "S1")["status"] == "FAIL"
    good = _write_fastp_json(tmp_path / "good.json", q30=0.90, dup=0.05)
    assert run_sample.sample_qc_gate_status(str(good), "S1")["status"] == "PASS"
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    assert run_sample.sample_qc_gate_status(str(broken), "S1")["status"] == "ERROR"


def test_sample_gate2_status_reuses_existing_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qc_dir = tmp_path / "assembly_qc"
    (qc_dir / "busco_result").mkdir(parents=True)
    _write_quast_tsv(qc_dir / "report.tsv", n50=30000)
    busco = _write_busco_json(qc_dir / "busco_result" / "short_summary.x.json", complete=99.0)
    monkeypatch.setattr(
        run_sample,
        "run_assembly_qc_status",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must reuse existing reports")),
    )
    assert run_sample.sample_gate2_status("asm.fasta", str(qc_dir), 2)["status"] == "FAIL"
    busco.write_text("{")
    assert run_sample.sample_gate2_status("asm.fasta", str(qc_dir), 2)["status"] == "ERROR"
    _write_busco_json(busco, complete=99.0)
    _write_quast_tsv(qc_dir / "report.tsv", n50=200000)
    assert run_sample.sample_gate2_status("asm.fasta", str(qc_dir), 2)["status"] == "PASS"


def test_sample_gate2_status_runs_from_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def _fake(asm, out_dir, **kw):
        seen.update(kw, asm=asm, out_dir=out_dir)
        return {"status": "ERROR", "n50": None, "busco_complete": None, "reason": "BUSCO failed"}

    monkeypatch.setattr(run_sample, "run_assembly_qc_status", _fake)
    res = run_sample.sample_gate2_status("asm.fasta", str(tmp_path / "assembly_qc"), 2)
    assert res["status"] == "ERROR"
    assert seen["min_n50"] == 50000 and seen["min_busco_completeness"] == 95.0
    assert seen["lineage"] == "bacillales_odb10" and seen["threads"] == 2
