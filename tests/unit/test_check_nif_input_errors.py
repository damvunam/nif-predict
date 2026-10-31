"""Unit tests for check_nif input errors (MU8: items 1-4, 6).

Style follows tests/unit/test_check_nif_exit.py: call main([...]), check
BOTH the return code AND the log lines. No network, no real BLAST/diamond
where monkeypatching can avoid them.

New functions are only reached through getattr(check_nif, "<name>") inside
the test body (never imported at the top), so each new-behaviour test
fails on code that does not have the function yet.
"""

from __future__ import annotations

import pytest

from nifpredict.nif import check_nif
from nifpredict.nif.check_nif import main

SAMPLE = "TESTX"

REF_FULL = (
    ">sp|T00001|NIFH_TEST Test NifH OS=Test OX=1 GN=nifH PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
    ">sp|T00002|NIFD_TEST Test NifD OS=Test OX=1 GN=nifD PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
    ">sp|T00003|NIFK_TEST Test NifK OS=Test OX=1 GN=nifK PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _ref(tmp_path, text=REF_FULL):
    return _write(tmp_path / "ref.faa", text)


def _assembly(tmp_path):
    return _write(tmp_path / "asm.fna", ">contig1\nACGTACGTACGTACGT\n")


def _log_text(out_dir):
    logs = sorted(out_dir.rglob("*_nif_check.log"))
    assert logs, f"no nif_check log under {out_dir}"
    return logs[0].read_text()


def _base_args(tmp_path, ref):
    return ["-s", SAMPLE, "--ref", str(ref), "-o", str(tmp_path / "out")]


def _assert_no_verdict(log, reason):
    assert "VERDICT:" not in log
    assert f"ERROR: no VERDICT - {reason}" in log


def _valid_hits(tmp_path):
    rows = []
    for ref_id, start, end in (
        ("sp|T00001|NIFH_TEST", 100, 1000),
        ("sp|T00002|NIFD_TEST", 1200, 2100),
        ("sp|T00003|NIFK_TEST", 2300, 3200),
    ):
        rows.append("\t".join(str(x) for x in (
            ref_id, "contig1", "99.0", 300, 300, "1e-50", "500", start, end, 1, 300)))
    return _write(tmp_path / "hits11.tsv", "\n".join(rows) + "\n")


def _valid_diamond_hits(tmp_path):
    rows = []
    for ref_id in ("sp|T00001|NIFH_TEST", "sp|T00002|NIFD_TEST", "sp|T00003|NIFK_TEST"):
        for i in range(25):
            rows.append("\t".join(str(x) for x in (
                f"read{i}", ref_id, "90.0", 45, "1e-20", "100")))
    return _write(tmp_path / "diamond_hits.tsv", "\n".join(rows) + "\n")


def _fake_reads(tmp_path, which=(1, 2)):
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir(exist_ok=True)
    for i in which:
        (reads_dir / f"{SAMPLE}_R{i}.trim.fastq.gz").write_bytes(b"fake")
    return reads_dir


# --- Item 1: -a pointing nowhere -------------------------------------------

def test_missing_assembly_exits_1_no_reads_fallback(tmp_path):
    # Reads are present, so on the old code this would silently produce a
    # reads-mode VERDICT despite -a being given.
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", "/khong/co.fna", "-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "assembly not found: /khong/co.fna")
    assert "falling back to reads" not in log


# --- Item 2: reads mode needs both R1 and R2 -------------------------------

def test_reads_missing_r2(tmp_path):
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path, which=(1,)))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "reads not found: TESTX_R2.trim.fastq.gz")


def test_reads_missing_r1(tmp_path):
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path, which=(2,)))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "reads not found: TESTX_R1.trim.fastq.gz")


def test_reads_none_found_keeps_old_message(tmp_path):
    # Guard: the both-missing case keeps the old branch (e) message, which
    # an existing test pins.
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(reads_dir)]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "no trimmed reads found")


# --- Item 3: malformed hits -------------------------------------------------

def test_tblastn_short_rows_malformed(tmp_path, monkeypatch):
    valid = _valid_hits(tmp_path).read_text()
    short1 = "sp|T00001|NIFH_TEST\tcontig1\t99.0\t300\t300\t1e-50\t500\t100\n"
    short2 = "sp|T00002|NIFD_TEST\tcontig1\t99.0\n"
    # One column short of 11: pins the 11-column threshold exactly.
    short3 = "sp|T00003|NIFK_TEST\tcontig1\t99.0\t300\t300\t1e-50\t500\t2300\t3200\t1\n"
    hits = _write(tmp_path / "hits_short.tsv", short1 + short2 + short3 + valid)
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    assert "VERDICT:" not in log
    assert "ERROR: no VERDICT - malformed hits: 3 line(s), first: " in log
    assert short1.strip() in log


def test_pure_old_format_keeps_old_message(tmp_path, monkeypatch):
    # Guard: a file whose rows are ALL 9-column is the known old format,
    # not corruption; its existing branch (and existing test) must survive.
    hits = _write(
        tmp_path / "hits9.tsv",
        "sp|T00001|NIFH_TEST\tcontig1\t99.0\t100\t300\t1e-30\t200\t1\t300\n",
    )
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "tblastn hits in old 9-column format")


def test_diamond_short_rows_malformed(tmp_path, monkeypatch):
    valid = _valid_diamond_hits(tmp_path).read_text()
    short1 = "readX\tsp|T00001|NIFH_TEST\t90.0\n"
    # One column short of 6: pins the 6-column threshold exactly.
    short2 = "readY\tsp|T00002|NIFD_TEST\t90.0\t45\t1e-20\n"
    hits = _write(tmp_path / "diamond_short.tsv", short1 + short2 + valid)
    monkeypatch.setattr(check_nif, "run_diamond_reads", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    assert "VERDICT:" not in log
    assert "ERROR: no VERDICT - malformed hits: 2 line(s), first: " in log
    assert short1.strip() in log


def test_diamond_all_rows_short_malformed(tmp_path, monkeypatch):
    # PM (31/10): diamond has no old-format branch, so a file whose rows are ALL
    # short must still be an error (not skipped as "old format" -> silent ABSENT).
    short1 = "readX\tsp|T00001|NIFH_TEST\t90.0\t45\t1e-20\n"
    short2 = "readY\tsp|T00002|NIFD_TEST\t90.0\t45\t1e-20\n"
    hits = _write(tmp_path / "diamond_all_short.tsv", short1 + short2)
    monkeypatch.setattr(check_nif, "run_diamond_reads", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    assert "VERDICT:" not in log
    assert "ERROR: no VERDICT - malformed hits: 2 line(s), first: " in log

def test_tblastn_bad_numeric_malformed(tmp_path, monkeypatch):
    rows = _valid_hits(tmp_path).read_text().splitlines()
    parts = rows[0].split("\t")
    parts[2] = "not_a_number"
    rows[0] = "\t".join(parts)
    hits = _write(tmp_path / "hits_badnum.tsv", "\n".join(rows) + "\n")
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "malformed hits: 1 line(s), first: ")
    assert "not_a_number" in log


def test_diamond_bad_numeric_malformed(tmp_path, monkeypatch):
    rows = _valid_diamond_hits(tmp_path).read_text().splitlines()
    parts = rows[0].split("\t")
    parts[2] = "not_a_number"
    rows[0] = "\t".join(parts)
    hits = _write(tmp_path / "diamond_badnum.tsv", "\n".join(rows) + "\n")
    monkeypatch.setattr(check_nif, "run_diamond_reads", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "malformed hits: 1 line(s), first: ")
    assert "not_a_number" in log


def test_malformed_first_line_truncated_to_200(tmp_path, monkeypatch):
    long_line = "A" * 250
    hits = _write(tmp_path / "hits_long.tsv", long_line + "\n" + _valid_hits(tmp_path).read_text())
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    error_lines = [ln for ln in log.splitlines() if "malformed hits:" in ln]
    assert len(error_lines) == 1
    first_part = error_lines[0].split("first: ", 1)[1]
    assert first_part == "A" * 200


def test_validate_hits_direct(tmp_path):
    validate_hits = getattr(check_nif, "validate_hits")
    hits = _write(tmp_path / "h.tsv", "a\tb\n" + _valid_hits(tmp_path).read_text())
    count, first = validate_hits(str(hits), 11, old_format_ok=True)
    assert count == 1
    assert first == "a\tb"
    empty = _write(tmp_path / "empty.tsv", "")
    assert validate_hits(str(empty), 11, old_format_ok=True) == (0, None)
    blank = _write(tmp_path / "blank.tsv", "\n\n")
    assert validate_hits(str(blank), 6, old_format_ok=False) == (0, None)


# --- Item 4: unexpected exception -------------------------------------------

def test_unexpected_exception_becomes_error_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_nif, "load_gene_map",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom-test")))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "unexpected RuntimeError: boom-test")
    captured = capsys.readouterr()
    assert "Traceback" in captured.err
    assert "RuntimeError: boom-test" in captured.err


def test_systemexit_passes_through(tmp_path, monkeypatch):
    # Guard, not a new behaviour: the old code also let SystemExit escape
    # (it had no catch at all); the fix must preserve that.
    monkeypatch.setattr(check_nif, "load_gene_map",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit(3)))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    with pytest.raises(SystemExit) as excinfo:
        main(args)
    assert excinfo.value.code == 3


# --- Valid behaviour unchanged ----------------------------------------------

def test_valid_hits_same_verdict(tmp_path, monkeypatch):
    hits = _valid_hits(tmp_path)
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "VERDICT: PRESENT (nifHDK clustered)" in log
    assert "ERROR: no VERDICT" not in log


def test_empty_hits_still_absent(tmp_path, monkeypatch):
    hits = _write(tmp_path / "hits_empty.tsv", "")
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "VERDICT: ABSENT" in log
    assert "ERROR: no VERDICT" not in log


def test_empty_diamond_hits_still_absent(tmp_path, monkeypatch):
    hits = _write(tmp_path / "diamond_empty.tsv", "")
    monkeypatch.setattr(check_nif, "run_diamond_reads", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "VERDICT: ABSENT" in log
    assert "ERROR: no VERDICT" not in log


def test_no_assembly_with_both_reads_goes_to_reads_mode(tmp_path, monkeypatch):
    hits = _valid_diamond_hits(tmp_path)
    monkeypatch.setattr(check_nif, "run_diamond_reads", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "Mode: reads (" in log
    assert "VERDICT: PRESENT (reads only - confirm with assembly)" in log
    assert "ERROR: no VERDICT" not in log
