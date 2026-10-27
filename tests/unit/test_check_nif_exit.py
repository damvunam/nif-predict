"""Unit tests for check_nif exit codes (MU1: no VERDICT must exit non-zero).

Every test calls main([...]) with tmp_path + monkeypatch, needs no network,
BLAST or diamond, and checks BOTH the return code AND the log: presence or
absence of a "VERDICT:" line and of the branch-specific
"ERROR: no VERDICT - ..." line.
"""

from __future__ import annotations

from pathlib import Path

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

REF_NO_NIFK = (
    ">sp|T00001|NIFH_TEST Test NifH OS=Test OX=1 GN=nifH PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
    ">sp|T00002|NIFD_TEST Test NifD OS=Test OX=1 GN=nifD PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _ref(tmp_path: Path, text: str = REF_FULL) -> Path:
    return _write(tmp_path / "ref.faa", text)


def _assembly(tmp_path: Path) -> Path:
    return _write(tmp_path / "asm.fna", ">contig1\nACGTACGTACGTACGT\n")


def _log_text(out_dir: Path) -> str:
    logs = sorted(out_dir.rglob("*_nif_check.log"))
    assert logs, f"no nif_check log under {out_dir}"
    return logs[0].read_text()


def _base_args(tmp_path: Path, ref: Path) -> list[str]:
    return ["-s", SAMPLE, "--ref", str(ref), "-o", str(tmp_path / "out")]


def _assert_no_verdict(log: str, reason: str) -> None:
    assert "VERDICT:" not in log
    assert f"ERROR: no VERDICT - {reason}" in log


# --- Branch (a): reference file missing or empty -----------------------------

def test_e1_ref_missing(tmp_path):
    rc = main(_base_args(tmp_path, tmp_path / "does_not_exist.faa"))
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "reference file missing or empty")


def test_e2_ref_empty(tmp_path):
    ref = _write(tmp_path / "empty.faa", "")
    rc = main(_base_args(tmp_path, ref))
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "reference file missing or empty")


# --- Branch (b): a gene has 0 reference proteins -----------------------------

def test_e3_ref_missing_nifk(tmp_path, monkeypatch):
    # run_tblastn is patched only so a (hypothetical) fall-through past
    # branch (b) cannot reach real BLAST; correct code returns before it.
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: None)
    args = _base_args(tmp_path, _ref(tmp_path, REF_NO_NIFK))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "a nif gene has 0 reference proteins")


# --- Branch (c): run_tblastn returns None ------------------------------------

def test_e4_tblastn_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: None)
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "tblastn failed")


# --- Branch (d): summarize_tblastn returns None (old 9-column hits) ----------

def test_e5_hits_old_9col_format(tmp_path, monkeypatch):
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


# --- Branch (e): reads mode, no trimmed reads found --------------------------

def test_e6_reads_dir_empty(tmp_path):
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(reads_dir)]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "no trimmed reads found")


# --- Branch (f): run_diamond_reads returns None ------------------------------

def _fake_reads(tmp_path: Path) -> Path:
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    for i in (1, 2):
        (reads_dir / f"{SAMPLE}_R{i}.trim.fastq.gz").write_bytes(b"fake")
    return reads_dir


def test_e7_diamond_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(check_nif, "run_diamond_reads", lambda *a, **k: None)
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    _assert_no_verdict(log, "diamond reads search failed")


# --- Success paths: a VERDICT (any verdict) exits 0 --------------------------

def test_e8_assembly_present_clustered(tmp_path, monkeypatch):
    rows = []
    for ref_id, start, end in (
        ("sp|T00001|NIFH_TEST", 100, 1000),
        ("sp|T00002|NIFD_TEST", 1200, 2100),
        ("sp|T00003|NIFK_TEST", 2300, 3200),
    ):
        rows.append("\t".join(str(x) for x in (
            ref_id, "contig1", "99.0", 300, 300, "1e-50", "500", start, end, 1, 300)))
    hits = _write(tmp_path / "hits11.tsv", "\n".join(rows) + "\n")
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "VERDICT: PRESENT (nifHDK clustered)" in log
    assert "ERROR: no VERDICT" not in log


def test_e9_assembly_absent_empty_hits(tmp_path, monkeypatch):
    hits = _write(tmp_path / "hits_empty.tsv", "")
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "VERDICT: ABSENT" in log
    assert "ERROR: no VERDICT" not in log


def test_e10_reads_present(tmp_path, monkeypatch):
    rows = []
    for ref_id in ("sp|T00001|NIFH_TEST", "sp|T00002|NIFD_TEST", "sp|T00003|NIFK_TEST"):
        for i in range(25):
            rows.append("\t".join(str(x) for x in (
                f"read{i}", ref_id, "90.0", 45, "1e-20", "100")))
    hits = _write(tmp_path / "diamond_hits.tsv", "\n".join(rows) + "\n")
    monkeypatch.setattr(check_nif, "run_diamond_reads", lambda *a, **k: str(hits))
    args = _base_args(tmp_path, _ref(tmp_path))
    args += ["-r", str(_fake_reads(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "VERDICT: PRESENT (reads only - confirm with assembly)" in log
    assert "ERROR: no VERDICT" not in log
