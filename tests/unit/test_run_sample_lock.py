"""Unit tests for W17: per-sample output directory locking in nifpredict.wgs.run_sample."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from nifpredict.wgs import run_sample


def _last_nonempty_line(captured_out: str) -> str:
    lines = [line.strip() for line in captured_out.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _parse_kv_file(path: Path) -> dict[str, str]:
    res: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            res[k.strip()] = v.strip()
    return res


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


# ==============================================================================
# 1. _pid_alive & _read_boot_id
# ==============================================================================


@pytest.mark.parametrize("bad_pid", [0, -1, -999])
def test_pid_alive_non_positive_returns_false_without_calling_kill(
    bad_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    kill_calls: list[tuple[int, int]] = []

    def _spy_kill(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))

    monkeypatch.setattr(os, "kill", _spy_kill)
    assert run_sample._pid_alive(bad_pid) is False
    assert kill_calls == []


def test_pid_alive_current_pid_is_true() -> None:
    assert run_sample._pid_alive(os.getpid()) is True


def test_pid_alive_dead_subprocess_is_false() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert run_sample._pid_alive(proc.pid) is False


def test_pid_alive_permission_error_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_perm(pid: int, sig: int) -> None:
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(os, "kill", _raise_perm)
    assert run_sample._pid_alive(12345) is True


def test_read_boot_id_returns_string_or_empty_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    val = run_sample._read_boot_id()
    assert isinstance(val, str)

    def _raise_oserror(*args, **kwargs):
        raise OSError("proc not mounted")

    monkeypatch.setattr("builtins.open", _raise_oserror)
    assert run_sample._read_boot_id() == ""


# ==============================================================================
# 2. acquire_run_lock
# ==============================================================================


def test_acquire_run_lock_first_time_ok(tmp_path: Path) -> None:
    sample_root = tmp_path / "out" / "S1"
    res = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1", "-d", "/data/raw reads"])

    assert res["status"] == "OK"
    lock_file = sample_root / ".run.lock"
    assert res["path"] == str(lock_file)
    assert lock_file.is_file()

    kv = _parse_kv_file(lock_file)
    assert set(kv.keys()) >= {"pid", "hostname", "boot_id", "started", "argv"}
    assert int(kv["pid"]) == os.getpid()
    assert kv["hostname"] == socket.gethostname()
    assert "T" in kv["started"]
    assert "-s S1" in kv["argv"]


def test_acquire_run_lock_rejects_live_pid_without_touching_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_root = tmp_path / "out" / "S1"
    first = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert first["status"] == "OK"

    extra_file = sample_root / "qc" / "existing.txt"
    extra_file.parent.mkdir(parents=True)
    extra_file.write_text("keep me\n", encoding="utf-8")

    lock_file = sample_root / ".run.lock"
    original_lock_text = lock_file.read_text(encoding="utf-8")
    original_kv = _parse_kv_file(lock_file)
    tree_before = sorted(sample_root.rglob("*"))

    second = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert second["status"] == "ERROR"

    out = capsys.readouterr().out
    last_line = _last_nonempty_line(out)
    expected = (
        f"STATUS: ERROR - lock - S1 is being written by pid {os.getpid()} "
        f"since {original_kv['started']}"
    )
    assert last_line == expected
    assert lock_file.read_text(encoding="utf-8") == original_lock_text
    assert sorted(sample_root.rglob("*")) == tree_before
    assert extra_file.read_text(encoding="utf-8") == "keep me\n"


def test_acquire_run_lock_takes_over_dead_pid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    dead_pid = proc.pid

    sample_root = tmp_path / "out" / "S1"
    sample_root.mkdir(parents=True)
    lock_file = sample_root / ".run.lock"
    lock_file.write_text(
        f"pid={dead_pid}\n"
        f"hostname={socket.gethostname()}\n"
        f"boot_id={run_sample._read_boot_id()}\n"
        "started=2026-01-01T00:00:00\n"
        "argv=-s S1\n",
        encoding="utf-8",
    )

    res = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert res["status"] == "OK"
    out = capsys.readouterr().out
    assert f"Warning: stale lock {lock_file} (pid {dead_pid} not running), taking over" in out

    kv = _parse_kv_file(lock_file)
    assert int(kv["pid"]) == os.getpid()


def test_acquire_run_lock_takes_over_different_boot_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_root = tmp_path / "out" / "S1"
    monkeypatch.setattr(run_sample, "_read_boot_id", lambda: "boot-old-111")
    first = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert first["status"] == "OK"

    monkeypatch.setattr(run_sample, "_read_boot_id", lambda: "boot-new-222")
    second = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert second["status"] == "OK"

    out = capsys.readouterr().out
    assert "Warning: stale lock" in out
    assert "boot_id" in out
    kv = _parse_kv_file(sample_root / ".run.lock")
    assert kv["boot_id"] == "boot-new-222"
    assert int(kv["pid"]) == os.getpid()


def test_acquire_run_lock_takes_over_different_hostname(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_root = tmp_path / "out" / "S1"
    sample_root.mkdir(parents=True)
    lock_file = sample_root / ".run.lock"
    lock_file.write_text(
        f"pid={os.getpid()}\n"
        "hostname=other-node.local\n"
        f"boot_id={run_sample._read_boot_id()}\n"
        "started=2026-01-01T00:00:00\n"
        "argv=-s S1\n",
        encoding="utf-8",
    )

    res = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert res["status"] == "OK"
    out = capsys.readouterr().out
    assert "Warning: stale lock" in out
    assert "hostname" in out

    kv = _parse_kv_file(lock_file)
    assert kv["hostname"] == socket.gethostname()
    assert int(kv["pid"]) == os.getpid()


@pytest.mark.parametrize(
    "corrupt_content",
    [
        "not_key_value_garbage_at_all\n!!!\n",
        "hostname=node1\nstarted=2026-01-01T00:00:00\n",
        "pid=abc\nhostname=node1\n",
        "pid=0\nhostname=node1\n",
        "pid=-5\nhostname=node1\n",
    ],
)
def test_acquire_run_lock_corrupt_lock_returns_error_and_preserves_file(
    tmp_path: Path, corrupt_content: str, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_root = tmp_path / "out" / "S1"
    sample_root.mkdir(parents=True)
    lock_file = sample_root / ".run.lock"
    lock_file.write_text(corrupt_content, encoding="utf-8")

    res = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert res["status"] == "ERROR"

    out = capsys.readouterr().out
    last_line = _last_nonempty_line(out)
    assert last_line.startswith(f"STATUS: ERROR - lock - cannot read {lock_file}:")
    assert lock_file.read_text(encoding="utf-8") == corrupt_content


# ==============================================================================
# 3. release_run_lock
# ==============================================================================


def test_release_run_lock_removes_own_lock_quietly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_root = tmp_path / "out" / "S1"
    lock = run_sample.acquire_run_lock(str(sample_root), ["-s", "S1"])
    assert lock["status"] == "OK"
    capsys.readouterr()

    released = run_sample.release_run_lock(lock["path"])
    assert released is True
    assert not Path(lock["path"]).exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_release_run_lock_does_not_remove_other_pid_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock_file = tmp_path / ".run.lock"
    other_pid = os.getpid() + 100000
    content = (
        f"pid={other_pid}\n"
        f"hostname={socket.gethostname()}\n"
        f"boot_id={run_sample._read_boot_id()}\n"
        "started=2026-01-01T00:00:00\n"
        "argv=-s S1\n"
    )
    lock_file.write_text(content, encoding="utf-8")

    released = run_sample.release_run_lock(str(lock_file))
    assert released is False
    assert lock_file.exists()
    assert lock_file.read_text(encoding="utf-8") == content
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# PM (W17 review): mutants "ignore hostname" and "ValueError not caught" survived the original suite
def test_release_run_lock_does_not_remove_same_pid_other_hostname(tmp_path: Path) -> None:
    lock_file = tmp_path / ".run.lock"
    content = (
        f"pid={os.getpid()}\n"
        "hostname=other-node.local\n"
        f"boot_id={run_sample._read_boot_id()}\n"
        "started=2026-01-01T00:00:00\n"
        "argv=-s S1\n"
    )
    lock_file.write_text(content, encoding="utf-8")

    assert run_sample.release_run_lock(str(lock_file)) is False
    assert lock_file.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("bad_content", ["", "hostname=node1\n", "pid=abc\nhostname=node1\n", "garbage\n"])
def test_release_run_lock_unparsable_lock_returns_false_and_keeps_file(
    tmp_path: Path, bad_content: str, capsys: pytest.CaptureFixture[str]
) -> None:
    lock_file = tmp_path / ".run.lock"
    lock_file.write_text(bad_content, encoding="utf-8")

    assert run_sample.release_run_lock(str(lock_file)) is False
    assert lock_file.read_text(encoding="utf-8") == bad_content
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_release_run_lock_missing_file_returns_false_quietly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does_not_exist" / ".run.lock"
    assert run_sample.release_run_lock(str(missing)) is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ==============================================================================
# 4. _install_sigterm_handler
# ==============================================================================


def test_install_sigterm_handler_main_thread_raises_143_and_restores() -> None:
    orig_handler = signal.getsignal(signal.SIGTERM)
    try:
        prev = run_sample._install_sigterm_handler()
        assert prev == orig_handler
        installed = signal.getsignal(signal.SIGTERM)
        assert callable(installed)
        with pytest.raises(SystemExit) as exc_info:
            installed(signal.SIGTERM, None)
        assert exc_info.value.code == 143
    finally:
        signal.signal(signal.SIGTERM, orig_handler)


def test_install_sigterm_handler_worker_thread_returns_none() -> None:
    orig_handler = signal.getsignal(signal.SIGTERM)
    result_holder: list[object] = ["sentinel"]

    def _worker() -> None:
        result_holder[0] = run_sample._install_sigterm_handler()

    try:
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        assert result_holder[0] is None
    finally:
        signal.signal(signal.SIGTERM, orig_handler)


# ==============================================================================
# 5. main([...]) integration with lock lifecycle
# ==============================================================================


def test_main_releases_lock_after_done_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    lock_path = tmp_path / "out" / "S1" / ".run.lock"

    rc = run_sample.main(argv)
    assert rc == 0
    assert not lock_path.exists()
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line == "DONE: S1 passed gate 1, gate 2 and identity; stopped before annotation"


def test_main_releases_lock_after_gate1_fail_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_sample,
        "sample_qc_gate_status",
        lambda j, s: {"status": "FAIL", "q30": 0.70, "dup": 0.05, "reason": "Q30 rate too low"},
    )
    lock_path = tmp_path / "out" / "S1" / ".run.lock"

    rc = run_sample.main(argv)
    assert rc == 1
    assert not lock_path.exists()
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: FAIL - gate1 - ")


def test_main_releases_lock_after_fastp_error_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(run_sample, "run_sample_qc", lambda r1, r2, q_d, s: (None, None, None))
    lock_path = tmp_path / "out" / "S1" / ".run.lock"

    rc = run_sample.main(argv)
    assert rc == 3
    assert not lock_path.exists()
    last_line = _last_nonempty_line(capsys.readouterr().out)
    assert last_line.startswith("STATUS: ERROR - fastp - ")


def test_main_releases_lock_when_step_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected assembly crash")

    monkeypatch.setattr(run_sample, "run_sample_assembly", _boom)
    lock_path = tmp_path / "out" / "S1" / ".run.lock"

    with pytest.raises(RuntimeError, match="unexpected assembly crash"):
        run_sample.main(argv)

    assert not lock_path.exists()


def test_main_returns_3_when_locked_by_live_pid_and_calls_no_pipeline_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    sample_root = tmp_path / "out" / "S1"
    sample_root.mkdir(parents=True)
    lock_path = sample_root / ".run.lock"

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock_content = (
            f"pid={proc.pid}\n"
            f"hostname={socket.gethostname()}\n"
            f"boot_id={run_sample._read_boot_id()}\n"
            "started=2026-05-01T12:00:00\n"
            "argv=-s S1 -d /tmp\n"
        )
        lock_path.write_text(lock_content, encoding="utf-8")

        for step_fn in (
            "run_sample_qc",
            "sample_qc_gate_status",
            "run_sample_assembly",
            "sample_gate2_status",
            "run_sample_contig_stats",
            "print_sample_coverage",
            "check_sample_identity",
        ):
            monkeypatch.setattr(
                run_sample,
                step_fn,
                lambda *a, _fn=step_fn, **kw: pytest.fail(f"{_fn} must not be called when lock fails"),
            )

        rc = run_sample.main(argv)
        assert rc == 3
        last_line = _last_nonempty_line(capsys.readouterr().out)
        assert last_line == f"STATUS: ERROR - lock - S1 is being written by pid {proc.pid} since 2026-05-01T12:00:00"
        assert lock_path.exists()
        assert lock_path.read_text(encoding="utf-8") == lock_content
    finally:
        proc.terminate()
        proc.wait()


def test_main_dry_run_does_not_create_sample_dir_or_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch) + ["--dry_run"]
    sample_root = tmp_path / "out" / "S1"
    lock_path = sample_root / ".run.lock"

    rc = run_sample.main(argv)
    assert rc == 0
    assert not lock_path.exists()
    assert not sample_root.exists()
    assert "[DRY-RUN] Sample: S1" in capsys.readouterr().out


def test_main_sigterm_system_exit_143_releases_lock_and_restores_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _setup_happy_mocks(tmp_path, monkeypatch)
    lock_path = tmp_path / "out" / "S1" / ".run.lock"
    orig_handler = signal.getsignal(signal.SIGTERM)

    def _raise_sigterm_exit(*args, **kwargs):
        assert lock_path.exists()
        raise SystemExit(143)

    monkeypatch.setattr(run_sample, "run_sample_assembly", _raise_sigterm_exit)

    try:
        with pytest.raises(SystemExit) as exc_info:
            run_sample.main(argv)
        assert exc_info.value.code == 143
        assert not lock_path.exists()
        assert signal.getsignal(signal.SIGTERM) == orig_handler
    finally:
        signal.signal(signal.SIGTERM, orig_handler)
