"""Unit tests for MU9: archive an incomplete assembly directory before a clean rerun."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import pytest

from nifpredict.wgs import run_sample


def _last_nonempty_line(captured_out: str) -> str:
    lines = [line.strip() for line in captured_out.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _tree_contents(root: Path) -> dict[str, bytes]:
    res: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            res[str(path.relative_to(root))] = path.read_bytes()
    return res


def _make_assemble_mock(calls: list) -> object:
    def _assemble(platform, assembly_dir, r1_path=None, r2_path=None, threads=None):
        # The assembly directory must already be empty at this point: the
        # incomplete directory was archived and a clean one was created.
        assert os.listdir(assembly_dir) == []
        calls.append(assembly_dir)
        fasta = os.path.join(assembly_dir, "assembly.fasta")
        Path(fasta).write_text(">contig1\nATGC\n", encoding="utf-8")
        return fasta

    return _assemble


def _setup_main_mocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list]:
    """Copied/adapted from _setup_happy_mocks in test_run_sample_lock.py.

    Every pipeline step is mocked except run_sample_assembly itself, which
    stays real; only assemble_genome is replaced. The CLI -o points at
    tmp_path, so the sample root is tmp_path / "S1".
    """
    r1 = str(tmp_path / "S1_R1.fastq.gz")
    r2 = str(tmp_path / "S1_R2.fastq.gz")
    trim_r1 = str(tmp_path / "out" / "S1" / "qc" / "S1_R1.trim.fastq.gz")
    trim_r2 = str(tmp_path / "out" / "S1" / "qc" / "S1_R2.trim.fastq.gz")
    fastp_json = str(tmp_path / "out" / "S1" / "qc" / "S1.fastp.json")
    ref_fna = str(tmp_path / "ref.fna")
    Path(ref_fna).write_text(">ref\nATGC\n", encoding="utf-8")
    calls: list = []

    monkeypatch.setattr(run_sample, "get_sample_raw_reads", lambda d, s: (r1, r2))
    monkeypatch.setattr(run_sample, "determine_platform", lambda p, r: "short")
    monkeypatch.setattr(run_sample, "run_sample_qc", lambda r1_p, r2_p, q_d, s: (trim_r1, trim_r2, fastp_json))
    monkeypatch.setattr(
        run_sample,
        "sample_qc_gate_status",
        lambda j, s: {"status": "PASS", "q30": 0.9, "dup": 0.01, "reason": "ok"},
    )
    monkeypatch.setattr(run_sample, "assemble_genome", _make_assemble_mock(calls))
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

    return ["-s", "S1", "-d", str(tmp_path), "-o", str(tmp_path), "-r", ref_fna], calls


# ==============================================================================
# archive_incomplete_dir
# ==============================================================================


def test_archive_nonexistent_returns_none(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    assert archive(str(tmp_path / "does_not_exist")) is None
    assert not (tmp_path / "does_not_exist").exists()


def test_archive_regular_file_returns_none_and_file_intact(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    file_path = tmp_path / "some_file.txt"
    file_path.write_text("keep me\n", encoding="utf-8")
    assert archive(str(file_path)) is None
    assert file_path.read_text(encoding="utf-8") == "keep me\n"
    assert list(tmp_path.glob("*.incomplete_*")) == []


def test_archive_empty_dir_returns_none_and_stays(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    empty = tmp_path / "assembly"
    empty.mkdir()
    assert archive(str(empty)) is None
    assert empty.is_dir()
    assert list(tmp_path.glob("*.incomplete_*")) == []


def test_archive_nested_spades_content_moves_every_file(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    src = tmp_path / "assembly"
    nested = src / "spades_assembly" / "K21"
    nested.mkdir(parents=True)
    (nested / "x.txt").write_text("old-x\n", encoding="utf-8")
    (src / "spades_assembly" / "other.txt").write_text("old-other\n", encoding="utf-8")
    before = _tree_contents(src)
    fixed_now = datetime.datetime(2026, 10, 31, 12, 0, 0)

    dest = archive(str(src), now=fixed_now)

    assert dest == str(tmp_path / "assembly.incomplete_20261031_120000")
    assert not src.exists()
    assert _tree_contents(Path(dest)) == before


def test_archive_hidden_file_only_still_renames(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    src = tmp_path / "assembly"
    src.mkdir()
    (src / ".hidden").write_text("hidden\n", encoding="utf-8")
    fixed_now = datetime.datetime(2026, 10, 31, 12, 0, 0)

    dest = archive(str(src), now=fixed_now)

    assert dest == str(tmp_path / "assembly.incomplete_20261031_120000")
    assert not src.exists()
    assert (Path(dest) / ".hidden").read_text(encoding="utf-8") == "hidden\n"


def test_archive_trailing_slash_same_result(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    src = tmp_path / "assembly"
    src.mkdir()
    (src / "x.txt").write_text("x\n", encoding="utf-8")
    fixed_now = datetime.datetime(2026, 10, 31, 12, 0, 0)

    dest = archive(str(src) + "/", now=fixed_now)

    assert dest == str(tmp_path / "assembly.incomplete_20261031_120000")
    assert not src.exists()
    assert (Path(dest) / "x.txt").read_text(encoding="utf-8") == "x\n"


def test_archive_collision_uses_suffix_2_and_keeps_existing(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    fixed_now = datetime.datetime(2026, 10, 31, 12, 0, 0)
    existing = tmp_path / "assembly.incomplete_20261031_120000"
    existing.mkdir()
    (existing / "marker.txt").write_text("existing\n", encoding="utf-8")
    src = tmp_path / "assembly"
    src.mkdir()
    (src / "x.txt").write_text("new\n", encoding="utf-8")

    dest = archive(str(src), now=fixed_now)

    assert dest == str(tmp_path / "assembly.incomplete_20261031_120000_2")
    assert (existing / "marker.txt").read_text(encoding="utf-8") == "existing\n"
    assert (Path(dest) / "x.txt").read_text(encoding="utf-8") == "new\n"


def test_archive_collision_uses_suffix_3_when_2_exists(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    fixed_now = datetime.datetime(2026, 10, 31, 12, 0, 0)
    for suffix in ("", "_2"):
        existing = tmp_path / f"assembly.incomplete_20261031_120000{suffix}"
        existing.mkdir()
        (existing / "marker.txt").write_text(f"existing{suffix}\n", encoding="utf-8")
    src = tmp_path / "assembly"
    src.mkdir()
    (src / "x.txt").write_text("new\n", encoding="utf-8")

    dest = archive(str(src), now=fixed_now)

    assert dest == str(tmp_path / "assembly.incomplete_20261031_120000_3")
    assert (tmp_path / "assembly.incomplete_20261031_120000" / "marker.txt").read_text(
        encoding="utf-8"
    ) == "existing\n"
    assert (tmp_path / "assembly.incomplete_20261031_120000_2" / "marker.txt").read_text(
        encoding="utf-8"
    ) == "existing_2\n"
    assert (Path(dest) / "x.txt").read_text(encoding="utf-8") == "new\n"


def test_archive_collision_dest_is_regular_file_still_avoids(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    fixed_now = datetime.datetime(2026, 10, 31, 12, 0, 0)
    blocker = tmp_path / "assembly.incomplete_20261031_120000"
    blocker.write_text("i am a file\n", encoding="utf-8")
    src = tmp_path / "assembly"
    src.mkdir()
    (src / "x.txt").write_text("new\n", encoding="utf-8")

    dest = archive(str(src), now=fixed_now)

    assert dest == str(tmp_path / "assembly.incomplete_20261031_120000_2")
    assert blocker.read_text(encoding="utf-8") == "i am a file\n"


def test_archive_collision_dest_is_broken_symlink_still_avoids(tmp_path: Path) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    fixed_now = datetime.datetime(2026, 10, 31, 12, 0, 0)
    blocker = tmp_path / "assembly.incomplete_20261031_120000"
    os.symlink(tmp_path / "nowhere_target", blocker)
    assert os.path.lexists(blocker)
    assert not blocker.exists()
    src = tmp_path / "assembly"
    src.mkdir()
    (src / "x.txt").write_text("new\n", encoding="utf-8")

    dest = archive(str(src), now=fixed_now)

    assert dest == str(tmp_path / "assembly.incomplete_20261031_120000_2")
    assert os.path.islink(blocker)


def test_archive_rename_oserror_propagates_and_source_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = getattr(run_sample, "archive_incomplete_dir")
    src = tmp_path / "assembly"
    src.mkdir()
    (src / "x.txt").write_text("old\n", encoding="utf-8")

    def _raise(*args, **kwargs):
        raise OSError("cannot rename")

    monkeypatch.setattr(os, "rename", _raise)
    with pytest.raises(OSError, match="cannot rename"):
        archive(str(src), now=datetime.datetime(2026, 10, 31, 12, 0, 0))
    assert src.is_dir()
    assert (src / "x.txt").read_text(encoding="utf-8") == "old\n"


# ==============================================================================
# run_sample_assembly
# ==============================================================================


def test_run_sample_assembly_incomplete_dir_archived_and_clean_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []
    monkeypatch.setattr(run_sample, "assemble_genome", _make_assemble_mock(calls))
    assembly_dir = tmp_path / "assembly"
    nested = assembly_dir / "spades_assembly" / "K21"
    nested.mkdir(parents=True)
    (nested / "x.txt").write_text("old\n", encoding="utf-8")
    before = _tree_contents(assembly_dir)

    result = run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2)

    assert result == str(assembly_dir / "assembly.fasta")
    assert (assembly_dir / "assembly.fasta").is_file()
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "Warning: incomplete assembly moved to " in out
    assert "; starting a clean assembly" in out
    archived = list(tmp_path.glob("assembly.incomplete_*"))
    assert len(archived) == 1
    assert _tree_contents(archived[0]) == before


def test_run_sample_assembly_existing_fasta_skips_without_rename_or_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []
    monkeypatch.setattr(run_sample, "assemble_genome", _make_assemble_mock(calls))
    assembly_dir = tmp_path / "assembly"
    assembly_dir.mkdir()
    fasta = assembly_dir / "assembly.fasta"
    fasta.write_text(">old\nATGC\n", encoding="utf-8")
    (assembly_dir / "other.txt").write_text("other\n", encoding="utf-8")

    result = run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2)

    assert result == str(fasta)
    assert calls == []
    assert list(tmp_path.glob("assembly.incomplete_*")) == []
    assert f"Warning: {fasta} exists, skipping assembly." in capsys.readouterr().out
    assert (assembly_dir / "other.txt").read_text(encoding="utf-8") == "other\n"


def test_run_sample_assembly_empty_dir_no_rename_no_new_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []
    monkeypatch.setattr(run_sample, "assemble_genome", _make_assemble_mock(calls))
    assembly_dir = tmp_path / "assembly"
    assembly_dir.mkdir()

    result = run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2)

    assert result == str(assembly_dir / "assembly.fasta")
    assert len(calls) == 1
    assert list(tmp_path.glob("assembly.incomplete_*")) == []
    assert "incomplete assembly moved to" not in capsys.readouterr().out


def test_run_sample_assembly_missing_dir_created_and_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list = []
    monkeypatch.setattr(run_sample, "assemble_genome", _make_assemble_mock(calls))
    assembly_dir = tmp_path / "assembly"

    result = run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2)

    assert result == str(assembly_dir / "assembly.fasta")
    assert assembly_dir.is_dir()
    assert len(calls) == 1


def test_run_sample_assembly_assemble_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_sample, "assemble_genome", lambda *a, **kw: None)
    assembly_dir = tmp_path / "assembly"
    assembly_dir.mkdir()

    result = run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2)

    assert result is None
    assert f"Error: Assembly failed for {assembly_dir}" in capsys.readouterr().out


def test_run_sample_assembly_rename_error_returns_none_no_call_dir_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []
    monkeypatch.setattr(run_sample, "assemble_genome", _make_assemble_mock(calls))
    assembly_dir = tmp_path / "assembly"
    assembly_dir.mkdir()
    (assembly_dir / "x.txt").write_text("old\n", encoding="utf-8")

    def _raise(*args, **kwargs):
        raise OSError("cannot rename")

    monkeypatch.setattr(os, "rename", _raise)

    result = run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2)

    assert result is None
    assert calls == []
    assert list(tmp_path.glob("assembly.incomplete_*")) == []
    assert assembly_dir.is_dir()
    assert (assembly_dir / "x.txt").read_text(encoding="utf-8") == "old\n"
    assert "Error: cannot move incomplete assembly" in capsys.readouterr().out


def test_run_sample_assembly_two_consecutive_do_not_overwrite_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list = []
    monkeypatch.setattr(run_sample, "assemble_genome", _make_assemble_mock(calls))
    assembly_dir = tmp_path / "assembly"

    assembly_dir.mkdir()
    (assembly_dir / "first.txt").write_text("first\n", encoding="utf-8")
    assert run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2) is not None

    (assembly_dir / "assembly.fasta").unlink()
    (assembly_dir / "second.txt").write_text("second\n", encoding="utf-8")
    assert run_sample.run_sample_assembly("short", str(assembly_dir), "r1", "r2", 2) is not None

    archived = sorted(tmp_path.glob("assembly.incomplete_*"))
    assert len(archived) == 2
    contents = [_tree_contents(d) for d in archived]
    assert {"first.txt": b"first\n"} in contents
    assert {"second.txt": b"second\n"} in contents
    assert len(calls) == 2


# ==============================================================================
# main([...])
# ==============================================================================


def test_main_incomplete_assembly_archived_and_lock_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv, calls = _setup_main_mocks(tmp_path, monkeypatch)
    sample_root = tmp_path / "S1"
    assembly_dir = sample_root / "assembly"
    nested = assembly_dir / "spades_assembly" / "K21"
    nested.mkdir(parents=True)
    (nested / "x.txt").write_text("old\n", encoding="utf-8")

    rc = run_sample.main(argv)

    assert rc == 0
    assert len(calls) == 1
    archived = list(sample_root.glob("assembly.incomplete_*"))
    assert len(archived) == 1
    assert (archived[0] / "spades_assembly" / "K21" / "x.txt").read_text(encoding="utf-8") == "old\n"
    assert not (sample_root / ".run.lock").exists()
    assert "Warning: incomplete assembly moved to " in capsys.readouterr().out


def test_main_rename_error_returns_3_and_lock_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv, calls = _setup_main_mocks(tmp_path, monkeypatch)
    sample_root = tmp_path / "S1"
    assembly_dir = sample_root / "assembly"
    assembly_dir.mkdir(parents=True)
    (assembly_dir / "x.txt").write_text("old\n", encoding="utf-8")

    def _raise(*args, **kwargs):
        raise OSError("cannot rename")

    monkeypatch.setattr(os, "rename", _raise)

    rc = run_sample.main(argv)

    assert rc == 3
    assert calls == []
    assert assembly_dir.is_dir()
    assert (assembly_dir / "x.txt").read_text(encoding="utf-8") == "old\n"
    assert not (sample_root / ".run.lock").exists()
    out = capsys.readouterr().out
    assert "Error: cannot move incomplete assembly" in out
    assert _last_nonempty_line(out) == "STATUS: ERROR - assembly - de novo assembly failed for S1"
