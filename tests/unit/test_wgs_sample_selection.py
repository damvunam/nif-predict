"""Unit tests for WGS sample raw-read selection, platform detection, and read-length peeking (W12)."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

# wgs_qc.py / run_wgs_sample.py import sibling modules by bare name (cwd = nifpredict/)
NIFPREDICT_DIR = Path(__file__).resolve().parents[2] / "nifpredict"
sys.path.insert(0, str(NIFPREDICT_DIR))

import run_wgs_sample  # noqa: E402
import wgs_loader  # noqa: E402
import wgs_qc  # noqa: E402


def _touch(path: Path) -> Path:
    """Creates an empty file at path."""
    path.write_text("")
    return path


def _fastq_text(read_lengths: list[int]) -> str:
    """Builds minimal valid FASTQ text with one 4-line record per read length."""
    lines: list[str] = []
    for idx, length in enumerate(read_lengths, start=1):
        lines.append(f"@read_{idx}")
        lines.append("A" * length)
        lines.append("+")
        lines.append("I" * length)
    return "\n".join(lines) + ("\n" if lines else "")


def _write_fastq(path: Path, read_lengths: list[int]) -> Path:
    """Writes minimal FASTQ records to plain or .gz path."""
    content = _fastq_text(read_lengths)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as handle:
            handle.write(content)
    else:
        path.write_text(content)
    return path


# ==============================================================================
# A. run_wgs_sample.get_sample_raw_reads(data_dir, sample_name)
# ==============================================================================


def test_get_sample_raw_reads_standard_pair_returns_r1_then_r2(tmp_path: Path) -> None:
    """A1: Directory with X35_1.fq.gz and X35_2.fq.gz returns (r1, r2) in sorted order."""
    r1 = _touch(tmp_path / "X35_1.fq.gz")
    r2 = _touch(tmp_path / "X35_2.fq.gz")
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (str(r1), str(r2))


@pytest.mark.parametrize("sep", ["_", ".", "-"])
def test_get_sample_raw_reads_accepts_valid_separators(tmp_path: Path, sep: str) -> None:
    """A2: Valid separators '_', '.', '-' immediately after sample_name are accepted."""
    r1 = _touch(tmp_path / f"X35{sep}R1.fastq.gz")
    r2 = _touch(tmp_path / f"X35{sep}R2.fastq.gz")
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (str(r1), str(r2))


def test_get_sample_raw_reads_prefix_collision_both_samples_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A3: Both G1_* and G1X_* present -> returns G1 pair and prints 2 Warning lines ignoring G1X."""
    # NOTE: find_raw_reads prints "Found 4 file(s) for sample 'G1'" before get_sample_raw_reads
    # filters out the G1X_* files, which can look misleading in logs.
    g1_1 = _touch(tmp_path / "G1_1.fq.gz")
    g1_2 = _touch(tmp_path / "G1_2.fq.gz")
    g1x_1 = _touch(tmp_path / "G1X_1.fq.gz")
    g1x_2 = _touch(tmp_path / "G1X_2.fq.gz")

    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "G1")
    assert res == (str(g1_1), str(g1_2))

    out_lines = capsys.readouterr().out.splitlines()
    warnings = [line for line in out_lines if line.startswith("Warning: ignoring ")]
    assert len(warnings) == 2
    assert any(str(g1x_1) in w and "(does not match sample name G1)" in w for w in warnings)
    assert any(str(g1x_2) in w and "(does not match sample name G1)" in w for w in warnings)


def test_get_sample_raw_reads_prefix_collision_only_longer_sample_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A4: Only C2X_1.fq.gz and C2X_2.fq.gz present; querying C2 returns (None, None) and never C2X files."""
    _touch(tmp_path / "C2X_1.fq.gz")
    _touch(tmp_path / "C2X_2.fq.gz")

    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "C2")
    assert res == (None, None)

    out_lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("Error: expected 2 fastq files for C2, found 0") for line in out_lines)


def test_get_sample_raw_reads_no_matching_files_returns_none_tuple(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A5: No matching files in directory -> returns (None, None)."""
    _touch(tmp_path / "OTHER_1.fq.gz")
    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35")
    assert res == (None, None)
    out = capsys.readouterr().out
    assert "Error: No fastq files found for 'X35'" in out


def test_get_sample_raw_reads_three_matching_files_returns_none_tuple(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A6: 3 matching files (X35_1, X35_2, X35_3) -> returns (None, None) and prints Error."""
    _touch(tmp_path / "X35_1.fq.gz")
    _touch(tmp_path / "X35_2.fq.gz")
    _touch(tmp_path / "X35_3.fq.gz")

    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35")
    assert res == (None, None)

    out_lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("Error: expected 2 fastq files for X35, found 3") for line in out_lines)


def test_get_sample_raw_reads_accepts_bare_sample_name_before_extension_dot(tmp_path: Path) -> None:
    """A7: X35.fq has '.' at base_name[len('X35')], so it is accepted alongside X35_2.fq."""
    # NOTE: Because '.' is in ("_", ".", "-"), a single-end or unnumbered file named `X35.fq`
    # passes the prefix separator check (`base_name[len("X35")] == "."`) and is paired with `X35_2.fq`.
    f_dot = _touch(tmp_path / "X35.fq")
    f_sub = _touch(tmp_path / "X35_2.fq")
    # In ASCII, '.' (46) < '_' (95), so X35.fq sorts before X35_2.fq
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (str(f_dot), str(f_sub))


def test_get_sample_raw_reads_mixed_separators_inverts_r1_r2_order(tmp_path: Path) -> None:
    """A8: Mixing separators (X_1.fq.gz and X.2.fq.gz) sorts '.' before '_', returning X.2.fq.gz as R1."""
    # NOTE: find_raw_reads sorts paths lexicographically (`sorted(found_files)`). Since ASCII '.' (46)
    # comes before '_' (95), `X.2.fq.gz` sorts before `X_1.fq.gz`, causing R1 and R2 to be swapped.
    r1_underscore = _touch(tmp_path / "X_1.fq.gz")
    r2_dot = _touch(tmp_path / "X.2.fq.gz")

    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X")
    assert res == (str(r2_dot), str(r1_underscore))


@pytest.mark.parametrize("ext", [".fastq", ".fastq.gz", ".fq", ".fq.gz"])
def test_find_raw_reads_supported_extensions(tmp_path: Path, ext: str) -> None:
    """A9a: All 4 supported extensions (.fastq, .fastq.gz, .fq, .fq.gz) are found by find_raw_reads."""
    f1 = _touch(tmp_path / f"X35_1{ext}")
    f2 = _touch(tmp_path / f"X35_2{ext}")
    assert wgs_loader.find_raw_reads(str(tmp_path), "X35") == [str(f1), str(f2)]
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (str(f1), str(f2))


def test_find_raw_reads_unsupported_extension_ignored(tmp_path: Path) -> None:
    """A9b: Unsupported extensions (e.g., .txt) are ignored."""
    _touch(tmp_path / "X35_1.txt")
    _touch(tmp_path / "X35_2.txt")
    assert wgs_loader.find_raw_reads(str(tmp_path), "X35") is None
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (None, None)


# ==============================================================================
# B. run_wgs_sample.determine_platform(platform, r1_path)
# ==============================================================================


def test_determine_platform_short_skips_peek_read_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: platform='short' returns 'short' immediately without calling peek_read_length or reading file."""
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("peek_read_length must not be called when platform='short'")

    monkeypatch.setattr(run_wgs_sample, "peek_read_length", _boom)
    missing_path = str(tmp_path / "nonexistent_R1.fq.gz")
    assert run_wgs_sample.determine_platform("short", missing_path) == "short"


def test_determine_platform_auto_150bp_returns_short(tmp_path: Path) -> None:
    """B2: platform='auto' on FASTQ with 150 bp reads -> returns 'short'."""
    fq = _write_fastq(tmp_path / "r1.fq", [150, 150])
    assert run_wgs_sample.determine_platform("auto", str(fq)) == "short"


def test_determine_platform_auto_boundary_299_vs_300(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """B3: Boundary 300 bp: 299 bp -> 'short'; 300 bp -> None with 'Error: long reads detected'."""
    fq_299 = _write_fastq(tmp_path / "r299.fq", [299])
    assert run_wgs_sample.determine_platform("auto", str(fq_299)) == "short"
    capsys.readouterr()

    fq_300 = _write_fastq(tmp_path / "r300.fq", [300])
    assert run_wgs_sample.determine_platform("auto", str(fq_300)) is None
    out = capsys.readouterr().out
    assert "Error: long reads detected" in out


def test_determine_platform_auto_empty_fastq_returns_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """B4: platform='auto' on empty FASTQ -> None with 'Error: could not determine read length'."""
    empty_fq = _write_fastq(tmp_path / "empty.fq", [])
    assert run_wgs_sample.determine_platform("auto", str(empty_fq)) is None
    out = capsys.readouterr().out
    assert "Error: could not determine read length" in out


# ==============================================================================
# C. wgs_qc.peek_read_length(fastq_path, n_reads=100)
# ==============================================================================


def test_peek_read_length_plain_and_gzip_identical(tmp_path: Path) -> None:
    """C1: Plain .fq and .fq.gz with 3 reads of lengths 100, 150, 200 both return 150.0."""
    fq_plain = _write_fastq(tmp_path / "sample.fq", [100, 150, 200])
    fq_gz = _write_fastq(tmp_path / "sample.fq.gz", [100, 150, 200])

    assert wgs_qc.peek_read_length(str(fq_plain)) == 150.0
    assert wgs_qc.peek_read_length(str(fq_gz)) == 150.0


def test_peek_read_length_stops_at_n_reads_100(tmp_path: Path) -> None:
    """C2: File with 150 reads and n_reads=100 averages only the first 100 reads."""
    # First 100 reads = 100 bp (avg 100.0); next 50 reads = 250 bp (overall avg would be 150.0)
    lengths = [100] * 100 + [250] * 50
    fq = _write_fastq(tmp_path / "reads150.fq", lengths)
    assert wgs_qc.peek_read_length(str(fq), n_reads=100) == 100.0


def test_peek_read_length_empty_file_returns_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """C3: Empty file returns None."""
    empty = _write_fastq(tmp_path / "empty.fq", [])
    assert wgs_qc.peek_read_length(str(empty)) is None
    out = capsys.readouterr().out
    assert f"Error: No reads found in {empty}" in out


def test_peek_read_length_missing_file_raises_filenotfounderror(tmp_path: Path) -> None:
    """C4: Non-existent file raises FileNotFoundError because open/gzip.open is not wrapped in try/except."""
    # NOTE: peek_read_length does not catch OSError / FileNotFoundError when opening fastq_path,
    # so passing a non-existent path raises FileNotFoundError instead of returning None.
    missing = tmp_path / "does_not_exist.fq.gz"
    with pytest.raises(FileNotFoundError):
        wgs_qc.peek_read_length(str(missing))
