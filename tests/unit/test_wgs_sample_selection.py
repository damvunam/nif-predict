"""Unit tests for WGS sample raw-read selection, platform detection, and read-length peeking (W12, updated W14)."""

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
# A. run_wgs_sample.get_sample_raw_reads(data_dir, sample_name) & get_mate_number
# ==============================================================================


def test_get_sample_raw_reads_standard_pair_returns_r1_then_r2(tmp_path: Path) -> None:
    """A1: Directory with X35_1.fq.gz and X35_2.fq.gz returns (r1, r2) in mate order."""
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


def test_get_sample_raw_reads_rejects_unnumbered_bare_sample_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A7: X35.fq + X35_2.fq -> (None, None) and prints Error: cannot tell R1 from R2 for X35."""
    _touch(tmp_path / "X35.fq")
    _touch(tmp_path / "X35_2.fq")

    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35")
    assert res == (None, None)

    out_lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("Error: cannot tell R1 from R2 for X35") for line in out_lines)


def test_get_sample_raw_reads_mixed_separators_returns_true_r1_then_r2(tmp_path: Path) -> None:
    """A8: Mixing separators (X_1.fq.gz and X.2.fq.gz) still returns (X_1.fq.gz, X.2.fq.gz) by mate number."""
    r1_underscore = _touch(tmp_path / "X_1.fq.gz")
    r2_dot = _touch(tmp_path / "X.2.fq.gz")

    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X")
    assert res == (str(r1_underscore), str(r2_dot))


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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("X35_1.fq.gz", 1),
        ("X35_R2.fastq.gz", 2),
        ("X35.r1.fq", 1),
        ("X35-2.fq", 2),
        ("X35_S1_L001_R1_001.fastq.gz", 1),
        ("X35.fq", None),
        ("X35_1_R2.fq.gz", None),
    ],
)
def test_get_mate_number_examples(filename: str, expected: int | None) -> None:
    """A10: get_mate_number returns 1, 2, or None across the 7 spec examples."""
    assert run_wgs_sample.get_mate_number(filename, "X35") == expected


def test_get_sample_raw_reads_r2_sorts_before_r1_still_returns_r1_first(tmp_path: Path) -> None:
    """A11: X35.R2.fq.gz sorts before X35_R1.fq.gz, but get_sample_raw_reads returns (X35_R1, X35.R2)."""
    r2 = _touch(tmp_path / "X35.R2.fq.gz")
    r1 = _touch(tmp_path / "X35_R1.fq.gz")
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (str(r1), str(r2))


def test_get_sample_raw_reads_duplicate_mate_numbers_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A12: Two files with the same mate number (X35_1.fq.gz and X35-1.fq.gz) -> (None, None) + Error."""
    _touch(tmp_path / "X35_1.fq.gz")
    _touch(tmp_path / "X35-1.fq.gz")

    res = run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35")
    assert res == (None, None)

    out_lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("Error: cannot tell R1 from R2") for line in out_lines)


def test_get_sample_raw_reads_illumina_style_filenames(tmp_path: Path) -> None:
    """A13: Illumina-style filenames X35_S1_L001_R1_001.fastq.gz + R2 -> returns (R1, R2)."""
    r1 = _touch(tmp_path / "X35_S1_L001_R1_001.fastq.gz")
    r2 = _touch(tmp_path / "X35_S1_L001_R2_001.fastq.gz")
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (str(r1), str(r2))


def test_get_sample_raw_reads_date_stamped_filenames(tmp_path: Path) -> None:
    """A14 (Claude, W14 review): digits inside a longer number (e.g. a date) are not mate numbers."""
    r1 = _touch(tmp_path / "X35_20231001_R1.fq.gz")
    r2 = _touch(tmp_path / "X35_20231001_R2.fq.gz")
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35") == (str(r1), str(r2))


def test_get_sample_raw_reads_sample_name_containing_mate_like_token(tmp_path: Path) -> None:
    """A15 (Claude, W14 review): the sample name itself is stripped before looking for the mate number (sample 'X35-2')."""
    r1 = _touch(tmp_path / "X35-2_R1.fq.gz")
    r2 = _touch(tmp_path / "X35-2_R2.fq.gz")
    assert run_wgs_sample.get_mate_number(str(r2), "X35-2") == 2
    assert run_wgs_sample.get_sample_raw_reads(str(tmp_path), "X35-2") == (str(r1), str(r2))


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
