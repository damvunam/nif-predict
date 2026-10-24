"""Unit tests for check_nif verdict logic (summarize_tblastn, is_nif_present_assembly, summarize_reads, is_nif_present_reads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nifpredict.check_nif import (
    is_nif_present_assembly,
    is_nif_present_reads,
    summarize_reads,
    summarize_tblastn,
)

GENE_MAP: dict[str, str] = {
    "ref_nifH_1": "nifH",
    "ref_nifH_2": "nifH",
    "ref_nifD_1": "nifD",
    "ref_nifK_1": "nifK",
    "ref_other": "other",
}


def _hit(
    contig: str,
    start: int,
    end: int,
    pident: float = 80.0,
    coverage: float = 0.95,
    ref: str = "ref_test",
    bitscore: float = 500.0,
) -> dict[str, Any]:
    """Helper to construct a single best-hit dictionary for is_nif_present_assembly."""
    return {
        "ref": ref,
        "contig": contig,
        "pident": pident,
        "coverage": coverage,
        "start": start,
        "end": end,
        "bitscore": bitscore,
    }


def _write_tsv(path: Path, rows: list[tuple[str, ...]]) -> Path:
    """Writes tabular rows to a TSV file in tmp_path."""
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return path


# ==============================================================================
# A. summarize_tblastn(hits_path, gene_map)
# ==============================================================================


def test_summarize_tblastn_keeps_highest_bitscore_not_first_row(tmp_path: Path) -> None:
    """A1: Multiple hits for the same gene -> keeps the one with highest bitscore (placed on row 2)."""
    hits = _write_tsv(
        tmp_path / "hits.tsv",
        [
            # qseqid, sseqid, pident, length, qlen, evalue, bitscore, sstart, send
            ("ref_nifH_1", "contig_1", "80.0", "280", "290", "1e-100", "400.0", "100", "940"),
            ("ref_nifH_2", "contig_2", "90.0", "285", "290", "1e-150", "550.0", "200", "1055"),
            ("ref_nifH_1", "contig_3", "85.0", "282", "290", "1e-120", "480.0", "300", "1146"),
        ],
    )
    best = summarize_tblastn(hits, GENE_MAP)
    assert set(best.keys()) == {"nifH"}
    assert best["nifH"]["ref"] == "ref_nifH_2"
    assert best["nifH"]["contig"] == "contig_2"
    assert best["nifH"]["bitscore"] == 550.0
    assert best["nifH"]["pident"] == 90.0


def test_summarize_tblastn_equal_bitscore_keeps_first_seen(tmp_path: Path) -> None:
    """A2: Two hits for the same gene with identical bitscore -> keeps the first one (>= skip check)."""
    hits = _write_tsv(
        tmp_path / "hits.tsv",
        [
            ("ref_nifH_1", "contig_first", "80.0", "280", "290", "1e-100", "500.0", "100", "940"),
            ("ref_nifH_2", "contig_second", "95.0", "290", "290", "1e-100", "500.0", "200", "1070"),
        ],
    )
    best = summarize_tblastn(hits, GENE_MAP)
    assert best["nifH"]["ref"] == "ref_nifH_1"
    assert best["nifH"]["contig"] == "contig_first"


def test_summarize_tblastn_coverage_calculation(tmp_path: Path) -> None:
    """A3: coverage == int(length) / int(qlen), e.g., 287 / 291."""
    # NOTE: coverage is computed as int(length) / int(qlen), where length is the BLAST alignment
    # length (which can include gaps) rather than (qend - qstart + 1), so gapped alignments can
    # slightly overestimate query coverage or exceed 1.0.
    hits = _write_tsv(
        tmp_path / "hits.tsv",
        [
            ("ref_nifH_1", "contig_1", "96.167", "287", "291", "1e-160", "565.0", "1000", "1860"),
        ],
    )
    best = summarize_tblastn(hits, GENE_MAP)
    assert best["nifH"]["coverage"] == 287 / 291


def test_summarize_tblastn_reverse_strand_normalizes_start_end(tmp_path: Path) -> None:
    """A4: Reverse-strand hit (sstart > send, e.g., 5000 and 4000) -> start == 4000, end == 5000."""
    hits = _write_tsv(
        tmp_path / "hits.tsv",
        [
            ("ref_nifD_1", "contig_1", "85.0", "450", "480", "1e-150", "600.0", "5000", "4000"),
        ],
    )
    best = summarize_tblastn(hits, GENE_MAP)
    assert best["nifD"]["start"] == 4000
    assert best["nifD"]["end"] == 5000


def test_summarize_tblastn_ignores_other_and_unmapped_qseqid(tmp_path: Path) -> None:
    """A5: qseqid mapped to 'other' or missing from gene_map is ignored."""
    hits = _write_tsv(
        tmp_path / "hits.tsv",
        [
            ("ref_other", "contig_1", "99.0", "290", "290", "1e-180", "700.0", "100", "970"),
            ("ref_unknown", "contig_1", "99.0", "290", "290", "1e-180", "700.0", "1000", "1870"),
        ],
    )
    assert summarize_tblastn(hits, GENE_MAP) == {}


def test_summarize_tblastn_ignores_short_rows_without_raising(tmp_path: Path) -> None:
    """A6: Rows with fewer than 9 columns are skipped without raising."""
    hits = _write_tsv(
        tmp_path / "hits.tsv",
        [
            ("ref_nifH_1", "contig_1", "90.0", "280", "290", "1e-100", "500.0", "100"),  # 8 cols
            ("only_one_col",),
            ("ref_nifK_1", "contig_1", "75.0", "480", "500", "1e-140", "520.0", "2000", "3440"),
        ],
    )
    best = summarize_tblastn(hits, GENE_MAP)
    assert set(best.keys()) == {"nifK"}


def test_summarize_tblastn_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    """A7: Empty file -> {}."""
    hits = tmp_path / "empty.tsv"
    hits.write_text("")
    assert summarize_tblastn(hits, GENE_MAP) == {}


# ==============================================================================
# B. is_nif_present_assembly(best)
# ==============================================================================


def test_is_nif_present_assembly_empty_best_returns_absent() -> None:
    """B1: {} -> 'ABSENT', exactly 3 reasons, each ending with ': no hit'."""
    verdict, reasons = is_nif_present_assembly({})
    assert verdict == "ABSENT"
    assert len(reasons) == 3
    assert reasons == [
        "nifH: no hit",
        "nifD: no hit",
        "nifK: no hit",
    ]
    assert all(r.endswith(": no hit") for r in reasons)


def test_is_nif_present_assembly_all_three_weak_hits_returns_absent() -> None:
    """B2: All 3 genes weak (pident 40) -> 'ABSENT', each reason contains 'weak hit'."""
    best = {
        "nifH": _hit("c1", 100, 900, pident=40.0, coverage=0.95),
        "nifD": _hit("c1", 1000, 2400, pident=40.0, coverage=0.95),
        "nifK": _hit("c1", 2500, 3900, pident=40.0, coverage=0.95),
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "ABSENT"
    assert len(reasons) == 3
    assert all("weak hit" in r for r in reasons)


def test_is_nif_present_assembly_exact_threshold_boundaries_pass() -> None:
    """B3a: pident == 50.0 and coverage == 0.7 on all 3 genes (same contig, close) -> 'PRESENT (nifHDK clustered)'."""
    best = {
        "nifH": _hit("c1", 100, 900, pident=50.0, coverage=0.7),
        "nifD": _hit("c1", 1000, 2400, pident=50.0, coverage=0.7),
        "nifK": _hit("c1", 2500, 3900, pident=50.0, coverage=0.7),
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PRESENT (nifHDK clustered)"
    assert all(": OK (" in r for r in reasons)


@pytest.mark.parametrize(
    ("pident", "coverage"),
    [
        (49.9, 0.7),
        (50.0, 0.69),
    ],
)
def test_is_nif_present_assembly_just_below_threshold_is_weak(
    pident: float, coverage: float
) -> None:
    """B3b: pident 49.9 or coverage 0.69 -> that gene is classified as weak."""
    best = {
        "nifH": _hit("c1", 100, 900, pident=pident, coverage=coverage),
        "nifD": _hit("c1", 1000, 2400, pident=80.0, coverage=0.95),
        "nifK": _hit("c1", 2500, 3900, pident=80.0, coverage=0.95),
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PARTIAL - check manually"
    assert "weak hit" in reasons[0]


@pytest.mark.parametrize("num_passed", [1, 2])
def test_is_nif_present_assembly_one_or_two_genes_passed_returns_partial(
    num_passed: int,
) -> None:
    """B4: 1 gene passed -> 'PARTIAL - check manually'; 2 genes passed -> 'PARTIAL - check manually'."""
    genes = ["nifH", "nifD", "nifK"]
    best = {
        genes[i]: _hit("c1", 1000 * (i + 1), 1000 * (i + 1) + 800)
        for i in range(num_passed)
    }
    verdict, _ = is_nif_present_assembly(best)
    assert verdict == "PARTIAL - check manually"


def test_is_nif_present_assembly_span_exactly_20000_is_clustered() -> None:
    """B5a: 3 genes passed, same contig, non-overlapping, span max(end) - min(start) == 20000 -> clustered."""
    best = {
        "nifH": _hit("c1", 1000, 1900),
        "nifD": _hit("c1", 5000, 6500),
        "nifK": _hit("c1", 19500, 21000),  # 21000 - 1000 == 20000
    }
    verdict, _ = is_nif_present_assembly(best)
    assert verdict == "PRESENT (nifHDK clustered)"


def test_is_nif_present_assembly_span_20001_is_not_clustered() -> None:
    """B5b: Span 20001 -> 'PRESENT (not clustered) - check manually' and reason contains 'max_gap'."""
    best = {
        "nifH": _hit("c1", 1000, 1900),
        "nifD": _hit("c1", 5000, 6500),
        "nifK": _hit("c1", 19500, 21001),  # 21001 - 1000 == 20001
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PRESENT (not clustered) - check manually"
    assert any("max_gap" in r for r in reasons)


def test_is_nif_present_assembly_three_genes_on_two_contigs_not_clustered() -> None:
    """B6: 3 genes passed across 2 different contigs -> 'PRESENT (not clustered) - check manually'."""
    best = {
        "nifH": _hit("contig_1", 1000, 1900),
        "nifD": _hit("contig_1", 2000, 3400),
        "nifK": _hit("contig_2", 1000, 2500),
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PRESENT (not clustered) - check manually"
    assert any("max_gap" in r for r in reasons)


def test_is_nif_present_assembly_overlap_boundary_50bp_allowed() -> None:
    """B7a: nifD=[1000, 2000], nifK=[1950, 3000] on same contig (50 bp overlap) -> not treated as overlap."""
    best = {
        "nifH": _hit("c1", 100, 900),
        "nifD": _hit("c1", 1000, 2000),
        "nifK": _hit("c1", 1950, 3000),  # 1950 + 50 < 2000 is False
    }
    verdict, _ = is_nif_present_assembly(best)
    assert verdict == "PRESENT (nifHDK clustered)"


def test_is_nif_present_assembly_overlap_51bp_triggers_partial() -> None:
    """B7b: nifD=[1000, 2000], nifK=[1949, 3000] on same contig (51 bp overlap) -> PARTIAL, 'same locus counted twice'."""
    best = {
        "nifH": _hit("c1", 100, 900),
        "nifD": _hit("c1", 1000, 2000),
        "nifK": _hit("c1", 1949, 3000),  # 1949 + 50 < 2000 is True
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PARTIAL - check manually"
    assert any("same locus counted twice" in r for r in reasons)


def test_is_nif_present_assembly_identical_coordinates_triggers_partial() -> None:
    """B7c: nifD and nifK have identical coordinates -> 'PARTIAL - check manually'."""
    best = {
        "nifH": _hit("c1", 100, 900),
        "nifD": _hit("c1", 1000, 2000),
        "nifK": _hit("c1", 1000, 2000),
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PARTIAL - check manually"
    assert any("same locus counted twice" in r for r in reasons)


def test_is_nif_present_assembly_same_coordinates_different_contigs_not_overlapping() -> None:
    """B7d: Overlapping coordinates on different contigs -> not treated as overlap."""
    best = {
        "nifH": _hit("c1", 100, 900),
        "nifD": _hit("c1", 1000, 2000),
        "nifK": _hit("c2", 1000, 2000),
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PRESENT (not clustered) - check manually"
    assert not any("same locus counted twice" in r for r in reasons)


def test_is_nif_present_assembly_overlap_check_precedes_cluster_check() -> None:
    """B8: All 3 genes pass, same contig, both overlapping AND within 20 kb -> 'PARTIAL - check manually'."""
    best = {
        "nifH": _hit("c1", 100, 900),
        "nifD": _hit("c1", 1000, 2400),
        "nifK": _hit("c1", 1200, 2600),  # span is 2500 <= 20000, but overlaps nifD
    }
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PARTIAL - check manually"
    assert any("same locus counted twice" in r for r in reasons)


def test_end_to_end_tblastn_file_to_clustered_verdict(tmp_path: Path) -> None:
    """B9: End-to-end from TSV file -> summarize_tblastn -> is_nif_present_assembly -> 'PRESENT (nifHDK clustered)'."""
    hits = _write_tsv(
        tmp_path / "operon_hits.tsv",
        [
            ("ref_nifH_1", "contig_28", "83.6", "287", "293", "1e-150", "510.0", "79284", "80144"),
            ("ref_nifD_1", "contig_28", "64.9", "482", "487", "1e-180", "620.0", "80244", "81690"),
            ("ref_nifK_1", "contig_28", "53.7", "514", "530", "1e-160", "540.0", "81790", "83332"),
        ],
    )
    best = summarize_tblastn(hits, GENE_MAP)
    verdict, reasons = is_nif_present_assembly(best)
    assert verdict == "PRESENT (nifHDK clustered)"
    assert len(reasons) == 3


# ==============================================================================
# C. summarize_reads(hits_path, gene_map)
# ==============================================================================


@pytest.mark.parametrize(
    ("pident", "length", "expected_reads", "expected_loose"),
    [
        ("60.0", "40", 1, 1),
        ("59.9", "40", 0, 1),
        ("60.0", "39", 0, 1),
    ],
)
def test_summarize_reads_threshold_boundaries(
    tmp_path: Path,
    pident: str,
    length: str,
    expected_reads: int,
    expected_loose: int,
) -> None:
    """C1: Read with pident==60.0 & length==40 counts in 'reads'; 59.9 or 39 counts only in 'reads_loose'."""
    hits = _write_tsv(
        tmp_path / "reads.tsv",
        [
            # qseqid (read), sseqid (ref protein), pident, length, evalue, bitscore
            ("read_1", "ref_nifH_1", pident, length, "1e-10", "80.0"),
        ],
    )
    summary = summarize_reads(hits, GENE_MAP)
    assert summary["nifH"]["reads"] == expected_reads
    assert summary["nifH"]["reads_loose"] == expected_loose


def test_summarize_reads_loose_includes_strict_reads(tmp_path: Path) -> None:
    """C2: reads_loose counts all rows for that gene, including rows that passed into reads."""
    # NOTE: summarize_reads does not deduplicate by read ID (qseqid). If a read has multiple HSPs
    # or if overlapping R1 and R2 mates of the same fragment both match the reference, both rows
    # are counted toward 'reads' and 'reads_loose'.
    hits = _write_tsv(
        tmp_path / "reads.tsv",
        [
            ("read_1", "ref_nifD_1", "75.0", "45", "1e-15", "95.0"),  # strict
            ("read_2", "ref_nifD_1", "60.0", "40", "1e-12", "82.0"),  # strict
            ("read_3", "ref_nifD_1", "45.0", "48", "1e-06", "50.0"),  # loose only
            ("read_4", "ref_nifD_1", "80.0", "30", "1e-08", "60.0"),  # loose only
        ],
    )
    summary = summarize_reads(hits, GENE_MAP)
    assert summary["nifD"]["reads"] == 2
    assert summary["nifD"]["reads_loose"] == 4


def test_summarize_reads_ignores_other_and_unmapped_sseqid(tmp_path: Path) -> None:
    """C3: sseqid (column 2) mapped to 'other' or missing from gene_map is not counted."""
    hits = _write_tsv(
        tmp_path / "reads.tsv",
        [
            ("read_1", "ref_other", "95.0", "50", "1e-20", "110.0"),
            ("read_2", "unmapped_ref", "95.0", "50", "1e-20", "110.0"),
            # Even if qseqid (col 1) looks like a valid reference ID, gene lookup uses col 2 (sseqid)
            ("ref_nifH_1", "ref_other", "95.0", "50", "1e-20", "110.0"),
        ],
    )
    summary = summarize_reads(hits, GENE_MAP)
    for g in ("nifH", "nifD", "nifK"):
        assert summary[g] == {"reads": 0, "reads_loose": 0, "median_pident": 0.0}


def test_summarize_reads_ignores_short_rows(tmp_path: Path) -> None:
    """C4: Rows with fewer than 6 columns are ignored without raising."""
    hits = _write_tsv(
        tmp_path / "reads.tsv",
        [
            ("read_1", "ref_nifK_1", "80.0", "45", "1e-15"),  # 5 cols
            ("short_row",),
            ("read_2", "ref_nifK_1", "80.0", "45", "1e-15", "90.0"),  # 6 cols
        ],
    )
    summary = summarize_reads(hits, GENE_MAP)
    assert summary["nifK"]["reads"] == 1
    assert summary["nifK"]["reads_loose"] == 1


def test_summarize_reads_always_returns_all_three_genes_on_empty_file(tmp_path: Path) -> None:
    """C5: Result always contains nifH, nifD, nifK initialized to zeros when no reads match."""
    hits = tmp_path / "empty_reads.tsv"
    hits.write_text("")
    summary = summarize_reads(hits, GENE_MAP)
    assert set(summary.keys()) == {"nifH", "nifD", "nifK"}
    for g in ("nifH", "nifD", "nifK"):
        assert summary[g] == {"reads": 0, "reads_loose": 0, "median_pident": 0.0}


def test_summarize_reads_median_pident_odd_and_even_counts(tmp_path: Path) -> None:
    """C6: median_pident for odd (61, 70, 65 -> 65.0) and even (61, 63, 65, 67 -> 65.0) counts."""
    # NOTE: For an even number of elements, check_nif.py computes `values[len(values) // 2]`,
    # which returns the upper-middle element (65.0 for [61, 63, 65, 67]) instead of the arithmetic
    # mean of the two middle elements ((63 + 65) / 2 = 64.0).
    hits = _write_tsv(
        tmp_path / "reads_median.tsv",
        [
            # Odd count for nifH: 61, 70, 65 -> sorted [61.0, 65.0, 70.0] -> index 1 is 65.0
            ("r1", "ref_nifH_1", "61.0", "45", "1e-10", "80.0"),
            ("r2", "ref_nifH_1", "70.0", "45", "1e-10", "80.0"),
            ("r3", "ref_nifH_1", "65.0", "45", "1e-10", "80.0"),
            # Even count for nifD: 61, 63, 65, 67 -> sorted [61.0, 63.0, 65.0, 67.0] -> index 2 is 65.0
            ("r4", "ref_nifD_1", "61.0", "45", "1e-10", "80.0"),
            ("r5", "ref_nifD_1", "63.0", "45", "1e-10", "80.0"),
            ("r6", "ref_nifD_1", "65.0", "45", "1e-10", "80.0"),
            ("r7", "ref_nifD_1", "67.0", "45", "1e-10", "80.0"),
        ],
    )
    summary = summarize_reads(hits, GENE_MAP)
    assert summary["nifH"]["median_pident"] == 65.0
    assert summary["nifD"]["median_pident"] == 65.0


# ==============================================================================
# D. is_nif_present_reads(summary)
# ==============================================================================


def _read_summary(
    nifh: int = 0, nifd: int = 0, nifk: int = 0
) -> dict[str, dict[str, int | float]]:
    """Helper to construct a summary dictionary for is_nif_present_reads."""
    return {
        "nifH": {"reads": nifh, "reads_loose": nifh, "median_pident": 75.0 if nifh else 0.0},
        "nifD": {"reads": nifd, "reads_loose": nifd, "median_pident": 75.0 if nifd else 0.0},
        "nifK": {"reads": nifk, "reads_loose": nifk, "median_pident": 75.0 if nifk else 0.0},
    }


def test_is_nif_present_reads_all_zero_returns_absent() -> None:
    """D1: All 3 genes with 0 reads -> 'ABSENT'."""
    verdict, reasons = is_nif_present_reads(_read_summary(0, 0, 0))
    assert verdict == "ABSENT"
    assert len(reasons) == 3


def test_is_nif_present_reads_boundary_20_passes_and_19_is_partial() -> None:
    """D2: All 3 genes == 20 reads -> PRESENT; one gene == 19 (others 20) -> 'PARTIAL - check manually'."""
    verdict_pass, _ = is_nif_present_reads(_read_summary(20, 20, 20))
    assert verdict_pass == "PRESENT (reads only - confirm with assembly)"

    verdict_partial, _ = is_nif_present_reads(_read_summary(20, 19, 20))
    assert verdict_partial == "PARTIAL - check manually"


def test_is_nif_present_reads_one_gene_passed_returns_partial() -> None:
    """D3: 1 gene >= 20 reads -> 'PARTIAL - check manually'."""
    verdict, _ = is_nif_present_reads(_read_summary(25, 0, 0))
    assert verdict == "PARTIAL - check manually"


def test_is_nif_present_reads_reasons_order_and_count() -> None:
    """D4: Always returns exactly 3 reasons in the order nifH, nifD, nifK."""
    # NOTE: The reason string hardcodes the text 'reads >=60% id' even if summarize_reads was
    # called with a custom min_pident threshold.
    verdict, reasons = is_nif_present_reads(_read_summary(20, 5, 0))
    assert verdict == "PARTIAL - check manually"
    assert len(reasons) == 3
    assert reasons[0].startswith("nifH: ")
    assert reasons[1].startswith("nifD: ")
    assert reasons[2].startswith("nifK: ")
