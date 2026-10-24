"""Unit tests for check_nif gene assignment (W4f: MK67I 'NIFK' must not count as nifK)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nifpredict.check_nif import is_nif_present_assembly, load_gene_map, summarize_tblastn

# Real 171-protein reference set (W4e). Lives only on the local machine (results/ is git-ignored).
REAL_REF = Path(__file__).resolve().parents[2] / "nifpredict" / "results" / "nif_check_v2" / "nif_ref_swissprot.faa"

HEADERS = {
    "sp|P00459|NIFH1_AZOVI": ("GN=nifH1", "nifH"),
    "sp|P00466|NIFH_KLEPN": ("GN=nifH", "nifH"),
    "sp|Q9AKT4|NIFH2_PAEDU": ("GN=nifH2", "nifH"),
    "sp|P07328|NIFD_KLEPN": ("GN=nifD", "nifD"),
    "sp|P00467|NIFK_KLEPN": ("GN=nifK", "nifK"),
    "sp|X00001|NIFK2_TEST": ("GN=nifK2", "nifK"),
    # MK67I (animal nucleolar protein): gene symbol NIFK in any case -> not nitrogenase
    "sp|Q9BYG3|MK67I_HUMAN": ("GN=NIFK", "other"),
    "sp|Q91VE6|MK67I_MOUSE": ("GN=Nifk", "other"),
    "sp|Q8JIY8|MK67I_DANRE": ("GN=nifk", "other"),
    # other non-targets
    "sp|P15335|NIFH2_AZOVI": ("GN=vnfH", "other"),
    "sp|X00002|NIFHD_TEST": ("GN=nifHD", "other"),
    "sp|Q5S7T7|TIM50_PHYIN": ("GN=TIM50", "other"),
    "sp|X00003|NOGN_TEST": ("", "other"),
}


def _write_fasta(path: Path) -> Path:
    lines = []
    for seq_id, (gn, _) in HEADERS.items():
        lines.append(f">{seq_id} Some protein OS=Test OX=1 {gn} PE=1 SV=1".replace("  ", " "))
        lines.append("MAAAAAAAAAA")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_gene_map_is_case_sensitive(tmp_path: Path) -> None:
    gene_map = load_gene_map(_write_fasta(tmp_path / "ref.faa"))
    assert gene_map == {seq_id: expected for seq_id, (_, expected) in HEADERS.items()}


def test_mk67i_hit_never_becomes_best_nifk(tmp_path: Path) -> None:
    """Decision path: a strong MK67I hit must be ignored, so the nifK call rests on a real NifK."""
    gene_map = load_gene_map(_write_fasta(tmp_path / "ref.faa"))
    hits = tmp_path / "hits.tsv"
    rows = [
        # qseqid, sseqid, pident, length, qlen, evalue, bitscore, sstart, send, qstart, qend
        ("sp|P00466|NIFH_KLEPN", "contig_1", "85", "290", "293", "1e-150", "500", "1000", "1870", "1", "290"),
        ("sp|P07328|NIFD_KLEPN", "contig_1", "70", "470", "483", "1e-200", "650", "2000", "3410", "1", "470"),
        ("sp|Q9BYG3|MK67I_HUMAN", "contig_9", "99", "290", "293", "1e-200", "900", "10", "880", "1", "290"),
    ]
    hits.write_text("\n".join("\t".join(r) for r in rows) + "\n")

    best = summarize_tblastn(hits, gene_map)
    assert best is not None
    assert "nifK" not in best
    verdict, _ = is_nif_present_assembly(best)
    assert verdict == "PARTIAL - check manually"


@pytest.mark.skipif(not REAL_REF.exists(), reason="171-protein reference set only exists on the local machine")
def test_real_reference_counts_after_w4f() -> None:
    gene_map = load_gene_map(REAL_REF)
    counts = {g: list(gene_map.values()).count(g) for g in ("nifH", "nifD", "nifK")}
    assert len(gene_map) == 171
    assert counts == {"nifH": 113, "nifD": 27, "nifK": 19}
