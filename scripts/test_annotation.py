#!/usr/bin/env python3
"""
Production-grade Pytest Integration Test Suite for HMM Annotation Pipeline.
"""

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Set, TypedDict

import pandas as pd
import pytest

from nifpredict.features import HMMAnnotator
from nifpredict.utils import load_config


class GeneProfileConfig(TypedDict):
    gene_symbol: str
    min_bitscore: float


HMM_GENE_MAPPING: Dict[str, GeneProfileConfig] = {
    "PF00142": {"gene_symbol": "nifH", "min_bitscore": 150.0},
    "PF00148": {"gene_symbol": "nifD", "min_bitscore": 200.0},
    "PF00149": {"gene_symbol": "nifK", "min_bitscore": 180.0},
    "nifh": {"gene_symbol": "nifH", "min_bitscore": 150.0},
    "nifd": {"gene_symbol": "nifD", "min_bitscore": 200.0},
    "nifk": {"gene_symbol": "nifK", "min_bitscore": 180.0},
}

REQUIRED_CORE_GENES: Set[str] = {"nifH", "nifD", "nifK"}
EXCLUDED_HMM_FILES: Set[str] = {"pfam-a.hmm", "pfam.hmm", "tigrfams.hmm"}


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def positive_control_fasta(config) -> Path:
    accession = "GCF_000021045.1"
    raw_dir = config.paths.raw_genomes_dir
    fasta_path = raw_dir / f"{accession}_protein.faa"

    if not fasta_path.exists():
        pytest.fail(f"[CRITICAL] Không tìm thấy tệp FASTA kiểm thử: {fasta_path}")
    if fasta_path.stat().st_size == 0:
        pytest.fail(f"[CRITICAL] Tệp FASTA kiểm thử bị rỗng (0 bytes): {fasta_path}")

    return fasta_path


def test_hmm_annotation_positive_control(config, positive_control_fasta):
    profiles_dir = config.paths.hmm_profiles_dir
    all_hmm_files = list(profiles_dir.glob("*.hmm"))

    # Bỏ qua Pfam-A.hmm để tối ưu tốc độ < 60s
    hmm_files: List[Path] = [
        f for f in all_hmm_files if f.name.lower() not in EXCLUDED_HMM_FILES
    ]

    assert len(hmm_files) > 0, f"Không tìm thấy HMM profile hợp lệ nào tại: {profiles_dir}"

    annotator = HMMAnnotator(config=config)
    all_hits: List[pd.DataFrame] = []

    tracemalloc.start()
    start_time = time.perf_counter()

    for hmm_file in hmm_files:
        if hmm_file.stat().st_size == 0:
            continue

        file_stem = hmm_file.stem
        stem_upper = file_stem.upper()
        lookup_key = stem_upper if stem_upper.startswith("PF") else file_stem.lower()

        profile_meta = HMM_GENE_MAPPING.get(
            lookup_key,
            {"gene_symbol": file_stem.upper(), "min_bitscore": 50.0},
        )

        df_hit = annotator.annotate_to_dataframe(
            protein_fasta=positive_control_fasta,
            hmm_profile=hmm_file,
        )

        if not df_hit.empty:
            score_col = "effective_score" if "effective_score" in df_hit.columns else "raw_score"
            if score_col in df_hit.columns:
                df_hit = df_hit[df_hit[score_col] >= profile_meta["min_bitscore"]]

            if not df_hit.empty:
                df_hit["target_gene"] = profile_meta["gene_symbol"]
                all_hits.append(df_hit)

    elapsed_time = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / (1024 * 1024)

    # Performance Assertions
    assert elapsed_time < 60.0, f"Thời gian thực thi vượt quá 60s: {elapsed_time:.2f}s"
    assert peak_ram_mb < 512.0, f"RAM tiêu thụ vượt quá 512MB: {peak_ram_mb:.2f}MB"

    # Data Integrity Assertions
    assert len(all_hits) > 0, "Không phát hiện bất kỳ HMM hit hợp lệ nào"

    df_final = pd.concat(all_hits, ignore_index=True)

    output_dir = config.paths.annotation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"{positive_control_fasta.stem}_nif_summary.tsv"
    df_final.to_csv(summary_file, sep="\t", index=False)

    # Biological Core Assertions
    detected_genes: Set[str] = set(df_final["target_gene"].unique())
    missing_genes = REQUIRED_CORE_GENES - detected_genes

    assert not missing_genes, (
        f"BIOLOGICAL ASSERTION FAILED: Thiếu các gen cố định đạm lõi {missing_genes}. "
        f"Các gen phát hiện được: {detected_genes}"
    )