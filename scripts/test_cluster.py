#!/usr/bin/env python3
"""
scripts/test_cluster.py
========================
Production-grade Automated Test Suite cho module ClusterFilter & Synteny Analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from nifpredict.features import ClusterFilter, HMMAnnotator
from nifpredict.utils import load_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NifPredict Synteny Cluster Test Suite")
    parser.add_argument("--accession", type=str, default="GCF_000021045.1")
    parser.add_argument("--gff-path", type=Path, default=None)
    parser.add_argument("--faa-path", type=Path, default=None)
    return parser.parse_args()


def run_unit_edge_cases(logger: logging.Logger) -> None:
    logger.info("--- RUNNING FIXTURE TESTS: Mock Edge Cases ---")
    filter_engine = ClusterFilter()

    mock_gff_1 = pd.DataFrame([
        {"gene_id": "g1", "seqid": "chr1", "start": 1000, "end": 2000, "strand": "-", "locus_tag": "L1", "protein_id": "L1"},
        {"gene_id": "g2", "seqid": "chr1", "start": 1950, "end": 3000, "strand": "-", "locus_tag": "L2", "protein_id": "L2"},
    ])
    mock_hits_1 = pd.DataFrame([
        {"target_protein": "L1", "gene_family": "PF00142", "seq_evalue": 1e-50},
        {"target_protein": "L2", "gene_family": "PF00148", "seq_evalue": 1e-50},
    ])
    clusters_1 = filter_engine.group_into_clusters(mock_hits_1, mock_gff_1)
    assert len(clusters_1) == 1, "FAIL Edge Case 1: Không nhóm được gen overlap trên mạch đảo (-)"

    mock_gff_contigs = pd.DataFrame([
        {"gene_id": "g10", "seqid": "contig_A", "start": 1000, "end": 2000, "strand": "+", "locus_tag": "LOC_10", "protein_id": "LOC_10"},
        {"gene_id": "g11", "seqid": "contig_B", "start": 2100, "end": 3000, "strand": "+", "locus_tag": "LOC_11", "protein_id": "LOC_11"},
    ])
    mock_hits_contigs = pd.DataFrame([
        {"target_protein": "LOC_10", "gene_family": "PF00142", "seq_evalue": 1e-50},
        {"target_protein": "LOC_11", "gene_family": "PF00148", "seq_evalue": 1e-50},
    ])
    clusters_contig = filter_engine.group_into_clusters(mock_hits_contigs, mock_gff_contigs)
    assert len(clusters_contig) == 0, "FAIL Edge Case 2: Đã tự động gộp nhầm 2 gen nằm trên 2 contig khác nhau!"

    logger.info("[PASS] Tất cả Mock Edge Cases đã đạt yêu cầu.")


def main() -> None:
    args = parse_args()
    logger = setup_logger("nifpredict.test_cluster")

    try:
        config = load_config(auto_create_dirs=False)
        run_unit_edge_cases(logger)

        accession = args.accession
        genomes_dir = config.paths.raw_genomes_dir
        hmm_dir = config.paths.hmm_profiles_dir

        gff_file = args.gff_path or (genomes_dir / f"{accession}_genomic.gff")
        faa_file = args.faa_path or (genomes_dir / f"{accession}_protein.faa")

        assert gff_file.exists(), f"Không tìm thấy file GFF: {gff_file}"
        assert faa_file.exists(), f"Không tìm thấy file FASTA: {faa_file}"

        annotator = HMMAnnotator(config=config)
        all_hits = []

        for profile_path in hmm_dir.glob("*.hmm"):
            if "pfam-a" in profile_path.name.lower():
                continue
            df_gene = annotator.annotate_to_dataframe(protein_fasta=faa_file, hmm_profile=profile_path)
            if not df_gene.empty:
                all_hits.append(df_gene)

        assert len(all_hits) > 0, "FAIL: Không tìm thấy bất kỳ HMM hit nào!"
        df_all_hits = pd.concat(all_hits, ignore_index=True)

        cluster_filter = ClusterFilter(config=config)
        df_gff = cluster_filter.parse_gff3(gff_file)
        clusters = cluster_filter.group_into_clusters(df_all_hits, df_gff)

        assert len(clusters) > 0, "FAIL: Thuật toán không phát hiện được cụm gen nào!"
        logger.info(f"=== PASSED: Phát hiện {len(clusters)} cụm gen synteny hợp lệ ===")
        sys.exit(0)

    except Exception as err:
        logger.critical(f"CRITICAL TEST FAILURE: {err}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()