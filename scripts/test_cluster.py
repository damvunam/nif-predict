#!/usr/bin/env python3
"""
scripts/test_cluster.py
========================
Production-grade Automated Test Suite cho module ClusterFilter & Synteny Analysis.

Khắc phục lỗi theo QA Audit v2:
1. Fix Bug False Positive (Test Case 2): Gán đúng locus_tag cho mock_hits contig.
2. Fix Bug Ghi đè file: Kiểm tra độc lập gff_path và faa_path, không tải đè dữ liệu local.
3. Thêm Pre-flight check: Kiểm tra sự tồn tại của các công cụ CLI (`datasets`, `unzip`, `hmmsearch`).
4. Đọc HMM profiles path động từ Pydantic config.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import pandas as pd

from nifpredict.features import ClusterFilter, HMMAnnotator
from nifpredict.utils import load_config, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="NifPredict Synteny Cluster Test Suite (CI/CD Ready)"
    )
    parser.add_argument(
        "--accession",
        type=str,
        default=None,
        help="NCBI accession override (Ví dụ: GCF_000021045.1)",
    )
    parser.add_argument(
        "--gff-path",
        type=Path,
        default=None,
        help="Đường dẫn file GFF3 local (nếu không muốn tải từ NCBI)",
    )
    parser.add_argument(
        "--faa-path",
        type=Path,
        default=None,
        help="Đường dẫn file Protein FASTA local",
    )
    return parser.parse_args()


def check_preflight_dependencies(tools: List[str], logger: logging.Logger) -> None:
    """
    KIỂM TRA TIỀN ĐIỀU KIỆN (Pre-flight Check):
    Đảm bảo môi trường runner CI/CD đã cài đặt đủ các công cụ phụ thuộc ngoài (External Binaries).
    """
    missing_tools = [tool for tool in tools if shutil.which(tool) is None]
    if missing_tools:
        raise EnvironmentError(
            f"Môi trường CI/CD thiếu các công cụ CLI bắt buộc: {missing_tools}. "
            "Vui lòng cài đặt trước khi chạy Test Suite!"
        )
    logger.info(f"[PRE-FLIGHT] Kiểm tra hoàn tất. Các CLI dependencies sẵn sàng: {tools}")


class SyntenyTestRunner:
    """Class quản lý quy trình kiểm thử Synteny & Cluster Filtering."""

    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Đọc tham số từ Pydantic Config linh hoạt (hỗ trợ cả dict lẫn BaseModel)
        cluster_cfg = getattr(config, "cluster_filter", {})
        if isinstance(config, dict):
            cluster_cfg = config.get("cluster_filter", {})

        self.max_gap_bp = getattr(cluster_cfg, "max_gap_bp", 10000)
        self.min_core_genes = getattr(cluster_cfg, "min_core_genes", 2)

        # Bộ gen catalytic core bắt buộc
        self.catalytic_core_families: Set[str] = {"PF00142", "PF00148", "PF02826"}

    def fetch_ncbi_dataset(
        self, accession: str, target_gff: Path, target_faa: Path, zip_dir: Path
    ) -> None:
        """Tải GFF3 và Protein FASTA từ NCBI Datasets CLI nếu file chưa tồn tại."""
        temp_zip = zip_dir / f"{accession}_data.zip"
        temp_extract = zip_dir / f"{accession}_extracted"

        try:
            cmd_download = [
                "datasets", "download", "genome", "accession", accession,
                "--include", "gff3,protein", "--filename", str(temp_zip)
            ]
            subprocess.run(cmd_download, check=True, capture_output=True, text=True)

            subprocess.run(
                ["unzip", "-o", str(temp_zip), "-d", str(temp_extract)],
                check=True, capture_output=True
            )

            ncbi_data_dir = temp_extract / "ncbi_dataset" / "data" / accession
            src_gff = ncbi_data_dir / "genomic.gff"
            src_faa = ncbi_data_dir / "protein.faa"

            assert src_gff.exists(), f"Không tìm thấy file genomic.gff trong gói NCBI: {src_gff}"
            assert src_faa.exists(), f"Không tìm thấy file protein.faa trong gói NCBI: {src_faa}"

            if not target_gff.exists():
                shutil.copy(src_gff, target_gff)
            if not target_faa.exists():
                shutil.copy(src_faa, target_faa)

            self.logger.info(f"Tải và trích xuất thành công dữ liệu NCBI cho accession: {accession}")

        except Exception as exc:
            raise RuntimeError(f"Thất bại khi lấy dữ liệu NCBI cho {accession}: {exc}") from exc
        finally:
            if temp_extract.exists():
                shutil.rmtree(temp_extract, ignore_errors=True)
            if temp_zip.exists():
                temp_zip.unlink(missing_ok=True)

    def validate_gff_sanity(self, df_gff: pd.DataFrame) -> None:
        """GFF3 Sanity Checks."""
        assert not df_gff.empty, "Tệp GFF3 rỗng hoặc không chứa bản ghi CDS/gene hợp lệ!"
        required_cols = {"seqid", "start", "end", "strand", "locus_tag"}
        missing = required_cols - set(df_gff.columns)
        assert not missing, f"Tệp GFF3 thiếu các cột cấu trúc: {missing}"

        invalid_coords = df_gff[df_gff["start"] > df_gff["end"]]
        assert len(invalid_coords) == 0, (
            f"Phát hiện {len(invalid_coords)} bản ghi GFF3 lỗi tọa độ (start > end)!"
        )

    def run_biological_assertions(self, clusters: List[Dict[str, Any]]) -> None:
        """Biological Assertions cho cụm gen."""
        assert len(clusters) > 0, "Thuật toán không phát hiện được cụm gen synteny nào!"

        has_valid_catalytic_cluster = False
        for cluster in clusters:
            genes = cluster.get("genes", [])
            families = set(cluster.get("gene_families", []))

            assert len(genes) >= self.min_core_genes, (
                f"Cụm {cluster.get('cluster_id')} có {len(genes)} gen, "
                f"nhỏ hơn ngưỡng min_core_genes ({self.min_core_genes})."
            )

            if not families.isdisjoint(self.catalytic_core_families):
                has_valid_catalytic_cluster = True

        assert has_valid_catalytic_cluster, (
            "QA FAIL: Các cụm lọc được không chứa bất kỳ gen catalytic core nào "
            f"thuộc bộ ({self.catalytic_core_families})!"
        )


def run_unit_edge_cases(logger: logging.Logger) -> None:
    """
    CHẠY FIXTURES FIX LỖI FALSE POSITIVE:
    Test Case 1: Strand (-) & Overlap.
    Test Case 2: Contig separation (Đã fix bug dữ liệu mock_hits).
    """
    logger.info("--- RUNNING FIXTURE TESTS: Mock Edge Cases ---")
    filter_engine = ClusterFilter(max_gap_bp=5000, min_core_genes=2)

    # Test Case 1: Overlapping genes trên mạch đảo (-)
    mock_gff_1 = pd.DataFrame([
        {"gene_id": "g1", "seqid": "chr1", "start": 1000, "end": 2000, "strand": "-", "locus_tag": "L1"},
        {"gene_id": "g2", "seqid": "chr1", "start": 1950, "end": 3000, "strand": "-", "locus_tag": "L2"},
    ])
    mock_hits_1 = pd.DataFrame([
        {"target_name": "L1", "gene_family": "PF00142", "evalue": 1e-50},
        {"target_name": "L2", "gene_family": "PF00148", "evalue": 1e-50},
    ])
    clusters_1 = filter_engine.group_into_clusters(mock_hits_1, mock_gff_1)
    assert len(clusters_1) == 1, "FAIL Edge Case 1: Không nhóm được gen overlap trên mạch đảo (-)"

    # Test Case 2: Phân tách Contig (ĐÃ FIX CRITICAL BUG #1)
    # Cấp đúng ID khớp giữa GFF3 và HMM hits cho cả 2 contig khác nhau
    mock_gff_contigs = pd.DataFrame([
        {"gene_id": "g10", "seqid": "contig_A", "start": 1000, "end": 2000, "strand": "+", "locus_tag": "LOC_10"},
        {"gene_id": "g11", "seqid": "contig_B", "start": 2100, "end": 3000, "strand": "+", "locus_tag": "LOC_11"},
    ])
    mock_hits_contigs = pd.DataFrame([
        {"target_name": "LOC_10", "gene_family": "PF00142", "evalue": 1e-50},
        {"target_name": "LOC_11", "gene_family": "PF00148", "evalue": 1e-50},
    ])
    clusters_contig = filter_engine.group_into_clusters(mock_hits_contigs, mock_gff_contigs)
    
    # Vì mỗi contig chỉ có 1 gen core, min_core_genes=2 sẽ loại bỏ cả 2 contig cô lập.
    # Kết quả kỳ vọng: 0 cụm gộp (thay vì gộp nhầm LOC_10 và LOC_11 thành 1 cụm).
    assert len(clusters_contig) == 0, (
        "FAIL Edge Case 2: Thuật toán đã tự động gộp nhầm 2 gen nằm trên 2 contig khác nhau!"
    )

    logger.info("[PASS] Tất cả Mock Edge Cases đã đạt yêu cầu và không còn False Positive.")


def get_config_path(config: Any, key: str, default: str) -> Path:
    """Helper đọc path từ config hỗ trợ cả dict và Pydantic BaseModel."""
    paths = config.paths if hasattr(config, "paths") else config.get("paths", {})
    if isinstance(paths, dict):
        return Path(paths.get(key, default))
    return Path(getattr(paths, key, default))


def main() -> None:
    """Main execution entrypoint chuẩn CI/CD."""
    args = parse_args()
    logger = setup_logger("nifpredict.test_cluster")

    try:
        # 1. Pre-flight check cho External CLI Dependencies (FIX ISSUE #4)
        check_preflight_dependencies(["datasets", "unzip", "hmmsearch"], logger)

        config = load_config()
        runner = SyntenyTestRunner(config, logger)

        # 2. Chạy Fixture Tests (Edge Cases)
        run_unit_edge_cases(logger)

        # 3. Đọc thư mục làm việc từ Config
        genomes_dir = get_config_path(config, "raw_genomes_dir", "data/raw_genomes")
        zip_dir = get_config_path(config, "raw_zip_dir", "data/raw_zip")
        annotation_dir = get_config_path(config, "annotation_dir", "data/annotation")
        hmm_dir = get_config_path(config, "hmm_profiles_dir", "data/hmm_profiles")  # FIX ISSUE #3

        for d in [genomes_dir, zip_dir, annotation_dir]:
            d.mkdir(parents=True, exist_ok=True)

        accession = args.accession or "GCF_000021045.1"

        # 4. Giải quyết đường dẫn và tránh đè file (FIX CRITICAL BUG #2)
        if args.gff_path:
            gff_file = args.gff_path
            assert gff_file.exists(), f"Tệp GFF3 local không tồn tại: {gff_file}"
        else:
            gff_file = genomes_dir / f"{accession}_genomic.gff"

        if args.faa_path:
            faa_file = args.faa_path
            assert faa_file.exists(), f"Tệp FASTA local không tồn tại: {faa_file}"
        else:
            faa_file = genomes_dir / f"{accession}_protein.faa"

        # Chỉ tải từ NCBI nếu ít nhất một trong hai file chưa sẵn sàng
        if not gff_file.exists() or not faa_file.exists():
            runner.fetch_ncbi_dataset(accession, gff_file, faa_file, zip_dir)

        # 5. HMM Search (Đọc HMM profiles động từ config - FIX ISSUE #3)
        annotator = HMMAnnotator()
        all_hits = []
        profiles = {
            "PF00142": hmm_dir / "nifH.hmm",
            "PF00148": hmm_dir / "nifD.hmm",
            "PF02826": hmm_dir / "nifK.hmm",
        }

        for gene_family, profile_path in profiles.items():
            if profile_path.exists():
                out_tbl = annotation_dir / f"{accession}_{gene_family}.tbl"
                annotator.run_hmmsearch(
                    protein_fasta=faa_file,
                    hmm_profile=profile_path,
                    output_tbl=out_tbl,
                )
                df_gene = annotator.parse_tblout(out_tbl)
                if not df_gene.empty:
                    df_gene["gene_family"] = gene_family
                    all_hits.append(df_gene)
            else:
                logger.warning(f"Không tìm thấy HMM profile tại: {profile_path}")

        assert len(all_hits) > 0, "FAIL: Không tìm thấy bất kỳ HMM hit nào trên bộ gen mẫu!"
        df_all_hits = pd.concat(all_hits, ignore_index=True)

        # 6. Parse GFF3 & Sanity check
        cluster_filter = ClusterFilter(max_gap_bp=runner.max_gap_bp, min_core_genes=runner.min_core_genes)
        df_gff = cluster_filter.parse_gff3(gff_file)
        runner.validate_gff_sanity(df_gff)

        # 7. Group clusters & Biological Assertions
        clusters = cluster_filter.group_into_clusters(df_all_hits, df_gff)
        runner.run_biological_assertions(clusters)

        logger.info("==================================================")
        logger.info(">>> KẾT QUẢ KIỂM THỬ: ALL QA CHECKS PASSED (Exit Code 0) <<<")
        logger.info("==================================================")
        sys.exit(0)

    except Exception as err:
        logger.critical(f"CRITICAL TEST FAILURE: {err}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()