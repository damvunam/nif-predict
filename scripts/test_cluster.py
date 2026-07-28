import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path

from nifpredict.utils import load_config, setup_logger
from nifpredict.features import HMMAnnotator, ClusterFilter


def main():
    logger = setup_logger("nifpredict.test_cluster")
    config = load_config()

    accession = "GCF_000021045.1"
    
    # 1. Khởi tạo và kiểm tra các thư mục cần thiết
    genomes_dir = Path(config["paths"]["raw_genomes_dir"])
    zip_dir = Path(config["paths"]["raw_zip_dir"])
    annotation_dir = Path(config["paths"]["annotation_dir"])
    
    genomes_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    protein_fasta = genomes_dir / f"{accession}_protein.faa"
    gff_file = genomes_dir / f"{accession}_genomic.gff"
    
    logger.info(f"=== BẮT ĐẦU TEST XÁC ĐỊNH CỤM GEN CHO ACCESSION: {accession} ===")

    # 2. Tải và trích xuất file GFF3 bám sát cấu trúc NCBI Datasets Package
    if not gff_file.exists():
        logger.info(f"Đang tải gói dữ liệu GFF3 cho accession: {accession}...")
        temp_zip = zip_dir / f"{accession}_gff_temp.zip"
        temp_extract_dir = zip_dir / f"{accession}_gff_temp"

        # Gọi CLI 'datasets' để tải gói GFF3
        cmd_download = [
            "datasets", "download", "genome", "accession", accession,
            "--include", "gff3", "--filename", str(temp_zip)
        ]
        subprocess.run(cmd_download, check=True)

        # Giải nén gói zip
        subprocess.run(["unzip", "-o", str(temp_zip), "-d", str(temp_extract_dir)], check=True)

        # Lấy file genomic.gff từ path chuẩn: ncbi_dataset/data/<accession>/genomic.gff
        expected_ncbi_gff = temp_extract_dir / "ncbi_dataset" / "data" / accession / "genomic.gff"

        if expected_ncbi_gff.exists():
            shutil.copy(expected_ncbi_gff, gff_file)
            logger.info(f"Đã trích xuất và lưu file GFF3 tại: {gff_file}")
        else:
            logger.error(f"Không tìm thấy file GFF3 tại đường dẫn chuẩn NCBI: {expected_ncbi_gff}")
            return

        # Dọn dẹp các tệp tạm
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        if temp_zip.exists():
            temp_zip.unlink()

    # 3. Chạy HMM search quét đồng thời 3 profiles (nifH, nifD, nifK)
    annotator = HMMAnnotator()
    all_hits = []
    
    profiles = {
        "PF00142": Path("data/hmm_profiles/nifH.hmm"),
        "PF00148": Path("data/hmm_profiles/nifD.hmm"),
        "PF02826": Path("data/hmm_profiles/nifK.hmm")
    }

    for gene_family, profile_path in profiles.items():
        if profile_path.exists():
            out_tbl = annotation_dir / f"{accession}_{gene_family}.tbl"
            
            # Khớp đúng chữ ký hàm: run_hmmsearch(protein_fasta, hmm_profile, output_tbl)
            annotator.run_hmmsearch(
                protein_fasta=protein_fasta, 
                hmm_profile=profile_path, 
                output_tbl=out_tbl
            )
            df_gene = annotator.parse_tblout(out_tbl)
            if not df_gene.empty:
                df_gene["gene_family"] = gene_family
                all_hits.append(df_gene)
        else:
            logger.warning(f"Chưa tìm thấy HMM profile tại: {profile_path}")

    if not all_hits:
        logger.error("Không ghi nhận được hits HMM nào!")
        return

    df_all_hits = pd.concat(all_hits, ignore_index=True)
    logger.info(f"Tổng số hits HMM ghi nhận trên toàn bộ các profiles: {len(df_all_hits)}")

    # 4. Parse GFF3 và Lọc Synteny Cluster
    cluster_filter = ClusterFilter(max_gap_bp=10000, min_core_genes=2)
    df_gff = cluster_filter.parse_gff3(gff_file)
    logger.info(f"Đã parse {len(df_gff)} bản ghi CDS/gene từ tệp GFF3.")

    clusters = cluster_filter.group_into_clusters(df_all_hits, df_gff)

    # 5. In kết quả JSON nghiệm thu
    logger.info("=== KẾT QUẢ TRÍCH XUẤT CỤM GEN (GENE CLUSTERS) ===")
    print(json.dumps(clusters, indent=2))


if __name__ == "__main__":
    main()
