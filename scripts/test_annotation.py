import pandas as pd
from pathlib import Path
from nifpredict.utils import load_config, setup_logger
from nifpredict.features import HMMAnnotator

def main():
    logger = setup_logger("nifpredict.test_annotation")
    config = load_config()

    accession = "GCF_000021045.1"
    protein_fasta = Path(config["paths"]["raw_genomes_dir"]) / f"{accession}_protein.faa"
    output_tbl = Path(config["paths"]["annotation_dir"]) / f"{accession}_nif_hits.tbl"
    
    logger.info(f"=== BẮT ĐẦU TEST ANNOTATION CHO ACCESSION: {accession} ===")

    if not protein_fasta.exists():
        logger.error(f"File protein FASTA chưa tồn tại tại {protein_fasta}. Vui lòng chạy Phase 2 trước!")
        return

    annotator = HMMAnnotator()

    # Kiểm tra xem đã có HMM profile chưa
    # Lưu ý: Ở đây ta kiểm tra file profile nifH.hmm trong data/hmm_profiles/
    hmm_profile = Path("data/hmm_profiles/nifH.hmm")

    if not hmm_profile.exists():
        logger.warning(f"Chưa tìm thấy {hmm_profile}. Đang khởi tạo file HMM profile thử nghiệm...")
        # Tạo file placeholder hướng dẫn người dùng
        hmm_profile.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Vui lòng tải hoặc cung cấp file .hmm chuẩn (ví dụ: Pfam PF00142 cho nifH) vào thư mục data/hmm_profiles/")
        return

    # Chạy HMM search
    result_path = annotator.run_hmmsearch(
        protein_fasta=protein_fasta,
        hmm_profile=hmm_profile,
        output_tbl=output_tbl,
        evalue_threshold=1e-5
    )

    if result_path and result_path.exists():
        df_hits = annotator.parse_tblout(result_path)
        logger.info("=== KIỂM THỬ ANNOTATION THÀNH CÔNG ===")
        logger.info(f"Tổng số hits phát hiện: {len(df_hits)}")
        if not df_hits.empty:
            print("\n--- BẢNG KẾT QUẢ TOP HITS ---")
            print(df_hits.head(10).to_string(index=False))
    else:
        logger.error("Chạy HMM search thất bại!")

if __name__ == "__main__":
    main()
