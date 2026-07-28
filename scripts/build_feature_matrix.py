import sys
import pandas as pd
from pathlib import Path

from nifpredict.pipeline import NifPredictor
from nifpredict.utils.logger import setup_logger

logger = setup_logger("nifpredict.scripts.build_features")

def main():
    if len(sys.argv) < 2:
        print("Sử dụng: python scripts/build_feature_matrix.py <PATH_TO_BATCH_FILE>")
        print("Ví dụ : python scripts/build_feature_matrix.py data/batch_accessions.txt")
        sys.exit(1)

    batch_file = Path(sys.argv[1])
    if not batch_file.exists():
        logger.error(f"Không tìm thấy file: {batch_file}")
        sys.exit(1)

    predictor = NifPredictor()

    with open(batch_file, "r") as f:
        accessions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    logger.info(f"=== BẮT ĐẦU TRÍCH XUẤT FEATURE MATRIX CHO {len(accessions)} ACCESSION ===")
    feature_rows = []

    for idx, acc in enumerate(accessions, start=1):
        logger.info(f"[{idx}/{len(accessions)}] Trích xuất đặc trưng cho Accession: {acc}")
        feat_dict = predictor.extract_sample_features(acc)
        if feat_dict:
            feature_rows.append(feat_dict)

    # Đóng gói DataFrame
    df_matrix = pd.DataFrame(feature_rows)

    # Lưu kết quả
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "feature_matrix.csv"
    
    df_matrix.to_csv(out_csv, index=False)
    
    logger.info("=" * 60)
    logger.info("        HOÀN THÀNH PHASE 6: FEATURE ENGINEERING MATRIX")
    logger.info("=" * 60)
    logger.info(f"Tổng số mẫu xử lý             : {len(df_matrix)}")
    logger.info(f"Số lượng chiều đặc trưng (Cols): {df_matrix.shape[1]}")
    logger.info(f"📁 File Ma trận lưu tại         : {out_csv}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()