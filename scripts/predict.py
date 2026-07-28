import sys
import json
from pathlib import Path
from nifpredict.pipeline import NifPredictor

def main():
    if len(sys.argv) < 2:
        print("Sử dụng:")
        print("  Chạy 1 mẫu:    python scripts/predict.py <ACCESSION>")
        print("  Chạy batch:    python scripts/predict.py --file <PATH_TO_FILE>")
        sys.exit(1)

    predictor = NifPredictor()

    if sys.argv[1] == "--file":
        file_path = Path(sys.argv[2])
        if not file_path.exists():
            print(f"Lỗi: Không tìm thấy file {file_path}")
            sys.exit(1)

        with open(file_path, "r") as f:
            accessions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        results = predictor.predict_batch(accessions)
    else:
        accession = sys.argv[1]
        results = [predictor.predict_accession(accession)]

    # Xuất file báo cáo (Step 4.5)
    json_path, csv_path = predictor.save_reports(results)

    # In thốn kê tổng quan ra Terminal
    total = len(results)
    successful = sum(1 for r in results if r.get("status") == "SUCCESS")
    bnf_positive = sum(1 for r in results if r.get("bnf_capable") is True)

    print("\n" + "=" * 55)
    print("        BÁO CÁO KẾT QUẢ DỰ ĐOÁN CỐ ĐỊNH ĐẠM (BNF)")
    print("=" * 55)
    print(f" Tổng số mẫu xử lý  : {total}")
    print(f" Xử lý thành công   : {successful}/{total}")
    print(f" Mẫu có BNF Positive : {bnf_positive}/{successful}")
    print("-" * 55)
    print(f" 📄 File JSON chi tiết: {json_path}")
    print(f" 📊 File CSV tổng quan : {csv_path}")
    print("=" * 55)

if __name__ == "__main__":
    main()