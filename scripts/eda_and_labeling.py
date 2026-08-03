import sys
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    input_file = Path("data/processed/feature_matrix.parquet")
    output_dir = Path("data/processed")
    reports_dir = Path("results/eda")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"❌ Lỗi: Không tìm thấy file {input_file}. Vui lòng chạy Phase 6 trước!")
        sys.exit(1)

    print("=" * 65)
    print("      PHASE 7: ADVANCED EDA, QUALITY CONTROL & GOLD LABELING")
    print("=" * 65)

    # 1. Nạp dữ liệu
    df = pd.read_parquet(input_file)
    print(f"📊 Tổng số mẫu nạp vào       : {len(df)}")

    # 2. Quality Control (QC)
    if "status" in df.columns:
        df_clean = df[df["status"] == "SUCCESS"].copy()
    else:
        df_clean = df.copy()
    print(f"✅ Số mẫu vượt qua QC        : {len(df_clean)} / {len(df)}")

    if df_clean.empty:
        print("⚠️ Cảnh báo: Không có mẫu nào có status == 'SUCCESS'. Dừng Phase 7!")
        sys.exit(0)

    # 3. Schema Validation & Gold-Standard Labeling
    if "complete_hdk_clusters" in df_clean.columns:
        df_clean["target_bnf"] = df_clean["complete_hdk_clusters"].apply(lambda x: 1 if x >= 1 else 0)
    elif "clusters_found" in df_clean.columns:
        df_clean["target_bnf"] = df_clean["clusters_found"].apply(lambda x: 1 if x > 0 else 0)
    else:
        print("⚠️ Không tìm thấy cột cụm gen ('complete_hdk_clusters' / 'clusters_found'). Khởi tạo mặc định target_bnf = 0.")
        df_clean["target_bnf"] = 0

    # 4. Biến đổi Log Transformation cho E-values (Log-Evalue transform)
    evalue_cols = [c for c in df_clean.columns if "evalue" in c]
    for col in evalue_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(1.0)
        # Ép E-value = 0 về số cực nhỏ (1e-300) để tránh lỗi chia log(0)
        safe_evalue = df_clean[col].apply(lambda x: 1e-300 if x <= 0 else x)
        df_clean[f"-log10_{col}"] = -np.log10(safe_evalue)

    # 5. Phân tích Thống kê Mô tả Chi tiết (Descriptive Statistics)
    numeric_cols = [
        "PF00142_max_bitscore", "PF00148_max_bitscore", "PF02826_max_bitscore",
        "-log10_PF00142_min_evalue", "-log10_PF00148_min_evalue", "-log10_PF02826_min_evalue",
        "clusters_found", "max_cluster_span_bp"
    ]
    
    # Chỉ lấy các cột số thực sự có trong DataFrame
    existing_num_cols = [c for c in numeric_cols if c in df_clean.columns]
    for c in existing_num_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0)

    # Thống kê tổng hợp: mean, std, median, min, max
    eda_summary = df_clean.groupby("target_bnf")[existing_num_cols].agg(
        ["mean", "std", "median", "min", "max"]
    ).T

    # 6. Hiển thị & Xuất kết quả
    print("\n[🏷️ PHÂN BỐ NHÃN TỰ ĐỘNG (TARGET DISTRIBUTION)]")
    label_counts = df_clean["target_bnf"].value_counts().rename(index={1: "Diazotroph (1)", 0: "Non-Diazotroph (0)"})
    print(label_counts)

    print("\n[📈 TỔNG HỢP THỐNG KÊ MÔ TẢ THEO LỚP (EDA SUMMARY)]")
    print(eda_summary.round(2))

    # Lưu tập dữ liệu đã làm sạch & gắn nhãn cho PHASE 8 ML
    clean_csv = output_dir / "labeled_dataset.csv"
    df_clean.to_csv(clean_csv, index=False)

    # Lưu báo cáo EDA thống kê chi tiết dạng CSV
    eda_csv = reports_dir / "eda_feature_stats.csv"
    eda_summary.to_csv(eda_csv)

    print("\n" + "=" * 65)
    print(f"📁 Dataset sạch sẵn sàng huấn luyện ML : {clean_csv}")
    print(f"📊 Báo cáo Thống kê EDA chi tiết       : {eda_csv}")
    print("=" * 65)


if __name__ == "__main__":
    main()