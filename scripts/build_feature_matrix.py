#!/usr/bin/env python3
"""
NifPredict Feature Matrix Builder Script (HPC Production-Grade).
Sửa lỗi: Tự động chạy HMM search, xuất kết quả ra data/interim/hmm_outputs/
và phòng chống crash StandardScaler khi ma trận đặc trưng bị rỗng.
"""

import argparse
import gc
import json
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from nifpredict.features.cluster_filter import ClusterFilter
from nifpredict.features.feature_extractor import GenomeFeatureExtractor
from nifpredict.features.hmm_annotator import HMMAnnotator
from nifpredict.utils.config import AppConfig, load_config
from nifpredict.utils.logger import get_logger

logger = get_logger("nifpredict.scripts.build_features")

_global_config: Optional[AppConfig] = None
_worker_annotator: Optional[HMMAnnotator] = None
_worker_cluster_filter: Optional[ClusterFilter] = None
_worker_extractor: Optional[GenomeFeatureExtractor] = None


def _init_worker() -> None:
    """Khởi tạo tài nguyên cached một lần duy nhất cho mỗi Worker Process."""
    global _global_config, _worker_annotator, _worker_cluster_filter, _worker_extractor
    _global_config = load_config(auto_create_dirs=True)
    _worker_annotator = HMMAnnotator(config=_global_config)
    _worker_cluster_filter = ClusterFilter(config=_global_config)
    _worker_extractor = GenomeFeatureExtractor(config=_global_config)


def _process_single_genome(
    accession_id: str,
    faa_path: Optional[Path],
    gff_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Xử lý 1 genome: Quét HMM, lưu output ra data/interim/hmm_outputs/, gom cụm synteny

    và trích xuất ma trận đặc trưng an toàn.
    """
    global _global_config, _worker_annotator, _worker_cluster_filter, _worker_extractor
    if not _worker_annotator or not _worker_cluster_filter or not _worker_extractor or not _global_config:
        _init_worker()

    if not faa_path or not faa_path.exists() or not gff_path or not gff_path.exists():
        logger.warning(
            f"[{accession_id}] Thiếu file FAA hoặc GFF đầu vào. "
            f"FAA: {faa_path}, GFF: {gff_path}"
        )
        return None

    try:
        hmm_dir = _global_config.paths.hmm_profiles_dir
        hmm_output_dir = _global_config.paths.hmmer_dir
        hmm_output_dir.mkdir(parents=True, exist_ok=True)

        all_hits: List[pd.DataFrame] = []

        # 1. Thực thi HMM Search bằng pyhmmer quét qua các tệp .hmm
        for hmm_file in hmm_dir.glob("*.hmm"):
            if "pfam-a" in hmm_file.name.lower():
                continue
            df_hit = _worker_annotator.annotate_to_dataframe(faa_path, hmm_file)
            if not df_hit.empty:
                all_hits.append(df_hit)

        df_all_hits = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()

        # 2. GHI FILE ĐẦU RA HMM SEARCH VÀO data/interim/hmm_outputs/
        output_tbl_path = hmm_output_dir / f"{accession_id}_hmm_hits.tsv"
        if not df_all_hits.empty:
            df_all_hits.to_csv(output_tbl_path, sep="\t", index=False)
        else:
            # Tạo file rỗng có header chuẩn để phục vụ provenance tracking
            pd.DataFrame(
                columns=["target_protein", "gene_family", "effective_score", "seq_evalue"]
            ).to_csv(output_tbl_path, sep="\t", index=False)

        # 3. Parse GFF3 & Lọc cụm gen Synteny
        df_gff = _worker_cluster_filter.parse_gff3(gff_path)

        clusters = []
        if not df_all_hits.empty and not df_gff.empty:
            clusters = _worker_cluster_filter.group_into_clusters(df_all_hits, df_gff)

        raw_record = {
            "accession": accession_id,
            "status": "SUCCESS",
            "df_all_hits": df_all_hits,
            "clusters": clusters,
            "pfam_domains": df_all_hits["gene_family"].tolist() if "gene_family" in df_all_hits.columns else [],
            "metadata": {},
        }

        # 4. Trích xuất Ma trận Đặc trưng
        if not _worker_extractor.is_fitted_:
            _worker_extractor.fit([raw_record])

        df_feat = _worker_extractor.transform([raw_record], return_sparse=False)

        # 5. CHỐT CHẶN AN TOÀN: Kiểm tra ma trận đặc trưng rỗng trước khi chuyển đổi
        # Bỏ qua cột 'accession' nếu có, kiểm tra số lượng cột đặc trưng thực tế
        feature_cols = [c for c in df_feat.columns if c != "accession"]
        if df_feat.empty or len(feature_cols) == 0:
            logger.warning(
                f"[{accession_id}] CẢNH BÁO: Ma trận đặc trưng rỗng (0 đặc trưng). "
                f"Bỏ qua mẫu để phòng chống lỗi crash StandardScaler!"
            )
            return None

        res_dict = df_feat.iloc[0].to_dict()
        res_dict["accession_id"] = accession_id
        res_dict["status"] = "SUCCESS"
        return res_dict

    except Exception as exc:
        logger.error(f"Lỗi khi xử lý Accession [{accession_id}]: {exc}", exc_info=True)
        return None


def index_genome_paths(
    genomes_dir: Path,
    annotation_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Index vị trí tệp FAA và GFF, ưu tiên tìm trong data/interim/annotations/ trước."""
    faa_map: Dict[str, Path] = {}
    gff_map: Dict[str, Path] = {}

    search_dirs = []
    if annotation_dir and annotation_dir.exists():
        search_dirs.append(annotation_dir)
    if genomes_dir and genomes_dir.exists():
        search_dirs.append(genomes_dir)

    for search_dir in search_dirs:
        for file in search_dir.iterdir():
            if not file.is_file():
                continue
            # Trích xuất mã accession chính xác
            stem = file.name.split("_genomic")[0].split("_protein")[0]
            if (file.suffix == ".faa" or file.name.endswith("_protein.faa")) and stem not in faa_map:
                faa_map[stem] = file
            elif (file.suffix == ".gff" or file.name.endswith("_genomic.gff")) and stem not in gff_map:
                gff_map[stem] = file

    return faa_map, gff_map


def get_completed_accessions(checkpoint_dir: Path) -> Set[str]:
    """Khôi phục trạng thái các accession đã được xử lý từ checkpoint parquet."""
    completed: Set[str] = set()
    if not checkpoint_dir.exists():
        return completed

    checkpoint_files = list(checkpoint_dir.glob("chk_*.parquet"))
    for chk_file in checkpoint_files:
        try:
            dataset = ds.dataset(chk_file, format="parquet")
            table = dataset.to_table(columns=["accession_id"])
            completed.update(table["accession_id"].to_pylist())
        except Exception as err:
            logger.warning(f"Không thể đọc file checkpoint {chk_file}: {err}")
    return completed


def stream_merge_checkpoints(
    checkpoint_dir: Path, output_path: Path, metadata_path: Path
) -> Tuple[int, int]:
    """Streaming merge tất cả tệp checkpoint thành file Parquet/CSV duy nhất."""
    chk_files = sorted(list(checkpoint_dir.glob("chk_*.parquet")))
    if not chk_files:
        raise FileNotFoundError("Không tìm thấy file checkpoint nào để hợp nhất.")

    datasets = [ds.dataset(f, format="parquet") for f in chk_files]
    unified_schema = pa.unify_schemas([d.schema for d in datasets])

    priority_cols = ["accession_id", "status", "complete_hdk_clusters", "clusters_found"]
    feature_names = sorted([name for name in unified_schema.names if name not in priority_cols])

    existing_priority = [col for col in priority_cols if col in unified_schema.names]
    final_schema_names = existing_priority + feature_names
    final_fields = [unified_schema.field(name) for name in final_schema_names]
    final_schema = pa.schema(final_fields)

    total_samples = sum(d.count_rows() for d in datasets)

    # Chốt chặn an toàn ở cấp độ tổng hợp ma trận
    if total_samples == 0 or len(feature_names) == 0:
        logger.warning(
            f"CẢNH BÁO NGHÊM TRỌNG: Tổng số mẫu={total_samples}, Số đặc trưng={len(feature_names)}. "
            f"Ma trận đặc trưng cuối cùng bị rỗng!"
        )

    def align_table(table: pa.Table) -> pa.Table:
        existing_cols = set(table.schema.names)
        for field in final_fields:
            if field.name not in existing_cols:
                null_array = pa.nulls(table.num_rows, type=field.type)
                table = table.append_column(field, null_array)
        return table.select(final_schema_names)

    is_parquet = output_path.suffix.lower() == ".parquet"
    if is_parquet:
        with pq.ParquetWriter(output_path, schema=final_schema, compression="snappy") as writer:
            for dataset in datasets:
                table = dataset.to_table()
                aligned_table = align_table(table)
                writer.write_table(aligned_table)
    else:
        with open(output_path, "wb") as out_file:
            for idx, dataset in enumerate(datasets):
                table = dataset.to_table()
                aligned_table = align_table(table)
                write_options = pcsv.WriteOptions(include_header=(idx == 0))
                pcsv.write_csv(aligned_table, out_file, write_options=write_options)

    metadata = {
        "total_samples": total_samples,
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "generated_at": datetime.now().isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return total_samples, len(feature_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="NifPredict HPC Feature Matrix Orchestrator")
    parser.add_argument("--input-file", "-i", type=Path, default=Path("data/batch_accessions.txt"))
    parser.add_argument("--genomes-dir", "-g", type=Path, default=Path("data/raw/genomes/"))
    parser.add_argument("--output-path", "-o", type=Path, default=Path("data/processed/feature_matrix.parquet"))
    parser.add_argument("--batch-size", "-b", type=int, default=500)
    parser.add_argument("--num-workers", "-w", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")

    args = parser.parse_args()
    config = load_config(auto_create_dirs=True)

    output_path: Path = args.output_path
    output_dir = output_path.parent
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "feature_names.json"

    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # Ưu tiên tìm tệp trong data/interim/annotations/, fallback về args.genomes_dir
    faa_map, gff_map = index_genome_paths(
        genomes_dir=args.genomes_dir,
        annotation_dir=config.paths.annotation_dir
    )

    if args.input_file and args.input_file.exists():
        with open(args.input_file, "r", encoding="utf-8") as f:
            all_accessions = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        all_accessions = sorted(list(set(faa_map.keys()).intersection(set(gff_map.keys()))))

    if args.force_rebuild:
        for chk in checkpoint_dir.glob("chk_*.parquet"):
            chk.unlink()
        completed_accessions = set()
    else:
        completed_accessions = get_completed_accessions(checkpoint_dir)

    pending_accessions = [acc for acc in all_accessions if acc not in completed_accessions]

    if pending_accessions:
        batches = [
            pending_accessions[i : i + args.batch_size]
            for i in range(0, len(pending_accessions), args.batch_size)
        ]

        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as executor:
            with tqdm(total=len(pending_accessions), desc="Building Features", unit="genome") as pbar:
                for batch_idx, batch_accs in enumerate(batches, start=1):
                    futures = {
                        executor.submit(_process_single_genome, acc, faa_map.get(acc), gff_map.get(acc)): acc
                        for acc in batch_accs
                    }

                    batch_results: List[Dict[str, Any]] = []
                    for future in as_completed(futures):
                        res = future.result()
                        if res:
                            batch_results.append(res)
                        pbar.update(1)

                    if batch_results:
                        df_batch = pd.DataFrame(batch_results)
                        batch_filename = f"chk_{session_id}_batch_{batch_idx:05d}.parquet"
                        df_batch.to_parquet(checkpoint_dir / batch_filename, engine="pyarrow", index=False)

                    del batch_results
                    gc.collect()

    total_samples, num_features = stream_merge_checkpoints(checkpoint_dir, output_path, metadata_path)

    if not args.keep_checkpoints:
        for chk in checkpoint_dir.glob("chk_*.parquet"):
            chk.unlink()
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass

    logger.info(
        f"Hoàn tất xuất ma trận đặc trưng thành công! "
        f"Mẫu (Samples): {total_samples}, Đặc trưng (Features): {num_features}"
    )


if __name__ == "__main__":
    main()