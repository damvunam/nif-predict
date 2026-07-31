#!/usr/bin/env python3
"""NifPredict Feature Matrix Builder Script (HPC Production-Grade).

Orchestrates end-to-end feature extraction with zero-RAM-spike PyArrow streaming,
dynamic schema unification with null padding, deterministic column ordering,
non-colliding session checkpointing, and automatic resource cleanup.
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

from nifpredict.features.hmm_annotator import HMMAnnotator
from nifpredict.features.cluster_filter import ClusterFilter
from nifpredict.utils.config import settings
from nifpredict.features.feature_extractor import FeatureExtractor
from nifpredict.utils.logger import setup_logger

logger = setup_logger("nifpredict.scripts.build_features")

# Biến toàn cục trong từng Worker Process (Cached Objects)
_worker_annotator: Optional[HMMAnnotator] = None
_worker_cluster_filter: Optional[ClusterFilter] = None
_worker_feature_extractor: Optional[FeatureExtractor] = None


def _init_worker() -> None:
    """Khởi tạo một lần duy nhất cho mỗi Worker Process khi spawn/fork."""
    global _worker_annotator, _worker_cluster_filter, _worker_feature_extractor
    _worker_annotator = HMMAnnotator(config=settings.hmm_config)
    _worker_cluster_filter = ClusterFilter(config=settings.filter_config)
    _worker_feature_extractor = FeatureExtractor(config=settings.feature_config)


def _process_single_genome(
    accession_id: str, genome_path: Optional[Path]
) -> Optional[Dict[str, Any]]:
    """Worker task xử lý 1 accession sử dụng các cached instances.

    Primary Column Fix: Đưa accession_id lên đầu Dictionary.
    """
    global _worker_annotator, _worker_cluster_filter, _worker_feature_extractor

    if not _worker_annotator or not _worker_cluster_filter or not _worker_feature_extractor:
        _init_worker()

    try:
        annotations = _worker_annotator.annotate(
            accession_id=accession_id, genome_path=genome_path
        )
        if not annotations:
            return None

        filtered_clusters = _worker_cluster_filter.filter(annotations)
        if not filtered_clusters:
            return None

        features = _worker_feature_extractor.extract(
            accession_id=accession_id, clusters=filtered_clusters
        )
        if not features:
            return None

        # Hotfix 1: accession_id làm Primary Key ở vị trí đầu tiên
        return {"accession_id": accession_id, **features}

    except Exception as exc:
        logger.error(f"Lỗi khi xử lý Accession [{accession_id}]: {exc}", exc_info=False)
        return None


def index_genome_paths(genomes_dir: Path) -> Dict[str, Path]:
    """Quét và index trước đường dẫn file genome bằng 1 lần duyệt đĩa duy nhất."""
    mapping: Dict[str, Path] = {}
    if not genomes_dir or not genomes_dir.exists():
        return mapping

    valid_extensions = {".fasta", ".fna", ".fa", ".gbk", ".gff"}
    for file in genomes_dir.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            mapping[file.stem] = file

    logger.info(f"Đã index thành công {len(mapping)} file genome từ đĩa.")
    return mapping


def get_completed_accessions(checkpoint_dir: Path) -> Set[str]:
    """Thu thập danh sách accession đã được lưu trữ thành công từ tất cả checkpoints."""
    completed: Set[str] = set()
    if not checkpoint_dir.exists():
        return completed

    checkpoint_files = list(checkpoint_dir.glob("chk_*.parquet"))
    if not checkpoint_files:
        return completed

    logger.info(f"Đang quét {len(checkpoint_files)} file checkpoint để khôi phục trạng thái...")
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
    """Hợp nhất các checkpoint bằng PyArrow với Unify Schema động & Null Padding Safety."""
    chk_files = sorted(list(checkpoint_dir.glob("chk_*.parquet")))
    if not chk_files:
        raise FileNotFoundError("Không tìm thấy file checkpoint nào để hợp nhất.")

    logger.info(f"Đang phân tích và hợp nhất Schema cho {len(chk_files)} checkpoints...")

    # 1. Unify Schema toàn cục từ tất cả các checkpoints
    datasets = [ds.dataset(f, format="parquet") for f in chk_files]
    schemas = [d.schema for d in datasets]
    unified_schema = pa.unify_schemas(schemas)

    # 2. Đảm bảo tính Deterministic: accession_id ở đầu, các feature được sort A-Z
    feature_names = sorted([name for name in unified_schema.names if name != "accession_id"])
    final_schema_names = ["accession_id"] + feature_names

    final_fields = [unified_schema.field(name) for name in final_schema_names]
    final_schema = pa.schema(final_fields)

    total_samples = sum(d.count_rows() for d in datasets)
    logger.info(f"Tổng mẫu: {total_samples} | Tổng chiều đặc trưng chuẩn hóa: {len(feature_names)}")

    # 3. Hàm Helper: Align Table bằng cách chèn Null Column cho các đặc trưng còn thiếu trong batch
    def align_table(table: pa.Table) -> pa.Table:
        existing_cols = set(table.schema.names)
        for field in final_fields:
            if field.name not in existing_cols:
                # Chèn null array với kiểu dữ liệu chuẩn xác của field đó
                null_array = pa.nulls(table.num_rows, type=field.type)
                table = table.append_column(field, null_array)
        # Sắp xếp lại danh sách cột theo đúng final_schema_names
        return table.select(final_schema_names)

    # 4. Stream ghi ra đĩa
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

    # 5. Ghi Metadata
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
    parser.add_argument(
        "--output-path", "-o", type=Path, default=Path("data/processed/feature_matrix.parquet")
    )
    parser.add_argument("--batch-size", "-b", type=int, default=500)
    parser.add_argument("--num-workers", "-w", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--force-rebuild", action="store_true", help="Xóa checkpoint cũ và chạy lại từ đầu.")
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Giữ lại thư mục checkpoint sau khi hợp nhất thành công.",
    )

    args = parser.parse_args()

    output_path: Path = args.output_path
    output_dir = output_path.parent
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "feature_names.json"

    # Tạo Session ID duy nhất cho lượt chạy này
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # 1. Thu thập dữ liệu & Indexing I/O
    genome_map = index_genome_paths(args.genomes_dir)

    if args.input_file and args.input_file.exists():
        with open(args.input_file, "r", encoding="utf-8") as f:
            all_accessions = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        all_accessions = sorted(list(genome_map.keys()))

    if not all_accessions:
        logger.error("Không có accession nào để xử lý.")
        sys.exit(1)

    # 2. Quản lý Checkpoints / Resume logic
    if args.force_rebuild:
        logger.warning("Cờ `--force-rebuild` được kích hoạt. Tiến hành xóa toàn bộ checkpoints cũ...")
        for chk in checkpoint_dir.glob("chk_*.parquet"):
            chk.unlink()
        completed_accessions = set()
    else:
        completed_accessions = get_completed_accessions(checkpoint_dir)

    pending_accessions = [acc for acc in all_accessions if acc not in completed_accessions]

    logger.info("=" * 60)
    logger.info("   NIFPREDICT ORCHESTRATION PIPELINE (HPC PRODUCTION)")
    logger.info(f"  • Session ID              : {session_id}")
    logger.info(f"  • Tổng Accessions         : {len(all_accessions)}")
    logger.info(f"  • Đã hoàn thành (Resume)  : {len(completed_accessions)}")
    logger.info(f"  • Cần xử lý (Pending)     : {len(pending_accessions)}")
    logger.info(f"  • Workers                 : {args.num_workers}")
    logger.info("=" * 60)

    # 3. Tiến hành trích xuất đặc trưng nếu còn pending accessions
    if pending_accessions:
        batches = [
            pending_accessions[i : i + args.batch_size]
            for i in range(0, len(pending_accessions), args.batch_size)
        ]

        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as executor:
            with tqdm(total=len(pending_accessions), desc="Orchestrating Pipeline", unit="genome") as pbar:
                for batch_idx, batch_accs in enumerate(batches, start=1):
                    futures = {
                        executor.submit(_process_single_genome, acc, genome_map.get(acc)): acc
                        for acc in batch_accs
                    }

                    batch_results: List[Dict[str, Any]] = []
                    for future in as_completed(futures):
                        res = future.result()
                        if res:
                            batch_results.append(res)
                        pbar.update(1)

                    # Ghi Checkpoint với Session ID + Batch Index tuyệt đối
                    if batch_results:
                        df_batch = pd.DataFrame(batch_results)
                        batch_filename = f"chk_{session_id}_batch_{batch_idx:05d}.parquet"
                        batch_filepath = checkpoint_dir / batch_filename
                        df_batch.to_parquet(batch_filepath, engine="pyarrow", index=False)

                    del batch_results
                    gc.collect()

    # 4. Stream Aggregate Phase (Zero-RAM Spike + Dynamic Schema Alignment)
    try:
        total_samples, num_features = stream_merge_checkpoints(
            checkpoint_dir=checkpoint_dir,
            output_path=output_path,
            metadata_path=metadata_path,
        )
    except Exception as err:
        logger.error(f"Lỗi nghiêm trọng trong quá trình hợp nhất checkpoints: {err}")
        sys.exit(1)

    # 5. Cleanup Checkpoint Files
    if not args.keep_checkpoints:
        logger.info("Đang tự động dọn dẹp các tệp checkpoint trung gian...")
        for chk in checkpoint_dir.glob("chk_*.parquet"):
            try:
                chk.unlink()
            except Exception as err:
                logger.warning(f"Không thể xóa checkpoint {chk}: {err}")

        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass
    else:
        logger.info(f"Checkpoints được giữ lại tại: `{checkpoint_dir}`")

    logger.info("=" * 60)
    logger.info("        HOÀN THÀNH XUẤT MA TRẬN ĐẶC TRƯNG TỔNG HỢP")
    logger.info(f"  • Mẫu lưu trữ thành công  : {total_samples}")
    logger.info(f"  • Số chiều đặc trưng      : {num_features}")
    logger.info(f"  • File lưu trữ Ma trận    : {output_path}")
    logger.info(f"  • File Metadata           : {metadata_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()