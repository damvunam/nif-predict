#!/usr/bin/env python3
"""
Kiểm thử quy trình tải, giải nén, tổ chức và xác minh dữ liệu NCBI.

Chế độ chạy:
- Batch (mặc định): đọc accession từ data/batch_accessions.txt.
- Đơn mẫu: truyền --accession GCF_xxxxxxxx.x hoặc GCA_xxxxxxxx.x.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from nifpredict.data import NCBIDownloader, NCBIExtractor
from nifpredict.utils.config import load_config
from nifpredict.utils.logger import setup_logger_from_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_FILE = PROJECT_ROOT / "data" / "batch_accessions.txt"


def cleanup_zip(zip_path: Optional[Path], logger: logging.Logger) -> None:
    """Xóa tệp ZIP tạm thời theo cách an toàn và idempotent."""
    if zip_path is None or not zip_path.exists():
        return

    try:
        zip_path.unlink()
        logger.info("Đã dọn dẹp tệp ZIP tạm thời: %s", zip_path)
    except OSError as exc:
        logger.warning("Không thể dọn dẹp tệp ZIP %s: %s", zip_path, exc)


def load_accessions(batch_file: Path, logger: logging.Logger) -> List[str]:
    """
    Đọc accession từ tệp batch.

    Dòng trống và dòng có ký tự đầu tiên (sau khoảng trắng) là ``#`` sẽ
    được bỏ qua. Thứ tự và các accession lặp lại được giữ nguyên.
    """
    try:
        with batch_file.open("r", encoding="utf-8") as handle:
            accessions = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Không tìm thấy tệp danh sách accession: {batch_file}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Không thể đọc tệp batch '{batch_file}': {exc}") from exc

    if not accessions:
        raise ValueError(f"Tệp batch không chứa accession hợp lệ: {batch_file}")

    logger.info("Đã đọc %d accession từ %s", len(accessions), batch_file)
    return accessions


def validate_extracted_files(
    accession: str,
    raw_genomes_dir: Path,
    raw_metadata_dir: Path,
    logger: logging.Logger,
) -> bool:
    """Kiểm tra sự tồn tại và dung lượng của các tệp đầu ra."""
    logger.info("=== 3. KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU ===")

    target_extensions = {".fna", ".faa", ".gff", ".jsonl", ".json"}
    valid_files: List[Tuple[Path, int]] = []
    invalid_files: List[Path] = []

    def scan_directory(directory: Path) -> None:
        if not directory.exists():
            return

        for file_path in directory.rglob("*"):
            try:
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in target_extensions
                    and accession in file_path.name
                ):
                    size_bytes = file_path.stat().st_size
                    if size_bytes > 0:
                        valid_files.append((file_path, size_bytes))
                    else:
                        invalid_files.append(file_path)
            except OSError as exc:
                logger.warning(
                    "Lỗi truy cập hệ thống tệp tại '%s': %s", file_path, exc
                )
                if accession in file_path.name:
                    invalid_files.append(file_path)

    scan_directory(raw_genomes_dir)
    scan_directory(raw_metadata_dir)

    logger.info("Tổng số tệp hợp lệ (> 0 bytes): %d", len(valid_files))
    for file_path, size_bytes in valid_files:
        logger.info(
            "  [OK] %s (%.2f KB) -> %s",
            file_path.name,
            size_bytes / 1024,
            file_path.parent,
        )

    if invalid_files:
        logger.error(
            "Tổng số tệp lỗi/rỗng/không thể đọc: %d", len(invalid_files)
        )
        for file_path in invalid_files:
            logger.error("  [FAIL] %s", file_path.name)
        return False

    if not valid_files:
        logger.error(
            "Không tìm thấy tệp dữ liệu hợp lệ cho accession '%s'", accession
        )
        return False

    return True


def run_test_workflow(
    accession: str,
    downloader: NCBIDownloader,
    extractor: NCBIExtractor,
    raw_genomes_dir: Path,
    raw_metadata_dir: Path,
    logger: logging.Logger,
    keep_zip: bool = False,
) -> bool:
    """Chạy quy trình tải, giải nén và xác minh cho một accession."""
    logger.info("=== BẮT ĐẦU WORKFLOW: %s ===", accession)
    zip_path: Optional[Path] = None

    try:
        logger.info("=== 1. TẢI ZIP PACKAGE TỪ NCBI ===")
        downloaded = downloader.download_genome_zip(accession)
        if not downloaded:
            logger.error("Không thể tải ZIP cho accession %s", accession)
            return False

        zip_path = Path(downloaded)
        if not zip_path.is_file():
            logger.error("Tệp ZIP không tồn tại tại %s", zip_path)
            return False

        try:
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            logger.info("Tải thành công: %s (%.2f MB)", zip_path, zip_size_mb)
        except OSError as exc:
            logger.warning("Không thể lấy dung lượng tệp ZIP: %s", exc)

        logger.info("=== 2. GIẢI NÉN VÀ SẮP XẾP DỮ LIỆU ===")
        metadata = extractor.extract_package(zip_path, accession)
        download_status = getattr(metadata, "download_status", None)
        if download_status != "SUCCESS":
            logger.error(
                "Giải nén không thành công cho %s (status=%s)",
                accession,
                download_status,
            )
            return False

        logger.info(
            "Thông tin trích xuất: Organism='%s', TaxID=%s, AssemblyLevel='%s'",
            getattr(metadata, "organism_name", "N/A"),
            getattr(metadata, "tax_id", "N/A"),
            getattr(metadata, "assembly_level", "N/A"),
        )

        is_valid = validate_extracted_files(
            accession=accession,
            raw_genomes_dir=raw_genomes_dir,
            raw_metadata_dir=raw_metadata_dir,
            logger=logger,
        )
        if is_valid:
            logger.info("=== HOÀN TẤT THÀNH CÔNG: %s ===", accession)
        else:
            logger.error("=== XÁC MINH THẤT BẠI: %s ===", accession)

        return is_valid

    except Exception as exc:
        logger.exception("Lỗi khi xử lý accession %s: %s", accession, exc)
        return False
    finally:
        if keep_zip:
            if zip_path is not None and zip_path.exists():
                logger.info("Giữ lại tệp ZIP theo cờ --keep-zip: %s", zip_path)
        else:
            cleanup_zip(zip_path, logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải và xác minh một hoặc nhiều NCBI genome assemblies."
    )
    parser.add_argument(
        "--accession",
        type=str,
        default=None,
        help=(
            "Chạy đơn mẫu với accession được chỉ định. Nếu bỏ qua, chương trình "
            "sẽ chạy batch từ data/batch_accessions.txt."
        ),
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=DEFAULT_BATCH_FILE,
        help=f"Tệp accession dùng cho batch (mặc định: {DEFAULT_BATCH_FILE})",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Giữ lại các tệp .zip sau khi giải nén.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger: logging.Logger = setup_logger_from_config("nifpredict.test_download")

    try:
        accessions = (
            [args.accession.strip()]
            if args.accession and args.accession.strip()
            else load_accessions(args.batch_file, logger)
        )
    except (OSError, ValueError) as exc:
        logger.critical("Không thể khởi tạo danh sách tải: %s", exc)
        sys.exit(2)

    try:
        config = load_config()
        raw_genomes_dir = Path(config.paths.raw["genomes_dir"])
        raw_metadata_dir = Path(config.paths.raw["metadata_dir"])
        raw_genomes_dir.mkdir(parents=True, exist_ok=True)
        raw_metadata_dir.mkdir(parents=True, exist_ok=True)
    except KeyError as exc:
        logger.critical("Thiếu key bắt buộc trong config paths.raw: %s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.critical("Không thể khởi tạo cấu hình hoặc thư mục dữ liệu: %s", exc)
        sys.exit(2)

    downloader: Optional[NCBIDownloader] = None
    successful: List[str] = []
    failed: List[str] = []

    try:
        downloader = NCBIDownloader(config=config, logger=logger)
        extractor = NCBIExtractor(config=config, logger=logger)

        progress = tqdm(
            accessions,
            desc="Downloading genomes",
            unit="genome",
            dynamic_ncols=True,
        )
        for accession in progress:
            progress.set_description(f"Processing {accession}")
            success = run_test_workflow(
                accession=accession,
                downloader=downloader,
                extractor=extractor,
                raw_genomes_dir=raw_genomes_dir,
                raw_metadata_dir=raw_metadata_dir,
                logger=logger,
                keep_zip=args.keep_zip,
            )

            if success:
                successful.append(accession)
            else:
                failed.append(accession)

            progress.set_postfix(
                success=len(successful),
                failed=len(failed),
                refresh=True,
            )
    except Exception as exc:
        logger.exception("Không thể khởi tạo hoặc hoàn tất batch download: %s", exc)
        sys.exit(2)
    finally:
        if downloader is not None:
            try:
                downloader.close()
            except Exception as exc:
                logger.warning("Không thể đóng NCBI downloader: %s", exc)

    logger.info("=== TỔNG KẾT BATCH DOWNLOAD ===")
    logger.info("Tổng số accession đã xử lý: %d", len(accessions))
    logger.info("Thành công: %d", len(successful))
    logger.info("Thất bại: %d", len(failed))

    if failed:
        logger.error("Danh sách accession thất bại: %s", ", ".join(failed))

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
