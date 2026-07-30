#!/usr/bin/env python3
"""
Script kiểm thử quy trình tải, giải nén, tổ chức và xác minh tính toàn vẹn dữ liệu NCBI.
Môi trường: Production-Grade (Tối ưu I/O, phòng thủ OSError, hỗ trợ CLI & Logging).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from nifpredict.utils.config import load_config
from nifpredict.utils.logger import setup_logger_from_config
from nifpredict.data import NCBIDownloader, NCBIExtractor


def cleanup_zip(zip_path: Optional[Path], logger: logging.Logger) -> None:
    """Xóa tệp ZIP tạm thời một cách an toàn và tối giản (Idempotent Cleanup)."""
    if zip_path and zip_path.exists():
        try:
            zip_path.unlink()
            logger.info(f"Đã dọn dẹp tệp ZIP tạm thời: {zip_path}")
        except OSError as e:
            logger.warning(f"Không thể dọn dẹp tệp ZIP {zip_path}: {e}")


def validate_extracted_files(
    accession: str,
    raw_genomes_dir: Path,
    raw_metadata_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Kiểm tra sự tồn tại và dung lượng (> 0 bytes) của các tệp đầu ra.
    
    Phòng thủ & Tối ưu:
    - Single-pass traversal O(N).
    - Cache stat() tránh System Call Leak.
    - Bắt OSError khi kiểm tra tệp (tránh dừng luồng do Broken Symlinks / Permission Denied).
    """
    logger.info("=== 3. KIỂM TRA TÍNH TOÀN VÊN DỮ LIỆU (VALIDATION) ===")

    target_extensions = {".fna", ".faa", ".gff", ".jsonl", ".json"}
    valid_files: List[Tuple[Path, int]] = []
    invalid_files: List[Path] = []

    def scan_directory(directory: Path) -> None:
        if not directory.exists():
            return
        
        for file in directory.rglob("*"):
            try:
                if file.is_file() and file.suffix.lower() in target_extensions:
                    if accession in file.name:
                        st = file.stat()
                        if st.st_size > 0:
                            valid_files.append((file, st.st_size))
                        else:
                            invalid_files.append(file)
            except OSError as err:
                logger.warning(f"Lỗi truy cập hệ thống tệp tại '{file}': {err}")
                if accession in file.name:
                    invalid_files.append(file)

    scan_directory(raw_genomes_dir)
    scan_directory(raw_metadata_dir)

    # Báo cáo kết quả
    logger.info(f"Tổng số tệp hợp lệ (> 0 bytes): {len(valid_files)}")
    for f, size_bytes in valid_files:
        size_kb = size_bytes / 1024
        logger.info(f"  [OK] {f.name} ({size_kb:.2f} KB) -> {f.parent}")

    if invalid_files:
        logger.error(f"Tổng số tệp lỗi/rỗng/không thể đọc (0 bytes / OSError): {len(invalid_files)}")
        for f in invalid_files:
            logger.error(f"  [FAIL] {f.name}")
        return False

    if not valid_files:
        logger.error(f"Thất bại: Không tìm thấy tệp dữ liệu hợp lệ nào cho Accession '{accession}'")
        return False

    return True


def run_test_workflow(accession: str, keep_zip: bool = False) -> bool:
    """Điều phối toàn bộ quy trình kiểm thử Tải -> Giải nén -> Dọn dẹp -> Xác minh."""
    logger: logging.Logger = setup_logger_from_config("nifpredict.test_download")

    # Step 0: Đọc cấu hình từ Pydantic AppConfig
    try:
        config = load_config()
        # Đọc đúng key theo file config.yaml của bạn
        raw_genomes_dir = Path(config.paths.raw["genomes_dir"])
        raw_metadata_dir = Path(config.paths.raw["metadata_dir"])
    except KeyError as e:
        logger.critical(f"Lỗi cấu hình: Thiếu key bắt buộc trong paths.raw: {e}")
        return False
    except Exception as e:
        logger.critical(f"Không thể tải file cấu hình hệ thống: {e}")
        return False

    # Tạo thư mục đích nếu chưa tồn tại
    raw_genomes_dir.mkdir(parents=True, exist_ok=True)
    raw_metadata_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== BẮT ĐẦU TEST WORKFLOW VỚI ACCESSION: {accession} ===")

    downloader = NCBIDownloader(config=config, logger=logger)
    extractor = NCBIExtractor(config=config, logger=logger)

    zip_path: Optional[Path] = None

    try:
        # Step 1: Tải Zip Package
        logger.info("=== 1. TẢI ZIP PACKAGE từ NCBI ===")
        downloaded = downloader.download_genome_zip(accession)
        if not downloaded:
            logger.error("Test Thất Bại: Không thể tải bản Zip từ NCBI!")
            return False

        zip_path = Path(downloaded)
        if not zip_path.exists():
            logger.error(f"Test Thất Bại: Tệp zip không tồn tại tại {zip_path}")
            return False

        try:
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            logger.info(f"Tải thành công: {zip_path} ({zip_size_mb:.2f} MB)")
        except OSError as e:
            logger.warning(f"Không thể lấy dung lượng tệp ZIP: {e}")

        # Step 2: Giải nén & Sắp xếp dữ liệu
        logger.info("=== 2. GIẢI NÉN VÀ SẮP XẾP DỮ LIỆU ===")
        metadata = extractor.extract_package(zip_path, accession)
        if metadata.download_status != "SUCCESS":
            logger.error(
                f"Test Thất Bại: Giải nén không thành công ({metadata.download_status})!"
            )
            return False

        organism = getattr(metadata, "organism_name", "N/A")
        tax_id = getattr(metadata, "tax_id", "N/A")
        level = getattr(metadata, "assembly_level", "N/A")
        logger.info(f"Thông tin trích xuất: Organism='{organism}', TaxID={tax_id}, AssemblyLevel='{level}'")

        # Step 3: Kiểm tra tính toàn vẹn dữ liệu
        is_valid = validate_extracted_files(
            accession=accession,
            raw_genomes_dir=raw_genomes_dir,
            raw_metadata_dir=raw_metadata_dir,
            logger=logger
        )

        if is_valid:
            logger.info(f"=== KIỂM THỬ HOÀN TẤT THÀNH CÔNG CHO ACCESSION: {accession} ===")
            return True
        return False

    except Exception as e:
        logger.exception(f"Lỗi ngoại lệ không mong muốn trong quá trình thực thi: {e}")
        return False
    finally:
        # Step 4: Dọn dẹp tệp ZIP duy nhất tại khối finally
        downloader.close()
        if not keep_zip:
            cleanup_zip(zip_path, logger)
        elif zip_path and zip_path.exists():
            logger.info(f"Giữ lại tệp ZIP theo cờ --keep-zip: {zip_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm thử quy trình tải, giải nén và phân loại dữ liệu NCBI Genomes."
    )
    parser.add_argument(
        "--accession",
        type=str,
        default="GCF_000021045.1",
        help="Mã NCBI Accession để chạy kiểm thử (Mặc định: GCF_000021045.1)"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Giữ lại tệp .zip sau khi giải nén (Mặc định: tự động xóa)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success = run_test_workflow(accession=args.accession, keep_zip=args.keep_zip)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()