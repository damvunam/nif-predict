from pathlib import Path
from nifpredict.utils import load_config, setup_logger
from nifpredict.data import NCBIDownloader, NCBIExtractor, NCBIResolver

def main():
    logger = setup_logger("nifpredict.test_download")
    config = load_config()

    # Dữ liệu test: Azotobacter vinelandii (Loài vi khuẩn cố định đạm tự do điển hình)
    target_accession = "GCF_000021045.1"
    logger.info(f"=== BẮT ĐẦU TEST PHASE 2 VỚI ACCESSION: {target_accession} ===")

    # 1. Khởi tạo các module
    downloader = NCBIDownloader()
    extractor = NCBIExtractor(
        raw_genomes_dir=Path(config["paths"]["raw_genomes_dir"]),
        raw_metadata_dir=Path(config["paths"]["raw_metadata_dir"])
    )

    # 2. Tải Zip Package
    zip_path = downloader.download_genome_zip(target_accession)
    if not zip_path:
        logger.error("Test Thất Bại: Không thể tải bản Zip từ NCBI!")
        return

    # 3. Giải nén và trích xuất Metadata
    metadata = extractor.extract_package(zip_path, target_accession)
    if metadata:
        logger.info("=== KIỂM THỬ THÀNH CÔNG ===")
        logger.info(f"Organism: {metadata.organism_name}")
        logger.info(f"TaxID: {metadata.tax_id}")
        logger.info(f"Assembly Level: {metadata.assembly_level}")
        logger.info(f"Status: {metadata.download_status}")
    else:
        logger.error("Test Thất Bại: Lỗi khi giải nén hoặc đọc metadata!")

if __name__ == "__main__":
    main()
