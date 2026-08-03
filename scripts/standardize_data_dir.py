#!/usr/bin/env python3
"""
Standardize and Clean Raw Data Directory for NifPredict.

Refactored according to Data Architecture Audit:
1. Data Loss Prevention: Supports multi-part extensions (.gz), uses MD5 checksums 
   for duplicate removal, and preserves NCBI data provenance files.
2. Architectural Safety: Eliminates source/target iteration conflicts and 
   maintains file extension consistency for downstream tools.
3. Idempotency: Fully repeatable without risk of data corruption or duplicate move loops.
4. Flat Layout Support: Scans and processes raw genome files (.fna, .gff, .faa) 
   directly under data/raw/genomes/ and standardizes them into interim annotations.
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Resolve project base directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nifpredict.utils.config import AppConfig, load_config
from nifpredict.utils.logger import get_logger

logger = get_logger("standardize_data_dir")

ACCESSION_REGEX = re.compile(r"(GC[FA]_\d+\.\d+)")


def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file in chunks for memory safety."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_accession(file_path: Path) -> Optional[str]:
    """Trích xuất mã Accession (GCF_xxx / GCA_xxx) từ tên tệp hoặc đường dẫn."""
    match = ACCESSION_REGEX.search(file_path.name)
    if match:
        return match.group(1)

    for part in file_path.parts:
        match = ACCESSION_REGEX.search(part)
        if match:
            return match.group(1)

    return None


class DataDirectoryStandardizer:
    """Production-grade standardizer for biological raw & interim data directories."""

    def __init__(self, config_path: Path, dry_run: bool = False, copy_mode: str = "copy") -> None:
        self.project_root = PROJECT_ROOT
        self.config_path = (
            config_path
            if config_path.is_absolute()
            else self.project_root / config_path
        )
        self.dry_run = dry_run
        self.copy_mode = copy_mode

        self.config: AppConfig = load_config(self.config_path, auto_create_dirs=True)
        self._init_paths()

        self.stats: Dict[str, int] = {
            "processed": 0,
            "moved": 0,
            "duplicates_removed": 0,
            "checksum_mismatches": 0,
            "dirs_cleaned": 0,
            "interim_standardized": 0,
        }

    def _init_paths(self) -> None:
        """Đồng bộ đường dẫn chuẩn trực tiếp từ Pydantic AppConfig."""
        self.target_genomes_dir = self.config.paths.raw_genomes_dir
        self.target_metadata_dir = self.config.paths.raw_metadata_dir
        self.target_zip_dir = self.config.paths.raw_zip_dir
        self.interim_annotation_dir = self.config.paths.annotation_dir

        self.external_source_dirs: List[Path] = [
            self.project_root / "data/raw_genomes",
            self.project_root / "data/raw_metadata",
            self.project_root / "data/raw/zips",
        ]

        self.root_zip = self.project_root / "ncbi_dataset.zip"

        if not self.dry_run:
            self.target_genomes_dir.mkdir(parents=True, exist_ok=True)
            self.target_metadata_dir.mkdir(parents=True, exist_ok=True)
            self.target_zip_dir.mkdir(parents=True, exist_ok=True)
            self.interim_annotation_dir.mkdir(parents=True, exist_ok=True)

    def extract_accession(self, file_path: Path) -> Optional[str]:
        """Extract NCBI accession (GCF_/GCA_) using parse_accession helper."""
        return parse_accession(file_path)

    def classify_and_resolve_target(
        self, file_path: Path
    ) -> Optional[Tuple[str, Path]]:
        """
        Classify file type and determine standard target filename and directory.
        Handles both uncompressed and gzipped (.gz) files.
        """
        accession = self.extract_accession(file_path)
        full_name = file_path.name.lower()

        # Preserve .gz extension if present
        is_gz = full_name.endswith(".gz")
        gz_ext = ".gz" if is_gz else ""

        # Strip .gz temporarily for inner extension checks
        base_name = full_name[:-3] if is_gz else full_name
        stem_ext = Path(base_name).suffix.lower()

        # 1. Zip archives
        if file_path.suffix.lower() == ".zip":
            std_name = f"{accession}.zip" if accession else file_path.name
            return std_name, self.target_zip_dir

        if not accession:
            return None

        # 2. Data Provenance & Metadata files
        if "dataset_catalog.json" in base_name:
            return f"{accession}_dataset_catalog.json{gz_ext}", self.target_metadata_dir
        if "md5sum.txt" in base_name:
            return f"{accession}_md5sum.txt{gz_ext}", self.target_metadata_dir

        # 3. Genomic sequence (.fna, .fa, .fasta)
        if stem_ext in [".fna", ".fa", ".fasta"] and "protein" not in base_name:
            return f"{accession}_genomic.fna{gz_ext}", self.target_genomes_dir

        # 4. Protein sequence (.faa)
        if stem_ext == ".faa" or "protein" in base_name:
            return f"{accession}_protein.faa{gz_ext}", self.target_genomes_dir

        # 5. Genomic Annotation (.gff, .gff3)
        if stem_ext in [".gff", ".gff3"]:
            return f"{accession}_genomic.gff{gz_ext}", self.target_genomes_dir

        # 6. Assembly Report (.jsonl)
        if stem_ext == ".jsonl" or "assembly_report" in base_name or "assembly_data_report" in base_name:
            return f"{accession}_assembly_report.jsonl{gz_ext}", self.target_metadata_dir

        return None

    def _safe_move_or_clean(self, src: Path, dst: Path) -> None:
        """
        Move src to dst safely. If dst exists, verify data integrity using MD5 checksum.
        Never delete src unless MD5 checksum matches dst exactly.
        """
        if src.resolve() == dst.resolve():
            return

        self.stats["processed"] += 1
        src_rel = src.relative_to(self.project_root) if src.is_relative_to(self.project_root) else src
        dst_rel = dst.relative_to(self.project_root) if dst.is_relative_to(self.project_root) else dst

        if dst.exists():
            src_md5 = compute_md5(src)
            dst_md5 = compute_md5(dst)

            if src_md5 == dst_md5:
                # Exact identical duplicate -> Safe to remove src
                msg = (
                    f"[DRY-RUN] Remove verified duplicate (MD5 match): {src_rel}"
                    if self.dry_run
                    else f"Removed verified duplicate (MD5 match): {src_rel}"
                )
                logger.info(msg)
                if not self.dry_run:
                    src.unlink()
                self.stats["duplicates_removed"] += 1
                return
            else:
                # MD5 Mismatch -> Risk of data corruption/different version
                logger.error(
                    f"CRITICAL: Checksum mismatch between {src_rel} and {dst_rel}. "
                    "Skipping operation to prevent data loss."
                )
                self.stats["checksum_mismatches"] += 1
                return

        # Perform safe move operation
        msg = (
            f"[DRY-RUN] Move: {src_rel} -> {dst_rel}"
            if self.dry_run
            else f"Moved: {src_rel} -> {dst_rel}"
        )
        logger.info(msg)

        if not self.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        self.stats["moved"] += 1

    def process_root_files(self) -> None:
        """Process and move stray root archives."""
        if self.root_zip.exists():
            target_dst = self.target_zip_dir / self.root_zip.name
            self._safe_move_or_clean(self.root_zip, target_dst)

    def scan_and_standardize_raw(self) -> None:
        """
        Scan external source directories and canonical target directories separately
        to avoid undefined behaviors from concurrent reads/writes.
        """
        # Phase 1: Process external non-standard folders
        for source_dir in self.external_source_dirs:
            if not source_dir.exists():
                continue

            files = [p for p in source_dir.rglob("*") if p.is_file()]
            for file_path in files:
                resolution = self.classify_and_resolve_target(file_path)
                if resolution:
                    std_filename, target_dir = resolution
                    dst_path = target_dir / std_filename
                    self._safe_move_or_clean(file_path, dst_path)

        # Phase 2: In-place standardization within canonical target directories
        target_dirs = [self.target_genomes_dir, self.target_metadata_dir]
        for target_dir in target_dirs:
            if not target_dir.exists():
                continue

            files = [p for p in target_dir.rglob("*") if p.is_file()]
            for file_path in files:
                resolution = self.classify_and_resolve_target(file_path)
                if resolution:
                    std_filename, dest_dir = resolution
                    dst_path = dest_dir / std_filename
                    if file_path.resolve() != dst_path.resolve():
                        self._safe_move_or_clean(file_path, dst_path)

    def populate_interim_annotations(self) -> None:
        """
        Quét dữ liệu phẳng từ target_genomes_dir, gom nhóm bộ 3 (.fna, .gff, .faa)
        và chuẩn hóa sang interim_annotation_dir.
        """
        logger.info(f"Đang đồng bộ dữ liệu phẳng từ '{self.target_genomes_dir}' sang '{self.interim_annotation_dir}'...")
        if not self.target_genomes_dir.exists():
            logger.warning(f"Thư mục raw genomes không tồn tại: {self.target_genomes_dir}")
            return

        grouped_files: Dict[str, Dict[str, Path]] = {}

        for file_path in self.target_genomes_dir.iterdir():
            if not file_path.is_file():
                continue

            acc = parse_accession(file_path)
            if not acc:
                continue

            if acc not in grouped_files:
                grouped_files[acc] = {}

            name_lower = file_path.name.lower()
            if name_lower.endswith(("_genomic.fna", ".fna", ".fasta", ".fa", "_genomic.fna.gz", ".fna.gz")):
                grouped_files[acc]["fna"] = file_path
            elif name_lower.endswith(("_genomic.gff", ".gff", ".gff3", "_genomic.gff.gz", ".gff.gz")):
                grouped_files[acc]["gff"] = file_path
            elif name_lower.endswith(("_protein.faa", ".faa", "_protein.faa.gz", ".faa.gz")):
                grouped_files[acc]["faa"] = file_path

        for acc, files in grouped_files.items():
            has_fna, has_gff, has_faa = "fna" in files, "gff" in files, "faa" in files
            if not (has_fna and has_gff and has_faa):
                missing = [k.upper() for k in ["fna", "gff", "faa"] if k not in files]
                logger.warning(f"[{acc}] Dữ liệu thô bị thiếu các tệp: {', '.join(missing)}")

            for ftype, src_path in files.items():
                is_gz = src_path.name.lower().endswith(".gz")
                gz_ext = ".gz" if is_gz else ""

                if ftype == "fna":
                    dest_name = f"{acc}_genomic.fna{gz_ext}"
                elif ftype == "gff":
                    dest_name = f"{acc}_genomic.gff{gz_ext}"
                elif ftype == "faa":
                    dest_name = f"{acc}_protein.faa{gz_ext}"
                else:
                    dest_name = f"{acc}_{src_path.name}"

                dest_path = self.interim_annotation_dir / dest_name

                msg = (
                    f"[DRY-RUN] Interim Sync: {src_path.name} -> {dest_path.name}"
                    if self.dry_run
                    else f"Interim Sync: {src_path.name} -> {dest_path.name}"
                )
                logger.info(msg)

                if not self.dry_run:
                    try:
                        if self.copy_mode == "symlink":
                            if dest_path.exists() or dest_path.is_symlink():
                                dest_path.unlink()
                            dest_path.symlink_to(src_path.resolve())
                        else:
                            shutil.copy2(src_path, dest_path)
                    except Exception as err:
                        logger.error(f"Lỗi khi đồng bộ {src_path.name} -> {dest_path.name}: {err}")

            self.stats["interim_standardized"] += 1

    def cleanup_empty_directories(self) -> None:
        """Clean empty subdirectories from external sources after migration."""
        temp_filenames = {"README.md", ".DS_Store"}

        for d in self.external_source_dirs:
            if not d.exists():
                continue

            for item in list(d.rglob("*")):
                if item.is_file() and item.name in temp_filenames:
                    item_rel = item.relative_to(self.project_root)
                    msg = (
                        f"[DRY-RUN] Clean temp file: {item_rel}"
                        if self.dry_run
                        else f"Cleaned temp file: {item_rel}"
                    )
                    logger.info(msg)
                    if not self.dry_run:
                        item.unlink()

            for sub_dir in sorted(list(d.rglob("*")), reverse=True):
                if sub_dir.is_dir():
                    try:
                        if not self.dry_run and not any(sub_dir.iterdir()):
                            sub_dir.rmdir()
                            self.stats["dirs_cleaned"] += 1
                        elif self.dry_run:
                            logger.info(
                                f"[DRY-RUN] Directory candidate for removal: {sub_dir.relative_to(self.project_root)}"
                            )
                    except OSError:
                        pass

            try:
                if not self.dry_run and not any(d.iterdir()):
                    d.rmdir()
                    self.stats["dirs_cleaned"] += 1
            except OSError:
                pass

    def run(self) -> None:
        """Run standardization process with full audit compliance."""
        logger.info(
            f"Starting data directory standardization (Dry-run: {self.dry_run}, Mode: {self.copy_mode})"
        )

        self.process_root_files()
        self.scan_and_standardize_raw()
        self.populate_interim_annotations()
        self.cleanup_empty_directories()

        interim_files_count = len(list(self.interim_annotation_dir.glob("*"))) if self.interim_annotation_dir.exists() else 0

        logger.info("=== Standardization Summary ===")
        logger.info(f"Files Processed        : {self.stats['processed']}")
        logger.info(f"Files Moved/Renamed    : {self.stats['moved']}")
        logger.info(f"Duplicates Cleaned     : {self.stats['duplicates_removed']}")
        logger.info(f"Checksum Mismatches    : {self.stats['checksum_mismatches']}")
        logger.info(f"Directories Cleaned    : {self.stats['dirs_cleaned']}")
        logger.info(f"Accessions Standardized: {self.stats['interim_standardized']}")
        logger.info(f"Interim Annotation Dir : {self.interim_annotation_dir} ({interim_files_count} files)")
        logger.info("===============================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize and consolidate raw data directories for NifPredict."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml relative to project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without modifying or moving any files.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Chế độ đồng bộ interim: 'copy' (sao chép tệp) hoặc 'symlink' (tạo liên kết mềm)",
    )

    args = parser.parse_args()

    standardizer = DataDirectoryStandardizer(
        config_path=Path(args.config),
        dry_run=args.dry_run,
        copy_mode=args.mode,
    )
    standardizer.run()


if __name__ == "__main__":
    main()