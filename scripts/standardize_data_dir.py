#!/usr/bin/env python3
"""
Standardize and Clean Raw Data Directory for NifPredict.

Refactored according to Data Architecture Audit:
1. Data Loss Prevention: Supports multi-part extensions (.gz), uses MD5 checksums 
   for duplicate removal, and preserves NCBI data provenance files.
2. Architectural Safety: Eliminates source/target iteration conflicts and 
   maintains file extension consistency for downstream tools.
3. Idempotency: Fully repeatable without risk of data corruption or duplicate move loops.
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Resolve project base directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logger import with fallback
from nifpredict.utils.config import load_config
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


class DataDirectoryStandardizer:
    """Production-grade standardizer for biological raw data directories."""

    def __init__(self, config_path: Path, dry_run: bool = False) -> None:
        self.project_root = PROJECT_ROOT
        self.config_path = (
            config_path
            if config_path.is_absolute()
            else self.project_root / config_path
        )
        self.dry_run = dry_run

        self.config = load_config(config_path)
        self._init_paths()

        self.stats: Dict[str, int] = {
            "processed": 0,
            "moved": 0,
            "duplicates_removed": 0,
            "checksum_mismatches": 0,
            "dirs_cleaned": 0,
        }

    def _load_config(self) -> dict:
        """Load project configuration YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_paths(self) -> None:
      """Đồng bộ đường dẫn chuẩn trực tiếp từ Pydantic AppConfig."""
      self.target_genomes_dir = self.config.paths.raw["genomes_dir"]
      self.target_metadata_dir = self.config.paths.raw["metadata_dir"]
      self.target_zip_dir = self.config.paths.raw["zip_dir"]

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

    def extract_accession(self, file_path: Path) -> Optional[str]:
        """Extract NCBI accession (GCF_/GCA_) from path or filename."""
        match = ACCESSION_REGEX.search(file_path.name)
        if match:
            return match.group(1)

        for part in file_path.parts:
            match = ACCESSION_REGEX.search(part)
            if match:
                return match.group(1)

        return None

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

        # 2. Data Provenance & Metadata files (preserves provenance instead of deleting)
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
        src_rel = src.relative_to(self.project_root)
        dst_rel = dst.relative_to(self.project_root)

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

    def scan_and_standardize(self) -> None:
        """
        Scan external source directories and canonical target directories separately
        to avoid undefined behaviors from concurrent reads/writes.
        """
        # Phase 1: Process external non-standard folders
        for source_dir in self.external_source_dirs:
            if not source_dir.exists():
                continue

            # Snapshot file list beforehand to avoid iterator mutation issues
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

    def cleanup_empty_directories(self) -> None:
        """Clean empty subdirectories from external sources after migration."""
        temp_filenames = {"README.md", ".DS_Store"}

        for d in self.external_source_dirs:
            if not d.exists():
                continue

            # Clean non-provenance temp files
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

            # Remove empty subdirectories bottom-up
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

            # Remove main source folder if empty
            try:
                if not self.dry_run and not any(d.iterdir()):
                    d.rmdir()
                    self.stats["dirs_cleaned"] += 1
            except OSError:
                pass

    def run(self) -> None:
        """Run standardization process with full audit compliance."""
        logger.info(
            f"Starting data directory standardization (Dry-run: {self.dry_run})"
        )

        self.process_root_files()
        self.scan_and_standardize()
        self.cleanup_empty_directories()

        logger.info("=== Standardization Summary ===")
        logger.info(f"Files Processed        : {self.stats['processed']}")
        logger.info(f"Files Moved/Renamed    : {self.stats['moved']}")
        logger.info(f"Duplicates Cleaned     : {self.stats['duplicates_removed']}")
        logger.info(f"Checksum Mismatches    : {self.stats['checksum_mismatches']}")
        logger.info(f"Directories Cleaned    : {self.stats['dirs_cleaned']}")
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

    args = parser.parse_args()

    standardizer = DataDirectoryStandardizer(
        config_path=Path(args.config),
        dry_run=args.dry_run,
    )
    standardizer.run()


if __name__ == "__main__":
    main()