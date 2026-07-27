import json
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

from nifpredict.utils.config import load_config
from nifpredict.utils.logger import setup_logger


@dataclass
class GenomeMetadata:
    """Structure chứa thông tin metadata cốt lõi của một Assembly."""
    accession: str
    organism_name: str
    tax_id: int
    assembly_level: str
    download_status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NCBIResolver:
    """Layer 1: Resolver - Chuyển đổi TaxID / Organism / Accession thành Accession chuẩn."""

    def __init__(self, session: requests.Session, base_url: str, logger=None) -> None:
        self.session = session
        self.base_url = base_url
        self.logger = logger or setup_logger("nifpredict.data.resolver")

    def resolve_taxon_to_accessions(self, taxon: str, limit: int = 10) -> List[str]:
        """Truy vấn NCBI Datasets API để lấy danh sách Accession từ TaxID hoặc tên sinh vật."""
        if taxon.startswith(("GCF_", "GCA_")):
            return [taxon.strip()]

        url = f"{self.base_url}/genome/taxon/{taxon}/dataset_report"
        params = {"page_size": limit}
        self.logger.info(f"Đang tra cứu Taxon/Organism: {taxon}...")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            reports = data.get("reports", [])
            accessions = [
                r["accession"] for r in reports 
                if "accession" in r
            ]
            self.logger.info(f"Tìm thấy {len(accessions)} accession cho taxon '{taxon}'")
            return accessions
        except requests.RequestException as e:
            self.logger.error(f"Lỗi khi tra cứu taxon '{taxon}': {e}")
            return []


class NCBIDownloader:
    """Layer 2: Downloader - Tải Zip Package từ NCBI với cơ chế Caching và Retry."""

    def __init__(self, config_path: str = "config/config.yaml", logger=None) -> None:
        self.config = load_config(config_path)
        self.ncbi_cfg = self.config.get("ncbi", {})
        self.base_url = self.ncbi_cfg.get("api_base_url", "https://api.ncbi.nlm.nih.gov/datasets/v2alpha")
        self.timeout = self.ncbi_cfg.get("timeout_seconds", 60)
        self.max_retries = self.ncbi_cfg.get("max_retries", 3)
        self.raw_zip_dir = Path(self.config["paths"]["raw_zip_dir"])
        self.raw_zip_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger("nifpredict.data.downloader")

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NifPredict-Bioinformatics-Platform/0.1.0"})

    def download_genome_zip(self, accession: str) -> Optional[Path]:
        """Tải Zip Package cho Accession chỉ định. Sử dụng Cache nếu file đã tồn tại."""
        output_zip_path = self.raw_zip_dir / f"{accession}.zip"

        # Caching logic
        if output_zip_path.exists() and output_zip_path.stat().st_size > 0:
            self.logger.info(f"Sử dụng bản Zip từ cache: {output_zip_path.name}")
            return output_zip_path

        url = f"{self.base_url}/genome/accession/{accession}/download"
        params = {
            "include_annotation_type": "GENOME_FASTA,PROT_FASTA",
            "filename": f"{accession}.zip"
        }

        self.logger.info(f"Đang tải Zip Package từ NCBI cho {accession}...")
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout, stream=True)
                response.raise_for_status()

                with open(output_zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.logger.info(f"Đã lưu thành công: {output_zip_path.name}")
                return output_zip_path
            except requests.RequestException as e:
                self.logger.warning(f"[Thử {attempt}/{self.max_retries}] Lỗi tải {accession}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        self.logger.error(f"Thất bại khi tải Archive ZIP cho {accession}")
        return None


class NCBIExtractor:
    """Layer 3: Extractor - Giải nén động và trích xuất Metadata chuẩn."""

    def __init__(self, raw_genomes_dir: Path, raw_metadata_dir: Path, logger=None) -> None:
        self.raw_genomes_dir = Path(raw_genomes_dir)
        self.raw_metadata_dir = Path(raw_metadata_dir)
        self.logger = logger or setup_logger("nifpredict.data.extractor")
        self.raw_genomes_dir.mkdir(parents=True, exist_ok=True)
        self.raw_metadata_dir.mkdir(parents=True, exist_ok=True)

    def extract_package(self, zip_path: Path, accession: str) -> Optional[GenomeMetadata]:
        """Duyệt động archive zip, trích xuất .fna, .faa và parse JSONL thành GenomeMetadata."""
        if not zip_path or not zip_path.exists():
            self.logger.error(f"Tệp tin ZIP không hợp lệ: {zip_path}")
            return None

        metadata = None
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    filename = Path(member).name

                    # 1. Trích xuất Genomic FASTA
                    if filename.endswith(("_genomic.fna", ".fna")):
                        target = self.raw_genomes_dir / f"{accession}_genomic.fna"
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        self.logger.info(f"Đã giải nén Genomic FASTA: {target.name}")

                    # 2. Trích xuất Protein FASTA
                    elif filename.endswith(("_protein.faa", ".faa")):
                        target = self.raw_genomes_dir / f"{accession}_protein.faa"
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        self.logger.info(f"Đã giải nén Protein FASTA: {target.name}")

                    # 3. Trích xuất & Parse Metadata JSONL
                    elif filename == "assembly_data_report.jsonl":
                        target = self.raw_metadata_dir / f"{accession}_assembly_report.jsonl"
                        raw_bytes = zf.read(member)
                        with open(target, "wb") as dst:
                            dst.write(raw_bytes)

                        # Parse dòng json đầu tiên để tạo Record
                        line = raw_bytes.decode("utf-8").splitlines()[0]
                        data = json.loads(line)
                        org = data.get("organism", {})
                        ass = data.get("assemblyInfo", {})

                        metadata = GenomeMetadata(
                            accession=accession,
                            organism_name=org.get("organismName", "Unknown"),
                            tax_id=org.get("taxId", 0),
                            assembly_level=ass.get("assemblyLevel", "Unknown"),
                            download_status="SUCCESS"
                        )
                        self.logger.info(f"Đã ghi nhận metadata cho: {metadata.organism_name}")

            return metadata
        except (zipfile.BadZipFile, json.JSONDecodeError, IndexError) as e:
            self.logger.error(f"Lỗi giải nén hoặc parse metadata cho {accession}: {e}")
            return None
