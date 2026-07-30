import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union

import requests
import requests.adapters import HTTPAdapter

from nifpredict.utils.config import load_config
from nifpredict.utils.logger import setup_logger


@dataclass
class GenomeMetadata:
    """Structure chứa thông tin metadata cốt lõi của một Assembly."""
    accession: str
    organism_name: str = "Unknown"
    tax_id: int = 0
    assembly_level: str = "Unknown"
    download_status: str = "PENDING"
    extracted_files: Dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RateLimiter:
    """Thresh-safe Rate Limiter"""

    def __init__(self, requests_per_second: float = 3.0) -> None:
        self.defay = 1.0 / requests_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self.defay:
                time.sleep(self.defay - elapsed)
            self._last_call = time.time()

class NCBIResolver:
    """Layer 1: Resolver - Chuyển đổi TaxID / Organism / Accession thành Accession chuẩn."""

    def __init__(self, session: requests.Session, base_url: str, rate_limiter: RateLimiter, logger=None) -> None:
        self.session = session
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self.logger = logger or setup_logger("nifpredict.data.resolver")

    def resolve_taxon_to_accessions(self, taxon: str, limit: int = 10) -> List[str]:
        """Truy vấn NCBI Datasets API để lấy danh sách Accession từ TaxID hoặc tên sinh vật."""
        if taxon.startswith(("GCF_", "GCA_")):
            return [taxon.strip()]

        url = f"{self.base_url}/genome/taxon/{taxon}/dataset_report"
        params = {"page_size": limit}
        self.logger.info(f"Đang tra cứu Taxon/Organism: {taxon}...")

        self.rate_limiter.wait()
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
    NON_RETRYABLE_STATUS_CODES: Set[int] = {400, 401, 403, 404, 410}

    def __init__(self, config_path: str = "config/config.yaml", logger=None) -> None:
        self.config = load_config(config_path)
        self.ncbi_cfg = self.config.get("ncbi", {})
        self.base_url = self.ncbi_cfg.get(
            "api_base_url", "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
        )
        self.timeout = self.ncbi_cfg.get("timeout_seconds", 60)
        self.max_retries = self.ncbi_cfg.get("max_retries", 3)

        # 1. API Key & Rate Limiter
        self.api_key = self.ncbi_cfg.get("api_key") or self.config.get("NCBI_API_KEY")
        requests_per_sec = 10.0 if self.api_key else 3.0
        self.rate_limiter = RateLimiter(requests_per_second=requests_per_sec)

        # 2. Connection Pool Size (Tránh nghẽn khi chạy đa luồng)
        self.session = requests.Session()
        headers = {"User-Agent": "NifPredict-Bioinformatics-Platform/0.1.0"}
        if self.api_key:
            headers["api-key"] = self.api_key
        self.session.headers.update(headers)

        max_conns = self.ncbi_cfg.get("max_connections", 32)
        adapter = HTTPAdapter(pool_connections=max_conns, pool_maxsize=max_conns)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        raw_base = Path(self.config.get("paths", {}).get("raw_dir", "data/raw"))
        self.raw_zip_dir = raw_base / "zips"
        self.raw_zip_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or setup_logger("nifpredict.data.downloader")

    def _verify_zip_integrity(self, zip_path: Path) -> bool:
        """Kiểm tra file ZIP bằng zipfile.is_zipfile và testzip()."""
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            return False
        if not zipfile.is_zipfile(zip_path):
            return False
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                return zf.testzip() is None
        except Exception:
            return False

    def download_genome_zip(self, accession: str) -> Optional[Path]:
        output_zip_path = self.raw_zip_dir / f"{accession}.zip"

        # Caching logic với Integrity Check
        if output_zip_path.exists():
            if self._verify_zip_integrity(output_zip_path):
                self.logger.info(f"Sử dụng bản Zip từ cache: {output_zip_path.name}")
                return output_zip_path
            else:
                self.logger.warning(f"File cache bị hỏng, tải lại: {output_zip_path.name}")
                output_zip_path.unlink(missing_ok=True)

        url = f"{self.base_url}/genome/accession/{accession}/download"
        params = {
            "include_annotation_type": ["GENOME_FASTA", "PROT_FASTA", "GENOME_GFF"],
            "filename": f"{accession}.zip",
        }
        temp_zip_path = output_zip_path.with_suffix(".zip.tmp")

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout, stream=True)

                if response.status_code in self.NON_RETRYABLE_STATUS_CODES:
                    self.logger.warning(f"Accession {accession} lỗi Client HTTP {response.status_code}")
                    return None

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()

                # Atomic Write (Ghi file .tmp)
                with open(temp_zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)

                # Verify trước khi thay thế file chính
                if self._verify_zip_integrity(temp_zip_path):
                    temp_zip_path.replace(output_zip_path)
                    return output_zip_path
                else:
                    temp_zip_path.unlink(missing_ok=True)

            except requests.RequestException as e:
                self.logger.warning(f"[Thử {attempt}/{self.max_retries}] Lỗi tải {accession}: {e}")
                if temp_zip_path.exists():
                    temp_zip_path.unlink(missing_ok=True)

            if attempt < self.max_retries:
                time.sleep(2**attempt)

        return None

    def close(self) -> None:
        if hasattr(self, "session") and self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class NCBIExtractor:
    def __init__(
        self,
        raw_genomes_dir: Union[Path, str],
        raw_metadata_dir: Union[Path, str],
        logger=None,
    ) -> None:
        self.raw_genomes_dir = Path(raw_genomes_dir)
        self.raw_metadata_dir = Path(raw_metadata_dir)
        self.logger = logger or setup_logger("nifpredict.data.extractor")
        self.raw_genomes_dir.mkdir(parents=True, exist_ok=True)
        self.raw_metadata_dir.mkdir(parents=True, exist_ok=True)

    def extract_package(
        self, zip_path: Path, accession: str
    ) -> Optional[GenomeMetadata]:
        if not zip_path or not zip_path.exists():
            return GenomeMetadata(accession=accession, download_status="FAILED")

        target_fna = self.raw_genomes_dir / f"{accession}_genomic.fna"
        target_faa = self.raw_genomes_dir / f"{accession}_protein.faa"
        target_gff = self.raw_genomes_dir / f"{accession}_genomic.gff"
        target_jsonl = self.raw_metadata_dir / f"{accession}_assembly_report.jsonl"

        # Dọn dẹp tệp cũ
        for target in (target_fna, target_faa, target_gff, target_jsonl):
            if target.exists():
                target.unlink()

        metadata = None
        extracted_paths: Dict[str, str] = {}

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    filename = Path(member).name

                    # 1. Ghép Genomic FASTA (dùng "ab" và thêm b"\n")
                    if filename.endswith(("_genomic.fna", ".fna")):
                        with zf.open(member) as src, open(target_fna, "ab") as dst:
                            dst.write(src.read())
                            dst.write(b"\n")
                        extracted_paths["genomic_fna"] = str(target_fna)

                    # 2. Ghép Protein FASTA
                    elif filename.endswith(("_protein.faa", ".faa")):
                        with zf.open(member) as src, open(target_faa, "ab") as dst:
                            dst.write(src.read())
                            dst.write(b"\n")
                        extracted_paths["protein_faa"] = str(target_faa)

                    # 3. Ghép Annotation GFF
                    elif filename.endswith((".gff", ".gff3")):
                        with zf.open(member) as src, open(target_gff, "ab") as dst:
                            dst.write(src.read())
                            dst.write(b"\n")
                        extracted_paths["genomic_gff"] = str(target_gff)

                    # 4. Parse Metadata JSONL an toàn
                    elif filename == "assembly_data_report.jsonl":
                        raw_bytes = zf.read(member)
                        with open(target_jsonl, "wb") as dst:
                            dst.write(raw_bytes)
                        extracted_paths["assembly_report"] = str(target_jsonl)

                        lines = [line.strip() for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
                        if lines:
                            data = json.loads(lines[0])
                            org = data.get("organism", {})
                            ass = data.get("assemblyInfo", {})
                            metadata = GenomeMetadata(
                                accession=accession,
                                organism_name=org.get("organismName", "Unknown"),
                                tax_id=org.get("taxId", 0),
                                assembly_level=ass.get("assemblyLevel", "Unknown"),
                                download_status="SUCCESS",
                            )

            # Validate không để file rỗng (0 bytes) đi vào pipeline
            for file_key, file_path_str in extracted_paths.items():
                p = Path(file_path_str)
                if not p.exists() or p.stat().st_size == 0:
                    p.unlink(missing_ok=True)
                    return GenomeMetadata(accession=accession, download_status="INVALID_OUTPUT")

            if not metadata:
                metadata = GenomeMetadata(accession=accession, download_status="SUCCESS")

            metadata.extracted_files = extracted_paths
            return metadata

        except (zipfile.BadZipFile, json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Lỗi giải nén {accession}: {e}")
            return GenomeMetadata(accession=accession, download_status="CORRUPTED")

class NCBIBatchPipeline:
    """Orchestrator: Điều phối tải và giải nén đa luồng an toàn."""

    def __init__(self, config_path: str = "config/config.yaml", logger=None) -> None:
        self.config = load_config(config_path)
        self.logger = logger or setup_logger("nifpredict.data.pipeline")

        raw_base = Path(self.config.get("paths", {}).get("raw_dir", "data/raw"))
        self.genomes_dir = raw_base / "genomes"
        self.metadata_dir = raw_base / "metadata"

        self.downloader = NCBIDownloader(config_path=config_path, logger=self.logger)
        self.extractor = NCBIExtractor(
            raw_genomes_dir=self.genomes_dir,
            raw_metadata_dir=self.metadata_dir,
            logger=self.logger,
        )

    def _load_accessions(self, source: Union[str, Path, List[str]]) -> List[str]:
        if isinstance(source, (str, Path)):
            file_path = Path(source)
            if not file_path.exists():
                return []
            with open(file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip() and not line.startswith("#")]
        elif isinstance(source, list):
            return [acc.strip() for acc in source if isinstance(acc, str) and acc.strip()]
        return []

    def _process_single_accession(self, accession: str) -> GenomeMetadata:
        zip_path = self.downloader.download_genome_zip(accession)
        if not zip_path:
            return GenomeMetadata(accession=accession, download_status="NOT_FOUND")
        return self.extractor.extract_package(zip_path, accession)

    def run_batch(
        self,
        source: Union[str, Path, List[str]],
        max_workers: int = 4,
    ) -> List[GenomeMetadata]:
        accessions = self._load_accessions(source)
        if not accessions:
            return []

        results: List[GenomeMetadata] = []
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_acc = {
                    executor.submit(self._process_single_accession, acc): acc
                    for acc in accessions
                }
                for future in as_completed(future_to_acc):
                    acc = future_to_acc[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(GenomeMetadata(accession=acc, download_status="FAILED"))
        finally:
            self.downloader.close()

        return results