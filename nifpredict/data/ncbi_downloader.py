import json
import logging
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import requests
from requests.adapters import HTTPAdapter

from nifpredict.utils.config import AppConfig, load_config
from nifpredict.utils.logger import get_logger


def _get_cfg_val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper truy xuất dữ liệu cấu hình linh hoạt cho cả Pydantic Object lẫn Python Dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class GenomeMetadata:
    """Structure lưu trữ thông tin metadata cốt lõi của một Assembly."""

    accession: str
    organism_name: str = "Unknown"
    tax_id: int = 0
    assembly_level: str = "Unknown"
    download_status: str = "PENDING"  # PENDING, SUCCESS, FAILED, NOT_FOUND, CORRUPTED, INVALID_OUTPUT
    extracted_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RateLimiter:
    """Thread-safe Rate Limiter đảm bảo tuân thủ giới hạn Request/giây của NCBI Datasets API."""

    def __init__(self, requests_per_second: float = 3.0) -> None:
        self.delay = 1.0 / requests_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_call = time.time()


class NCBIResolver:
    """Layer 1: Resolver - Chuyển đổi TaxID / Organism / Accession thành Accession chuẩn."""

    def __init__(
        self,
        session: requests.Session,
        base_url: str,
        rate_limiter: RateLimiter,
        logger=None,
    ) -> None:
        self.session = session
        self.base_url = str(base_url).rstrip("/")  # Khắc phục lỗi double slash
        self.rate_limiter = rate_limiter
        self.logger = logger or get_logger("nifpredict.data.resolver")

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
            accessions = [r["accession"] for r in reports if "accession" in r]
            self.logger.info(f"Tìm thấy {len(accessions)} accession cho taxon '{taxon}'")
            return accessions
        except requests.RequestException as e:
            self.logger.error(f"Lỗi khi tra cứu taxon '{taxon}': {e}")
            return []


class NCBIDownloader:
    """Layer 2: Downloader - Tải Zip Package từ NCBI với Retry, Rate Limiting và Atomic Writes."""

    NON_RETRYABLE_STATUS_CODES: Set[int] = {400, 401, 403, 404, 410}

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or load_config()
        self.logger = logger or get_logger("nifpredict.data.downloader")

        ncbi_cfg = self.config.ncbi
        self.base_url = ncbi_cfg.api_base_url.rstrip("/")
        self.timeout = ncbi_cfg.timeout_seconds
        self.max_retries = ncbi_cfg.max_retries
        self.api_key = ncbi_cfg.api_key

        requests_per_sec = float(ncbi_cfg.rate_limit_per_sec if self.api_key else 3)
        self.rate_limiter = RateLimiter(requests_per_second=requests_per_sec)

        self.session = requests.Session()
        headers = {"User-Agent": "NifPredict-Bioinformatics-Platform/0.1.0"}
        if self.api_key:
            headers["api-key"] = self.api_key
        self.session.headers.update(headers)

        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Trỏ đúng đường dẫn zip_dir từ config.yaml
        self.raw_zip_dir = self.config.paths.raw["zip_dir"]
        self.raw_zip_dir.mkdir(parents=True, exist_ok=True)

    def _verify_zip_integrity(self, zip_path: Path) -> bool:
        """Xác minh tính toàn vẹn của tệp ZIP."""
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            return False
        if not zipfile.is_zipfile(zip_path):
            return False
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                return zf.testzip() is None
        except Exception as e:
            self.logger.debug(f"Xác minh tệp ZIP thất bại cho {zip_path.name}: {e}")
            return False

    def download_genome_zip(self, accession: str) -> Optional[Path]:
        """Tải Zip Package cho Accession với cơ chế Atomic Write, Rate Limiting và Selective Retry."""
        output_zip_path = self.raw_zip_dir / f"{accession}.zip"

        if output_zip_path.exists():
            if self._verify_zip_integrity(output_zip_path):
                self.logger.info(f"Sử dụng bản Zip hợp lệ từ cache: {output_zip_path.name}")
                return output_zip_path
            else:
                self.logger.warning(
                    f"Tệp Zip cache bị hỏng/không hợp lệ: {output_zip_path.name}. Tiến hành tải lại..."
                )
                output_zip_path.unlink(missing_ok=True)

        url = f"{self.base_url}/genome/accession/{accession}/download"
        params = {
            "include_annotation_type": ["GENOME_FASTA", "PROT_FASTA", "GENOME_GFF"],
            "filename": f"{accession}.zip",
        }

        temp_zip_path = output_zip_path.with_suffix(".zip.tmp")

        self.logger.info(f"Đang tải Zip Package cho {accession}...")
        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.get(
                    url, params=params, timeout=self.timeout, stream=True
                )

                if response.status_code in self.NON_RETRYABLE_STATUS_CODES:
                    self.logger.warning(
                        f"Accession {accession} trả về HTTP {response.status_code} (Không khả thi để retry)."
                    )
                    return None

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    self.logger.warning(
                        f"Gặp NCBI Rate Limit (429). Tạm dừng {retry_after}s trước khi thử lại..."
                    )
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()

                # Atomic Write
                with open(temp_zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)

                if self._verify_zip_integrity(temp_zip_path):
                    temp_zip_path.replace(output_zip_path)
                    self.logger.info(f"Tải và xác thực thành công: {output_zip_path.name}")
                    return output_zip_path
                else:
                    self.logger.warning(
                        f"[Thử {attempt}/{self.max_retries}] Tệp ZIP tải về cho {accession} bị lỗi cấu trúc."
                    )
                    temp_zip_path.unlink(missing_ok=True)

            except requests.RequestException as e:
                self.logger.warning(
                    f"[Thử {attempt}/{self.max_retries}] Lỗi kết nối mạng với {accession}: {e}"
                )
                if temp_zip_path.exists():
                    temp_zip_path.unlink(missing_ok=True)

            if attempt < self.max_retries:
                time.sleep(2**attempt)

        self.logger.error(f"Thất bại hoàn toàn khi tải Archive ZIP cho {accession}")
        return None

    def close(self) -> None:
        """Đóng session mạng an toàn."""
        if hasattr(self, "session") and self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class NCBIExtractor:
    """Layer 3: Extractor - Giải nén, ghép file nối tiếp (concatenation) và validate đầu ra."""

    def __init__(
        self,
        raw_genomes_dir: Optional[Union[Path, str]] = None,
        raw_metadata_dir: Optional[Union[Path, str]] = None,
        config: Optional[AppConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or load_config()
        self.logger = logger or get_logger("nifpredict.data.extractor")

        self.raw_genomes_dir = Path(
            raw_genomes_dir or self.config.paths.raw["genomes_dir"]
        )
        self.raw_metadata_dir = Path(
            raw_metadata_dir or self.config.paths.raw["metadata_dir"]
        )

        self.raw_genomes_dir.mkdir(parents=True, exist_ok=True)
        self.raw_metadata_dir.mkdir(parents=True, exist_ok=True)

    def extract_package(
        self, zip_path: Path, accession: str
    ) -> Optional[GenomeMetadata]:
        """Duyệt archive ZIP, ghép các contigs/plasmids tách rời và validate kích thước file xuất."""
        if not zip_path or not zip_path.exists():
            return GenomeMetadata(accession=accession, download_status="FAILED")

        target_fna = self.raw_genomes_dir / f"{accession}_genomic.fna"
        target_faa = self.raw_genomes_dir / f"{accession}_protein.faa"
        target_gff = self.raw_genomes_dir / f"{accession}_genomic.gff"
        target_jsonl = self.raw_metadata_dir / f"{accession}_assembly_report.jsonl"

        for target in (target_fna, target_faa, target_gff, target_jsonl):
            if target.exists():
                target.unlink()

        metadata = None
        extracted_paths: Dict[str, str] = {}

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    filename = Path(member).name

                    if filename.endswith(("_genomic.fna", ".fna")):
                        with zf.open(member) as src, open(target_fna, "ab") as dst:
                            dst.write(src.read())
                            dst.write(b"\n")
                        extracted_paths["genomic_fna"] = str(target_fna)

                    elif filename.endswith(("_protein.faa", ".faa")):
                        with zf.open(member) as src, open(target_faa, "ab") as dst:
                            dst.write(src.read())
                            dst.write(b"\n")
                        extracted_paths["protein_faa"] = str(target_faa)

                    elif filename.endswith((".gff", ".gff3")):
                        with zf.open(member) as src, open(target_gff, "ab") as dst:
                            dst.write(src.read())
                            dst.write(b"\n")
                        extracted_paths["genomic_gff"] = str(target_gff)

                    elif filename == "assembly_data_report.jsonl":
                        raw_bytes = zf.read(member)
                        with open(target_jsonl, "wb") as dst:
                            dst.write(raw_bytes)
                        extracted_paths["assembly_report"] = str(target_jsonl)

                        lines = [
                            line.strip()
                            for line in raw_bytes.decode("utf-8").splitlines()
                            if line.strip()
                        ]
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

            for file_key, file_path_str in extracted_paths.items():
                p = Path(file_path_str)
                if not p.exists() or p.stat().st_size == 0:
                    self.logger.error(
                        f"File giải nén '{file_key}' cho {accession} bị rỗng (0 bytes): {p}"
                    )
                    p.unlink(missing_ok=True)
                    return GenomeMetadata(
                        accession=accession, download_status="INVALID_OUTPUT"
                    )

            if not metadata:
                metadata = GenomeMetadata(
                    accession=accession, download_status="SUCCESS"
                )

            metadata.extracted_files = extracted_paths
            self.logger.info(f"Đã giải nén và validate thành công cho: {accession}")
            return metadata

        except (zipfile.BadZipFile, json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Lỗi giải nén hoặc parse metadata cho {accession}: {e}")
            for target in (target_fna, target_faa, target_gff, target_jsonl):
                if target.exists():
                    target.unlink(missing_ok=True)
            return GenomeMetadata(accession=accession, download_status="CORRUPTED")


class NCBIBatchPipeline:
    """Orchestrator: Điều phối tải và giải nén đa luồng an toàn."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or load_config()
        self.logger = logger or get_logger("nifpredict.data.pipeline")

        self.downloader = NCBIDownloader(config=self.config, logger=self.logger)
        self.extractor = NCBIExtractor(config=self.config, logger=self.logger)

    def _load_accessions(self, source: Union[str, Path, List[str]]) -> List[str]:
        """Đọc danh sách accession ID từ tệp text hoặc Python List."""
        if isinstance(source, (str, Path)):
            file_path = Path(source)
            if not file_path.exists():
                self.logger.error(f"Tệp danh sách accession không tồn tại: {file_path}")
                return []
            with open(file_path, "r", encoding="utf-8") as f:
                return [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        elif isinstance(source, list):
            return [acc.strip() for acc in source if isinstance(acc, str) and acc.strip()]
        return []

    def _process_single_accession(self, accession: str) -> GenomeMetadata:
        """Quy trình đơn lẻ cho một Accession."""
        zip_path = self.downloader.download_genome_zip(accession)
        if not zip_path:
            return GenomeMetadata(accession=accession, download_status="NOT_FOUND")

        return self.extractor.extract_package(zip_path, accession)

    def run_batch(
        self,
        source: Union[str, Path, List[str]],
        max_workers: int = 4,
    ) -> List[GenomeMetadata]:
        """Khởi chạy pipeline tải hàng loạt với ThreadPoolExecutor."""
        accessions = self._load_accessions(source)
        if not accessions:
            self.logger.warning("Danh sách Accession trống.")
            return []

        self.logger.info(
            f"Bắt đầu pipeline cho {len(accessions)} accessions ({max_workers} threads)..."
        )
        results: List[GenomeMetadata] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_acc = {
                executor.submit(self._process_single_accession, acc): acc
                for acc in accessions
            }

            for future in as_completed(future_to_acc):
                acc = future_to_acc[future]
                try:
                    meta = future.result()
                    results.append(meta)
                except Exception as exc:
                    self.logger.error(
                        f"Ngoại lệ chưa được xử lý khi làm việc với {acc}: {exc}"
                    )
                    results.append(
                        GenomeMetadata(accession=acc, download_status="FAILED")
                    )

        success_count = sum(1 for r in results if r.download_status == "SUCCESS")
        self.logger.info(
            f"Hoàn thành Batch Pipeline! Thành công: {success_count}/{len(accessions)}"
        )
        return results

    def close(self) -> None:
        """Giải phóng tài nguyên mạng của downloader."""
        if hasattr(self, "downloader") and self.downloader:
            self.downloader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()