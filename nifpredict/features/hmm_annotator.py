import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from nifpredict.utils.config import load_config
from nifpredict.utils.logger import setup_logger


class HMMAnnotator:
    """Engine chạy HMM search (hmmsearch từ gói hmmer) để phát hiện các gen cố định đạm (nif genes)."""

    def __init__(self, config_path: str = "config/config.yaml", logger=None) -> None:
        self.config = load_config(config_path)
        self.logger = logger or setup_logger("nifpredict.features.hmm_annotator")
        
        # Thư mục lưu kết quả annotation
        self.annotation_dir = Path(self.config["paths"]["annotation_dir"])
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

    def run_hmmsearch(self, protein_fasta: Path, hmm_profile: Path, output_tbl: Path, evalue_threshold: float = 1e-5) -> Optional[Path]:
        """Thực thi lệnh hmmsearch từ HMMER suite."""
        if not protein_fasta.exists():
            self.logger.error(f"Không tìm thấy file protein FASTA: {protein_fasta}")
            return None
        if not hmm_profile.exists():
            self.logger.error(f"Không tìm thấy file HMM profile: {hmm_profile}")
            return None

        cmd = [
            "hmmsearch",
            "--tblout", str(output_tbl),
            "-E", str(evalue_threshold),
            str(hmm_profile),
            str(protein_fasta)
        ]

        self.logger.info(f"Đang chạy HMMsearch với profile {hmm_profile.name} trên {protein_fasta.name}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"HMMsearch hoàn tất thành công. Kết quả lưu tại: {output_tbl.name}")
            return output_tbl
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi chạy hmmsearch: {e.stderr}")
            return None
        except FileNotFoundError:
            self.logger.error("Không tìm thấy chương trình 'hmmsearch'. Hãy đảm bảo gói 'hmmer' đã được cài đặt trên hệ thống!")
            return None

    def parse_tblout(self, tbl_path: Path) -> pd.DataFrame:
        """Parse định dạng bảng đầu ra (.tblout) của hmmsearch thành DataFrame chuẩn."""
        records = []
        if not tbl_path.exists():
            return pd.DataFrame()

        with open(tbl_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 19:
                    target_name = parts[0]
                    query_name = parts[3]
                    evalue = float(parts[4])
                    score = float(parts[5])
                    records.append({
                        "target_protein": target_name,
                        "gene_family": query_name,
                        "evalue": evalue,
                        "score": score
                    })

        df = pd.DataFrame(records)
        self.logger.info(f"Đã trích xuất {len(df)} hits từ file bảng kết quả.")
        return df
