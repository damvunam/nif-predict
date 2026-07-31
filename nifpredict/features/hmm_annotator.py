"""
Module: nifpredict.features.hmm_annotator
Description: Production-Ready HMM Annotation Engine for NifPredict optimized for HPC/Slurm.
             Uses PyHMMER C-bindings, zero-copy streaming memory execution, Pydantic configuration,
             and biology-aware cross-profile overlap resolution.
"""

import gc
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator

import pyhmmer
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, SequenceFile
from pyhmmer.plan7 import HMMFile

from nifpredict.utils.config import AppConfig, load_config

logger = logging.getLogger("nifpredict.features.hmm_annotator")


# ----------------------------------------------------------------------
# 1. Configuration Validation Model (Pydantic)
# ----------------------------------------------------------------------
class HMMConfig(BaseModel):
    max_seq_evalue: float = Field(default=1e-5, ge=0.0)
    max_dom_evalue: float = Field(default=1e-5, ge=0.0)
    min_bitscore: float = Field(default=80.0, ge=0.0)
    min_query_coverage: float = Field(default=0.7, ge=0.0, le=1.0)
    min_target_coverage: float = Field(default=0.3, ge=0.0, le=1.0)
    max_overlap_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    use_bias_correction: bool = Field(default=True)
    use_trusted_cutoffs: bool = Field(default=True)

    @model_validator(mode="before")
    @classmethod
    def map_alignment_thresholds_schema(cls, data: Any) -> Any:
        """Mapping tự động tên trường từ AlignmentThresholds (config.yaml) sang HMMConfig."""
        if isinstance(data, dict):
            mapped_data = data.copy()

            if "e_value_max" in mapped_data:
                e_val = mapped_data.pop("e_value_max")
                mapped_data.setdefault("max_seq_evalue", e_val)
                mapped_data.setdefault("max_dom_evalue", e_val)

            if "min_bit_score" in mapped_data:
                mapped_data.setdefault("min_bitscore", mapped_data.pop("min_bit_score"))

            if "min_coverage" in mapped_data:
                cov = mapped_data.pop("min_coverage")
                mapped_data.setdefault("min_query_coverage", cov)

            return mapped_data
        return data


# ----------------------------------------------------------------------
# 2. Output Data Structure
# ----------------------------------------------------------------------
@dataclass(slots=True)
class DomainHit:
    """Cấu trúc dữ liệu lưu trữ 1 Domain Hit hợp lệ đã trải qua bộ lọc."""
    target_protein: str
    target_len: int
    gene_family: str
    hmm_len: int
    seq_evalue: float
    dom_evalue: float
    raw_score: float
    bias: float
    effective_score: float  # raw_score - bias
    query_cov: float
    target_cov: float
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int


# ----------------------------------------------------------------------
# 3. Core Engine
# ----------------------------------------------------------------------
class HMMAnnotator:
    """
    Động cơ chú giải HMM cấp Production dành cho hạ tầng HPC/Slurm Cluster.
    Thực thi HMM search trực tiếp trên RAM bằng pyhmmer, tự động quản lý tài nguyên CPU
    và phân giải cạnh tranh chéo giữa các họ gen nif/vnf/anf/fix.
    """

    def __init__(
    self,
    config: Optional[Union[AppConfig, Dict[str, Any]]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    cpus: int = 0,
    ) -> None:
        if isinstance(config, AppConfig):
            app_cfg = config
            align_dict = app_cfg.biological_thresholds.alignment.model_dump()
            self.cpus = cpus or app_cfg.computing.max_threads
        elif isinstance(config, dict):
            align_dict = config
            self.cpus = cpus
        elif config_dict is not None:
            align_dict = config_dict
            self.cpus = cpus
        else:
            app_cfg = load_config()
            align_dict = app_cfg.biological_thresholds.alignment.model_dump()
            self.cpus = cpus or app_cfg.computing.max_threads

        try:
            self.cfg = HMMConfig(**align_dict)
        except ValidationError as e:
            logger.error(f"Cấu hình HMMConfig không hợp lệ: {e}")
            raise e

        if self.cpus == 0:
            slurm_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "0"))
            self.cpus = slurm_cpus

        self.alphabet = Alphabet.amino()

    def validate_fasta(self, faa_path: Path) -> bool:
        """Kiểm tra tính hợp lệ của tệp FASTA trước khi đẩy vào C engine."""
        if not faa_path.is_file():
            logger.error(f"File FASTA không tồn tại: {faa_path}")
            return False
        
        if faa_path.stat().st_size == 0:
            logger.warning(f"File FASTA rỗng (0 bytes): {faa_path.name}")
            return False

        try:
            with SequenceFile(faa_path, digital=True, alphabet=self.alphabet) as seq_file:
                first_seq = seq_file.read_block()
                if len(first_seq) == 0:
                    logger.warning(f"File FASTA không chứa chuỗi protein hợp lệ: {faa_path.name}")
                    return False
            return True
        except Exception as err:
            logger.error(f"File FASTA bị lỗi định dạng hoặc corrupted ({faa_path.name}): {err}")
            return False

    @staticmethod
    def _calculate_coverage(start: int, end: int, total_length: int) -> float:
        """Tính tỷ lệ bao phủ (Coverage)."""
        if total_length <= 0:
            return 0.0
        return round((abs(end - start) + 1) / total_length, 4)

    def _resolve_cross_family_overlaps(
        self, hits: List[DomainHit]
    ) -> List[DomainHit]:
        """
        Giải quyết cạnh tranh Cross-Profile (VD: nifD vs vnfD vs anfD) và Overlapping Domains
        trên cùng một chuỗi Target Protein.
        
        Chiến lược: Sắp xếp theo Effective Bit-Score (Score - Bias) giảm dần.
        Hit nào có Effective Score cao nhất trên cùng vùng tọa độ Envelope sẽ được chọn.
        """
        if not hits:
            return []

        sorted_hits = sorted(hits, key=lambda x: x.effective_score, reverse=True)
        selected_hits: List[DomainHit] = []

        for candidate in sorted_hits:
            has_overlap = False
            cand_len = candidate.env_to - candidate.env_from + 1

            for existing in selected_hits:
                overlap_start = max(candidate.env_from, existing.env_from)
                overlap_end = min(candidate.env_to, existing.env_to)

                if overlap_start <= overlap_end:
                    overlap_len = overlap_end - overlap_start + 1
                    if (overlap_len / cand_len) > self.cfg.max_overlap_fraction:
                        has_overlap = True
                        break

            if not has_overlap:
                selected_hits.append(candidate)

        return selected_hits

    def annotate_faa(
        self, protein_fasta: Path, hmm_profile: Path
    ) -> List[DomainHit]:
        """
        Quét tệp protein FASTA với tập hợp HMM profile bằng pyhmmer (Streaming Mode).
        
        :param protein_fasta: Đường dẫn tệp .faa
        :param hmm_profile: Đường dẫn tệp HMM Profile DB
        :return: Danh sách DomainHit sạch đã qua bộ lọc và khử trùng lấp.
        """
        protein_fasta, hmm_profile = Path(protein_fasta), Path(hmm_profile)

        # 1. FASTA & HMM Validation
        if not self.validate_fasta(protein_fasta):
            return []
        if not hmm_profile.is_file():
            logger.error(f"File HMM profile không tồn tại: {hmm_profile}")
            return []

        start_time = time.perf_counter()
        raw_hits_by_target: Dict[str, List[DomainHit]] = {}
        total_raw_domains = 0

        # Xác định tham số bit_cutoffs ban đầu
        bit_cutoffs_param = "tc" if self.cfg.use_trusted_cutoffs else None

        logger.info(
            f"Bắt đầu HMM Search | Sequence: {protein_fasta.name} | "
            f"Profiles: {hmm_profile.name} | Mode TC: {self.cfg.use_trusted_cutoffs} | "
            f"Threads: {self.cpus or 'Auto'}"
        )

        try:
            # 2. Đọc Sequence thành C Digital Block (Zero-copy RAM)
            with SequenceFile(protein_fasta, digital=True, alphabet=self.alphabet) as seq_file:
                sequences: DigitalSequenceBlock = seq_file.read_block()

            # 3. Mở HMM File và khởi tạo Streaming Generator
            with HMMFile(hmm_profile) as hmm_file:
                try:
                    top_hits_stream = pyhmmer.hmmer.hmmsearch(
                        queries=hmm_file,
                        sequences=sequences,
                        cpus=self.cpus,
                        bit_cutoffs=bit_cutoffs_param
                    )
                except ValueError:
                    # Fallback an toàn nếu HMM profile tùy chỉnh thiếu header Trusted Cutoff (TC)
                    logger.warning(
                        f"HMM Profile {hmm_profile.name} không hỗ trợ Trusted Cutoffs (TC). "
                        f"Tự động chuyển sang lọc Bit-Score thủ công (min_bitscore={self.cfg.min_bitscore})."
                    )
                    # Tua lại vị trí đầu file HMM để đọc lại generator
                    hmm_file.rewind()
                    top_hits_stream = pyhmmer.hmmer.hmmsearch(
                        queries=hmm_file,
                        sequences=sequences,
                        cpus=self.cpus,
                        bit_cutoffs=None
                    )
                    bit_cutoffs_param = None

                # 4. Duyệt trực tiếp Streaming Generator (Tối ưu Memory Footprint)
                for top_hits in top_hits_stream:
                    query_name = top_hits.query.name.decode("utf-8")
                    query_len = top_hits.query.length

                    for hit in top_hits:
                        target_name = hit.name.decode("utf-8")
                        target_len = len(sequences[hit.index])

                        # Lấy Raw Score và tính toán Effective Score (Bias Correction)
                        raw_score = hit.score
                        bias = hit.bias
                        eff_score = (raw_score - bias) if self.cfg.use_bias_correction else raw_score

                        # Nếu KHÔNG sử dụng TC (hoặc rơi vào fallback), kiểm tra ngưỡng min_bitscore
                        if bit_cutoffs_param is None and eff_score < self.cfg.min_bitscore:
                            continue
                        
                        # Lọc Sequence E-value
                        if hit.evalue > self.cfg.max_seq_evalue:
                            continue

                        for domain in hit.domains:
                            # Lọc Domain E-value (i-evalue)
                            if domain.i_evalue > self.cfg.max_dom_evalue:
                                continue

                            # Tính toán Coverage
                            q_cov = self._calculate_coverage(
                                domain.alignment.hmm_from, domain.alignment.hmm_to, query_len
                            )
                            t_cov = self._calculate_coverage(
                                domain.alignment.target_from, domain.alignment.target_to, target_len
                            )

                            if q_cov < self.cfg.min_query_coverage or t_cov < self.cfg.min_target_coverage:
                                continue

                            # Tạo DomainHit sử dụng thuộc tính C-struct chuẩn của PyHMMER
                            hit_obj = DomainHit(
                                target_protein=target_name,
                                target_len=target_len,
                                gene_family=query_name,
                                hmm_len=query_len,
                                seq_evalue=hit.evalue,
                                dom_evalue=domain.i_evalue,
                                raw_score=raw_score,
                                bias=bias,
                                effective_score=eff_score,
                                query_cov=q_cov,
                                target_cov=t_cov,
                                ali_from=domain.alignment.target_from,
                                ali_to=domain.alignment.target_to,
                                env_from=domain.env_from,
                                env_to=domain.env_to
                            )

                            raw_hits_by_target.setdefault(target_name, []).append(hit_obj)
                            total_raw_domains += 1

            # 5. Khử trùng lấp Cross-Profile & Chọn Best Hit per Domain Region
            filtered_hits: List[DomainHit] = []
            for target_name, target_hits in raw_hits_by_target.items():
                resolved = self._resolve_cross_family_overlaps(target_hits)
                filtered_hits.extend(resolved)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Hoàn thành {protein_fasta.name} trong {elapsed:.2f}s | "
                f"Raw Hits: {total_raw_domains} -> Valid Unique Hits: {len(filtered_hits)}"
            )

            return filtered_hits

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi thực thi HMM Annotator trên {protein_fasta.name}: {str(e)}", exc_info=True)
            return []
        finally:
            gc.collect()

    def annotate_to_dataframe(
        self, protein_fasta: Path, hmm_profile: Path
    ) -> pd.DataFrame:
        """
        Thực hiện chú giải và trả về kết quả dưới dạng Pandas DataFrame hoàn thiện.
        """
        hits = self.annotate_faa(protein_fasta, hmm_profile)
        if not hits:
            return pd.DataFrame(columns=[field for field in DomainHit.__dataclass_fields__.keys()])
        
        return pd.DataFrame([asdict(h) for h in hits])