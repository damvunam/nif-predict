"""
Module: nifpredict.features.hmm_annotator
Description: Production-Ready HMM Annotation Engine using PyHMMER C-bindings.
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


@dataclass(slots=True)
class DomainHit:
    target_protein: str
    target_len: int
    gene_family: str
    hmm_len: int
    seq_evalue: float
    dom_evalue: float
    raw_score: float
    bias: float
    effective_score: float
    query_cov: float
    target_cov: float
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int


class HMMAnnotator:
    def __init__(
        self,
        config: Optional[Union[AppConfig, Dict[str, Any]]] = None,
        cpus: int = 0,
    ) -> None:
        if isinstance(config, AppConfig):
            app_cfg = config
            align_dict = app_cfg.biological_thresholds.alignment.model_dump()
            self.cpus = cpus or app_cfg.computing.max_threads
        elif isinstance(config, dict):
            align_dict = config
            self.cpus = cpus
        else:
            app_cfg = load_config(auto_create_dirs=False)
            align_dict = app_cfg.biological_thresholds.alignment.model_dump()
            self.cpus = cpus or app_cfg.computing.max_threads

        try:
            self.cfg = HMMConfig(**align_dict)
        except ValidationError as e:
            logger.error(f"Cấu hình HMMConfig không hợp lệ: {e}")
            raise e

        if self.cpus == 0:
            self.cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "0"))

        self.alphabet = Alphabet.amino()

    def validate_fasta(self, faa_path: Path) -> bool:
        if not faa_path.is_file() or faa_path.stat().st_size == 0:
            return False
        try:
            with SequenceFile(faa_path, digital=True, alphabet=self.alphabet) as seq_file:
                first_seq = seq_file.read_block()
                return len(first_seq) > 0
        except Exception as err:
            logger.error(f"File FASTA không hợp lệ ({faa_path.name}): {err}")
            return False

    @staticmethod
    def _calculate_coverage(start: int, end: int, total_length: int) -> float:
        if total_length <= 0:
            return 0.0
        return round((abs(end - start) + 1) / total_length, 4)

    def _resolve_cross_family_overlaps(self, hits: List[DomainHit]) -> List[DomainHit]:
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

    def annotate_faa(self, protein_fasta: Path, hmm_profile: Path) -> List[DomainHit]:
        protein_fasta, hmm_profile = Path(protein_fasta), Path(hmm_profile)
        if not self.validate_fasta(protein_fasta) or not hmm_profile.is_file():
            return []

        start_time = time.perf_counter()
        raw_hits_by_target: Dict[str, List[DomainHit]] = {}
        bit_cutoffs_param = "tc" if self.cfg.use_trusted_cutoffs else None

        try:
            with SequenceFile(protein_fasta, digital=True, alphabet=self.alphabet) as seq_file:
                sequences: DigitalSequenceBlock = seq_file.read_block()

            with HMMFile(hmm_profile) as hmm_file:
                try:
                    top_hits_stream = pyhmmer.hmmer.hmmsearch(
                        queries=hmm_file, sequences=sequences, cpus=self.cpus, bit_cutoffs=bit_cutoffs_param
                    )
                except ValueError:
                    hmm_file.rewind()
                    top_hits_stream = pyhmmer.hmmer.hmmsearch(
                        queries=hmm_file, sequences=sequences, cpus=self.cpus, bit_cutoffs=None
                    )
                    bit_cutoffs_param = None

                for top_hits in top_hits_stream:
                    query_name = top_hits.query.name.decode("utf-8")
                    query_len = top_hits.query.length

                    for hit in top_hits:
                        target_name = hit.name.decode("utf-8")
                        target_len = len(sequences[hit.index])
                        raw_score = hit.score
                        bias = hit.bias
                        eff_score = (raw_score - bias) if self.cfg.use_bias_correction else raw_score

                        if bit_cutoffs_param is None and eff_score < self.cfg.min_bitscore:
                            continue
                        if hit.evalue > self.cfg.max_seq_evalue:
                            continue

                        for domain in hit.domains:
                            if domain.i_evalue > self.cfg.max_dom_evalue:
                                continue

                            q_cov = self._calculate_coverage(domain.alignment.hmm_from, domain.alignment.hmm_to, query_len)
                            t_cov = self._calculate_coverage(domain.alignment.target_from, domain.alignment.target_to, target_len)

                            if q_cov < self.cfg.min_query_coverage or t_cov < self.cfg.min_target_coverage:
                                continue

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
                                env_to=domain.env_to,
                            )
                            raw_hits_by_target.setdefault(target_name, []).append(hit_obj)

            filtered_hits: List[DomainHit] = []
            for target_hits in raw_hits_by_target.values():
                filtered_hits.extend(self._resolve_cross_family_overlaps(target_hits))

            return filtered_hits
        except Exception as e:
            logger.error(f"Lỗi khi thực thi HMM Annotator trên {protein_fasta.name}: {e}", exc_info=True)
            return []
        finally:
            gc.collect()

    def annotate_to_dataframe(self, protein_fasta: Path, hmm_profile: Path) -> pd.DataFrame:
        hits = self.annotate_faa(protein_fasta, hmm_profile)
        if not hits:
            return pd.DataFrame(columns=[field for field in DomainHit.__dataclass_fields__.keys()])
        return pd.DataFrame([asdict(h) for h in hits])