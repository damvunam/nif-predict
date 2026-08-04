"""Production-ready HMM annotation engine using PyHMMER C bindings."""

import gc
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyhmmer
from pydantic import BaseModel, Field, ValidationError, model_validator
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, SequenceFile
from pyhmmer.plan7 import HMMFile

from nifpredict.utils.config import AppConfig, load_config


logger = logging.getLogger("nifpredict.features.hmm_annotator")


class HMMConfig(BaseModel):
    """Validated thresholds used by the HMM annotation pipeline."""

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
        """Map legacy alignment-threshold names to the current schema."""
        if isinstance(data, dict):
            mapped_data = data.copy()
            if "e_value_max" in mapped_data:
                evalue = mapped_data.pop("e_value_max")
                mapped_data.setdefault("max_seq_evalue", evalue)
                mapped_data.setdefault("max_dom_evalue", evalue)
            if "min_bit_score" in mapped_data:
                mapped_data.setdefault(
                    "min_bitscore",
                    mapped_data.pop("min_bit_score"),
                )
            if "min_coverage" in mapped_data:
                coverage = mapped_data.pop("min_coverage")
                mapped_data.setdefault("min_query_coverage", coverage)
            return mapped_data
        return data


@dataclass(slots=True)
class DomainHit:
    """One filtered HMM domain match."""

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
    """Search protein FASTA files against one or more profile HMMs."""

    def __init__(
        self,
        config: Optional[Union[AppConfig, Dict[str, Any]]] = None,
        cpus: int = 0,
    ) -> None:
        if cpus < 0:
            raise ValueError("cpus must be zero or a positive integer")

        if isinstance(config, AppConfig):
            alignment_config = (
                config.biological_thresholds.alignment.model_dump()
            )
            self.cpus = cpus or config.computing.max_threads
        elif isinstance(config, dict):
            alignment_config = config
            self.cpus = cpus
        else:
            app_config = load_config(auto_create_dirs=False)
            alignment_config = (
                app_config.biological_thresholds.alignment.model_dump()
            )
            self.cpus = cpus or app_config.computing.max_threads

        try:
            self.cfg = HMMConfig(**alignment_config)
        except ValidationError:
            logger.exception("Invalid HMMConfig")
            raise

        if self.cpus == 0:
            slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                try:
                    self.cpus = int(slurm_cpus)
                except ValueError as exc:
                    raise ValueError(
                        "SLURM_CPUS_PER_TASK must contain an integer"
                    ) from exc
                if self.cpus < 1:
                    raise ValueError(
                        "SLURM_CPUS_PER_TASK must be a positive integer"
                    )

        self.alphabet = Alphabet.amino()

    def validate_fasta(self, faa_path: Path) -> bool:
        """Return whether a FASTA file contains at least one protein."""
        faa_path = Path(faa_path)
        if not faa_path.is_file() or faa_path.stat().st_size == 0:
            return False

        try:
            with SequenceFile(
                faa_path,
                digital=True,
                alphabet=self.alphabet,
            ) as sequence_file:
                first_sequence = sequence_file.read()
                return first_sequence is not None
        except (OSError, ValueError) as exc:
            logger.error("Invalid FASTA file %s: %s", faa_path, exc)
            return False

    @staticmethod
    def _as_text(value: Any) -> str:
        """Normalize PyHMMER names across pre-0.12 and 0.12+ releases."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _calculate_coverage(start: int, end: int, total_length: int) -> float:
        if total_length <= 0:
            return 0.0
        return round((abs(end - start) + 1) / total_length, 4)

    @staticmethod
    def _query_length(query: Any) -> int:
        """Read the model length from an HMM or profile object."""
        if hasattr(query, "M"):
            return int(query.M)
        if hasattr(query, "length"):
            return int(query.length)
        raise AttributeError("PyHMMER query object has no model length")

    @staticmethod
    def _target_length(hit: Any, sequences: DigitalSequenceBlock) -> int:
        """Read target length with a fallback for older PyHMMER versions."""
        if hasattr(hit, "length"):
            return int(hit.length)
        return len(sequences[hit.index])

    @staticmethod
    def _has_trusted_cutoffs(query: Any) -> bool:
        """Return whether a profile contains model-specific TC thresholds."""
        cutoffs = getattr(query, "cutoffs", None)
        if cutoffs is None:
            return False

        available = getattr(cutoffs, "trusted_available", None)
        if callable(available):
            return bool(available())
        return getattr(cutoffs, "trusted", None) is not None

    def _resolve_cross_family_overlaps(
        self,
        hits: List[DomainHit],
    ) -> List[DomainHit]:
        """Keep the best-scoring domain when domain envelopes overlap."""
        if not hits:
            return []

        sorted_hits = sorted(
            hits,
            key=lambda item: (
                item.effective_score,
                -item.dom_evalue,
                item.query_cov,
            ),
            reverse=True,
        )
        selected_hits: List[DomainHit] = []

        for candidate in sorted_hits:
            candidate_start = min(candidate.env_from, candidate.env_to)
            candidate_end = max(candidate.env_from, candidate.env_to)
            candidate_length = candidate_end - candidate_start + 1
            has_conflicting_overlap = False

            for existing in selected_hits:
                existing_start = min(existing.env_from, existing.env_to)
                existing_end = max(existing.env_from, existing.env_to)
                existing_length = existing_end - existing_start + 1
                overlap_start = max(candidate_start, existing_start)
                overlap_end = min(candidate_end, existing_end)

                if overlap_start <= overlap_end:
                    overlap_length = overlap_end - overlap_start + 1
                    shorter_length = min(candidate_length, existing_length)
                    overlap_fraction = overlap_length / shorter_length
                    if overlap_fraction > self.cfg.max_overlap_fraction:
                        has_conflicting_overlap = True
                        break

            if not has_conflicting_overlap:
                selected_hits.append(candidate)

        return sorted(
            selected_hits,
            key=lambda item: (
                item.target_protein,
                item.env_from,
                item.env_to,
                item.gene_family,
            ),
        )

    def _search(
        self,
        queries: List[Any],
        sequences: DigitalSequenceBlock,
        use_trusted_cutoffs: bool,
    ) -> List[Any]:
        """Run PyHMMER and consume the lazy iterator inside error handling."""
        options: Dict[str, Any] = {}
        if use_trusted_cutoffs:
            options["bit_cutoffs"] = "trusted"
        else:
            options["bit_cutoffs"] = None
            options["E"] = self.cfg.max_seq_evalue
            options["domE"] = self.cfg.max_dom_evalue

        return list(
            pyhmmer.hmmer.hmmsearch(
                queries=queries,
                sequences=sequences,
                cpus=self.cpus,
                **options,
            )
        )

    def annotate_faa(
        self,
        protein_fasta: Path,
        hmm_profile: Path,
    ) -> List[DomainHit]:
        """Annotate one protein FASTA file with the supplied HMM profile(s)."""
        protein_fasta = Path(protein_fasta)
        hmm_profile = Path(hmm_profile)

        if not self.validate_fasta(protein_fasta):
            raise ValueError(
                f"Protein FASTA is missing, empty, or invalid: {protein_fasta}"
            )
        if not hmm_profile.is_file() or hmm_profile.stat().st_size == 0:
            raise FileNotFoundError(
                f"HMM profile is missing or empty: {hmm_profile}"
            )

        start_time = time.perf_counter()
        raw_hits_by_target: Dict[str, List[DomainHit]] = {}

        try:
            with SequenceFile(
                protein_fasta,
                digital=True,
                alphabet=self.alphabet,
            ) as sequence_file:
                sequences: DigitalSequenceBlock = sequence_file.read_block()

            with HMMFile(hmm_profile) as hmm_file:
                queries = list(hmm_file)

            if not queries:
                raise ValueError(f"HMM profile contains no models: {hmm_profile}")

            use_trusted_cutoffs = self.cfg.use_trusted_cutoffs and all(
                self._has_trusted_cutoffs(query) for query in queries
            )
            if self.cfg.use_trusted_cutoffs and not use_trusted_cutoffs:
                logger.warning(
                    "At least one model in %s has no trusted cutoff; "
                    "using configured E-value and bit-score thresholds",
                    hmm_profile,
                )

            top_hits_collection = self._search(
                queries,
                sequences,
                use_trusted_cutoffs,
            )

            for top_hits in top_hits_collection:
                query = top_hits.query
                query_name = self._as_text(query.name)
                query_length = self._query_length(query)

                for hit in top_hits:
                    target_name = self._as_text(hit.name)
                    target_length = self._target_length(hit, sequences)
                    raw_score = float(
                        getattr(hit, "pre_score", hit.score + hit.bias)
                    )
                    bias = float(hit.bias)
                    effective_score = (
                        float(hit.score)
                        if self.cfg.use_bias_correction
                        else raw_score
                    )

                    if (
                        not use_trusted_cutoffs
                        and effective_score < self.cfg.min_bitscore
                    ):
                        continue
                    if hit.evalue > self.cfg.max_seq_evalue:
                        continue

                    for domain in hit.domains:
                        if domain.i_evalue > self.cfg.max_dom_evalue:
                            continue

                        alignment = domain.alignment
                        query_coverage = self._calculate_coverage(
                            alignment.hmm_from,
                            alignment.hmm_to,
                            query_length,
                        )
                        target_coverage = self._calculate_coverage(
                            alignment.target_from,
                            alignment.target_to,
                            target_length,
                        )

                        if (
                            query_coverage < self.cfg.min_query_coverage
                            or target_coverage < self.cfg.min_target_coverage
                        ):
                            continue

                        raw_hits_by_target.setdefault(target_name, []).append(
                            DomainHit(
                                target_protein=target_name,
                                target_len=target_length,
                                gene_family=query_name,
                                hmm_len=query_length,
                                seq_evalue=float(hit.evalue),
                                dom_evalue=float(domain.i_evalue),
                                raw_score=raw_score,
                                bias=bias,
                                effective_score=effective_score,
                                query_cov=query_coverage,
                                target_cov=target_coverage,
                                ali_from=int(alignment.target_from),
                                ali_to=int(alignment.target_to),
                                env_from=int(domain.env_from),
                                env_to=int(domain.env_to),
                            )
                        )

            filtered_hits: List[DomainHit] = []
            for target_hits in raw_hits_by_target.values():
                filtered_hits.extend(
                    self._resolve_cross_family_overlaps(target_hits)
                )

            logger.info(
                "Annotated %s against %s: %d hits in %.2f seconds",
                protein_fasta.name,
                hmm_profile.name,
                len(filtered_hits),
                time.perf_counter() - start_time,
            )
            return filtered_hits
        except Exception:
            logger.exception(
                "HMM annotation failed for %s against %s",
                protein_fasta,
                hmm_profile,
            )
            raise
        finally:
            gc.collect()

    def annotate_to_dataframe(
        self,
        protein_fasta: Path,
        hmm_profile: Path,
    ) -> pd.DataFrame:
        """Return filtered domain hits as a stable-schema DataFrame."""
        hits = self.annotate_faa(protein_fasta, hmm_profile)
        columns = list(DomainHit.__dataclass_fields__)
        if not hits:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([asdict(hit) for hit in hits], columns=columns)