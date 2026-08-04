"""
Module: nifpredict.features.cluster_filter
Description: Synteny analysis and nitrogenase gene cluster filtering for NifPredict.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import BaseModel

from nifpredict.features.gff_mapping import merge_hits_with_gff, parse_gff3
from nifpredict.utils.config import AppConfig, PFAM_TO_GENE_MAP, load_config
from nifpredict.utils.logger import get_logger

logger = get_logger("nifpredict.features.cluster_filter")


class GeneClusterResult(BaseModel):
    cluster_id: str
    seqid: str
    start: int
    end: int
    span_bp: int
    gene_count: int
    gene_families: List[str]
    protein_ids: List[str]
    strands: List[str]
    strand_switches: int
    cluster_type: str
    has_catalytic_core: bool
    is_truncated: bool
    truncated_side: Optional[str] = None
    quality_status: str


class ClusterFilter:
    CORE_H_FAMILIES: Set[str] = {"nifH", "vnfH", "anfH", "PF00142"}
    CORE_D_FAMILIES: Set[str] = {"nifD", "vnfD", "anfD", "PF00148"}
    CORE_K_FAMILIES: Set[str] = {"nifK", "vnfK", "anfK", "PF02826"}

    VNF_SPECIFIC: Set[str] = {"vnfG", "vnfEN", "vnfH", "vnfD", "vnfK", "PF05911"}
    ANF_SPECIFIC: Set[str] = {"anfG", "anfH", "anfD", "anfK", "PF05910"}

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config(auto_create_dirs=False)
        synteny_cfg = self.config.biological_thresholds.synteny

        self.max_gap_bp: int = synteny_cfg.max_intergenic_distance_bp
        self.max_divergent_gap_bp: int = synteny_cfg.max_divergent_gap_bp
        self.min_core_genes: int = synteny_cfg.min_core_genes_required
        self.edge_margin_bp: int = synteny_cfg.contig_edge_margin_bp

    @staticmethod
    def parse_gff3(gff_path: Path) -> pd.DataFrame:
        """Compatibility entry point for the dedicated GFF mapping module."""
        return parse_gff3(gff_path)

    @staticmethod
    def _merge_hits_with_gff(
        df_hits: pd.DataFrame, df_gff: pd.DataFrame
    ) -> pd.DataFrame:
        return merge_hits_with_gff(df_hits, df_gff)

    def group_into_clusters(
        self,
        df_hits: pd.DataFrame,
        df_gff: pd.DataFrame,
        contig_lengths: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        if df_hits.empty:
            return []

        source_path = df_gff.attrs.get("source_path", "GFF3")
        merged = self._merge_hits_with_gff(df_hits, df_gff)
        match_stats = merged.attrs.get("match_stats", {})
        if merged.empty:
            examples = match_stats.get("unmatched_protein_ids", [])[:5]
            raise ValueError(
                f"Không thể khớp HMM hits với {source_path}. "
                f"Ví dụ protein ID không khớp: {examples}"
            )

        unmatched_ids = match_stats.get("unmatched_protein_ids", [])
        if unmatched_ids:
            logger.warning(
                "Có %d protein ID từ HMM không khớp %s; vẫn xử lý các hit đã khớp. "
                "Ví dụ: %s",
                len(unmatched_ids),
                source_path,
                unmatched_ids[:5],
            )

        merged = merged.sort_values(by=["seqid", "start", "end"]).reset_index(drop=True)

        raw_clusters: List[List[Dict[str, Any]]] = []
        current_cluster: List[Dict[str, Any]] = []
        current_max_end = 0

        for row in merged.itertuples(index=False):
            row_dict = row._asdict()
            if not current_cluster:
                current_cluster.append(row_dict)
                current_max_end = row_dict["end"]
                continue

            last_gene = current_cluster[-1]
            same_seq = row_dict["seqid"] == last_gene["seqid"]
            gap = max(0, row_dict["start"] - current_max_end - 1)

            # In genomic order, - followed by + is divergent (<- ->).
            is_divergent = last_gene["strand"] == "-" and row_dict["strand"] == "+"
            effective_max_gap = (
                self.max_divergent_gap_bp if is_divergent else self.max_gap_bp
            )

            if same_seq and gap <= effective_max_gap:
                current_cluster.append(row_dict)
                current_max_end = max(current_max_end, row_dict["end"])
            else:
                raw_clusters.append(current_cluster)
                current_cluster = [row_dict]
                current_max_end = row_dict["end"]

        if current_cluster:
            raw_clusters.append(current_cluster)

        valid_clusters: List[Dict[str, Any]] = []
        for idx, cluster_genes in enumerate(raw_clusters, 1):
            cluster_info = self._analyze_cluster(
                cluster_id=f"cluster_{idx:04d}",
                cluster_genes=cluster_genes,
                contig_lengths=contig_lengths,
            )

            is_valid_complete = (
                len(cluster_info["gene_families"]) >= self.min_core_genes
                and cluster_info["has_catalytic_core"]
            )
            is_valid_truncated = (
                cluster_info["is_truncated"]
                and len(cluster_info["gene_families"]) >= 1
                and cluster_info["has_any_core_subunit"]
            )

            if is_valid_complete or is_valid_truncated:
                valid_clusters.append(cluster_info)

        return valid_clusters

    def _resolve_cluster_type(self, raw_families: Set[str]) -> Tuple[str, bool, bool]:
        mapped_families = set(raw_families)
        for family in raw_families:
            if family in PFAM_TO_GENE_MAP:
                mapped_families.add(PFAM_TO_GENE_MAP[family])

        has_h = bool(mapped_families.intersection(self.CORE_H_FAMILIES))
        has_d = bool(mapped_families.intersection(self.CORE_D_FAMILIES))
        has_k = bool(mapped_families.intersection(self.CORE_K_FAMILIES))

        has_catalytic_core = sum((has_h, has_d, has_k)) >= 2
        has_any_core_subunit = has_h or has_d or has_k

        vnf_score = len(mapped_families.intersection(self.VNF_SPECIFIC))
        anf_score = len(mapped_families.intersection(self.ANF_SPECIFIC))
        nif_score = sum(1 for family in mapped_families if family.startswith("nif"))

        if (
            vnf_score > 0
            and vnf_score >= anf_score
            and ("vnfG" in mapped_families or "PF05911" in mapped_families)
        ):
            cluster_type = "vnf"
        elif (
            anf_score > 0
            and anf_score > vnf_score
            and ("anfG" in mapped_families or "PF05910" in mapped_families)
        ):
            cluster_type = "anf"
        elif nif_score > 0 or has_catalytic_core:
            cluster_type = "nif"
        else:
            cluster_type = "accessory"

        return cluster_type, has_catalytic_core, has_any_core_subunit

    def _analyze_cluster(
        self,
        cluster_id: str,
        cluster_genes: List[Dict[str, Any]],
        contig_lengths: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        seqid = cluster_genes[0]["seqid"]
        start_pos = min(gene["start"] for gene in cluster_genes)
        end_pos = max(gene["end"] for gene in cluster_genes)
        span_bp = end_pos - start_pos + 1

        gene_families = list(
            dict.fromkeys(gene.get("gene_family", "unknown") for gene in cluster_genes)
        )
        protein_ids = list(
            dict.fromkeys(
                gene.get("protein_id")
                or gene.get("gff_protein_id")
                or gene.get("locus_tag", "unknown")
                for gene in cluster_genes
            )
        )
        # One protein accession may represent several genomic CDS loci, while
        # one locus may receive multiple HMM family hits. Count and orient CDS
        # loci, not unique protein sequences or raw HMM rows.
        loci: Dict[Any, Dict[str, Any]] = {}
        for gene in cluster_genes:
            locus_key = gene.get("_feature_row_id")
            if locus_key is None:
                locus_key = (gene["seqid"], gene["start"], gene["end"], gene["strand"])
            loci.setdefault(locus_key, gene)

        strands = [gene["strand"] for gene in loci.values()]
        strand_switches = sum(
            1 for index in range(len(strands) - 1) if strands[index] != strands[index + 1]
        )

        cluster_type, has_catalytic_core, has_any_core_subunit = self._resolve_cluster_type(
            set(gene_families)
        )

        is_5prime_truncated = start_pos <= self.edge_margin_bp
        is_3prime_truncated = False
        if contig_lengths and seqid in contig_lengths:
            is_3prime_truncated = (
                contig_lengths[seqid] - end_pos
            ) <= self.edge_margin_bp

        is_truncated = is_5prime_truncated or is_3prime_truncated
        if is_5prime_truncated and is_3prime_truncated:
            truncated_side = "both"
        elif is_5prime_truncated:
            truncated_side = "5prime"
        elif is_3prime_truncated:
            truncated_side = "3prime"
        else:
            truncated_side = None

        if has_catalytic_core and not is_truncated:
            quality_status = "complete"
        elif is_truncated and has_any_core_subunit:
            quality_status = "partial_truncated"
        else:
            quality_status = "accessory_only"

        result = GeneClusterResult(
            cluster_id=cluster_id,
            seqid=seqid,
            start=start_pos,
            end=end_pos,
            span_bp=span_bp,
            gene_count=len(loci),
            gene_families=gene_families,
            protein_ids=protein_ids,
            strands=strands,
            strand_switches=strand_switches,
            cluster_type=cluster_type,
            has_catalytic_core=has_catalytic_core,
            is_truncated=is_truncated,
            truncated_side=truncated_side,
            quality_status=quality_status,
        )

        result_dict = result.model_dump()
        result_dict["has_any_core_subunit"] = has_any_core_subunit
        return result_dict