"""
Module: nifpredict.features.cluster_filter
Description: Synteny analysis and nitrogenase gene cluster filtering for NifPredict.
"""

import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import BaseModel

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
        records: List[Dict[str, Any]] = []
        if not gff_path.exists():
            return pd.DataFrame()

        with open(gff_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) != 9 or parts[2] != "CDS":
                    continue

                seqid, _, _, start, end, _, strand, _, attributes_raw = parts
                attr_dict: Dict[str, str] = {}
                for item in attributes_raw.split(";"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        attr_dict[k.strip()] = urllib.parse.unquote(v.strip())

                protein_id = attr_dict.get("protein_id") or attr_dict.get("ID")
                locus_tag = attr_dict.get("locus_tag") or attr_dict.get("ID")

                records.append({
                    "seqid": seqid,
                    "feature_type": "CDS",
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "protein_id": protein_id,
                    "locus_tag": locus_tag,
                })
        return pd.DataFrame(records)

    def _merge_hits_with_gff(self, df_hits: pd.DataFrame, df_gff: pd.DataFrame) -> pd.DataFrame:
        hits = df_hits.copy()
        if "target_protein" in hits.columns:
            hits = hits.rename(columns={"target_protein": "protein_id"})
        if "target_name" in hits.columns and "protein_id" not in hits.columns:
            hits = hits.rename(columns={"target_name": "protein_id"})

        gff_prot = df_gff.dropna(subset=["protein_id"])
        merged_prot = pd.merge(hits, gff_prot, on="protein_id", how="inner")

        matched_proteins = set(merged_prot["protein_id"].dropna())
        unmatched_hits = hits[~hits["protein_id"].isin(matched_proteins)]

        merged_locus = pd.DataFrame()
        if not unmatched_hits.empty and "locus_tag" in df_gff.columns:
            unmatched_renamed = unmatched_hits.rename(columns={"protein_id": "locus_tag"})
            gff_locus = df_gff.dropna(subset=["locus_tag"])
            merged_locus = pd.merge(unmatched_renamed, gff_locus, on="locus_tag", how="inner")

        combined = pd.concat([merged_prot, merged_locus], ignore_index=True)
        return combined.drop_duplicates(subset=["seqid", "start", "end", "protein_id"])

    def group_into_clusters(
        self,
        df_hits: pd.DataFrame,
        df_gff: pd.DataFrame,
        contig_lengths: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        merged = self._merge_hits_with_gff(df_hits, df_gff)
        if merged.empty:
            logger.warning("Không thể khớp dữ liệu HMM hits và GFF3!")
            return []

        merged = merged.sort_values(by=["seqid", "start", "end"]).reset_index(drop=True)

        raw_clusters: List[List[Dict[str, Any]]] = []
        current_cluster: List[Dict[str, Any]] = []
        current_max_end: int = 0

        for row in merged.itertuples(index=False):
            row_dict = row._asdict()
            if not current_cluster:
                current_cluster.append(row_dict)
                current_max_end = row_dict["end"]
                continue

            last_gene = current_cluster[-1]
            same_seq = row_dict["seqid"] == last_gene["seqid"]
            strand_flip = row_dict["strand"] != last_gene["strand"]
            gap = max(0, row_dict["start"] - current_max_end)

            effective_max_gap = self.max_divergent_gap_bp if strand_flip else self.max_gap_bp

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
        # Tự động map Pfam ID sang Gene Symbol
        mapped_families = set()
        for fam in raw_families:
            mapped_families.add(fam)
            if fam in PFAM_TO_GENE_MAP:
                mapped_families.add(PFAM_TO_GENE_MAP[fam])

        has_h = bool(mapped_families.intersection(self.CORE_H_FAMILIES))
        has_d = bool(mapped_families.intersection(self.CORE_D_FAMILIES))
        has_k = bool(mapped_families.intersection(self.CORE_K_FAMILIES))

        has_catalytic_core = (has_h + has_d + has_k) >= 2
        has_any_core_subunit = has_h or has_d or has_k

        vnf_score = len(mapped_families.intersection(self.VNF_SPECIFIC))
        anf_score = len(mapped_families.intersection(self.ANF_SPECIFIC))
        nif_score = sum(1 for f in mapped_families if f.startswith("nif"))

        if vnf_score > 0 and vnf_score >= anf_score and ("vnfG" in mapped_families or "PF05911" in mapped_families):
            cluster_type = "vnf"
        elif anf_score > 0 and anf_score > vnf_score and ("anfG" in mapped_families or "PF05910" in mapped_families):
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
        start_pos = min(g["start"] for g in cluster_genes)
        end_pos = max(g["end"] for g in cluster_genes)
        span_bp = end_pos - start_pos + 1

        gene_families = list(
            dict.fromkeys(g.get("gene_family", "unknown") for g in cluster_genes)
        )
        protein_ids = [
            g.get("protein_id") or g.get("locus_tag", "unknown")
            for g in cluster_genes
        ]
        strands = [g["strand"] for g in cluster_genes]
        strand_switches = sum(1 for i in range(len(strands) - 1) if strands[i] != strands[i + 1])

        cluster_type, has_catalytic_core, has_any_core_subunit = self._resolve_cluster_type(set(gene_families))

        is_5prime_truncated = start_pos <= self.edge_margin_bp
        is_3prime_truncated = False
        if contig_lengths and seqid in contig_lengths:
            is_3prime_truncated = (contig_lengths[seqid] - end_pos) <= self.edge_margin_bp

        is_truncated = is_5prime_truncated or is_3prime_truncated
        truncated_side = "both" if (is_5prime_truncated and is_3prime_truncated) else ("5prime" if is_5prime_truncated else ("3prime" if is_3prime_truncated else None))

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
            gene_count=len(cluster_genes),
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

        res_dict = result.model_dump()
        res_dict["has_any_core_subunit"] = has_any_core_subunit
        return res_dict