"""
Module phân tích synteny và lọc cụm gen cố định đạm (nif/fix/anf/vnf) cho NifPredict.

Đã xử lý triệt để:
1. Giữ lại cụm gen bị cắt đứt ở ranh giới contig (MAGs truncation rescue).
2. Merge đa khóa (multi-key merge: protein_id -> locus_tag fallback) không bỏ sót bản ghi.
3. Tách cụm gen phân kỳ khi đổi chiều mạch ADN (divergent operon splitting).
4. Phân giải nhiễu chéo giữa các hệ thống Nitrogenase (Nif vs Vnf vs Anf resolution).
"""

import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from nifpredict.utils.config import NifPredictConfig, load_config
from nifpredict.utils.logger import setup_logger


class GeneClusterResult(BaseModel):
    """Schema Pydantic đại diện cho kết quả phân tích của một cụm gen."""

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
    cluster_type: str  # "nif", "vnf", "anf", "accessory"
    has_catalytic_core: bool
    is_truncated: bool
    truncated_side: Optional[str] = None  # "5prime", "3prime", "both", None
    quality_status: str  # "complete", "partial_truncated", "accessory_only"


class ClusterFilter:
    """
    Module phân tích Synteny và lọc các Gene Cluster cố định đạm đạt chuẩn Production.
    """

    # Họ gen xúc tác lõi chuẩn hóa (Generic catalytic core families)
    CORE_H_FAMILIES: Set[str] = {"nifH", "vnfH", "anfH"}
    CORE_D_FAMILIES: Set[str] = {"nifD", "vnfD", "anfD"}
    CORE_K_FAMILIES: Set[str] = {"nifK", "vnfK", "anfK"}

    # Gen đặc trưng phân biệt hệ thống nitrogenase thay thế (Vnf/Anf specific structural subunits)
    VNF_SPECIFIC: Set[str] = {"vnfG", "vnfEN", "vnfH", "vnfD", "vnfK"}
    ANF_SPECIFIC: Set[str] = {"anfG", "anfH", "anfD", "anfK"}

    def __init__(
        self,
        config: Optional[NifPredictConfig] = None,
        config_path: str = "config/config.yaml",
        logger: Optional[Any] = None,
    ) -> None:
        """Khởi tạo ClusterFilter với cấu hình Pydantic v2."""
        self.logger = logger or setup_logger("nifpredict.features.cluster_filter")

        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Đọc tham số từ config với giá trị fallback an toàn
        self.max_gap_bp: int = getattr(
            self.config.cluster, "max_intergenic_distance_bp", 10000
        )
        self.max_divergent_gap_bp: int = getattr(
            self.config.cluster, "max_divergent_gap_bp", 1500
        )
        self.min_core_genes: int = getattr(
            self.config.cluster, "min_core_genes", 2
        )
        self.edge_margin_bp: int = getattr(
            self.config.cluster, "contig_edge_margin_bp", 1000
        )

    @staticmethod
    def parse_gff3(gff_path: Path) -> pd.DataFrame:
        """Parse tệp GFF3 chuẩn, chỉ lấy CDS và unquote chuỗi URL-encoded."""
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

                records.append(
                    {
                        "seqid": seqid,
                        "feature_type": "CDS",
                        "start": int(start),
                        "end": int(end),
                        "strand": strand,
                        "protein_id": protein_id,
                        "locus_tag": locus_tag,
                    }
                )

        return pd.DataFrame(records)

    def _merge_hits_with_gff(
        self, df_hits: pd.DataFrame, df_gff: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Khắc phục Lỗi 2: Merge 2 lớp (Two-pass Merge) để không bỏ sót bản ghi lai.
        Pass 1: Merge theo protein_id.
        Pass 2: Lấy các hits chưa merge được, tiếp tục merge theo locus_tag.
        """
        hits = df_hits.copy()
        if "target_protein" in hits.columns:
            hits = hits.rename(columns={"target_protein": "protein_id"})

        # Pass 1: Merge theo protein_id
        gff_prot = df_gff.dropna(subset=["protein_id"])
        merged_prot = pd.merge(hits, gff_prot, on="protein_id", how="inner")

        # Xác định các hits chưa khớp ở Pass 1
        matched_proteins = set(merged_prot["protein_id"].dropna())
        unmatched_hits = hits[~hits["protein_id"].isin(matched_proteins)]

        # Pass 2: Merge theo locus_tag cho các hits còn lại
        merged_locus = pd.DataFrame()
        if not unmatched_hits.empty and "locus_tag" in unmatched_hits.columns:
            gff_locus = df_gff.dropna(subset=["locus_tag"])
            merged_locus = pd.merge(
                unmatched_hits, gff_locus, on="locus_tag", how="inner"
            )

        # Gộp hai tập kết quả và loại bỏ bản ghi trùng lặp
        combined = pd.concat([merged_prot, merged_locus], ignore_index=True)
        return combined.drop_duplicates(subset=["seqid", "start", "end", "protein_id"])

    def group_into_clusters(
        self,
        df_hits: pd.DataFrame,
        df_gff: pd.DataFrame,
        contig_lengths: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Gom nhóm cụm gen synteny có xử lý Divergent Operons và MAG Truncation."""
        merged = self._merge_hits_with_gff(df_hits, df_gff)

        if merged.empty:
            self.logger.warning("Không thể khớp dữ liệu HMM hits và GFF3!")
            return []

        # Sắp xếp CDS theo thứ tự tọa độ trên contig
        merged = merged.sort_values(by=["seqid", "start", "end"]).reset_index(
            drop=True
        )

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

            # Tính khoảng cách intergenic chuẩn xác
            gap = max(0, row_dict["start"] - current_max_end)

            # Khắc phục Lỗi 3: Ngưỡng gap nghiêm ngặt hơn khi có sự đảo chiều mạch ADN
            effective_max_gap = (
                self.max_divergent_gap_bp if strand_flip else self.max_gap_bp
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

        # Đánh giá và lọc cụm gen
        valid_clusters: List[Dict[str, Any]] = []
        rejected_count = 0

        for idx, cluster_genes in enumerate(raw_clusters, 1):
            cluster_info = self._analyze_cluster(
                cluster_id=f"cluster_{idx:04d}",
                cluster_genes=cluster_genes,
                contig_lengths=contig_lengths,
            )

            # Khắc phục Lỗi 1: Cứu các cụm bị đứt gãy ranh giới contig (MAG Truncation Rescue)
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
            else:
                rejected_count += 1

        self.logger.info(
            f"Lọc synteny hoàn tất: Giữ lại {len(valid_clusters)} cụm gen (bao gồm cụm partial/truncated), "
            f"loại bỏ {rejected_count} cụm nhiễu/không chức năng."
        )

        return valid_clusters

    def _resolve_cluster_type(self, families: Set[str]) -> Tuple[str, bool, bool]:
        """
        Khắc phục Lỗi 4: Phân giải nhiễu chéo (Homology Cross-Talk) giữa Nif / Vnf / Anf.
        Sử dụng phân nhóm họ gen gốc (Generic subunits) và gen đặc trưng (Specific subunits).
        """
        has_h = bool(families.intersection(self.CORE_H_FAMILIES))
        has_d = bool(families.intersection(self.CORE_D_FAMILIES))
        has_k = bool(families.intersection(self.CORE_K_FAMILIES))

        # Có ít nhất 2/3 tiểu đơn vị xúc tác
        has_catalytic_core = (has_h + has_d + has_k) >= 2
        has_any_core_subunit = has_h or has_d or has_k

        # Đếm số lượng gen đặc trưng của từng hệ thống
        vnf_score = len(families.intersection(self.VNF_SPECIFIC))
        anf_score = len(families.intersection(self.ANF_SPECIFIC))
        nif_score = sum(1 for f in families if f.startswith("nif"))

        # Quyết định cluster_type dựa trên điểm đặc trưng cao nhất
        if vnf_score > 0 and vnf_score >= anf_score and "vnfG" in families:
            cluster_type = "vnf"
        elif anf_score > 0 and anf_score > vnf_score and "anfG" in families:
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
        """Phân tích tổng hợp đặc tính sinh học, chiều mạch và chất lượng cụm gen."""
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

        strand_switches = sum(
            1 for i in range(len(strands) - 1) if strands[i] != strands[i + 1]
        )

        # Phân giải loại cụm gen & bộ ba xúc tác
        families_set = set(gene_families)
        cluster_type, has_catalytic_core, has_any_core_subunit = (
            self._resolve_cluster_type(families_set)
        )

        # Kiểm tra ranh giới contig (Contig Edge Truncation)
        is_5prime_truncated = start_pos <= self.edge_margin_bp
        is_3prime_truncated = False

        if contig_lengths and seqid in contig_lengths:
            contig_len = contig_lengths[seqid]
            is_3prime_truncated = (contig_len - end_pos) <= self.edge_margin_bp

        is_truncated = is_5prime_truncated or is_3prime_truncated
        truncated_side = None
        if is_5prime_truncated and is_3prime_truncated:
            truncated_side = "both"
        elif is_5prime_truncated:
            truncated_side = "5prime"
        elif is_3prime_truncated:
            truncated_side = "3prime"

        # Phân loại trạng thái chất lượng cụm (Quality Status)
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