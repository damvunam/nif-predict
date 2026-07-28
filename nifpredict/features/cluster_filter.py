import re
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from nifpredict.utils.config import load_config
from nifpredict.utils.logger import setup_logger


class ClusterFilter:
    """
    Module lọc và gom nhóm các gen nif thành Gene Clusters dựa trên tọa độ GFF3 chuẩn và quy tắc sinh học synteny.
    """

    def __init__(
        self, 
        max_gap_bp: int = 10000, 
        min_core_genes: int = 2,
        config_path: str = "config/config.yaml", 
        logger=None
    ) -> None:
        self.config = load_config(config_path)
        self.logger = logger or setup_logger("nifpredict.features.cluster_filter")
        self.max_gap_bp = max_gap_bp
        self.min_core_genes = min_core_genes

    @staticmethod
    def parse_gff3(gff_path: Path) -> pd.DataFrame:
        """Parse file GFF3 9 cột chuẩn của NCBI thành DataFrame."""
        records = []
        if not gff_path.exists():
            return pd.DataFrame()

        with open(gff_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) == 9 and parts[2] in ["CDS", "gene"]:
                    seqid, source, feature_type, start, end, score, strand, phase, attributes = parts
                    
                    protein_id_match = re.search(r"protein_id=([^;]+)", attributes)
                    protein_id = protein_id_match.group(1) if protein_id_match else None

                    locus_match = re.search(r"locus_tag=([^;]+)", attributes)
                    locus_tag = locus_match.group(1) if locus_match else None

                    records.append({
                        "seqid": seqid,
                        "feature_type": feature_type,
                        "start": int(start),
                        "end": int(end),
                        "strand": strand,
                        "protein_id": protein_id,
                        "locus_tag": locus_tag
                    })

        return pd.DataFrame(records)

    def group_into_clusters(self, df_hits: pd.DataFrame, df_gff: pd.DataFrame) -> List[Dict[str, Any]]:
        """Gộp các hits gen nif dựa trên vị trí tọa độ GFF thành các Gene Cluster."""
        if df_hits.empty or df_gff.empty:
            self.logger.warning("Dữ liệu đầu vào hits hoặc gff rỗng!")
            return []

        # Chuẩn hóa tên cột đồng nhất
        hits_df = df_hits.copy()
        if "target_protein" in hits_df.columns:
            hits_df = hits_df.rename(columns={"target_protein": "protein_id"})

        # Merge thông tin vị trí từ GFF vào hits dựa trên protein_id
        merged = pd.merge(hits_df, df_gff.dropna(subset=["protein_id"]), on="protein_id", how="inner")
        
        if merged.empty:
            self.logger.warning("Không khớp được protein_id giữa HMM hits và GFF!")
            return []

        # Sắp xếp các gen theo thứ tự tọa độ tăng dần trên bộ gen
        merged = merged.sort_values(by=["seqid", "start"]).reset_index(drop=True)

        raw_clusters = []
        current_cluster = []

        for _, row in merged.iterrows():
            if not current_cluster:
                current_cluster.append(row.to_dict())
                continue

            last_gene = current_cluster[-1]

            same_seq = (row["seqid"] == last_gene["seqid"])
            gap = row["start"] - last_gene["end"]

            if same_seq and gap <= self.max_gap_bp:
                current_cluster.append(row.to_dict())
            else:
                raw_clusters.append(current_cluster)
                current_cluster = [row.to_dict()]

        if current_cluster:
            raw_clusters.append(current_cluster)

        # Lọc các cluster thỏa mãn điều kiện min_core_genes
        valid_clusters = []
        for c in raw_clusters:
            formatted = self._format_cluster(c)
            if len(formatted["gene_families"]) >= self.min_core_genes:
                valid_clusters.append(formatted)

        self.logger.info(f"Phát hiện {len(valid_clusters)} cụm gen đạt chuẩn synteny & core genes.")
        return valid_clusters

    def _format_cluster(self, cluster_genes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Đóng gói thông tin cụm gen (giữ nguyên thứ tự gen xuất hiện sinh học)."""
        seqid = cluster_genes[0]["seqid"]
        start_pos = cluster_genes[0]["start"]
        end_pos = max(g["end"] for g in cluster_genes)
        
        # Giữ đúng thứ tự gen xuất hiện trên cụm operon
        genes = list(dict.fromkeys(g["gene_family"] for g in cluster_genes))
        targets = [g["protein_id"] for g in cluster_genes]
        strands = list(dict.fromkeys(g["strand"] for g in cluster_genes))

        span_bp = end_pos - start_pos + 1

        return {
            "seqid": seqid,
            "start": start_pos,
            "end": end_pos,
            "span_bp": span_bp,
            "strands": strands,
            "gene_count": len(cluster_genes),
            "gene_families": genes,
            "protein_ids": targets
        }
