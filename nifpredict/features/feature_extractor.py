import pandas as pd
import numpy as np
from typing import Dict, Any, List


class GenomeFeatureExtractor:
    """PHASE 6: Feature Engineering Matrix.
    
    Trích xuất ma trận đặc trưng số từ kết quả HMM Search, Synteny Clusters và Metadata.
    """

    def __init__(self) -> None:
        self.gene_families = ["PF00142", "PF00148", "PF02826"]  # nifH, nifD, nifK

    def extract_features(
        self, 
        accession: str, 
        df_all_hits: pd.DataFrame, 
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Tạo Feature Vector hoàn chỉnh cho 1 Accession."""
        
        features: Dict[str, Any] = {"accession": accession}

        # 1. Trích xuất chỉ số HMM cho từng họ gen (Bit-score, E-value, Count)
        for gene in self.gene_families:
            if not df_all_hits.empty and "gene_family" in df_all_hits.columns:
                df_gene = df_all_hits[df_all_hits["gene_family"] == gene]
            else:
                df_gene = pd.DataFrame()

            if not df_gene.empty:
                features[f"{gene}_hit_count"] = len(df_gene)
                features[f"{gene}_max_bitscore"] = float(df_gene["score"].max()) if "score" in df_gene else 0.0
                features[f"{gene}_min_evalue"] = float(df_gene["evalue"].min()) if "evalue" in df_gene else 1.0
            else:
                features[f"{gene}_hit_count"] = 0
                features[f"{gene}_max_bitscore"] = 0.0
                features[f"{gene}_min_evalue"] = 1.0

        # 2. Trích xuất chỉ số Cụm gen Synteny & Khoảng cách sinh học
        features["clusters_found"] = len(clusters)
        
        if clusters:
            spans = [c.get("span_bp", 0) for c in clusters]
            gene_counts = [c.get("gene_count", 0) for c in clusters]
            
            features["max_cluster_span_bp"] = int(np.max(spans))
            features["avg_cluster_span_bp"] = float(np.mean(spans))
            features["max_cluster_gene_count"] = int(np.max(gene_counts))
            
            # Đếm số cụm chứa trọn vẹn bộ ba lõi nifHDK
            complete_clusters = sum(
                1 for c in clusters 
                if set(self.gene_families).issubset(set(c.get("gene_families", [])))
            )
            features["complete_hdk_clusters"] = complete_clusters
        else:
            features["max_cluster_span_bp"] = 0
            features["avg_cluster_span_bp"] = 0.0
            features["max_cluster_gene_count"] = 0
            features["complete_hdk_clusters"] = 0

        return features