"""
Module: nifpredict.features.feature_extractor
Description: Production-ready Multi-layered Feature Engineering Matrix.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nifpredict.utils.config import AppConfig, load_config
from nifpredict.utils.logger import get_logger

logger = get_logger("nifpredict.features.feature_extractor")


def compute_neg_log10_evalue(evalue_series: pd.Series, epsilon: float = 1e-300) -> float:
    if evalue_series.empty:
        return 0.0
    min_evalue = float(evalue_series.min())
    safe_evalue = max(0.0, min_evalue) + epsilon
    return float(-np.log10(safe_evalue))


class MarkerClusterExtractor:
    def __init__(
        self,
        core_pfams: Optional[Dict[str, str]] = None,
        aux_pfams: Optional[Dict[str, str]] = None,
        epsilon: float = 1e-300,
    ) -> None:
        self.core_pfams = core_pfams or {"nifH": "PF00142", "nifD": "PF00148", "nifK": "PF02826"}
        self.aux_pfams = aux_pfams or {
            "anfG": "PF05910", "vnfG": "PF05911", "fixA": "PF01202",
            "fixB": "PF02525", "fixC": "PF02526", "fixX": "PF01802"
        }
        self.epsilon = epsilon

    def extract_single_record(
        self,
        accession: str,
        df_hits: pd.DataFrame,
        clusters: List[Dict[str, Any]],
        status: str = "SUCCESS",
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {"accession": accession, "status": status}
        found_core_count = 0
        total_core_genes = len(self.core_pfams)

        # 1. Trích xuất chỉ số Core Genes (nifHDK)
        for gene_name, pfam_id in self.core_pfams.items():
            if df_hits is not None and not df_hits.empty:
                if "pfam_id" in df_hits.columns:
                    df_gene = df_hits[df_hits["pfam_id"] == pfam_id]
                elif "gene_family" in df_hits.columns:
                    df_gene = df_hits[df_hits["gene_family"].isin([pfam_id, gene_name])]
                else:
                    df_gene = pd.DataFrame()
            else:
                df_gene = pd.DataFrame()

            hit_count = len(df_gene)
            max_bitscore = (
                float(df_gene["effective_score"].max()) if (not df_gene.empty and "effective_score" in df_gene.columns)
                else (float(df_gene["raw_score"].max()) if (not df_gene.empty and "raw_score" in df_gene.columns) else 0.0)
            )
            min_evalue = (
                float(df_gene["seq_evalue"].min()) if (not df_gene.empty and "seq_evalue" in df_gene.columns)
                else 1.0
            )
            neg_log_evalue = compute_neg_log10_evalue(
                df_gene["seq_evalue"] if (not df_gene.empty and "seq_evalue" in df_gene.columns) else pd.Series(dtype=float),
                epsilon=self.epsilon,
            )

            # Cột tên gen chuẩn
            features[f"marker_core_{gene_name}_count"] = hit_count
            features[f"marker_core_{gene_name}_max_bitscore"] = max_bitscore
            features[f"marker_core_{gene_name}_neg_log_evalue"] = neg_log_evalue

            # Bí danh tên Pfam ID (bắt buộc cho eda_and_labeling.py)
            features[f"{pfam_id}_max_bitscore"] = max_bitscore
            features[f"{pfam_id}_min_evalue"] = min_evalue
            features[f"-log10_{pfam_id}_min_evalue"] = neg_log_evalue

            if hit_count > 0:
                found_core_count += 1

        features["operon_core_completeness_score"] = float(found_core_count / total_core_genes)

        # 2. Trích xuất chỉ số Auxiliary Genes (Bổ sung đoạn bị thiếu)
        for gene_name, pfam_id in self.aux_pfams.items():
            if df_hits is not None and not df_hits.empty and "gene_family" in df_hits.columns:
                df_gene = df_hits[df_hits["gene_family"].isin([pfam_id, gene_name])]
            else:
                df_gene = pd.DataFrame()
            features[f"marker_aux_{gene_name}_count"] = len(df_gene)

        # 3. Trích xuất chỉ số Cụm Gen Synteny (Đưa NẰM NGOÀI các vòng lặp)
        num_clusters = len(clusters) if clusters else 0
        features["clusters_found"] = num_clusters
        features["clusters_found_count"] = num_clusters

        # Tính toán biến spans và complete_clusters trước khi gán
        spans = [c.get("span_bp", 0) for c in clusters] if clusters else []
        gene_counts = [c.get("gene_count", 0) for c in clusters] if clusters else []

        core_pfams_set = set(self.core_pfams.values())
        complete_clusters = sum(
            1 for c in clusters 
            if c.get("has_catalytic_core", False) 
            or core_pfams_set.issubset(set(c.get("gene_families", [])))
            or {"nifH", "nifD", "nifK"}.issubset(set(c.get("gene_families", [])))
        ) if clusters else 0

        if clusters:
            features["max_cluster_span_bp"] = float(np.max(spans))
            features["cluster_max_span_bp"] = float(np.max(spans))
            features["cluster_avg_span_bp"] = float(np.mean(spans))
            features["cluster_max_gene_count"] = int(np.max(gene_counts))
            features["complete_hdk_clusters"] = complete_clusters
            features["cluster_complete_hdk_count"] = complete_clusters
        else:
            features["max_cluster_span_bp"] = 0.0
            features["cluster_max_span_bp"] = 0.0
            features["cluster_avg_span_bp"] = 0.0
            features["cluster_max_gene_count"] = 0
            features["complete_hdk_clusters"] = 0
            features["cluster_complete_hdk_count"] = 0

        return features


class GenomeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        max_pfam_features: int = 1000,
        taxonomic_cols: Optional[List[str]] = None,
        numeric_meta_cols: Optional[List[str]] = None,
        epsilon: float = 1e-300,
        min_taxon_freq: float = 0.01,
    ) -> None:
        self.app_config = config or load_config(auto_create_dirs=False)
        self.max_pfam_features = max_pfam_features
        self.epsilon = epsilon

        self.taxonomic_cols = taxonomic_cols or ["phylum", "class", "order", "family"]
        self.numeric_meta_cols = numeric_meta_cols or ["optimal_temperature", "optimal_ph"]

        self.marker_extractor = MarkerClusterExtractor(epsilon=self.epsilon)

        self.pfam_vectorizer = TfidfVectorizer(
            max_features=self.max_pfam_features,
            token_pattern=r"\b\w+\b",
            lowercase=False,
            sublinear_tf=True,
        )
        self.cat_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_taxon_freq,
            sparse_output=False,
        )
        
        # HOTFIX: Bổ sung keep_empty_features=True và fill_value=0.0
        # Ngăn SimpleImputer xóa cột khi gặp toàn bộ giá trị NaN trong sample đơn lẻ
        self.num_imputer = SimpleImputer(
            strategy="median",
            fill_value=0.0,
            keep_empty_features=True,
        )
        self.num_scaler = StandardScaler()

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _extract_layer1_df(self, raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
        rows = [
            self.marker_extractor.extract_single_record(
                rec["accession"],
                rec.get("df_all_hits", pd.DataFrame()),
                rec.get("clusters", []),
                status=rec.get("status", "SUCCESS")
            )
            for rec in raw_records
        ]
        return pd.DataFrame(rows)

    def fit(
        self, raw_records: List[Dict[str, Any]], y: Optional[Any] = None
    ) -> "GenomeFeatureExtractor":
        pfam_texts = [" ".join(rec.get("pfam_domains", [])) for rec in raw_records]
        if any(pfam_texts):
            self.pfam_vectorizer.fit(pfam_texts)
        else:
            self.pfam_vectorizer.fit(["dummy_pfam"])

        meta_rows = [rec.get("metadata", {}) for rec in raw_records]
        df_meta = pd.DataFrame(meta_rows)

        for col in self.taxonomic_cols:
            if col not in df_meta.columns:
                df_meta[col] = "unknown"
        df_tax = df_meta[self.taxonomic_cols].fillna("unknown").astype(str)
        self.cat_encoder.fit(df_tax)

        for col in self.numeric_meta_cols:
            if col not in df_meta.columns:
                df_meta[col] = np.nan
        df_num = df_meta[self.numeric_meta_cols].astype(float)
        num_imputed = self.num_imputer.fit_transform(df_num)
        self.num_scaler.fit(num_imputed)

        self.is_fitted_ = True
        return self

    def transform(
        self, raw_records: List[Dict[str, Any]], return_sparse: bool = False
    ) -> Union[pd.DataFrame, csr_matrix]:
        if not self.is_fitted_:
            raise RuntimeError("Extractor chưa được fit! Vui lòng gọi .fit() trước.")

        accessions = [rec["accession"] for rec in raw_records]
        df_l1 = self._extract_layer1_df(raw_records)

        pfam_texts = [" ".join(rec.get("pfam_domains", [])) for rec in raw_records]
        pfam_sparse = self.pfam_vectorizer.transform(pfam_texts)
        pfam_cols = [f"pfam_tfidf_{feat}" for feat in self.pfam_vectorizer.get_feature_names_out()]

        meta_rows = [rec.get("metadata", {}) for rec in raw_records]
        df_meta = pd.DataFrame(meta_rows)

        for col in self.taxonomic_cols:
            if col not in df_meta.columns:
                df_meta[col] = "unknown"
        df_tax = df_meta[self.taxonomic_cols].fillna("unknown").astype(str)
        tax_encoded = self.cat_encoder.transform(df_tax)
        tax_cols = [f"tax_{feat}" for feat in self.cat_encoder.get_feature_names_out()]

        for col in self.numeric_meta_cols:
            if col not in df_meta.columns:
                df_meta[col] = np.nan
        df_num = df_meta[self.numeric_meta_cols].astype(float)
        num_imputed = self.num_imputer.transform(df_num)
        num_scaled = self.num_scaler.transform(num_imputed)
        num_cols = [f"meta_num_{col}" for col in self.numeric_meta_cols]

        df_dense = pd.concat(
            [
                df_l1.reset_index(drop=True),
                pd.DataFrame(tax_encoded, columns=tax_cols),
                pd.DataFrame(num_scaled, columns=num_cols),
            ],
            axis=1,
        )
        df_dense["accession"] = accessions

        non_feature_cols = {"accession", "status"}
        self.feature_names_ = [c for c in df_dense.columns if c not in non_feature_cols] + pfam_cols

        if return_sparse:
            dense_vals = df_dense.drop(columns=["accession"]).values
            full_matrix = hstack([csr_matrix(dense_vals), pfam_sparse], format="csr")
            return full_matrix

        df_pfam = pd.DataFrame(pfam_sparse.toarray(), columns=pfam_cols)
        df_final = pd.concat([df_dense, df_pfam], axis=1)
        return df_final

    def fit_transform(
        self,
        raw_records: List[Dict[str, Any]],
        y: Optional[Any] = None,
        return_sparse: bool = False,
    ) -> Union[pd.DataFrame, csr_matrix]:
        return self.fit(raw_records, y).transform(raw_records, return_sparse=return_sparse)