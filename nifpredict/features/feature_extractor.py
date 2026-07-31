"""
Module: nifpredict.features.feature_extractor
Description: Production-ready Multi-layered Feature Engineering Matrix for NifPredict.
             Designed for high scalability, memory efficiency (Sparse Matrix support),
             data leakage prevention, and full Scikit-Learn API compatibility.
Author: Senior ML & Bioinformatics Engineer
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nifpredict.utils.config import config
from nifpredict.utils.logger import logger


def compute_neg_log10_evalue(evalue_series: pd.Series, epsilon: float = 1e-300) -> float:
    """Tính toán -log10(evalue) an toàn sinh học và tránh lỗi log(0).

    Phép biến đổi: f(E) = -log10(evalue + epsilon)
    """
    if evalue_series.empty:
        return 0.0
    min_evalue = float(evalue_series.min())
    # Giới hạn dưới epsilon để tránh log10(0.0) khi HMMER3 trả về 0.0
    safe_evalue = max(0.0, min_evalue) + epsilon
    return float(-np.log10(safe_evalue))


class MarkerClusterExtractor:
    """Tầng 1: Trích xuất chỉ số sinh học từ Marker Genes và Synteny Clusters (Sample-level)."""

    def __init__(
        self,
        core_pfams: Optional[Dict[str, str]] = None,
        aux_pfams: Optional[Dict[str, str]] = None,
        epsilon: float = 1e-300,
    ) -> None:
        self.core_pfams = core_pfams or getattr(
            config,
            "CORE_PFAMS",
            {"nifH": "PF00142", "nifD": "PF00148", "nifK": "PF02826"},
        )
        self.aux_pfams = aux_pfams or getattr(
            config,
            "AUX_PFAMS",
            {
                "anfG": "PF05910",
                "vnfG": "PF05911",
                "fixA": "PF01202",
                "fixB": "PF02525",
                "fixC": "PF02526",
                "fixX": "PF01802",
            },
        )
        self.epsilon = epsilon

    def extract_single_record(
        self,
        accession: str,
        df_hits: pd.DataFrame,
        clusters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Trích xuất feature vector Tầng 1 với định danh Schema cứng."""
        features: Dict[str, Any] = {"accession": accession}

        if df_hits.empty:
            logger.warning(f"[{accession}] df_hits rỗng. Gán đặc trưng Marker Genes mặc định.")

        # --- 1.1 Core Marker Genes (nifHDK) ---
        found_core_count = 0
        total_core_genes = len(self.core_pfams)

        for gene_name, pfam_id in self.core_pfams.items():
            if not df_hits.empty and "pfam_id" in df_hits.columns:
                df_gene = df_hits[df_hits["pfam_id"] == pfam_id]
            elif not df_hits.empty and "gene_family" in df_hits.columns:
                df_gene = df_hits[df_hits["gene_family"] == pfam_id]
            else:
                df_gene = pd.DataFrame()

            hit_count = len(df_gene)
            features[f"marker_core_{gene_name}_count"] = hit_count
            features[f"marker_core_{gene_name}_max_bitscore"] = (
                float(df_gene["score"].max()) if (not df_gene.empty and "score" in df_gene) else 0.0
            )
            features[f"marker_core_{gene_name}_neg_log_evalue"] = compute_neg_log10_evalue(
                df_gene["evalue"] if (not df_gene.empty and "evalue" in df_gene) else pd.Series(dtype=float),
                epsilon=self.epsilon,
            )

            if hit_count > 0:
                found_core_count += 1

        features["operon_core_completeness_score"] = float(found_core_count / total_core_genes)

        # --- 1.2 Auxiliary Marker Genes ---
        for gene_name, pfam_id in self.aux_pfams.items():
            if not df_hits.empty and "pfam_id" in df_hits.columns:
                df_gene = df_hits[df_hits["pfam_id"] == pfam_id]
            elif not df_hits.empty and "gene_family" in df_hits.columns:
                df_gene = df_hits[df_hits["gene_family"] == pfam_id]
            else:
                df_gene = pd.DataFrame()

            features[f"marker_aux_{gene_name}_count"] = len(df_gene)

        # --- 1.3 Synteny Clusters Metrics ---
        features["clusters_found_count"] = len(clusters)
        if clusters:
            spans = [c.get("span_bp", 0) for c in clusters]
            gene_counts = [c.get("gene_count", 0) for c in clusters]

            features["cluster_max_span_bp"] = float(np.max(spans))
            features["cluster_avg_span_bp"] = float(np.mean(spans))
            features["cluster_max_gene_count"] = int(np.max(gene_counts))

            core_pfam_ids = set(self.core_pfams.values())
            complete_clusters = sum(
                1 for c in clusters if core_pfam_ids.issubset(set(c.get("gene_families", [])))
            )
            features["cluster_complete_hdk_count"] = int(complete_clusters)
        else:
            features["cluster_max_span_bp"] = 0.0
            features["cluster_avg_span_bp"] = 0.0
            features["cluster_max_gene_count"] = 0
            features["cluster_complete_hdk_count"] = 0

        return features


class GenomeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Pipeline tổng hợp ma trận đặc trưng đa tầng cho NifPredict.

    Hỗ trợ xử lý bộ nhớ tối ưu (Sparse Matrix), phòng chống Data Leakage và kiểm soát số chiều.
    """

    def __init__(
        self,
        max_pfam_features: int = 1000,
        taxonomic_cols: Optional[List[str]] = None,
        numeric_meta_cols: Optional[List[str]] = None,
        epsilon: float = 1e-300,
        min_taxon_freq: float = 0.01,
    ) -> None:
        self.max_pfam_features = getattr(config, "MAX_PFAM_FEATURES", max_pfam_features)
        self.epsilon = getattr(config, "EVALUE_EPSILON", epsilon)

        # Chỉ lấy cấp độ Taxonomy kiểm soát được chiều (mặc định bỏ Genus/Species)
        self.taxonomic_cols = taxonomic_cols or getattr(
            config, "TAXONOMIC_LEVELS", ["phylum", "class", "order", "family"]
        )
        self.numeric_meta_cols = numeric_meta_cols or getattr(
            config, "NUMERIC_META_COLS", ["optimal_temperature", "optimal_ph"]
        )

        self.marker_extractor = MarkerClusterExtractor(epsilon=self.epsilon)

        # Layer 2: Functional Profile (Bag-of-Domains) -> Xuất ra Sparse Matrix
        self.pfam_vectorizer = TfidfVectorizer(
            max_features=self.max_pfam_features,
            token_pattern=r"\b\w+\b",
            lowercase=False,
            sublinear_tf=True,
        )

        # Layer 3: Taxonomy & Metadata (One-Hot Encoder giảm bùng nổ chiều)
        self.cat_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_taxon_freq,
            sparse_output=False,
        )
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _extract_layer1_df(self, raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Chuyển đổi dữ liệu Tầng 1 thành DataFrame chuẩn hóa."""
        rows = [
            self.marker_extractor.extract_single_record(
                rec["accession"],
                rec.get("df_all_hits", pd.DataFrame()),
                rec.get("clusters", []),
            )
            for rec in raw_records
        ]
        df_l1 = pd.DataFrame(rows)
        if "accession" in df_l1.columns:
            df_l1.set_index("accession", inplace=True)
        return df_l1

    def fit(
        self, raw_records: List[Dict[str, Any]], y: Optional[Any] = None
    ) -> "GenomeFeatureExtractor":
        """Fit các transformers độc lập trên tập Train."""
        logger.info(f"Bắt đầu fit GenomeFeatureExtractor trên {len(raw_records)} mẫu...")

        # 1. Fit Tầng 2: Pfam Bag-of-Domains
        pfam_texts = [" ".join(rec.get("pfam_domains", [])) for rec in raw_records]
        self.pfam_vectorizer.fit(pfam_texts)

        # 2. Fit Tầng 3: Taxonomy & Metadata
        meta_rows = [rec.get("metadata", {}) for rec in raw_records]
        df_meta = pd.DataFrame(meta_rows)

        # Fit Categorical Taxonomy
        for col in self.taxonomic_cols:
            if col not in df_meta.columns:
                logger.warning(f"Thiếu cột Taxonomy '{col}' trong metadata. Tự động điền 'unknown'.")
                df_meta[col] = "unknown"
        df_tax = df_meta[self.taxonomic_cols].fillna("unknown").astype(str)
        self.cat_encoder.fit(df_tax)

        # Fit Numerical Metadata
        for col in self.numeric_meta_cols:
            if col not in df_meta.columns:
                logger.warning(f"Thiếu cột Metadata số '{col}'. Tự động điền NaN.")
                df_meta[col] = np.nan
        df_num = df_meta[self.numeric_meta_cols]
        num_imputed = self.num_imputer.fit_transform(df_num)
        self.num_scaler.fit(num_imputed)

        self.is_fitted_ = True
        logger.info("Hoàn tất fit() transformers thành công.")
        return self

    def transform(
        self, raw_records: List[Dict[str, Any]], return_sparse: bool = False
    ) -> Union[pd.DataFrame, csr_matrix]:
        """Transform dữ liệu thô sang Ma trận Đặc trưng hoàn chỉnh.

        Args:
            raw_records: Danh sách dữ liệu thô đầu vào.
            return_sparse: Nếu True, trả về scipy.sparse.csr_matrix để tối ưu RAM.
                           Nếu False, trả về pd.DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("Extractor chưa được fit! Vui lòng gọi hàm .fit() trước.")

        accessions = [rec["accession"] for rec in raw_records]
        logger.info(f"Đang transform đặc trưng cho {len(accessions)} mẫu...")

        # --- Transform Layer 1 ---
        df_l1 = self._extract_layer1_df(raw_records)

        # --- Transform Layer 2 (Sparse Pfam Profile) ---
        pfam_texts = [" ".join(rec.get("pfam_domains", [])) for rec in raw_records]
        pfam_sparse = self.pfam_vectorizer.transform(pfam_texts)
        pfam_cols = [f"pfam_tfidf_{feat}" for feat in self.pfam_vectorizer.get_feature_names_out()]

        # --- Transform Layer 3 ---
        meta_rows = [rec.get("metadata", {}) for rec in raw_records]
        df_meta = pd.DataFrame(meta_rows)

        # Taxonomy
        for col in self.taxonomic_cols:
            if col not in df_meta.columns:
                df_meta[col] = "unknown"
        df_tax = df_meta[self.taxonomic_cols].fillna("unknown").astype(str)
        tax_encoded = self.cat_encoder.transform(df_tax)
        tax_cols = [f"tax_{feat}" for feat in self.cat_encoder.get_feature_names_out()]

        # Numeric Metadata
        for col in self.numeric_meta_cols:
            if col not in df_meta.columns:
                df_meta[col] = np.nan
        df_num = df_meta[self.numeric_meta_cols]
        num_imputed = self.num_imputer.transform(df_num)
        num_scaled = self.num_scaler.transform(num_imputed)
        num_cols = [f"meta_num_{col}" for col in self.numeric_meta_cols]

        # Ghép Layer 1 và Layer 3 thành Dense DataFrame
        df_dense = pd.concat(
            [
                df_l1,
                pd.DataFrame(tax_encoded, index=accessions, columns=tax_cols),
                pd.DataFrame(num_scaled, index=accessions, columns=num_cols),
            ],
            axis=1,
        )

        self.feature_names_ = list(df_dense.columns) + pfam_cols

        if return_sparse:
            # Chuyển đổi toàn bộ sang scipy.sparse.csr_matrix để tối ưu RAM
            dense_sparse = csr_matrix(df_dense.values)
            full_matrix = hstack([dense_sparse, pfam_sparse], format="csr")
            logger.info(f"Hoàn thành Ma trận Thưa (CSR Matrix). Kích thước: {full_matrix.shape}")
            return full_matrix

        # Trả về Pandas DataFrame đầy đủ
        df_pfam = pd.DataFrame(pfam_sparse.toarray(), index=accessions, columns=pfam_cols)
        df_final = pd.concat([df_dense, df_pfam], axis=1)
        logger.info(f"Hoàn thành DataFrame. Kích thước: {df_final.shape}")
        return df_final

    def fit_transform(
        self,
        raw_records: List[Dict[str, Any]],
        y: Optional[Any] = None,
        return_sparse: bool = False,
    ) -> Union[pd.DataFrame, csr_matrix]:
        """Thực hiện fit và transform đồng thời."""
        return self.fit(raw_records, y).transform(raw_records, return_sparse=return_sparse)

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Trả về danh sách tên đặc trưng theo chuẩn Scikit-Learn API."""
        if not self.is_fitted_:
            raise RuntimeError("Extractor chưa được fit! Vui lòng gọi hàm .fit() trước.")
        return np.array(self.feature_names_, dtype=object)