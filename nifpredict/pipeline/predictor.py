import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from nifpredict.utils import load_config, setup_logger
from nifpredict.data import NCBIDownloader, NCBIExtractor
from nifpredict.features import HMMAnnotator, ClusterFilter
from nifpredict.features import GenomeFeatureExtractor


class NifPredictor:
    """Core Pipeline dự đoán khả năng cố định đạm (BNF) từ bộ gen vi khuẩn."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = load_config(config_path)
        self.logger = setup_logger("nifpredict.pipeline")
        
        self.downloader = NCBIDownloader()
        
        # 1. Gán thuộc tính instance cho raw_genomes_dir & raw_metadata_dir
        self.raw_genomes_dir = Path(self.config.get("raw_genomes_dir", "data/raw_genomes"))
        self.raw_metadata_dir = Path(self.config.get("raw_metadata_dir", "data/raw_metadata"))

        self.extractor = NCBIExtractor(
            raw_genomes_dir=self.raw_genomes_dir, 
            raw_metadata_dir=self.raw_metadata_dir
        )
        self.annotator = HMMAnnotator()
        self.cluster_filter = ClusterFilter(max_gap_bp=10000, min_core_genes=2)
        self.feature_extractor = GenomeFeatureExtractor()

        self.profiles = {
            "PF00142": Path("data/hmm_profiles/nifH.hmm"),
            "PF00148": Path("data/hmm_profiles/nifD.hmm"),
            "PF02826": Path("data/hmm_profiles/nifK.hmm")
        }

    def predict_accession(self, accession: str) -> Dict[str, Any]:
        """Thực thi toàn bộ pipeline cho 1 mã RefSeq Accession."""
        self.logger.info(f"=== BẮT ĐẦU DỰ ĐOÁN CHO ACCESSION: {accession} ===")

        # 1. Tải gói dữ liệu bộ gen
        zip_path = self.downloader.download_genome_zip(accession)
        if not zip_path:
            return {"accession": accession, "status": "FAILED", "error": "Download failed"}

        # 2. Giải nén FASTA protein & Metadata
        metadata = self.extractor.extract_package(zip_path, accession)
        if not metadata:
            return {"accession": accession, "status": "FAILED", "error": "Extraction or metadata parsing failed"}

        # Xác định đường dẫn file protein sau khi giải nén
        protein_fasta = self.extractor.raw_genomes_dir / f"{accession}_protein.faa"

        # Kiểm tra sự tồn tại của file protein FAA
        if not protein_fasta.exists():
            return {"accession": accession, "status": "FAILED", "error": f"Missing protein FAA file: {protein_fasta.name}"}

        # 3. Quét HMM Search các gen cố định đạm
        annotation_dir = Path(self.config.get("paths", {}).get("annotation_dir", "data/annotation"))
        annotation_dir.mkdir(parents=True, exist_ok=True)
        
        all_hits = []
        for gene_family, profile_path in self.profiles.items():
            if profile_path.exists():
                out_tbl = annotation_dir / f"{accession}_{gene_family}.tbl"
                self.annotator.run_hmmsearch(protein_fasta, profile_path, out_tbl)
                df_gene = self.annotator.parse_tblout(out_tbl)
                if not df_gene.empty:
                    df_gene["gene_family"] = gene_family
                    all_hits.append(df_gene)

        if not all_hits:
            return {
                "accession": accession,
                "status": "SUCCESS",
                "organism_name": metadata.organism_name,
                "bnf_capable": False,
                "confidence_score_percent": 0.0,
                "clusters_found": 0,
                "clusters": []
            }

        df_all_hits = pd.concat(all_hits, ignore_index=True)

        # 4. Gom cụm Synteny Clusters (Cần file GFF3 nếu ClusterFilter yêu cầu)
        gff_file = self.extractor.raw_genomes_dir / f"{accession}_genomic.gff"
        df_gff = self.cluster_filter.parse_gff3(gff_file) if gff_file.exists() else pd.DataFrame()
        clusters = self.cluster_filter.group_into_clusters(df_all_hits, df_gff)

        # 5. Tính điểm BNF Confidence Score & Kết luận
        score, is_capable = self._calculate_bnf_score(clusters)

        return {
            "accession": accession,
            "organism_name": metadata.organism_name,
            "status": "SUCCESS",
            "bnf_capable": is_capable,
            "confidence_score_percent": score,
            "clusters_found": len(clusters),
            "clusters": clusters
        }

    def _calculate_bnf_score(self, clusters: list) -> Tuple[float, bool]:
        """Thuật toán đánh giá điểm tin cậy khả năng cố định đạm (BNF Score)."""
        if not clusters:
            return 0.0, False

        max_score = 0.0
        for cluster in clusters:
            families = set(cluster.get("gene_families", []))
            score = 0.0

            if "PF00142" in families: score += 35.0  # nifH
            if "PF00148" in families: score += 35.0  # nifD
            if "PF02826" in families: score += 30.0  # nifK

            if score > max_score:
                max_score = score

        is_capable = max_score >= 70.0
        return max_score, is_capable

    def predict_batch(self, accessions: list[str]) -> list[Dict[str, Any]]:
        """Thực thi dự đoán hàng loạt cho danh sách Accession."""
        self.logger.info(f"=== BẮT ĐẦU DỰ ĐOÁN BATCH CHO {len(accessions)} ACCESSION ===")
        results = []

        for idx, acc in enumerate(accessions, start=1):
            acc = acc.strip()
            if not acc or acc.startswith("#"):
                continue

            self.logger.info(f"[{idx}/{len(accessions)}] Đang xử lý Accession: {acc}")
            try:
                res = self.predict_accession(acc)
                results.append(res)
            except Exception as e:
                self.logger.error(f"Lỗi không xác định khi xử lý {acc}: {e}")
                results.append({
                    "accession": acc,
                    "status": "FAILED",
                    "error": str(e)
                })

        return results

    def save_reports(self, results: list[Dict[str, Any]], output_dir: str = "results") -> Tuple[Path, Path]:
        """Xuất kết quả dự đoán ra file JSON chi tiết và CSV tổng quan."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Lưu file JSON chi tiết
        json_file = out_path / "bnf_prediction_summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 2. Lưu file CSV tổng quan
        csv_file = out_path / "bnf_prediction_summary.csv"
        rows = []
        for r in results:
            rows.append({
                "accession": r.get("accession"),
                "organism_name": r.get("organism_name", "N/A"),
                "status": r.get("status"),
                "bnf_capable": r.get("bnf_capable", False),
                "confidence_score_percent": r.get("confidence_score_percent", 0.0),
                "clusters_found": r.get("clusters_found", 0),
                "error": r.get("error", "")
            })

        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(csv_file, index=False)

        self.logger.info(f"Đã lưu Báo cáo JSON tại: {json_file}")
        self.logger.info(f"Đã lưu Báo cáo CSV tại: {csv_file}")

        return json_file, csv_file

    def extract_sample_features(self, accession: str) -> Optional[Dict[str, Any]]:
        """Helper bao đóng quy trình tải, giải nén, quét HMM và trích xuất Feature Vector."""
        try:
            # 1. Tải & Giải nén
            zip_path = self.downloader.download_genome_zip(accession)
            if not zip_path:
                return {"accession": accession, "status": "DOWNLOAD_FAILED"}

            metadata = self.extractor.extract_package(zip_path, accession)
            if not metadata:
                return {"accession": accession, "status": "EXTRACTION_FAILED"}

            # 2. Lấy đường dẫn file protein và GFF
            protein_fasta = self.raw_genomes_dir / f"{accession}_protein.faa"
            gff_file = self.raw_genomes_dir / f"{accession}_genomic.gff"

            if not protein_fasta.exists():
                return {"accession": accession, "status": "MISSING_PROTEIN_FAA"}

            # 3. Quét HMM Search qua danh sách profiles (Đồng bộ với predict_accession)
            annotation_dir = Path(self.config.get("paths", {}).get("annotation_dir", "data/annotation"))
            annotation_dir.mkdir(parents=True, exist_ok=True)

            all_hits = []
            for gene_family, profile_path in self.profiles.items():
                if profile_path.exists():
                    out_tbl = annotation_dir / f"{accession}_{gene_family}.tbl"
                    self.annotator.run_hmmsearch(protein_fasta, profile_path, out_tbl)
                    df_gene = self.annotator.parse_tblout(out_tbl)
                    if not df_gene.empty:
                        df_gene["gene_family"] = gene_family
                        all_hits.append(df_gene)

            df_all_hits = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()

            # 4. Gom cụm Synteny
            df_gff = self.cluster_filter.parse_gff3(gff_file) if gff_file.exists() else pd.DataFrame()
            clusters = self.cluster_filter.group_into_clusters(df_all_hits, df_gff)

            # 5. Trích xuất Feature Vector cho Phase 6
            feat_dict = self.feature_extractor.extract_features(accession, df_all_hits, clusters)
            feat_dict["organism_name"] = metadata.organism_name
            feat_dict["status"] = "SUCCESS"
            return feat_dict

        except Exception as e:
            self.logger.error(f"Lỗi trích xuất đặc trưng cho {accession}: {e}")
            return {"accession": accession, "status": f"ERROR: {str(e)}"}