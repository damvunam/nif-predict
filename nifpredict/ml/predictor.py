import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from nifpredict.common import paths
from nifpredict.common.config import PFAM_TO_GENE_MAP, load_config
from nifpredict.common.logger import setup_logger_from_config
from nifpredict.ml.data import NCBIDownloader, NCBIExtractor
from nifpredict.ml.features import ClusterFilter, GenomeFeatureExtractor, HMMAnnotator


# NAME của model Pfam trong file .hmm -> gen nif tương ứng.
# annotate_faa đặt gene_family bằng NAME (Fer4_NifH, Oxidored_nitro,
# 2-Hacid_dh_C), không phải accession Pfam (PF00142/148/02826).
# Regex nif([HDK]) chỉ cứu được Fer4_NifH; Oxidored_nitro và 2-Hacid_dh_C
# không chứa chuỗi "nif" nên không map sẽ rớt khỏi cụm, đối chứng dương
# Azotobacter vinelandii DJ ra âm tính giả dù HMM tìm đủ hit.
PFAM_NAME_TO_GENE_MAP = {
    "Fer4_NifH": "nifH",
    "Oxidored_nitro": "nifD",
    "2-Hacid_dh_C": "nifK",
}


def _canonical_gene_family(value: Any) -> str:
    """Chuẩn hóa tên family của một HMM hit về gen nifH/nifD/nifK (nếu được).

    ``HMMAnnotator.annotate_faa`` đặt ``gene_family`` bằng NAME của model
    trong file .hmm, không phải accession Pfam. Với Pfam, NAME có thể là
    accession (``PF00142``), ký hiệu gen (``nifH``) hoặc tên Pfam
    (``Fer4_NifH``). Không chuẩn hóa thì cả ClusterFilter lẫn bước chấm
    điểm đều không nhận ra hit, cho âm tính giả dù HMM đã tìm thấy gen.
    """
    text = str(value).strip()
    if text in PFAM_TO_GENE_MAP:
        return PFAM_TO_GENE_MAP[text]
    if text in PFAM_NAME_TO_GENE_MAP:
        return PFAM_NAME_TO_GENE_MAP[text]
    if text in {"nifH", "nifD", "nifK"}:
        return text
    if text == "nitrogenase_reductase":
        return "nifH"
    match = re.search(r"nif([HDK])", text, re.IGNORECASE)
    if match:
        return "nif" + match.group(1).upper()
    return text


def _score_families(families: set[str]) -> tuple[float, bool]:
    """Chấm điểm BNF từ tập gen đã chuẩn hóa: nifH 35 + nifD 35 + nifK 30, đủ >=70."""
    canonical = {_canonical_gene_family(f) for f in families}
    score = 0.0
    if "nifH" in canonical:
        score += 35.0
    if "nifD" in canonical:
        score += 35.0
    if "nifK" in canonical:
        score += 30.0
    return score, score >= 70.0


class NifPredictor:
    """Core Pipeline dự đoán khả năng cố định đạm (BNF) từ bộ gen vi khuẩn."""

    def __init__(self, config_path: str = str(paths.CONFIG_YAML)) -> None:
        self.config = load_config(config_path)
        self.logger = setup_logger_from_config(__name__)

        self.downloader = NCBIDownloader()

        # 1. Gán thuộc tính instance cho raw_genomes_dir & raw_metadata_dir.
        # AppConfig là Pydantic model, không phải dict: đường dẫn lấy qua
        # self.config.paths.<thuộc tính>, không dùng getattr fallback.
        # Trước đây getattr luôn rơi về fallback legacy data/raw_genomes,
        # trong khi config.yaml và build_feature_matrix.py dùng data/raw/genomes.
        self.raw_genomes_dir = Path(self.config.paths.raw_genomes_dir)
        self.raw_metadata_dir = Path(self.config.paths.raw_metadata_dir)

        self.extractor = NCBIExtractor(
            raw_genomes_dir=self.raw_genomes_dir, 
            raw_metadata_dir=self.raw_metadata_dir
        )
        self.annotator = HMMAnnotator()
        self.cluster_filter = ClusterFilter()
        self.feature_extractor = GenomeFeatureExtractor()

        self.profiles = {
            "PF00142": paths.ML_HMM_PROFILES / "nifH.hmm",
            "PF00148": paths.ML_HMM_PROFILES / "nifD.hmm",
            "PF02826": paths.ML_HMM_PROFILES / "nifK.hmm"
        }

    def _scan_hmm_profiles(
        self, protein_fasta: Path, accession: str
    ) -> pd.DataFrame:
        """Quét protein FASTA bằng 3 profile nifH/D/K qua HMMAnnotator thật.

        API cũ ``run_hmmsearch``/``parse_tblout`` không tồn tại trong repo
        (HMMAnnotator chỉ có ``annotate_faa``/``annotate_to_dataframe``),
        và code cũ bỏ qua lặng lẽ profile thiếu (``if exists`` rồi skip),
        nên lần chạy trước trả SUCCESS với score 0 dù chưa quét gì cả.
        Giờ profile thiếu là lỗi rõ ràng, không phải âm tính giả.
        """
        missing = [
            str(path)
            for path in self.profiles.values()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing HMM profile(s): " + ", ".join(missing)
                + ". Run: python scripts/ml/setup_hmm_profiles.py"
            )

        annotation_dir = Path(self.config.paths.annotation_dir)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        hit_frames: list[pd.DataFrame] = []
        for pfam_accession, profile_path in self.profiles.items():
            df_gene = self.annotator.annotate_to_dataframe(
                protein_fasta, profile_path
            )
            if df_gene.empty:
                continue
            # annotate_faa đặt gene_family = NAME của model trong file .hmm;
            # chuẩn hóa về nifH/nifD/nifK trước khi gom cụm và chấm điểm.
            df_gene = df_gene.copy()
            df_gene["gene_family"] = df_gene["gene_family"].map(
                _canonical_gene_family
            )
            hit_frames.append(df_gene)

        if not hit_frames:
            return pd.DataFrame()

        df_all_hits = pd.concat(hit_frames, ignore_index=True)
        df_all_hits.to_csv(
            annotation_dir / f"{accession}_hmm_hits.tsv", sep="\t", index=False
        )
        return df_all_hits

    def _annotate_and_cluster(
        self, protein_fasta: Path, accession: str
    ) -> tuple[pd.DataFrame, list]:
        """Quét HMM rồi gom cụm synteny; trả (df_all_hits, clusters)."""
        df_all_hits = self._scan_hmm_profiles(protein_fasta, accession)
        if df_all_hits.empty:
            return df_all_hits, []

        gff_file = self.extractor.raw_genomes_dir / f"{accession}_genomic.gff"
        if not gff_file.is_file():
            # Không có GFF thì không gom cụm được, nhưng hit HMM vẫn có giá
            # trị: chấm từ tập gen tìm thấy, ghi rõ là không có synteny.
            families = {
                _canonical_gene_family(v)
                for v in df_all_hits["gene_family"].dropna()
            }
            return df_all_hits, [
                {
                    "cluster_id": "ungrouped_hits",
                    "gene_families": sorted(families),
                    "note": "no GFF available; scored from HMM hits without synteny",
                }
            ]

        df_gff = self.cluster_filter.parse_gff3(gff_file)
        clusters = self.cluster_filter.group_into_clusters(df_all_hits, df_gff)
        return df_all_hits, clusters

    def predict_accession(self, accession: str) -> dict[str, Any]:
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

        # 3. Quét HMM các gen cố định đạm (annotate_to_dataframe, không phải
        # API cũ run_hmmsearch/parse_tblout đã không còn tồn tại).
        try:
            df_all_hits, clusters = self._annotate_and_cluster(
                protein_fasta, accession
            )
        except FileNotFoundError as exc:
            return {"accession": accession, "status": "FAILED", "error": str(exc)}

        if df_all_hits.empty:
            return {
                "accession": accession,
                "status": "SUCCESS",
                "organism_name": metadata.organism_name,
                "bnf_capable": False,
                "confidence_score_percent": 0.0,
                "clusters_found": 0,
                "clusters": []
            }

        # 4. Tính điểm BNF Confidence Score & Kết luận
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

    def _calculate_bnf_score(self, clusters: list) -> tuple[float, bool]:
        """Thuật toán đánh giá điểm tin cậy khả năng cố định đạm (BNF Score)."""
        if not clusters:
            return 0.0, False

        max_score = 0.0
        for cluster in clusters:
            families = set(cluster.get("gene_families", []))
            score, _ = _score_families(families)
            if score > max_score:
                max_score = score

        is_capable = max_score >= 70.0
        return max_score, is_capable

    def predict_batch(self, accessions: list[str]) -> list[dict[str, Any]]:
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

    def save_reports(self, results: list[dict[str, Any]], output_dir: str = str(paths.ML_RESULTS)) -> tuple[Path, Path]:
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

    def extract_sample_features(self, accession: str) -> dict[str, Any] | None:
        """Helper bao đóng quy trình tải, giải nén, quét HMM và trích xuất Feature Vector."""
        try:
            # 1. Tải & Giải nén
            zip_path = self.downloader.download_genome_zip(accession)
            if not zip_path:
                return {"accession": accession, "status": "DOWNLOAD_FAILED"}

            metadata = self.extractor.extract_package(zip_path, accession)
            if not metadata:
                return {"accession": accession, "status": "EXTRACTION_FAILED"}

            # 2. Lấy đường dẫn file protein (GFF được helper tự tìm)
            protein_fasta = self.raw_genomes_dir / f"{accession}_protein.faa"

            if not protein_fasta.exists():
                return {"accession": accession, "status": "MISSING_PROTEIN_FAA"}

            # 3. Quét HMM và gom cụm synteny (dùng chung helper với
            # predict_accession, qua API thật annotate_to_dataframe).
            df_all_hits, clusters = self._annotate_and_cluster(
                protein_fasta, accession
            )

            # 4. Trích xuất Feature Vector cho Phase 6
            feat_dict = self.feature_extractor.extract_features(accession, df_all_hits, clusters)
            feat_dict["organism_name"] = metadata.organism_name
            feat_dict["status"] = "SUCCESS"
            return feat_dict

        except Exception as e:
            self.logger.error(f"Lỗi trích xuất đặc trưng cho {accession}: {e}")
            return {"accession": accession, "status": f"ERROR: {str(e)}"}