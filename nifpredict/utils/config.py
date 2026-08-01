"""
Module: nifpredict.utils.config
Description: Centralized Pydantic v2 Configuration Management for NifPredict.
             Supports environment variable overrides and path resolution.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import yaml

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ánh xạ chuẩn 2 chiều giữa Pfam Accession và Gene Symbol
PFAM_TO_GENE_MAP: Dict[str, str] = {
    "PF00142": "nifH",
    "PF00148": "nifD",
    "PF02826": "nifK",
    "PF05910": "anfG",
    "PF05911": "vnfG",
    "PF01202": "fixA",
    "PF02525": "fixB",
    "PF02526": "fixC",
    "PF01802": "fixX",
}

GENE_TO_PFAM_MAP: Dict[str, str] = {v: k for k, v in PFAM_TO_GENE_MAP.items()}


class ProjectConfig(BaseModel):
    name: str
    version: str
    environment: Literal["development", "testing", "production"]
    description: str


class ComputingConfig(BaseModel):
    max_threads: int = Field(gt=0, description="Số CPU threads phải > 0")
    chunk_size: int = Field(gt=0)
    enable_gpu: bool


class PathsConfig(BaseModel):
    base_dir: Optional[Path] = Field(default_factory=lambda: PROJECT_ROOT)
    raw: Dict[str, Path]
    interim: Dict[str, Path]
    processed: Dict[str, Path]
    databases: Dict[str, Path]
    models: Path
    log_dir: Path

    @model_validator(mode="after")
    def resolve_absolute_paths(self) -> "PathsConfig":
        base = (self.base_dir or PROJECT_ROOT).resolve()

        def _to_abs(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        self.models = _to_abs(self.models)
        self.log_dir = _to_abs(self.log_dir)

        for category in [self.raw, self.interim, self.processed, self.databases]:
            for key, path_val in category.items():
                category[key] = _to_abs(path_val)
        return self

    @property
    def raw_genomes_dir(self) -> Path:
        return self.raw["genomes_dir"]

    @property
    def raw_metadata_dir(self) -> Path:
        return self.raw["metadata_dir"]

    @property
    def raw_zip_dir(self) -> Path:
        return self.raw["zip_dir"]

    @property
    def annotation_dir(self) -> Path:
        return self.interim["annotation_dir"]

    @property
    def hmmer_dir(self) -> Path:
        return self.interim["hmmer_dir"]

    @property
    def hmm_profiles_dir(self) -> Path:
        db_path = self.databases.get("hmm_profiles")
        fallback_path = (PROJECT_ROOT / "data" / "hmm_profiles").resolve()
        if db_path and db_path.exists():
            return db_path
        if fallback_path.exists():
            return fallback_path
        return db_path or fallback_path


class AlignmentThresholds(BaseModel):
    use_trusted_cutoffs: bool = True
    e_value_max: float = Field(default=1e-10, gt=0.0)
    min_coverage: float = Field(default=0.7, ge=0.0, le=1.0)
    min_identity: float = Field(default=0.35, ge=0.0, le=1.0)
    min_bit_score: float = Field(default=80.0, ge=0.0)


class SyntenyThresholds(BaseModel):
    max_intergenic_distance_bp: int = Field(default=5000, gt=0)
    max_divergent_gap_bp: int = Field(default=1500, gt=0)
    min_core_genes_required: int = Field(default=2, gt=0)
    strand_sensitive: bool = False
    contig_edge_margin_bp: int = Field(default=1000, ge=0)


class GeneSystem(BaseModel):
    core_genes: List[str]
    accessory_genes: List[str]


class BiologicalThresholdsConfig(BaseModel):
    alignment: AlignmentThresholds
    synteny: SyntenyThresholds
    gene_systems: Dict[str, GeneSystem]


class SplitStrategyConfig(BaseModel):
    train_ratio: float = Field(ge=0.0, le=1.0)
    validation_ratio: float = Field(ge=0.0, le=1.0)
    test_ratio: float = Field(ge=0.0, le=1.0)
    split_by_taxon: bool
    taxonomy_level: Literal["species", "genus", "family", "order", "class", "phylum"]

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "SplitStrategyConfig":
        total = round(self.train_ratio + self.validation_ratio + self.test_ratio, 5)
        if total != 1.0:
            raise ValueError(f"Tổng tỷ lệ phân chia train/val/test phải bằng 1.0 (Hiện tại: {total})")
        return self


class ModelTrainingConfig(BaseModel):
    cv_folds: int = Field(gt=1)
    imbalance_handling: Literal["None", "ClassWeight", "SMOTE"]
    evaluation_metrics: str


class MachineLearningConfig(BaseModel):
    random_seed: int
    split_strategy: SplitStrategyConfig
    model_training: ModelTrainingConfig


class NCBIConfig(BaseModel):
    api_base_url: str
    api_key_env_var: str
    timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    rate_limit_per_sec: int = Field(gt=0)

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env_var)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: str
    file_path: Path

    @model_validator(mode="after")
    def resolve_file_path(self) -> "LoggingConfig":
        if not self.file_path.is_absolute():
            self.file_path = (PROJECT_ROOT / self.file_path).resolve()
        return self


class AppConfig(BaseModel):
    project: ProjectConfig
    computing: ComputingConfig
    paths: PathsConfig
    ncbi: NCBIConfig
    biological_thresholds: BiologicalThresholdsConfig
    machine_learning: MachineLearningConfig
    logging: LoggingConfig

    def create_directories(self) -> None:
        all_dirs = [
            self.paths.models,
            self.paths.log_dir,
            self.logging.file_path.parent,
            *self.paths.raw.values(),
            *self.paths.interim.values(),
            *self.paths.processed.values(),
            *self.paths.databases.values(),
        ]
        for directory in all_dirs:
            directory.mkdir(parents=True, exist_ok=True)


class EnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NIFPREDICT_", env_file=".env", extra="ignore")

    enable_gpu: Optional[bool] = None
    max_threads: Optional[int] = None
    log_level: Optional[str] = None


@lru_cache(maxsize=1)
def load_config(
    config_path: Optional[Union[str, Path]] = None,
    auto_create_dirs: bool = True,
) -> AppConfig:
    if config_path is None:
        resolved_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        resolved_path = Path(config_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {resolved_path}")

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Lỗi định dạng YAML tại {resolved_path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError(f"File cấu hình tại {resolved_path} phải là Dictionary.")

    env_settings = EnvOverrides()
    if env_settings.enable_gpu is not None:
        raw_data["computing"]["enable_gpu"] = env_settings.enable_gpu
    if env_settings.max_threads is not None:
        raw_data["computing"]["max_threads"] = env_settings.max_threads
    if env_settings.log_level is not None:
        raw_data["logging"]["level"] = env_settings.log_level

    config = AppConfig(**raw_data)
    if auto_create_dirs:
        config.create_directories()

    return config


# Singleton Instance hỗ trợ Import trực tiếp
settings: AppConfig = load_config(auto_create_dirs=False)