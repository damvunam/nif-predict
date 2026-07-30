from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import yaml

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Định vị Project Root độc lập hoàn toàn với CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# Sub-models
class ProjectConfig(BaseModel):         # luu thong tin dinh danh
    name: str
    version: str
    environment: Literal["development", "testing", "production"]
    description: str


class ComputingConfig(BaseModel):        # quan ly tai nguyen phan cung
    max_threads: int = Field(gt=0, description="Số CPU threads phải > 0")
    chunk_size: int = Field(gt=0)
    enable_gpu: bool


class PathsConfig(BaseModel):           # quan ly toan bo cau truc thu muc
    base_dir: Optional[Path] = Field(default_factory=lambda: PROJECT_ROOT)
    raw: Dict[str, Path]
    interim: Dict[str, Path]
    processed: Dict[str, Path]
    databases: Dict[str, Path]
    models: Path
    log_dir: Path

    @model_validator(mode="after")   
    def resolve_absolute_paths(self) -> "PathsConfig":
        """Chuyển đổi toàn bộ đường dẫn tương đối thành tuyệt đối dựa trên base_dir."""
        base = self.base_dir.resolve()

        def _to_abs(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        self.models = _to_abs(self.models)
        self.log_dir = _to_abs(self.log_dir)

        for category in [self.raw, self.interim, self.processed, self.databases]:
            for key, path_val in category.items():
                category[key] = _to_abs(path_val)
        return self

# cac nguong~ sinh hoc
class AlignmentThresholds(BaseModel):       # nguong ve alignment
    use_trusted_cutoffs: bool
    e_value_max: float = Field(gt=0.0)
    min_coverage: float = Field(ge=0.0, le=1.0)
    min_identity: float = Field(ge=0.0, le=1.0)
    min_bit_score: float = Field(ge=0.0)


class SyntenyThresholds(BaseModel):         # nguong ve synteny
    max_intergenic_distance_bp: int = Field(gt=0)
    min_core_genes_required: int = Field(gt=0)
    strand_sensitive: bool


class GeneSystem(BaseModel):
    core_genes: List[str]
    accessory_genes: List[str]


class BiologicalThresholdsConfig(BaseModel):
    alignment: AlignmentThresholds
    synteny: SyntenyThresholds
    gene_systems: Dict[str, GeneSystem]


# ML
class SplitStrategyConfig(BaseModel):               # chia tep train/val/test theo taxonomy
    train_ratio: float = Field(ge=0.0, le=1.0)
    validation_ratio: float = Field(ge=0.0, le=1.0)
    test_ratio: float = Field(ge=0.0, le=1.0)
    split_by_taxon: bool
    taxonomy_level: Literal["species", "genus", "family", "order", "class", "phylum"]

    @model_validator(mode="after")                  # tu dong cong tong
    def validate_split_ratios(self) -> "SplitStrategyConfig":
        total = round(self.train_ratio + self.validation_ratio + self.test_ratio, 5)
        if total != 1.0:
            raise ValueError(f"Tổng tỷ lệ phân chia train/val/test phải bằng 1.0 (Hiện tại: {total})")
        return self


class ModelTrainingConfig(BaseModel):               # quan ly cac tham so khi train model
    cv_folds: int = Field(gt=1)
    imbalance_handling: Literal["None", "ClassWeight", "SMOTE"]
    evaluation_metrics: str


class MachineLearningConfig(BaseModel):             
    random_seed: int
    split_strategy: SplitStrategyConfig
    model_training: ModelTrainingConfig


# ncbi
class NCBIConfig(BaseModel):            # quan ly tham so ket noi NCBI API v2
    api_base_url: str
    api_key_env_var: str
    timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    rate_limit_per_sec: int = Field(gt=0)

    @property           # doc api_key sach
    def api_key(self) -> Optional[str]:
        """Đọc NCBI API key từ môi trường runtime."""
        return os.getenv(self.api_key_env_var)


class LoggingConfig(BaseModel):         # quan ly log
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: str
    file_path: Path

    @model_validator(mode="after")      
    def resolve_file_path(self) -> "LoggingConfig":
        """Đảm bảo file_path được resolve thành đường dẫn tuyệt đối theo PROJECT_ROOT."""
        if not self.file_path.is_absolute():
            self.file_path = (PROJECT_ROOT / self.file_path).resolve()
        return self


# Root Configuration Class
class AppConfig(BaseModel):
    """Root model quản lý toàn bộ cấu hình dự án NifPredict."""

    project: ProjectConfig
    computing: ComputingConfig
    paths: PathsConfig
    ncbi: NCBIConfig
    biological_thresholds: BiologicalThresholdsConfig
    machine_learning: MachineLearningConfig
    logging: LoggingConfig

    def create_directories(self) -> None:
        """Tự động tạo mới hạ tầng thư mục làm việc nếu chưa tồn tại."""
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


# Environment Overrides
class EnvOverrides(BaseSettings):
    """Cấu hình đè từ môi trường runtime (Tiền tố: NIF_)."""

    model_config = SettingsConfigDict(env_prefix="NIFPREDICT_", env_file=".env", extra="ignore")

    enable_gpu: Optional[bool] = None
    max_threads: Optional[int] = None
    log_level: Optional[str] = None


# Loader API
@lru_cache(maxsize=1)
def load_config(
    config_path: Optional[Union[str, Path]] = None,
    auto_create_dirs: bool = True,
) -> AppConfig:
    """Tải, validate, đè biến môi trường và cache cấu hình ứng dụng.

    Args:
        config_path: Đường dẫn tới file config YAML tùy chỉnh. Nếu None sẽ dùng default.
        auto_create_dirs: Khởi tạo thư mục tự động trên đĩa (Đặt False khi test).

    Returns:
        AppConfig: Cấu hình ứng dụng đã được validate hoàn chỉnh.
    """
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

    # Đè tham số từ biến môi trường nếu có
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