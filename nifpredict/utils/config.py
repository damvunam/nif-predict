from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: Path | str = "config/config.yaml") -> Dict[str, Any]:
    """
    Tải và xác thực an toàn cấu hình YAML dự án.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None or not isinstance(config, dict):
        raise ValueError(f"File cấu hình tại {path.resolve()} không hợp lệ hoặc rỗng.")

    return config
