"""
Module: nifpredict.utils.logger
Description: Hệ thống ghi log tập trung cho dự án NifPredict.
             Hỗ trợ Dual-Handler (Console + RotatingFile), ANSI Color Formatter,
             An toàn cho Multiprocessing (QueueHandler/QueueListener),
             và cấu hình linh hoạt qua biến môi trường NIFPREDICT_LOG_LEVEL.
"""

import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

# Format chuẩn hóa chứa đầy đủ ngữ cảnh cho Debug & Multiprocessing
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | PID:%(process)-6d | "
    "%(filename)s:%(lineno)d (%(funcName)s) | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredConsoleFormatter(logging.Formatter):
    """Custom Formatter hỗ trợ ANSI color cho Output Console/Terminal."""

    COLOR_CODES = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m"  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_CODES.get(record.levelname, self.RESET)
        original_msg = super().format(record)
        return f"{color}{original_msg}{self.RESET}"


def setup_logger_from_config(
    name: str = "nifpredict",
    log_file: Optional[Union[Path, str]] = None,
    level: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # Mặc định 10 MB per file
    backup_count: int = 5,              # Tối đa 5 file backup (Tổng ~50MB)
    propagate: bool = False,
    force_reconfig: bool = False,
) -> logging.Logger:
    """Khởi tạo và cấu hình Logger tập trung cho dự án NifPredict.

    Args:
        name: Tên logger (Khuyên dùng __name__ hoặc namespace 'nifpredict').
        log_file: Đường dẫn file log (ví dụ: 'logs/nifpredict.log').
        level: Cấp độ ghi log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Nếu để None sẽ ưu tiên đọc từ biến môi trường NIFPREDICT_LOG_LEVEL.
        max_bytes: Dung lượng tối đa mỗi file log trước khi xoay vòng.
        backup_count: Số lượng file log lưu trữ lại.
        propagate: Lan truyền log lên logger root hay không (Default: False).
        force_reconfig: Ép buộc xóa handlers cũ và cấu hình lại từ đầu.

    Returns:
        logging.Logger: Instance logger đã hoàn tất cấu hình.
    """
    logger = logging.getLogger(name)

    # 1. Xác định Log Level (Thứ tự ưu tiên: Parameter > Environment Var > Default INFO)
    env_level = level or os.getenv("NIFPREDICT_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, env_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = propagate

    # 2. Kiểm tra nếu đã được cấu hình và không yêu cầu reconfig
    if logger.hasHandlers() and not force_reconfig:
        return logger

    # Xóa sạch handlers cũ nếu yêu cầu cấu hình lại
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. Console Handler (Truyền ra stdout kèm màu sắc)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredConsoleFormatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)
    )
    logger.addHandler(console_handler)

    # 4. Rotating File Handler (Ghi file xoay vòng an toàn chống tràn đĩa)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Utility nhận logger chuẩn hóa theo module (Khuyên dùng get_logger(__name__)).

    Args:
        name: Tên của module gọi logger (thường là __name__).

    Returns:
        logging.Logger: Logger tương ứng với namespace.
    """
    return logging.getLogger(name)


def setup_worker_queue_logger(
    queue: Any,
    level: Optional[str] = None
) -> logging.Logger:
    """Cấu hình logger cho các Worker process trong môi trường Multiprocessing.
    Gửi toàn bộ log qua QueueHandler để Main Process xử lý bằng QueueListener,
    tránh xung đột I/O lock trên tệp tin.

    Args:
        queue: multiprocessing.Queue dùng để đẩy log.
        level: Cấp độ ghi log (Tùy chọn, mặc định lấy theo NIFPREDICT_LOG_LEVEL).

    Returns:
        logging.Logger: Worker logger an toàn đa tiến trình.
    """
    logger = logging.getLogger("nifpredict")
    logger.handlers.clear()
    logger.addHandler(QueueHandler(queue))

    env_level = level or os.getenv("NIFPREDICT_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, env_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    logger.propagate = False
    return logger


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """Utility helper hỗ trợ ghi nhận log lỗi kèm trọn vẹn Stack Trace cho Debug.

    Args:
        logger: Logger instance đang sử dụng.
        message: Thông báo ngữ cảnh về lỗi.
        exc: Exception object bắt được.
    """
    logger.error(f"{message}: {str(exc)}", exc_info=True)