"""
scripts/setup_hmm_profiles.py

Quy trình quản lý, kiểm tra toàn vẹn, tự động tải xuống, chỉ mục hóa (hmmpress)
và xuất runtime manifest cho các tệp HMM Profile trong dự án NifPredict.

Author: NifPredict Team
"""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from nifpredict.common import paths

# Tích hợp cấu hình và logger của NifPredict
try:
    from nifpredict.common.config import settings
    from nifpredict.common.logger import logger
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("NifPredict")

    class FallbackSettings:
        HMM_PROFILES_DIR: Path = paths.ML_HMM_PROFILES

    settings = FallbackSettings()


@dataclass(frozen=True)
class HMMProfileSpec:
    """Cấu hình định danh và tiêu chuẩn kỹ thuật cho từng HMM Profile."""

    name: str
    filename: str
    category: str
    is_core: bool = False
    cutoff_flag: str = "-cut_ga"  # Options: '-cut_ga', '-cut_tc'
    download_url: str | None = None
    expected_sha256: str | None = None


# Danh mục HMM Profile chuẩn hóa
HMM_REGISTRY: list[HMMProfileSpec] = [
    # 1. Gen lõi cố định đạm Mo-nitrogenase
    HMMProfileSpec("nifH", "nifH.hmm", "core_nif", is_core=True),
    HMMProfileSpec("nifD", "nifD.hmm", "core_nif", is_core=True),
    HMMProfileSpec("nifK", "nifK.hmm", "core_nif", is_core=True),
    # nifE/nifN/nifB không phải is_core: predictor chỉ quét nifH/D/K,
    # 3 gen này nằm trong accessory_genes của config.yaml, không có URL
    # tải riêng và không có mục trích từ Pfam-A. Để is_core=True thì
    # setup luôn exit 1 sau khi đã trích xong 3 profile cần thiết.
    HMMProfileSpec("nifE", "nifE.hmm", "core_nif"),
    HMMProfileSpec("nifN", "nifN.hmm", "core_nif"),
    HMMProfileSpec("nifB", "nifB.hmm", "core_nif"),
    # 2. Alternative Nitrogenase - V-nitrogenase
    HMMProfileSpec("vnfH", "vnfH.hmm", "alternative_nitrogenase"),
    HMMProfileSpec("vnfD", "vnfD.hmm", "alternative_nitrogenase"),
    HMMProfileSpec("vnfK", "vnfK.hmm", "alternative_nitrogenase"),
    HMMProfileSpec("vnfG", "vnfG.hmm", "alternative_nitrogenase"),
    # 3. Alternative Nitrogenase - Fe-only Nitrogenase
    HMMProfileSpec("anfH", "anfH.hmm", "alternative_nitrogenase"),
    HMMProfileSpec("anfD", "anfD.hmm", "alternative_nitrogenase"),
    HMMProfileSpec("anfK", "anfK.hmm", "alternative_nitrogenase"),
    HMMProfileSpec("anfG", "anfG.hmm", "alternative_nitrogenase"),
    # 4. Hệ thống phụ trợ / Vận chuyển electron
    HMMProfileSpec("fixA", "fixA.hmm", "electron_transport"),
    HMMProfileSpec("fixB", "fixB.hmm", "electron_transport"),
    HMMProfileSpec("fixC", "fixC.hmm", "electron_transport"),
    HMMProfileSpec("fixX", "fixX.hmm", "electron_transport"),
    # 5. Cơ sở dữ liệu Pfam tổng quan
    HMMProfileSpec(
        name="Pfam-A",
        filename="Pfam-A.hmm",
        category="general_pfam",
        is_core=False,
        cutoff_flag="-cut_ga",
        # r1031: ghim một bản phát hành Pfam (thư mục bản mới nhất đổi theo thời gian, không tái lập được);
        # cùng phiên bản với dòng PFAM_A trong data/public/MANIFEST.tsv
        download_url="https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.2/Pfam-A.hmm.gz",
        expected_sha256=None,
    ),
]

HMM_INDEX_EXTENSIONS: set[str] = {".h3f", ".h3i", ".h3m", ".h3p"}


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Xác minh mã SHA-256 Checksum của tệp."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower() == expected_hash.lower()


def download_and_extract_profile(
    spec: HMMProfileSpec, dest_dir: Path, timeout: int = 60
) -> bool:
    """Tải xuống tệp .gz với timeout cố định, kiểm tra SHA-256 và giải nén."""
    if not spec.download_url:
        logger.error(
            f"Tệp {spec.filename} không tồn tại và không có URL cấu hình."
        )
        return False

    target_hmm = dest_dir / spec.filename
    gz_tmp_path = dest_dir / f"{spec.filename}.gz"

    logger.info(
        f"Đang tải {spec.name} từ nguồn: {spec.download_url} (timeout={timeout}s)..."
    )

    try:
        with urllib.request.urlopen(
            spec.download_url, timeout=timeout
        ) as response, open(gz_tmp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        if spec.expected_sha256:
            logger.info(f"Đang kiểm tra SHA-256 cho {gz_tmp_path.name}...")
            if not verify_sha256(gz_tmp_path, spec.expected_sha256):
                logger.error(f"Xác minh SHA-256 thất bại cho {gz_tmp_path.name}!")
                gz_tmp_path.unlink(missing_ok=True)
                return False

        logger.info(f"Đang giải nén {gz_tmp_path.name} -> {target_hmm.name}...")
        with gzip.open(gz_tmp_path, "rb") as f_in, open(
            target_hmm, "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)

        gz_tmp_path.unlink(missing_ok=True)
        logger.info(f"Hoàn tất tải và giải nén {target_hmm.name}")
        return True

    except Exception as err:
        logger.error(f"Lỗi khi tải/giải nén {spec.name}: {str(err)}")
        gz_tmp_path.unlink(missing_ok=True)
        return False


def is_hmm_pressed(hmm_path: Path) -> bool:
    """Kiểm tra tệp .hmm đã có đủ 4 tệp nhị phân VÀ không bị Stale hay chưa."""
    if not hmm_path.is_file():
        return False

    hmm_mtime = hmm_path.stat().st_mtime
    parent_dir = hmm_path.parent
    base_stem = hmm_path.stem

    for ext in HMM_INDEX_EXTENSIONS:
        idx_option1 = parent_dir / f"{hmm_path.name}{ext}"
        idx_option2 = parent_dir / f"{base_stem}{ext}"

        idx_file = idx_option1 if idx_option1.is_file() else idx_option2

        if not idx_file.is_file() or idx_file.stat().st_mtime < hmm_mtime:
            return False

    return True


def run_hmmpress_safe(hmm_path: Path, force: bool = False) -> bool:
    """Thực thi hmmpress an toàn với FileLock.
    
    Mặc định force=False để tôn trọng kiểm tra is_hmm_pressed trong FileLock.
    """
    hmmpress_bin = shutil.which("hmmpress")
    if not hmmpress_bin:
        logger.error("Không tìm thấy 'hmmpress' trong PATH!")
        return False

    lock_file = hmm_path.with_suffix(".lock")

    try:
        with FileLock(str(lock_file), timeout=600):
            # Nếu đã pressed và không yêu cầu ép buộc rebuild -> bỏ qua
            if is_hmm_pressed(hmm_path) and not force:
                logger.info(
                    f"Tệp {hmm_path.name} đã có chỉ mục hợp lệ. Bỏ qua hmmpress."
                )
                return True

            cmd = [hmmpress_bin]
            if force or not is_hmm_pressed(hmm_path):
                cmd.append("-f")
            cmd.append(str(hmm_path.resolve()))

            logger.info(f"Đang thực thi hmmpress cho {hmm_path.name}...")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Chỉ mục hóa thành công: {hmm_path.name}")
            return True

    except Timeout:
        logger.error(f"Timeout chờ Lock cho tệp {hmm_path.name}.")
        return False
    except subprocess.CalledProcessError as err:
        logger.error(f"Lỗi hmmpress {hmm_path.name}:\n{err.stderr.strip()}")
        return False


def generate_manifest(
    ready_profiles: dict[str, HMMProfileSpec], output_dir: Path
) -> Path:
    """Xuất tệp runtime manifest (hmm_manifest.json) phục vụ các worker phía sau."""
    manifest_path = output_dir / "hmm_manifest.json"
    manifest_data = {}

    for name, spec in ready_profiles.items():
        data = asdict(spec)
        data["path"] = str((output_dir / spec.filename).resolve())
        manifest_data[name] = data

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Đã tạo tệp Runtime Manifest tại: {manifest_path.resolve()}")
    return manifest_path


# Accession Pfam của 3 gen lõi mà NifPredictor cần. Các spec nifH/nifD/nifK
# trong HMM_REGISTRY không có download_url riêng: chúng được trích ra từ
# Pfam-A.hmm (xem extract_profile_from_pfam). Trước đây không có bước trích
# nào, nên setup_hmm luôn exit 1 vì thiếu gen lõi, và predictor không bao
# giờ có profile để quét.
CORE_PFAM_ACCESSIONS: dict[str, str] = {
    "nifH": "PF00142",
    "nifD": "PF00148",
    "nifK": "PF02826",
}


def extract_profile_from_pfam(
    pfam_path: Path, accession: str, dest_path: Path
) -> bool:
    """Trích 1 model khỏi Pfam-A.hmm theo accession, ghi ra file .hmm riêng.

    Pfam-A.hmm là file text, mỗi model là 1 khối kết thúc bằng ``//``.
    Đọc theo dòng (không nạp cả file ~2,2 GB vào RAM), giữ khối nào có
    dòng ``ACC`` trùng accession (bỏ hậu tố phiên bản, vd PF00142.22).
    Không cần hmmfetch: pyhmmer đọc trực tiếp file .hmm text.
    """
    wanted = accession.split(".")[0]
    found = False
    try:
        with (
            open(pfam_path, encoding="utf-8", errors="strict") as src,
            open(dest_path, "w", encoding="utf-8") as dst,
        ):
            block: list[str] = []
            block_acc: str | None = None
            for line in src:
                block.append(line)
                if line.startswith("ACC"):
                    parts = line.split()
                    if len(parts) >= 2:
                        block_acc = parts[1].split(".")[0]
                if line.rstrip() == "//":
                    if block_acc == wanted:
                        dst.writelines(block)
                        found = True
                    block = []
                    block_acc = None
    except OSError as err:
        logger.error(f"Không đọc/ghi được khi trích {accession}: {err}")
        return False
    if not found:
        dest_path.unlink(missing_ok=True)
        logger.error(f"Không tìm thấy accession {accession} trong {pfam_path}")
    return found


def setup_hmm_pipeline(
    profiles_dir: Path, force_rebuild: bool = False
) -> dict[str, HMMProfileSpec]:
    """Quy trình tổng thể quản lý, chỉ mục hóa và tạo manifest cho HMM Profiles."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Khởi chạy quy trình HMM Profiles tại: {profiles_dir.resolve()}"
    )

    # Trích 3 profile lõi từ Pfam-A trước vòng lặp chính: Pfam-A đứng cuối
    # registry nhưng nifH/D/K lại đứng đầu, và chúng không có URL riêng.
    pfam_spec = next(
        (s for s in HMM_REGISTRY if s.name == "Pfam-A"), None
    )
    pfam_file = profiles_dir / "Pfam-A.hmm"
    if pfam_spec is not None and not pfam_file.is_file():
        if not download_and_extract_profile(pfam_spec, profiles_dir, timeout=60):
            logger.critical("Không tải được Pfam-A.hmm, không trích được profile lõi")
    if pfam_file.is_file():
        for name, accession in CORE_PFAM_ACCESSIONS.items():
            dest = profiles_dir / f"{name}.hmm"
            if not dest.is_file():
                logger.info(f"Trích {accession} từ Pfam-A.hmm -> {dest.name}")
                extract_profile_from_pfam(pfam_file, accession, dest)

    ready_profiles: dict[str, HMMProfileSpec] = {}
    missing_core: list[str] = []

    for spec in HMM_REGISTRY:
        hmm_file = profiles_dir / spec.filename

        if not hmm_file.is_file():
            if spec.download_url:
                if not download_and_extract_profile(
                    spec, profiles_dir, timeout=60
                ):
                    if spec.is_core:
                        missing_core.append(spec.filename)
                    continue
            else:
                msg = f"Thiếu tệp HMM Profile: {spec.filename} [{spec.category}]"
                if spec.is_core:
                    logger.critical(msg)
                    missing_core.append(spec.filename)
                else:
                    logger.warning(msg)
                continue

        # hmmpress chỉ phục vụ tool CLI của HMMER; không có code nào trong
        # repo gọi hmmsearch/hmmfetch CLI (mọi thứ qua pyhmmer, đọc thẳng
        # file .hmm). Thiếu binary thì bỏ qua chỉ mục thay vì exit 1.
        if shutil.which("hmmpress") is None:
            logger.warning(
                "Không tìm thấy 'hmmpress' trong PATH, bỏ qua chỉ mục "
                f"cho {spec.filename} (pyhmmer không cần)"
            )
        elif not is_hmm_pressed(hmm_file) or force_rebuild:
            logger.warning(
                f"Đang xử lý chỉ mục cho {spec.filename} (force_rebuild={force_rebuild})..."
            )
            if not run_hmmpress_safe(hmm_file, force=force_rebuild):
                if spec.is_core:
                    missing_core.append(spec.filename)
                continue

        ready_profiles[spec.name] = spec

    if missing_core:
        logger.critical(f"Dừng hệ thống: Thiếu các gen lõi: {missing_core}")
        sys.exit(1)

    # Ghi nhận manifest để các runner khác sử dụng
    generate_manifest(ready_profiles, profiles_dir)

    logger.info("Hoàn tất thiết lập HMM Profiles cho NifPredict!")
    return ready_profiles


def main() -> None:
    """Hàm điều khiển chính hỗ trợ cờ CLI."""
    parser = argparse.ArgumentParser(
        description="NifPredict HMM Profile Setup & Management Script"
    )
    parser.add_argument(
        "--rebuild-index",
        "-f",
        action="store_true",
        help="Ép buộc chỉ mục lại toàn bộ các tệp HMM profile.",
    )
    args = parser.parse_args()

    hmm_directory = paths.ML_HMM_PROFILES
    setup_hmm_pipeline(hmm_directory, force_rebuild=args.rebuild_index)


if __name__ == "__main__":
    main()