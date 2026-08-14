"""Acquisition and provenance validation for local HMM-panel inputs."""

from __future__ import annotations

import math
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from ._hmm_panel_files import (
    _coerce_roots,
    _resolve_unique_model_file,
    hmmstat_valid,
    normalized_hmm_sha256,
    parse_hmm_header,
    parse_stockholm,
    sha256_file,
)
from ._hmm_panel_types import (
    DownloadedModelInputs,
    HMMHeader,
    HMMModelSpec,
    HMMPanelAuditError,
    HMMPanelSpec,
    ModelInputAudit,
)


def _download_verified_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    timeout: float,
    overwrite: bool,
) -> bool:
    """Atomically download one file and require its versioned checksum."""
    if not url.startswith("https://"):
        raise HMMPanelAuditError(f"Panel input URL must use HTTPS: {url}")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0.0
    ):
        raise HMMPanelAuditError("Download timeout must be a positive number")

    if destination.is_file():
        observed_sha256 = sha256_file(destination)
        if observed_sha256 == expected_sha256:
            return False
        if not overwrite:
            raise HMMPanelAuditError(
                f"Existing panel input has an unexpected SHA-256: {destination}"
            )
    elif destination.exists():
        raise HMMPanelAuditError(
            f"Panel input destination is not a regular file: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
            delete=False,
        ) as temporary_handle:
            temporary_path = Path(temporary_handle.name)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary_handle)

        observed_sha256 = sha256_file(temporary_path)
        if observed_sha256 != expected_sha256:
            raise HMMPanelAuditError(
                f"Downloaded panel input failed SHA-256 validation: {url}"
            )
        temporary_path.replace(destination)
        temporary_path = None
        return True
    except HMMPanelAuditError:
        raise
    except (OSError, urllib.error.URLError) as error:
        raise HMMPanelAuditError(
            f"Cannot download panel input '{url}': {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_panel_inputs(
    panel: HMMPanelSpec,
    destination_root: Path | str,
    *,
    timeout: float = 60.0,
    overwrite: bool = False,
) -> tuple[DownloadedModelInputs, ...]:
    """Download only the versioned HMM and SEED files declared by the panel."""
    root = Path(destination_root)
    model_directory = root / "models"
    seed_directory = root / "seeds"
    results: list[DownloadedModelInputs] = []

    for model in panel.models:
        hmm_path = model_directory / f"{model.accession}.HMM"
        seed_path = seed_directory / f"{model.accession}.SEED"
        hmm_url = f"{panel.model_base_url.rstrip('/')}/{hmm_path.name}"
        seed_url = f"{panel.seed_base_url.rstrip('/')}/{seed_path.name}"
        hmm_downloaded = _download_verified_file(
            hmm_url,
            hmm_path,
            expected_sha256=model.checksums.hmm_raw_sha256,
            timeout=timeout,
            overwrite=overwrite,
        )
        seed_downloaded = _download_verified_file(
            seed_url,
            seed_path,
            expected_sha256=model.checksums.seed_raw_sha256,
            timeout=timeout,
            overwrite=overwrite,
        )
        results.append(
            DownloadedModelInputs(
                role=model.role,
                accession=model.accession,
                hmm_path=hmm_path,
                seed_path=seed_path,
                hmm_downloaded=hmm_downloaded,
                seed_downloaded=seed_downloaded,
            )
        )
    return tuple(results)


def _compare_header(header: HMMHeader, model: HMMModelSpec) -> list[str]:
    errors: list[str] = []
    expected_trusted_cutoff = (model.sequence_cutoff, model.domain_cutoff)
    expected_noise_cutoff = (model.noise_cutoff, model.noise_cutoff)
    checks = (
        (header.name, model.hmm_name, "NAME"),
        (header.accession, model.accession, "ACC"),
        (header.length, model.hmm_length, "LENG"),
        (header.sequence_count, model.seed_sequence_count, "NSEQ"),
        (header.gathering_cutoff, expected_trusted_cutoff, "GA"),
        (header.trusted_cutoff, expected_trusted_cutoff, "TC"),
        (header.noise_cutoff, expected_noise_cutoff, "NC"),
    )
    for observed, expected, field in checks:
        if observed != expected:
            errors.append(
                f"{field} mismatch: expected {expected!r}, observed {observed!r}"
            )
    return errors


def audit_panel_inputs(
    panel: HMMPanelSpec,
    *,
    hmm_roots: Path | str | Iterable[Path | str],
    seed_roots: Path | str | Iterable[Path | str],
    check_normalized: bool = True,
    check_hmmstat: bool = True,
) -> tuple[ModelInputAudit, ...]:
    """Validate all local HMM and SEED files against versioned provenance."""
    hmm_directories = _coerce_roots(hmm_roots)
    seed_directories = _coerce_roots(seed_roots)
    audits: list[ModelInputAudit] = []

    for model in panel.models:
        hmm_path = _resolve_unique_model_file(hmm_directories, model.accession, "HMM")
        seed_path = _resolve_unique_model_file(
            seed_directories, model.accession, "SEED"
        )
        errors: list[str] = []

        raw_hmm_digest = sha256_file(hmm_path)
        if raw_hmm_digest != model.checksums.hmm_raw_sha256:
            errors.append("Raw HMM SHA-256 mismatch")
        raw_seed_digest = sha256_file(seed_path)
        if raw_seed_digest != model.checksums.seed_raw_sha256:
            errors.append("Raw SEED SHA-256 mismatch")

        header = parse_hmm_header(hmm_path)
        errors.extend(_compare_header(header, model))
        records = parse_stockholm(seed_path)
        if len(records) != model.seed_sequence_count:
            errors.append(
                "SEED count mismatch: expected "
                f"{model.seed_sequence_count}, observed {len(records)}"
            )

        normalized_digest: str | None = None
        if check_normalized:
            normalized_digest = normalized_hmm_sha256(hmm_path)
            if normalized_digest != model.checksums.hmmconvert_sha256:
                errors.append("Normalized HMM SHA-256 mismatch")

        validity: bool | None = None
        if check_hmmstat:
            validity = hmmstat_valid(hmm_path)
            if not validity:
                errors.append("hmmstat rejected the HMM")

        audits.append(
            ModelInputAudit(
                role=model.role,
                accession=model.accession,
                hmm_path=hmm_path,
                seed_path=seed_path,
                raw_hmm_sha256=raw_hmm_digest,
                normalized_hmm_sha256=normalized_digest,
                raw_seed_sha256=raw_seed_digest,
                seed_sequence_count=len(records),
                hmmstat_valid=validity,
                errors=tuple(errors),
            )
        )
    return tuple(audits)
