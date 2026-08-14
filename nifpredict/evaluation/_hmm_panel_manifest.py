"""Strict loading and validation of versioned HMM-panel manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

from ._hmm_panel_types import (
    HMMModelSpec,
    HMMPanelAuditError,
    HMMPanelSpec,
    ModelChecksums,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HMMPanelAuditError(f"{location} must be a mapping")
    return value


def _require_string(mapping: Mapping[str, object], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HMMPanelAuditError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _require_integer(mapping: Mapping[str, object], key: str, location: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HMMPanelAuditError(f"{location}.{key} must be an integer")
    return value


def _require_float(mapping: Mapping[str, object], key: str, location: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HMMPanelAuditError(f"{location}.{key} must be numeric")
    return float(value)


def _validate_sha256(value: str, location: str) -> str:
    digest = value.lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise HMMPanelAuditError(f"{location} must be a 64-character SHA-256")
    return digest


def load_panel_spec(path: Path | str) -> HMMPanelSpec:
    """Load and strictly validate a versioned HMM-panel YAML manifest."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HMMPanelAuditError(
            f"Cannot read panel manifest '{config_path}': {error}"
        ) from error
    except yaml.YAMLError as error:
        raise HMMPanelAuditError(
            f"Panel manifest '{config_path}' is invalid YAML: {error}"
        ) from error

    root = _require_mapping(raw, "manifest")
    schema_version = _require_integer(root, "schema_version", "manifest")
    if schema_version != 1:
        raise HMMPanelAuditError(
            f"Unsupported panel schema_version={schema_version}; expected 1"
        )

    panel = _require_mapping(root.get("panel"), "manifest.panel")
    raw_models = root.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise HMMPanelAuditError("manifest.models must be a non-empty list")

    models: list[HMMModelSpec] = []
    for index, raw_model in enumerate(raw_models):
        location = f"manifest.models[{index}]"
        model = _require_mapping(raw_model, location)
        raw_symbols = model.get("gene_symbols")
        if (
            not isinstance(raw_symbols, list)
            or not raw_symbols
            or any(not isinstance(symbol, str) or not symbol for symbol in raw_symbols)
        ):
            raise HMMPanelAuditError(
                f"{location}.gene_symbols must be a non-empty string list"
            )

        checksum_mapping = _require_mapping(
            model.get("checksums"), f"{location}.checksums"
        )
        checksum_location = f"{location}.checksums"
        checksums = ModelChecksums(
            hmm_raw_sha256=_validate_sha256(
                _require_string(checksum_mapping, "hmm_raw_sha256", checksum_location),
                f"{checksum_location}.hmm_raw_sha256",
            ),
            hmmconvert_sha256=_validate_sha256(
                _require_string(
                    checksum_mapping, "hmmconvert_sha256", checksum_location
                ),
                f"{checksum_location}.hmmconvert_sha256",
            ),
            seed_raw_sha256=_validate_sha256(
                _require_string(checksum_mapping, "seed_raw_sha256", checksum_location),
                f"{checksum_location}.seed_raw_sha256",
            ),
        )
        hmm_length = _require_integer(model, "hmm_length", location)
        seed_count = _require_integer(model, "seed_sequence_count", location)
        if hmm_length < 1 or seed_count < 1:
            raise HMMPanelAuditError(
                f"{location} lengths and sequence counts must be positive"
            )

        models.append(
            HMMModelSpec(
                role=_require_string(model, "role", location),
                accession=_require_string(model, "accession", location),
                source_identifier=_require_string(model, "source_identifier", location),
                hmm_name=_require_string(model, "hmm_name", location),
                gene_symbols=tuple(raw_symbols),
                hmm_length=hmm_length,
                sequence_cutoff=_require_float(model, "sequence_cutoff", location),
                domain_cutoff=_require_float(model, "domain_cutoff", location),
                noise_cutoff=_require_float(model, "noise_cutoff", location),
                family_type=_require_string(model, "family_type", location),
                product=_require_string(model, "product", location),
                seed_sequence_count=seed_count,
                checksums=checksums,
            )
        )

    roles = [model.role for model in models]
    accessions = [model.accession for model in models]
    if len(set(roles)) != len(roles):
        raise HMMPanelAuditError("manifest.models contains duplicate roles")
    if len(set(accessions)) != len(accessions):
        raise HMMPanelAuditError("manifest.models contains duplicate accessions")

    cutoff_policy = _require_string(panel, "cutoff_policy", "manifest.panel")
    if cutoff_policy != "trusted":
        raise HMMPanelAuditError(
            f"Unsupported cutoff_policy='{cutoff_policy}'; expected 'trusted'"
        )

    return HMMPanelSpec(
        schema_version=schema_version,
        panel_id=_require_string(panel, "id", "manifest.panel"),
        provider=_require_string(panel, "provider", "manifest.panel"),
        release_name=_require_string(panel, "release_name", "manifest.panel"),
        release_date=_require_string(panel, "release_date", "manifest.panel"),
        registry_url=_require_string(panel, "registry_url", "manifest.panel"),
        model_base_url=_require_string(panel, "model_base_url", "manifest.panel"),
        seed_base_url=_require_string(panel, "seed_base_url", "manifest.panel"),
        cutoff_policy=cutoff_policy,
        normalized_with=_require_string(panel, "normalized_with", "manifest.panel"),
        models=tuple(models),
    )
