"""Reproducible validation utilities for the curated nitrogenase HMM panel."""

from __future__ import annotations

import gzip
import hashlib
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import yaml

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HMM_HEADER_FIELDS = {"NAME", "ACC", "DESC", "LENG", "NSEQ", "GA", "TC", "NC"}
FASTA_SUFFIXES = (
    "_protein.faa.gz",
    "_protein.faa",
    ".faa.gz",
    ".faa",
)


class HMMPanelAuditError(ValueError):
    """Raised when panel provenance or an audited biological file is invalid."""


class ExternalHMMToolError(RuntimeError):
    """Raised when a required HMMER executable cannot validate a model."""


@dataclass(frozen=True)
class ModelChecksums:
    """Expected SHA-256 digests for one model and its seed alignment."""

    hmm_raw_sha256: str
    hmmconvert_sha256: str
    seed_raw_sha256: str


@dataclass(frozen=True)
class HMMModelSpec:
    """Validated model metadata loaded from the versioned panel manifest."""

    role: str
    accession: str
    source_identifier: str
    hmm_name: str
    gene_symbols: tuple[str, ...]
    hmm_length: int
    sequence_cutoff: float
    domain_cutoff: float
    noise_cutoff: float
    family_type: str
    product: str
    seed_sequence_count: int
    checksums: ModelChecksums


@dataclass(frozen=True)
class HMMPanelSpec:
    """Panel-level provenance and its ordered collection of models."""

    schema_version: int
    panel_id: str
    provider: str
    release_name: str
    release_date: str
    registry_url: str
    model_base_url: str
    seed_base_url: str
    cutoff_policy: str
    normalized_with: str
    models: tuple[HMMModelSpec, ...]


@dataclass(frozen=True)
class StockholmRecord:
    """One aligned and ungapped sequence parsed from a Stockholm file."""

    identifier: str
    aligned_sequence: str
    sequence: str


@dataclass(frozen=True)
class HMMHeader:
    """Relevant metadata parsed from the text header of an HMMER model."""

    name: str
    accession: str
    description: str
    length: int
    sequence_count: int
    gathering_cutoff: tuple[float, float]
    trusted_cutoff: tuple[float, float]
    noise_cutoff: tuple[float, float]


@dataclass(frozen=True)
class ModelInputAudit:
    """Observed validation result for one HMM and its seed alignment."""

    role: str
    accession: str
    hmm_path: Path
    seed_path: Path
    raw_hmm_sha256: str
    normalized_hmm_sha256: str | None
    raw_seed_sha256: str
    seed_sequence_count: int
    hmmstat_valid: bool | None
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Return whether every enabled validation check passed."""
        return not self.errors


@dataclass(frozen=True)
class FastaRecord:
    """A protein FASTA record with its source file."""

    source: Path
    identifier: str
    sequence: str


@dataclass(frozen=True)
class SeedReference:
    """Provenance for one ungapped seed sequence."""

    role: str
    model_accession: str
    seed_identifier: str
    sequence_sha256: str


@dataclass(frozen=True)
class ExactSeedMatch:
    """An exact local-protein match to one or more panel seed sequences."""

    accession: str
    protein_id: str
    protein_source: Path
    sequence_sha256: str
    seed_references: tuple[SeedReference, ...]


@dataclass(frozen=True)
class ExactLeakageAudit:
    """Summary of exact sequence overlap between evaluation proteins and seeds."""

    total_seed_sequences: int
    unique_seed_hashes: int
    cross_model_duplicate_hashes: int
    local_proteins_scanned: int
    matches: tuple[ExactSeedMatch, ...]


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


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def sha256_file(path: Path | str) -> str:
    """Calculate the SHA-256 digest of a file without loading it into memory."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    """Hash an uppercase ungapped biological sequence deterministically."""
    return hashlib.sha256(sequence.upper().encode("ascii")).hexdigest()


def ungap_stockholm_sequence(aligned_sequence: str) -> str:
    """Normalize a Stockholm protein sequence for identity-based comparison."""
    sequence = "".join(
        character for character in aligned_sequence if character not in {"-", ".", "~"}
    ).upper()
    if not sequence:
        raise HMMPanelAuditError("Stockholm sequence becomes empty after ungapping")
    if not sequence.isalpha():
        raise HMMPanelAuditError(
            "Stockholm sequence contains unsupported non-alphabetic residues"
        )
    return sequence


def parse_stockholm(path: Path | str) -> tuple[StockholmRecord, ...]:
    """Parse sequential or interleaved Stockholm 1.0 alignments safely."""
    stockholm_path = Path(path)
    fragments: dict[str, list[str]] = {}
    saw_header = False
    saw_terminator = False

    try:
        with _open_text(stockholm_path) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if not saw_header:
                    if line != "# STOCKHOLM 1.0":
                        raise HMMPanelAuditError(
                            f"'{stockholm_path}' is not Stockholm 1.0"
                        )
                    saw_header = True
                    continue
                if line == "//":
                    saw_terminator = True
                    break
                if line.startswith("#"):
                    continue

                fields = line.split()
                if len(fields) != 2:
                    raise HMMPanelAuditError(
                        f"Malformed Stockholm sequence row at "
                        f"{stockholm_path}:{line_number}"
                    )
                identifier, fragment = fields
                fragments.setdefault(identifier, []).append(fragment)
    except OSError as error:
        raise HMMPanelAuditError(
            f"Cannot read Stockholm file '{stockholm_path}': {error}"
        ) from error

    if not saw_header or not saw_terminator:
        raise HMMPanelAuditError(f"Stockholm file '{stockholm_path}' is incomplete")
    if not fragments:
        raise HMMPanelAuditError(
            f"Stockholm file '{stockholm_path}' contains no sequences"
        )

    records = tuple(
        StockholmRecord(
            identifier=identifier,
            aligned_sequence="".join(parts),
            sequence=ungap_stockholm_sequence("".join(parts)),
        )
        for identifier, parts in fragments.items()
    )
    aligned_lengths = {len(record.aligned_sequence) for record in records}
    if len(aligned_lengths) != 1:
        raise HMMPanelAuditError(
            f"Stockholm file '{stockholm_path}' has inconsistent aligned lengths"
        )
    return records


def _parse_cutoff(value: str, field: str, path: Path) -> tuple[float, float]:
    tokens = value.split()
    if len(tokens) < 2:
        raise HMMPanelAuditError(f"HMM '{path}' has malformed {field} cutoff")
    try:
        return float(tokens[0]), float(tokens[1])
    except ValueError as error:
        raise HMMPanelAuditError(
            f"HMM '{path}' has non-numeric {field} cutoff"
        ) from error


def parse_hmm_header(path: Path | str) -> HMMHeader:
    """Read provenance and cutoff fields from an HMMER text model."""
    hmm_path = Path(path)
    observed: dict[str, str] = {}
    try:
        with _open_text(hmm_path) as handle:
            first_line = handle.readline().strip()
            if not first_line.startswith("HMMER3/"):
                raise HMMPanelAuditError(f"'{hmm_path}' is not an HMMER3 text model")
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.startswith("HMM "):
                    break
                fields = line.split(maxsplit=1)
                if len(fields) == 2 and fields[0] in HMM_HEADER_FIELDS:
                    observed[fields[0]] = fields[1].strip()
    except OSError as error:
        raise HMMPanelAuditError(f"Cannot read HMM '{hmm_path}': {error}") from error

    missing = sorted(HMM_HEADER_FIELDS - set(observed))
    if missing:
        raise HMMPanelAuditError(
            f"HMM '{hmm_path}' is missing header field(s): {', '.join(missing)}"
        )
    try:
        length = int(observed["LENG"])
        sequence_count = int(observed["NSEQ"])
    except ValueError as error:
        raise HMMPanelAuditError(
            f"HMM '{hmm_path}' has invalid LENG or NSEQ"
        ) from error

    return HMMHeader(
        name=observed["NAME"],
        accession=observed["ACC"],
        description=observed["DESC"],
        length=length,
        sequence_count=sequence_count,
        gathering_cutoff=_parse_cutoff(observed["GA"], "GA", hmm_path),
        trusted_cutoff=_parse_cutoff(observed["TC"], "TC", hmm_path),
        noise_cutoff=_parse_cutoff(observed["NC"], "NC", hmm_path),
    )


def normalized_hmm_sha256(path: Path | str, *, executable: str = "hmmconvert") -> str:
    """Hash the normalized HMMER text emitted by ``hmmconvert``."""
    hmm_path = Path(path)
    try:
        result = subprocess.run(
            [executable, str(hmm_path)],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ExternalHMMToolError(f"Cannot execute '{executable}': {error}") from error
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExternalHMMToolError(f"'{executable}' failed for '{hmm_path}': {message}")
    return hashlib.sha256(result.stdout).hexdigest()


def hmmstat_valid(path: Path | str, *, executable: str = "hmmstat") -> bool:
    """Return whether ``hmmstat`` accepts an HMM without errors."""
    hmm_path = Path(path)
    try:
        result = subprocess.run(
            [executable, str(hmm_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ExternalHMMToolError(f"Cannot execute '{executable}': {error}") from error
    return result.returncode == 0


def _coerce_roots(roots: Path | str | Iterable[Path | str]) -> tuple[Path, ...]:
    if isinstance(roots, (str, Path)):
        result = (Path(roots),)
    else:
        result = tuple(Path(root) for root in roots)
    if not result:
        raise HMMPanelAuditError("At least one input root is required")
    missing = [str(root) for root in result if not root.is_dir()]
    if missing:
        raise HMMPanelAuditError("Input root(s) do not exist: " + ", ".join(missing))
    return result


def _resolve_unique_model_file(
    roots: Sequence[Path], accession: str, suffix: str
) -> Path:
    filename = f"{accession}.{suffix}"
    matches = sorted(
        {
            candidate.resolve()
            for root in roots
            for candidate in root.rglob(filename)
            if candidate.is_file()
        }
    )
    if not matches:
        raise HMMPanelAuditError(f"Cannot locate required file '{filename}'")
    if len(matches) > 1:
        raise HMMPanelAuditError(
            f"Required file '{filename}' is ambiguous: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


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


def iter_fasta_records(paths: Iterable[Path | str]) -> Iterator[FastaRecord]:
    """Stream validated protein FASTA records from plain or gzip files."""
    for raw_path in paths:
        path = Path(raw_path)
        identifier: str | None = None
        sequence_parts: list[str] = []
        with _open_text(path) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if identifier is not None:
                        sequence = "".join(sequence_parts).upper()
                        if not sequence or not sequence.isalpha():
                            raise HMMPanelAuditError(
                                f"Invalid FASTA sequence '{identifier}' in '{path}'"
                            )
                        yield FastaRecord(path, identifier, sequence)
                    header = line[1:].strip()
                    if not header:
                        raise HMMPanelAuditError(
                            f"Empty FASTA header at {path}:{line_number}"
                        )
                    identifier = header.split()[0]
                    sequence_parts = []
                else:
                    if identifier is None:
                        raise HMMPanelAuditError(
                            f"Sequence precedes FASTA header at {path}:{line_number}"
                        )
                    sequence_parts.append(line)

        if identifier is not None:
            sequence = "".join(sequence_parts).upper()
            if not sequence or not sequence.isalpha():
                raise HMMPanelAuditError(
                    f"Invalid FASTA sequence '{identifier}' in '{path}'"
                )
            yield FastaRecord(path, identifier, sequence)


def accession_from_fasta_path(path: Path | str) -> str:
    """Infer the assembly accession from a standardized protein FASTA name."""
    name = Path(path).name
    for suffix in FASTA_SUFFIXES:
        if name.endswith(suffix):
            accession = name[: -len(suffix)]
            if accession:
                return accession
    raise HMMPanelAuditError(
        f"Cannot infer accession from non-standard FASTA filename '{name}'"
    )


def build_seed_hash_index(
    panel: HMMPanelSpec,
    seed_roots: Path | str | Iterable[Path | str],
) -> dict[str, tuple[SeedReference, ...]]:
    """Index every ungapped panel seed by its exact sequence digest."""
    roots = _coerce_roots(seed_roots)
    index: dict[str, list[SeedReference]] = defaultdict(list)
    for model in panel.models:
        seed_path = _resolve_unique_model_file(roots, model.accession, "SEED")
        for record in parse_stockholm(seed_path):
            digest = sequence_sha256(record.sequence)
            index[digest].append(
                SeedReference(
                    role=model.role,
                    model_accession=model.accession,
                    seed_identifier=record.identifier,
                    sequence_sha256=digest,
                )
            )
    return {digest: tuple(references) for digest, references in index.items()}


def audit_exact_seed_leakage(
    panel: HMMPanelSpec,
    *,
    seed_roots: Path | str | Iterable[Path | str],
    protein_fastas: Iterable[Path | str],
) -> ExactLeakageAudit:
    """Audit exact sequence overlap without retaining all local proteins in memory."""
    seed_index = build_seed_hash_index(panel, seed_roots)
    matches: list[ExactSeedMatch] = []
    local_count = 0

    for protein in iter_fasta_records(protein_fastas):
        local_count += 1
        digest = sequence_sha256(protein.sequence)
        references = seed_index.get(digest)
        if references:
            matches.append(
                ExactSeedMatch(
                    accession=accession_from_fasta_path(protein.source),
                    protein_id=protein.identifier,
                    protein_source=protein.source,
                    sequence_sha256=digest,
                    seed_references=references,
                )
            )

    total_seed_sequences = sum(len(references) for references in seed_index.values())
    cross_model_duplicates = sum(
        len({reference.role for reference in references}) > 1
        for references in seed_index.values()
    )
    return ExactLeakageAudit(
        total_seed_sequences=total_seed_sequences,
        unique_seed_hashes=len(seed_index),
        cross_model_duplicate_hashes=cross_model_duplicates,
        local_proteins_scanned=local_count,
        matches=tuple(matches),
    )


__all__ = [
    "ExactLeakageAudit",
    "ExactSeedMatch",
    "ExternalHMMToolError",
    "FastaRecord",
    "HMMHeader",
    "HMMModelSpec",
    "HMMPanelAuditError",
    "HMMPanelSpec",
    "ModelChecksums",
    "ModelInputAudit",
    "SeedReference",
    "StockholmRecord",
    "accession_from_fasta_path",
    "audit_exact_seed_leakage",
    "audit_panel_inputs",
    "build_seed_hash_index",
    "hmmstat_valid",
    "iter_fasta_records",
    "load_panel_spec",
    "normalized_hmm_sha256",
    "parse_hmm_header",
    "parse_stockholm",
    "sequence_sha256",
    "sha256_file",
    "ungap_stockholm_sequence",
]
