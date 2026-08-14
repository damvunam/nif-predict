"""Low-level parsers, hashing, discovery, and HMMER validation helpers."""

from __future__ import annotations

import gzip
import hashlib
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TextIO

from ._hmm_panel_types import (
    ExternalHMMToolError,
    FastaRecord,
    HMMHeader,
    HMMPanelAuditError,
    StockholmRecord,
)

HMM_HEADER_FIELDS = {"NAME", "ACC", "DESC", "LENG", "NSEQ", "GA", "TC", "NC"}
FASTA_SUFFIXES = (
    "_protein.faa.gz",
    "_protein.faa",
    ".faa.gz",
    ".faa",
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
