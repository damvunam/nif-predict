"""Input discovery and annotation preparation for feature matrix builds."""

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from nifpredict.utils.logger import get_logger


LOGGER = get_logger("nifpredict.workflows.feature_matrix.inputs")
ACCESSION_PATTERN = re.compile(r"GC[AF]_\d+\.\d+")
PathMap = Dict[str, Path]
PathMaps = Tuple[PathMap, PathMap, PathMap]


def load_accessions(input_file: Path) -> List[str]:
    """Load accessions while ignoring blank lines and comment lines."""
    try:
        with input_file.open("r", encoding="utf-8") as handle:
            accessions = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Accession file not found: {input_file}") from exc
    except OSError as exc:
        raise OSError(f"Could not read accession file '{input_file}': {exc}") from exc

    if not accessions:
        raise ValueError(f"No accessions found in {input_file}")

    invalid = [item for item in accessions if not ACCESSION_PATTERN.fullmatch(item)]
    if invalid:
        raise ValueError(
            "Invalid NCBI assembly accession(s): " + ", ".join(sorted(set(invalid)))
        )
    return accessions


def _extract_accession(path: Path) -> Optional[str]:
    match = ACCESSION_PATTERN.search(str(path))
    return match.group(0) if match else None


def _candidate_score(
    path: Path,
    file_type: str,
    root_priority: int,
) -> Tuple[int, int, str]:
    canonical_markers = {
        "faa": ("_protein.faa",),
        "gff": ("_genomic.gff", "_genomic.gff3"),
        "fna": ("_genomic.fna",),
    }
    name = path.name.lower()
    is_canonical = any(
        name.endswith(marker) for marker in canonical_markers[file_type]
    )
    return root_priority, 0 if is_canonical else 1, str(path)


def _select_candidate(
    path_map: PathMap,
    score_map: Dict[str, Tuple[int, int, str]],
    accession: str,
    path: Path,
    file_type: str,
    root_priority: int,
) -> None:
    score = _candidate_score(path, file_type, root_priority)
    if accession not in score_map or score < score_map[accession]:
        path_map[accession] = path
        score_map[accession] = score


def index_genome_paths(
    genomes_dir: Path,
    annotation_dir: Optional[Path] = None,
) -> PathMaps:
    """Recursively index FAA, GFF, and genomic FASTA files by accession."""
    faa_map: PathMap = {}
    gff_map: PathMap = {}
    fna_map: PathMap = {}
    faa_scores: Dict[str, Tuple[int, int, str]] = {}
    gff_scores: Dict[str, Tuple[int, int, str]] = {}
    fna_scores: Dict[str, Tuple[int, int, str]] = {}

    search_roots: List[Path] = []
    if annotation_dir is not None and annotation_dir.exists():
        search_roots.append(annotation_dir)
    if genomes_dir.exists() and genomes_dir not in search_roots:
        search_roots.append(genomes_dir)

    for root_priority, search_root in enumerate(search_roots):
        try:
            paths: Iterable[Path] = search_root.rglob("*")
            for path in paths:
                if not path.is_file():
                    continue

                accession = _extract_accession(path)
                if accession is None:
                    continue

                name = path.name.lower()
                if name.endswith(".faa"):
                    _select_candidate(
                        faa_map, faa_scores, accession, path, "faa", root_priority
                    )
                elif name.endswith((".gff", ".gff3")):
                    _select_candidate(
                        gff_map, gff_scores, accession, path, "gff", root_priority
                    )
                elif name.endswith((".fna", ".fa", ".fasta")):
                    if "cds_from_genomic" in name or "rna_from_genomic" in name:
                        continue
                    _select_candidate(
                        fna_map, fna_scores, accession, path, "fna", root_priority
                    )
        except OSError as exc:
            LOGGER.warning("Could not scan input directory %s: %s", search_root, exc)

    return faa_map, gff_map, fna_map


def _validate_nonempty_file(path: Path, label: str) -> None:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{label} file is missing or empty: {path}")
    except OSError as exc:
        raise RuntimeError(f"Could not validate {label} file '{path}': {exc}") from exc


def _run_prodigal(
    accession: str,
    fna_path: Path,
    annotation_dir: Path,
    prodigal_executable: str,
    prodigal_mode: str,
) -> Tuple[Path, Path]:
    """Generate FAA and GFF files atomically with Prodigal."""
    annotation_dir.mkdir(parents=True, exist_ok=True)
    final_faa = annotation_dir / f"{accession}_protein.faa"
    final_gff = annotation_dir / f"{accession}_genomic.gff"

    if final_faa.exists() and final_gff.exists():
        _validate_nonempty_file(final_faa, "FAA")
        _validate_nonempty_file(final_gff, "GFF")
        return final_faa, final_gff

    token = uuid.uuid4().hex
    temporary_faa = annotation_dir / f".{accession}.{token}.faa.tmp"
    temporary_gff = annotation_dir / f".{accession}.{token}.gff.tmp"
    command = [
        prodigal_executable,
        "-i",
        str(fna_path),
        "-a",
        str(temporary_faa),
        "-o",
        str(temporary_gff),
        "-f",
        "gff",
        "-p",
        prodigal_mode,
        "-q",
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Prodigal failed for {accession} with exit code "
                f"{completed.returncode}: {details}"
            )

        _validate_nonempty_file(temporary_faa, "temporary FAA")
        _validate_nonempty_file(temporary_gff, "temporary GFF")
        os.replace(temporary_faa, final_faa)
        os.replace(temporary_gff, final_gff)
        return final_faa, final_gff
    finally:
        for temporary_path in (temporary_faa, temporary_gff):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def prepare_input_files(
    accessions: Sequence[str],
    faa_map: PathMap,
    gff_map: PathMap,
    fna_map: PathMap,
    annotation_dir: Path,
    annotate_missing: bool,
    prodigal_mode: str,
) -> Tuple[PathMap, PathMap, Dict[str, str]]:
    """Ensure that each possible accession has FAA and GFF inputs."""
    failures: Dict[str, str] = {}
    missing = list(
        dict.fromkeys(
            accession
            for accession in accessions
            if accession not in faa_map or accession not in gff_map
        )
    )
    if not missing:
        return faa_map, gff_map, failures

    if not annotate_missing:
        for accession in missing:
            failures[accession] = (
                f"Missing FAA or GFF input. FAA={faa_map.get(accession)}, "
                f"GFF={gff_map.get(accession)}"
            )
        return faa_map, gff_map, failures

    prodigal_executable = shutil.which("prodigal")
    if prodigal_executable is None:
        message = (
            "Prodigal is required to generate missing FAA/GFF files but was not "
            "found in PATH. Install it with 'conda install -c bioconda prodigal'."
        )
        return faa_map, gff_map, {accession: message for accession in missing}

    for accession in tqdm(
        missing,
        desc="Preparing annotations",
        unit="genome",
        dynamic_ncols=True,
    ):
        fna_path = fna_map.get(accession)
        if fna_path is None:
            failures[accession] = "Missing FAA/GFF and no genomic FASTA was found"
            continue

        try:
            faa_path, gff_path = _run_prodigal(
                accession,
                fna_path,
                annotation_dir,
                prodigal_executable,
                prodigal_mode,
            )
            faa_map[accession] = faa_path
            gff_map[accession] = gff_path
            LOGGER.info("Generated Prodigal annotation for %s", accession)
        except Exception as exc:
            failures[accession] = str(exc)
            LOGGER.exception("Could not annotate %s: %s", accession, exc)

    return faa_map, gff_map, failures
