"""Parallel HMM and gene-cluster extraction for feature matrix builds."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from nifpredict.features.cluster_filter import ClusterFilter
from nifpredict.features.hmm_annotator import HMMAnnotator
from nifpredict.utils.config import AppConfig, load_config
from nifpredict.utils.logger import get_logger


LOGGER = get_logger("nifpredict.workflows.feature_matrix.extraction")
HMM_OUTPUT_COLUMNS = [
    "target_protein",
    "gene_family",
    "effective_score",
    "seq_evalue",
]

_worker_config: Optional[AppConfig] = None
_worker_annotator: Optional[HMMAnnotator] = None
_worker_cluster_filter: Optional[ClusterFilter] = None


def _init_worker() -> None:
    """Initialize process-local resources once per worker."""
    global _worker_config, _worker_annotator, _worker_cluster_filter

    _worker_config = load_config(auto_create_dirs=True)
    _worker_annotator = HMMAnnotator(config=_worker_config)
    _worker_cluster_filter = ClusterFilter(config=_worker_config)


def _require_nonempty_file(path: Path, label: str) -> None:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{label} file is missing or empty: {path}")
    except OSError as exc:
        raise RuntimeError(f"Could not validate {label} file '{path}': {exc}") from exc


def _process_single_genome(
    accession: str,
    faa_path: Path,
    gff_path: Path,
) -> Dict[str, Any]:
    """Run HMM annotation and cluster detection for one genome."""
    global _worker_config, _worker_annotator, _worker_cluster_filter

    if (
        _worker_config is None
        or _worker_annotator is None
        or _worker_cluster_filter is None
    ):
        _init_worker()

    assert _worker_config is not None
    assert _worker_annotator is not None
    assert _worker_cluster_filter is not None

    _require_nonempty_file(faa_path, "FAA")
    _require_nonempty_file(gff_path, "GFF")

    hmm_dir = Path(_worker_config.paths.hmm_profiles_dir)
    hmm_output_dir = Path(_worker_config.paths.hmmer_dir)
    hmm_output_dir.mkdir(parents=True, exist_ok=True)
    hmm_files = sorted(
        path
        for path in hmm_dir.glob("*.hmm")
        if "pfam-a" not in path.name.lower()
    )
    if not hmm_files:
        raise FileNotFoundError(f"No HMM profile files found in {hmm_dir}")

    hit_frames: List[pd.DataFrame] = []
    for hmm_file in hmm_files:
        hits = _worker_annotator.annotate_to_dataframe(faa_path, hmm_file)
        if not hits.empty:
            hit_frames.append(hits)

    all_hits = (
        pd.concat(hit_frames, ignore_index=True)
        if hit_frames
        else pd.DataFrame(columns=HMM_OUTPUT_COLUMNS)
    )
    output_path = hmm_output_dir / f"{accession}_hmm_hits.tsv"
    all_hits.to_csv(output_path, sep="\t", index=False)

    gff_frame = _worker_cluster_filter.parse_gff3(gff_path)
    clusters: List[Any] = []
    if not all_hits.empty and not gff_frame.empty:
        clusters = _worker_cluster_filter.group_into_clusters(all_hits, gff_frame)

    pfam_domains = (
        all_hits["gene_family"].dropna().astype(str).tolist()
        if "gene_family" in all_hits.columns
        else []
    )
    return {
        "accession": accession,
        "status": "SUCCESS",
        "df_all_hits": all_hits,
        "clusters": clusters,
        "pfam_domains": pfam_domains,
        "metadata": {},
    }


def extract_raw_records(
    accessions: Sequence[str],
    faa_map: Dict[str, Path],
    gff_map: Dict[str, Path],
    initial_failures: Dict[str, str],
    num_workers: int,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Extract raw records in parallel while isolating per-genome failures."""
    failures = dict(initial_failures)
    runnable = list(
        dict.fromkeys(
            accession
            for accession in accessions
            if accession not in failures
            and accession in faa_map
            and accession in gff_map
        )
    )
    records_by_accession: Dict[str, Dict[str, Any]] = {}
    batches = [
        runnable[index : index + batch_size]
        for index in range(0, len(runnable), batch_size)
    ]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
    ) as executor:
        with tqdm(
            total=len(runnable),
            desc="Extracting raw features",
            unit="genome",
            dynamic_ncols=True,
        ) as progress:
            for batch in batches:
                futures = {
                    executor.submit(
                        _process_single_genome,
                        accession,
                        faa_map[accession],
                        gff_map[accession],
                    ): accession
                    for accession in batch
                }
                for future in as_completed(futures):
                    accession = futures[future]
                    try:
                        records_by_accession[accession] = future.result()
                    except Exception as exc:
                        failures[accession] = str(exc)
                        LOGGER.error(
                            "Feature extraction failed for %s: %s",
                            accession,
                            exc,
                            exc_info=True,
                        )
                    finally:
                        progress.update(1)
                        progress.set_postfix(
                            success=len(records_by_accession),
                            failed=len(failures),
                            refresh=True,
                        )

    ordered_records = [
        records_by_accession[accession]
        for accession in accessions
        if accession in records_by_accession
    ]
    return ordered_records, failures
