"""Every filesystem location used by NifPredict. No other module hard-codes a path.

Layout (relative to the repo root):
    data/public/<kind>/<accession>/   public inputs, listed in data/public/MANIFEST.tsv (git-ignored)
    results/<pipeline>/<sample>/<step>/   all outputs (git-ignored)
    tests/baseline/<sample>/          expected check_nif results of the 5 control genomes (tracked)
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "config"
CONFIG_YAML = CONFIG_DIR / "config.yaml"
HMM_PANEL_CONFIG = CONFIG_DIR / "hmm_panel_pgap_19.yaml"

DATA_DIR = REPO_ROOT / "data"
PUBLIC_DIR = DATA_DIR / "public"
MANIFEST = PUBLIC_DIR / "MANIFEST.tsv"
READS_DIR = PUBLIC_DIR / "reads"
GENOMES_DIR = PUBLIC_DIR / "genomes"
PROTEINS_DIR = PUBLIC_DIR / "proteins"
BUSCO_DIR = PUBLIC_DIR / "busco"
NIF_REF_FAA = PROTEINS_DIR / "nif_ref_171" / "nif_ref_swissprot.faa"

RESULTS_DIR = REPO_ROOT / "results"
WGS_RESULTS = RESULTS_DIR / "wgs"
NIF_RESULTS = RESULTS_DIR / "nif"

TESTS_DIR = REPO_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"
BASELINE_DIR = TESTS_DIR / "baseline"
SCRIPTS_DIR = REPO_ROOT / "scripts"
ML_SCRIPTS_DIR = SCRIPTS_DIR / "ml"

# Default reference for the WGS identity check: B. velezensis FZB42
DEFAULT_IDENTITY_ACCESSION = "GCF_000015785.2"

# ML (paused): same locations as before the restructure
ML_BATCH_ACCESSIONS = DATA_DIR / "batch_accessions.txt"
ML_RAW_GENOMES = DATA_DIR / "raw" / "genomes"
ML_LEGACY_RAW_GENOMES = DATA_DIR / "raw_genomes"
ML_LEGACY_RAW_METADATA = DATA_DIR / "raw_metadata"
ML_RAW_ZIPS = DATA_DIR / "raw" / "zips"
ML_ROOT_ZIP = REPO_ROOT / "ncbi_dataset.zip"
ML_ANNOTATION = DATA_DIR / "annotation"
ML_INTERIM = DATA_DIR / "interim"
ML_PROCESSED = DATA_DIR / "processed"
ML_LABELS = DATA_DIR / "labels"
ML_HMM_PROFILES = DATA_DIR / "hmm_profiles"
ML_HMM_PANELS = DATA_DIR / "hmm_panels"
ML_RESULTS = RESULTS_DIR
ML_MODELS = REPO_ROOT / "models"
ML_MLRUNS = REPO_ROOT / "mlruns"


def genome_fna(accession: str) -> Path:
    """Genomic FASTA of an NCBI Datasets package stored under data/public/genomes/<accession>/."""
    package = GENOMES_DIR / accession / "ncbi_dataset" / "data" / accession
    matches = sorted(package.glob("*_genomic.fna"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected 1 *_genomic.fna in {package}, found {len(matches)}; "
            f"fetch it with: python scripts/fetch_public_data.py (see {MANIFEST.relative_to(REPO_ROOT)})"
        )
    return matches[0]


def reads_dir(accession: str) -> Path:
    """Folder holding the reads of one public run (data/public/reads/<accession>/)."""
    return READS_DIR / accession


def wgs_dir(sample: str, step: str) -> Path:
    """Output folder of one WGS step for one sample: results/wgs/<sample>/<step>/."""
    return WGS_RESULTS / sample / step


def nif_dir(sample: str, mode: str, root: Path | str | None = None) -> Path:
    """Output folder of check_nif for one sample and mode (assembly/reads): results/nif/<sample>/<mode>/."""
    return Path(root if root is not None else NIF_RESULTS) / sample / mode
