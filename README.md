# NifPredict

Detect nitrogen-fixation (nif) genes in bacterial genomes, from raw WGS reads to a nifH/nifD/nifK verdict.
MSc thesis project, 2026–2027. The ML benchmark (`nifpredict/ml/`) is paused.

## Layout

```text
nif-predict/
├── nifpredict/
│   ├── common/      paths.py (every path), config.py, logger.py
│   ├── wgs/         run_sample.py (runner), loader, qc, assembly, annotation, contig_stats
│   ├── nif/         check_nif.py (tblastn on assemblies, DIAMOND on reads)
│   └── ml/          data, features, labeling, models, evaluation, workflows, predictor (paused)
├── scripts/         fetch_public_data.py, check_baseline.sh, ml/ (ML entry points)
├── tests/           unit/, integration/, fixtures/, baseline/ (expected results of 5 controls)
├── config/          config.yaml, hmm_panel_pgap_19.yaml (ML)
├── data/public/     public inputs, data/public/<kind>/<accession>/ (git-ignored; MANIFEST.tsv is tracked)
└── results/         all outputs, results/<pipeline>/<sample>/<step>/ (git-ignored)
```

Rules: run everything from the repo root; imports are absolute (`nifpredict.wgs...`); no path is hard-coded
outside `nifpredict/common/paths.py`.

## Setup

- Python ≥ 3.10, ≤ 3.12 venv for code and tests (`numpy<2` has no wheels for 3.13):
  `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- WGS tools + package in one conda env (fastp, unicycler, SPAdes, quast, busco, skani, BLAST+, DIAMOND,
  seqtk, seqkit, NCBI `datasets`, pinned versions): from the repo root,
  `micromamba create -y -n nifwgs -f environment.yml && micromamba activate nifwgs`.
  The solve itself is not covered by the test suite (see `tests/unit/test_mu6_env_files.py` for the static checks).
- prokka/bakta for annotation are not in `environment.yml`; the maintainer's own env is `prokka_test`.

## Public data

`data/public/MANIFEST.tsv` lists every public input (accession, source, date, md5, download or cut command).

```bash
PYTHONPATH=. .venv/bin/python scripts/fetch_public_data.py --verify            # md5 of what is on disk
PYTHONPATH=. .venv/bin/python scripts/fetch_public_data.py --id PSON_RAW_1 PSON_RAW_2 PSON_1 PSON_2
PYTHONPATH=. .venv/bin/python scripts/fetch_public_data.py --group controls --dest /tmp/fetch_test --dry-run
```

## Run

```bash
# WGS: QC (gate 1) -> assembly -> QUAST/BUSCO (gate 2) -> contig stats, coverage -> identity (skani)
python -m nifpredict.wgs.run_sample -s PSON -d data/public/reads/SRR17868980 \
    -r data/public/genomes/GCF_021312615.1/ncbi_dataset/data/GCF_021312615.1/GCF_021312615.1_ASM2131261v1_genomic.fna
#   -> results/wgs/PSON/{qc,assembly,assembly_qc,identity}/ ; -n prints the plan without running tools

# nif genes on an assembly (tblastn) or, without -a, on trimmed reads (DIAMOND)
python -m nifpredict.nif.check_nif -s PSON -a results/wgs/PSON/assembly/assembly.fasta
#   -> results/nif/PSON/assembly/ (log with VERDICT, tblastn_hits.tsv)
```

## Check after any change

```bash
bash scripts/check_baseline.sh      # in the tools env; pytest uses .venv/bin/python
```

pytest, PSON runner on its existing outputs (must print DONE and create no file), and `check_nif` on the
5 controls (CTRL_AVIN, CTRL_PWLY, CTRL_BSUB, PSON, PSON_REF) compared with `tests/baseline/`.
Output: `results/checks/<timestamp>/summary.txt`.
