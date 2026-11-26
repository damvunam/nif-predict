"""Single source of truth for WGS / check_nif thresholds.

Before this module, the same numbers were hard-coded in several places:
Gate 2 (N50 50000, BUSCO 95, lineage bacillales_odb10) appeared in
``nifpredict/wgs/qc.py`` defaults and in ``nifpredict/wgs/run_sample.py``
four more times; ANI 95 and the check_nif cutoffs were repeated the same
way. Changing one threshold meant editing many places and risking drift.
Values here are unchanged from the old literals; only their location moved.

Deliberately NOT merged with ``config/config.yaml``: the YAML thresholds
are for the ML HMM pipeline (HMMER e-value 1e-10, bit score 80, ...),
while these are for different tools (tblastn e-value 1e-5, QUAST/BUSCO,
skani ANI). Same name does not mean same value; keep the two systems apart.
"""

# Gate 1: fastp read QC (nifpredict/wgs/qc.py)
GATE1_MIN_Q30 = 0.8
GATE1_MAX_DUPLICATION = 0.2

# Gate 2: assembly QC (QUAST + BUSCO)
GATE2_MIN_N50 = 50000
GATE2_MIN_BUSCO_COMPLETENESS = 95.0
GATE2_BUSCO_LINEAGE = "bacillales_odb10"

# Identity check: skani ANI against the reference genome
IDENTITY_MIN_ANI = 95.0

# Platform detection: reads at or above this length are treated as long reads
LONG_READ_MIN_LENGTH = 300

# check_nif: tblastn on assemblies and DIAMOND blastx on reads
NIF_TBLASTN_EVALUE = "1e-5"
NIF_DIAMOND_EVALUE = "1e-5"
NIF_MIN_PIDENT_ASSEMBLY = 50.0
NIF_MIN_COVERAGE = 0.7
NIF_MAX_GAP = 20000
NIF_MIN_OVERLAP = 50
NIF_MIN_PIDENT_READS = 60.0
NIF_MIN_ALN_LEN = 40
NIF_MIN_READS = 20
