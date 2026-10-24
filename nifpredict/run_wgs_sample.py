# run_wgs_sample.py

import argparse
import glob
import os
import re
import subprocess
import sys

from wgs_loader import find_raw_reads
from wgs_qc import (
    check_assembly_quality,
    check_qc_report,
    get_fasta_total_length,
    get_read_coverage,
    peek_read_length,
    run_assembly_qc_gate,
    run_fastp,
)
from wgs_assembly import assemble_genome
import contig_stats


def get_mate_number(file_path, sample_name):
    """Extract mate number (1 or 2) from a raw read filename after stripping sample prefix and extension.

    Args:
        file_path: Path or filename of the fastq file (str).
        sample_name: Sample prefix to strip from the beginning of the basename (str).

    Returns:
        int: 1 or 2 if unambiguously identified, or None if missing or ambiguous.
    """
    # 1. Strip directory and sample_name prefix
    base_name = os.path.basename(file_path)
    if base_name.startswith(sample_name):
        remainder = base_name[len(sample_name):]
    else:
        remainder = base_name

    # 2. Strip supported fastq extensions (longest first)
    for ext in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if remainder.endswith(ext):
            remainder = remainder[:-len(ext)]
            break

    # 3. Find all mate tokens matching [._-]?[Rr]?[12] bounded by separator or end of string
    matches = re.findall(r"(?:^|[._-])[Rr]?([12])(?=[._-]|$)", remainder)
    unique_mates = {int(m) for m in matches}
    if len(unique_mates) == 1:
        return unique_mates.pop()
    return None


def get_sample_raw_reads(data_dir, sample_name):
    """Find raw paired-end short reads for a sample, returning (r1_path, r2_path).

    Args:
        data_dir: Path to directory containing raw reads.
        sample_name: Identifier of the sample.

    Returns:
        tuple: (r1_path, r2_path) ordered as mate 1 then mate 2, or (None, None) if not found,
               count != 2, or R1/R2 mate numbers cannot be unambiguously determined.
    """
    # FIX: do not swallow exceptions or guess return types; find_raw_reads returns sorted list or None
    # try:
    #     res = find_raw_reads(data_dir, sample_name)
    # except Exception:
    #     return None, None, None
    found_files = find_raw_reads(data_dir, sample_name)
    if found_files is None:
        return None, None

    # FIX: filter by exact sample prefix followed by '_', '.' or '-' to avoid prefix collision (e.g. G1 matching G1X_*)
    matched_files = []
    for file_path in found_files:
        base_name = os.path.basename(file_path)
        if base_name.startswith(sample_name) and len(base_name) > len(sample_name) and base_name[len(sample_name)] in ("_", ".", "-"):
            matched_files.append(file_path)
        else:
            print(f"Warning: ignoring {file_path} (does not match sample name {sample_name})")

    # FIX: short-read branch requires exactly 2 files after filtering
    # if len(found_files) != 2:
    #     print(f"Error: expected 2 fastq files for {sample_name}, found {len(found_files)}: {found_files}")
    #     return None, None
    if len(matched_files) != 2:
        print(f"Error: expected 2 fastq files for {sample_name}, found {len(matched_files)}: {matched_files}")
        return None, None

    # FIX (W14): assign R1/R2 by parsed mate number instead of lexicographic sort order
    # return matched_files[0], matched_files[1]
    mate_0 = get_mate_number(matched_files[0], sample_name)
    mate_1 = get_mate_number(matched_files[1], sample_name)
    if mate_0 is None or mate_1 is None or mate_0 == mate_1:
        print(f"Error: cannot tell R1 from R2 for {sample_name}: {matched_files}")
        return None, None

    if mate_0 == 1:
        return matched_files[0], matched_files[1]
    return matched_files[1], matched_files[0]


def determine_platform(platform, r1_path):
    """Determine platform ('short') from argument or peeked read length.

    Args:
        platform: Platform string ('auto' or 'short').
        r1_path: Path to read 1 fastq file.

    Returns:
        str: 'short' if valid short reads, or None if invalid/unsupported.
    """
    if platform != "auto":
        return platform

    read_len = peek_read_length(r1_path)
    # FIX: stop with Error if peek_read_length returns None or if reads >= 300 bp (long reads unsupported)
    # if read_len and read_len < 300:
    #     return "short"
    # return "long"
    if read_len is None:
        print(f"Error: could not determine read length from {r1_path}")
        return None

    # Threshold 300 bp distinguishes standard Illumina short reads from long reads
    if read_len >= 300:
        print("Error: long reads detected, not supported by run_wgs_sample.py yet")
        return None

    return "short"


def run_sample_qc(r1_path, r2_path, qc_dir, sample_name):
    """Run fastp trimming on raw reads if outputs do not already exist.

    Args:
        r1_path: Path to raw read 1.
        r2_path: Path to raw read 2.
        qc_dir: Output directory for QC results.
        sample_name: Name of sample.

    Returns:
        tuple: (trim_r1_path, trim_r2_path, fastp_json_path) or (None, None, None) if fastp failed.
    """
    trim_r1_path = os.path.join(qc_dir, f"{sample_name}_R1.trim.fastq.gz")
    trim_r2_path = os.path.join(qc_dir, f"{sample_name}_R2.trim.fastq.gz")
    fastp_json_path = os.path.join(qc_dir, f"{sample_name}.fastp.json")

    # 1. Skip fastp if outputs already exist
    if os.path.exists(trim_r1_path) and os.path.exists(trim_r2_path) and os.path.exists(fastp_json_path):
        print(f"Warning: {trim_r1_path} exists, skipping fastp.")
        return trim_r1_path, trim_r2_path, fastp_json_path

    # 2. Run fastp and use its returned dict
    os.makedirs(qc_dir, exist_ok=True)
    # FIX: check return value of run_fastp (dict or None) instead of ignoring it
    # run_fastp(r1_path, r2_path, qc_dir, sample_name=sample_name)
    # return trim_r1, trim_r2, fastp_json
    fastp_res = run_fastp(r1_path, r2_path, qc_dir, sample_name=sample_name)
    if fastp_res is None:
        return None, None, None

    return fastp_res["r1"], fastp_res["r2"], fastp_res["json"]


def is_sample_qc_gate_passed(fastp_json_path, sample_name):
    """Evaluate Gate 1 quality metrics from fastp report.

    Args:
        fastp_json_path: Path to fastp json report.
        sample_name: Name of sample.

    Returns:
        bool: True if PASS, False otherwise.
    """
    if not os.path.exists(fastp_json_path):
        print(f"Error: {fastp_json_path} not found for Gate 1 QC check.")
        return False

    # FIX: check_qc_report returns bool directly; remove tuple/dict guessing
    # is_pass = check_qc_report(fastp_json_path)
    # if isinstance(is_pass, tuple):
    #     is_pass = is_pass[0]
    # elif isinstance(is_pass, dict):
    #     is_pass = is_pass.get("pass", True)
    is_pass = check_qc_report(fastp_json_path)

    if not is_pass:
        print(f"Error: Gate 1 FAIL for {sample_name} based on {fastp_json_path}")
        return False
    return True


def run_sample_assembly(platform, assembly_dir, trim_r1_path, trim_r2_path, threads):
    """Run de novo assembly if output assembly FASTA does not exist.

    Args:
        platform: Sequencing platform ('short').
        assembly_dir: Output directory for assembly.
        trim_r1_path: Path to trimmed read 1.
        trim_r2_path: Path to trimmed read 2.
        threads: Number of threads.

    Returns:
        str: Path to assembly FASTA, or None if failed.
    """
    assembly_fasta_path = os.path.join(assembly_dir, "assembly.fasta")

    # 1. Skip if assembly already exists
    if os.path.exists(assembly_fasta_path):
        print(f"Warning: {assembly_fasta_path} exists, skipping assembly.")
        return assembly_fasta_path

    # 2. Run assembly and use returned path
    os.makedirs(assembly_dir, exist_ok=True)
    # FIX: use return value of assemble_genome (str or None) directly
    # assemble_genome(platform, assembly_dir, r1_path=trim_r1, r2_path=trim_r2, threads=threads)
    # if not os.path.exists(assembly_fasta):
    #     print(f"Error: Assembly failed, {assembly_fasta} not found.")
    #     return None
    # return assembly_fasta
    assembled_path = assemble_genome(
        platform,
        assembly_dir,
        r1_path=trim_r1_path,
        r2_path=trim_r2_path,
        threads=threads,
    )
    if assembled_path is None:
        print(f"Error: Assembly failed for {assembly_dir}")
        return None
    return assembled_path


def is_sample_gate2_passed(assembly_fasta_path, assembly_qc_dir, threads):
    """Run Gate 2 assembly evaluation (QUAST + BUSCO) or evaluate existing reports.

    Args:
        assembly_fasta_path: Path to assembly FASTA.
        assembly_qc_dir: Output directory for assembly QC.
        threads: Number of threads.

    Returns:
        bool: True if Gate 2 PASS, False otherwise.
    """
    quast_tsv_path = os.path.join(assembly_qc_dir, "report.tsv")
    # FIX: match only .json BUSCO summaries as run_busco does, avoiding .txt files
    # busco_summaries = glob.glob(os.path.join(assembly_qc_dir, "busco_result", "short_summary*"))
    busco_summaries = glob.glob(os.path.join(assembly_qc_dir, "busco_result", "short_summary*.json"))

    # 1. Reuse existing reports if available
    if os.path.exists(quast_tsv_path) and busco_summaries:
        print(f"Warning: {quast_tsv_path} exists, skipping QUAST and BUSCO run.")
        return check_assembly_quality(
            quast_tsv_path,
            busco_summaries[0],
            min_n50=50000,               # Floor threshold against fragmented assembly
            min_busco_completeness=95.0, # Gate 2 completeness threshold
        )

    # 2. Run Gate 2 evaluation from scratch
    os.makedirs(assembly_qc_dir, exist_ok=True)
    return run_assembly_qc_gate(
        assembly_fasta_path,
        assembly_qc_dir,
        lineage="bacillales_odb10",      # Default Bacillales lineage
        min_n50=50000,                   # Floor threshold
        min_busco_completeness=95.0,     # Gate 2 threshold
        threads=threads,
    )


def run_sample_contig_stats(assembly_fasta_path, contig_stats_tsv_path):
    """Generate contig-level statistics table if not already present.

    Args:
        assembly_fasta_path: Path to assembly FASTA.
        contig_stats_tsv_path: Output path for TSV file.

    Returns:
        bool: True if completed or skipped.
    """
    if os.path.exists(contig_stats_tsv_path):
        print(f"Warning: {contig_stats_tsv_path} exists, skipping contig stats.")
        return True

    os.makedirs(os.path.dirname(contig_stats_tsv_path), exist_ok=True)
    records = contig_stats.parse_fasta(assembly_fasta_path)
    contig_stats.write_tsv(records, contig_stats_tsv_path)
    contig_stats.print_summary(records)
    return True


def print_sample_coverage(fastp_json_path, assembly_fasta_path, ref_genome_path):
    """Prints read coverage (x) against the assembly and the reference genome. Print only, never stops the run.

    Args:
        fastp_json_path: Path to fastp JSON report (str).
        assembly_fasta_path: Path to assembled FASTA file (str).
        ref_genome_path: Path to reference genome FASTA file (str).

    Returns:
        None.
    """
    # 1. Coverage against assembly length
    asm_size = get_fasta_total_length(assembly_fasta_path)
    asm_cov = get_read_coverage(fastp_json_path, asm_size)
    if asm_cov is not None:
        print(
            f"COVERAGE: {asm_cov['coverage']:.1f}x vs assembly "
            f"({asm_cov['total_bases']:,} bases after fastp / {asm_cov['genome_size']:,} bp)"
        )
    else:
        print("Warning: COVERAGE vs assembly not computed")

    # 2. Coverage against reference genome length, if the reference exists
    if ref_genome_path and os.path.exists(ref_genome_path):
        ref_size = get_fasta_total_length(ref_genome_path)
        ref_cov = get_read_coverage(fastp_json_path, ref_size)
        if ref_cov is not None:
            print(
                f"COVERAGE: {ref_cov['coverage']:.1f}x vs reference "
                f"({ref_cov['total_bases']:,} bases after fastp / {ref_cov['genome_size']:,} bp, {ref_genome_path})"
            )
        else:
            print("Warning: COVERAGE vs reference not computed")
    else:
        print(f"Warning: COVERAGE vs reference skipped, not found: {ref_genome_path}")

    return None


def check_sample_identity(assembly_fasta_path, ref_genome_path, identity_tsv_path, threads=10):
    """Check identity against reference genome using skani dist.

    Args:
        assembly_fasta_path: Path to assembly FASTA.
        ref_genome_path: Path to reference genome FASTA.
        identity_tsv_path: Path to output TSV.
        threads: Number of CPU threads (default: 10).

    Returns:
        dict: {"status": "PASS"|"FAIL"|"ERROR", "ani": float or None, "af_ref": float or None, "af_query": float or None}.
    """
    # FIX: separate 5 distinct outcomes (missing ref, skani error, empty result, ANI < 95, ANI >= 95)
    # 1. Check reference file exists
    if not os.path.exists(ref_genome_path):
        print(f"IDENTITY: ERROR - reference not found: {ref_genome_path}")
        return {"status": "ERROR", "ani": None, "af_ref": None, "af_query": None}

    # 2. Run skani dist if output does not exist
    if os.path.exists(identity_tsv_path):
        print(f"Warning: {identity_tsv_path} exists, skipping skani dist.")
    else:
        os.makedirs(os.path.dirname(identity_tsv_path), exist_ok=True)
        cmd = [
            "skani", "dist",
            assembly_fasta_path, ref_genome_path,
            "-o", identity_tsv_path,
            "-t", str(threads),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("IDENTITY: ERROR - skani failed")
            if e.stderr:
                print(e.stderr.strip())
            # FIX: remove partial/empty TSV if skani failed so subsequent runs do not mistake it for valid empty output
            if os.path.exists(identity_tsv_path):
                os.remove(identity_tsv_path)
            return {"status": "ERROR", "ani": None, "af_ref": None, "af_query": None}
        except FileNotFoundError as e:
            print(f"IDENTITY: ERROR - skani failed: {e}")
            # FIX: remove partial/empty TSV if skani failed
            if os.path.exists(identity_tsv_path):
                os.remove(identity_tsv_path)
            return {"status": "ERROR", "ani": None, "af_ref": None, "af_query": None}

    # 3. Parse TSV results
    if not os.path.exists(identity_tsv_path):
        print("IDENTITY: ERROR - skani failed (output TSV not created)")
        return {"status": "ERROR", "ani": None, "af_ref": None, "af_query": None}

    try:
        with open(identity_tsv_path, "r") as f:
            header = f.readline()
            lines = [line.strip().split("\t") for line in f if line.strip()]
    except OSError as e:
        print(f"IDENTITY: ERROR - skani failed to read {identity_tsv_path}: {e}")
        return {"status": "ERROR", "ani": None, "af_ref": None, "af_query": None}

    if not lines:
        print(f"IDENTITY: FAIL - no ANI with {ref_genome_path} (below skani screening threshold)")
        return {"status": "FAIL", "ani": None, "af_ref": None, "af_query": None}

    ani = float(lines[0][2])
    af_ref = float(lines[0][3])
    af_query = float(lines[0][4])

    # 95% is the recognized species-level ANI threshold
    if ani < 95.0:
        print(
            f"IDENTITY: FAIL - ANI={ani:.2f}% < 95 with {ref_genome_path} "
            f"(align_fraction ref/query={af_ref:.2f}%/{af_query:.2f}%)"
        )
        return {"status": "FAIL", "ani": ani, "af_ref": af_ref, "af_query": af_query}

    print(
        f"IDENTITY: PASS - ANI={ani:.2f}% with {ref_genome_path} "
        f"(align_fraction ref/query={af_ref:.2f}%/{af_query:.2f}%)"
    )
    return {"status": "PASS", "ani": ani, "af_ref": af_ref, "af_query": af_query}


if __name__ == "__main__":
    default_threads = min(10, os.cpu_count() or 1) # Cap at 10 threads per machine policy
    default_ref_path = "results/identity_check/refs/B_velezensis/ncbi_dataset/data/GCF_000015785.2/GCF_000015785.2_ASM1578v2_genomic.fna" # Default FZB42 reference

    parser = argparse.ArgumentParser(
        description="Run complete WGS pipeline for a sample (QC, assembly, stats, identity check)."
    )
    parser.add_argument("-s", "--sample", required=True, help="Sample identifier (e.g. G1X, X35)")
    parser.add_argument("-d", "--data_dir", default="resources/raw", help="Folder containing raw reads (default: resources/raw)")
    parser.add_argument("-o", "--out_dir", default="results", help="Folder for outputs (default: results)")
    # FIX: restrict platform choices to auto and short until long-read support is implemented
    # parser.add_argument("-p", "--platform", default="auto", choices=["auto", "short", "long", "hybrid"], help="Sequencing platform (default: auto)")
    parser.add_argument("-p", "--platform", default="auto", choices=["auto", "short"], help="Sequencing platform (default: auto)")
    parser.add_argument("-t", "--threads", type=int, default=default_threads, help=f"CPU threads (default: {default_threads})")
    parser.add_argument("-r", "--ref_genome", default=default_ref_path, help="Reference genome FNA for identity check (default: FZB42)")
    parser.add_argument("-n", "--dry_run", action="store_true", help="Print steps and paths without running tools")

    args = parser.parse_args()

    # 1. Find raw reads
    r1_path, r2_path = get_sample_raw_reads(args.data_dir, args.sample)
    if not r1_path or not r2_path:
        sys.exit(1)

    # 2. Determine platform
    platform = determine_platform(args.platform, r1_path)
    if platform is None:
        sys.exit(1)

    # 3. Plan output paths
    qc_dir = os.path.join(args.out_dir, "qc")
    trim_r1_path = os.path.join(qc_dir, f"{args.sample}_R1.trim.fastq.gz")
    trim_r2_path = os.path.join(qc_dir, f"{args.sample}_R2.trim.fastq.gz")
    fastp_json_path = os.path.join(qc_dir, f"{args.sample}.fastp.json")
    assembly_dir = os.path.join(args.out_dir, "assembly", args.sample)
    assembly_fasta_path = os.path.join(assembly_dir, "assembly.fasta")
    assembly_qc_dir = os.path.join(args.out_dir, "assembly_qc", args.sample)
    contig_stats_tsv_path = os.path.join(assembly_dir, "contig_stats.tsv")
    identity_dir = os.path.join(args.out_dir, "identity_check", args.sample)
    # FIX: tie identity output filename to reference stem instead of hardcoded skani_vs_refs.tsv
    # identity_tsv = os.path.join(identity_dir, "skani_vs_refs.tsv")
    ref_stem = os.path.splitext(os.path.basename(args.ref_genome))[0]
    identity_tsv_path = os.path.join(identity_dir, f"skani_vs_{ref_stem}.tsv")

    # 4. Dry run mode
    if args.dry_run:
        print(f"[DRY-RUN] Sample: {args.sample}")
        print(f"[DRY-RUN] Raw reads: R1={r1_path}, R2={r2_path}")
        print(f"[DRY-RUN] Platform: {platform}")
        print(f"[DRY-RUN] Step 1 (fastp QC): {trim_r1_path}, {trim_r2_path}, {fastp_json_path}")
        print(f"[DRY-RUN] Step 2 (Gate 1 QC check): {fastp_json_path}")
        print(f"[DRY-RUN] Step 3 (Assembly): {assembly_fasta_path} (platform: {platform}, threads: {args.threads})")
        print(f"[DRY-RUN] Step 4 (Gate 2 QUAST+BUSCO): {assembly_qc_dir} (min_n50=50000, min_busco=95.0)")
        print(f"[DRY-RUN] Step 5 (Contig stats): {contig_stats_tsv_path}")
        print(f"[DRY-RUN] Step 5b (Coverage): {fastp_json_path} vs {assembly_fasta_path} and {args.ref_genome}")
        print(f"[DRY-RUN] Step 6 (Identity check): {identity_tsv_path} vs {args.ref_genome}")
        sys.exit(0)

    # 5. Step 1: QC fastp
    trim_r1_path, trim_r2_path, fastp_json_path = run_sample_qc(r1_path, r2_path, qc_dir, args.sample)
    if not trim_r1_path or not trim_r2_path or not fastp_json_path:
        sys.exit(1)

    # 6. Step 2: Gate 1 check
    if not is_sample_qc_gate_passed(fastp_json_path, args.sample):
        sys.exit(1)

    # 7. Step 3: De novo assembly
    assembly_fasta_path = run_sample_assembly(platform, assembly_dir, trim_r1_path, trim_r2_path, args.threads)
    if not assembly_fasta_path:
        sys.exit(1)

    # 8. Step 4: Gate 2 assembly QC
    if not is_sample_gate2_passed(assembly_fasta_path, assembly_qc_dir, args.threads):
        print(f"Error: Gate 2 FAIL for {args.sample}: assembly quality below thresholds.")
        sys.exit(1)

    # 9. Step 5: Contig stats
    run_sample_contig_stats(assembly_fasta_path, contig_stats_tsv_path)

    # 9b. Step 5b: Print read coverage (print only, not a gate)
    print_sample_coverage(fastp_json_path, assembly_fasta_path, args.ref_genome)

    # 10. Step 6: Identity check
    identity_res = check_sample_identity(assembly_fasta_path, args.ref_genome, identity_tsv_path, args.threads)
    if identity_res["status"] != "PASS":
        sys.exit(1)

    # FIX: print completion line when all gates and identity check pass
    print(f"DONE: {args.sample} passed gate 1, gate 2 and identity; stopped before annotation")
