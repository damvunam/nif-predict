#wgs_qc.py
import gzip
import subprocess
import os
import json
import glob

def peek_read_length(fastq_path, n_reads=100):
    """
    Estimates average read length by reading the first N reads of a fastq file.

    Args:
        fastq_path (str): Path to a .fastq or .fastq.gz file.
        n_reads (int): Number of reads to sample (default 100).

    Returns:
        float: Average read length in bp, or None if the file has no reads.
    """
    # 1. Open file, handling gzip transparently
    if fastq_path.endswith(".gz"):
        handle = gzip.open(fastq_path, "rt")
    else:
        handle = open(fastq_path, "rt")

    # 2. Read sequence lines (every 4th line in a fastq record)
    lengths = []
    with handle:
        for i, line in enumerate(handle):
            if i % 4 == 1:
                lengths.append(len(line.strip()))
            if len(lengths) >= n_reads:
                break

    # 3. Report result
    if not lengths:
        print(f"Error: No reads found in {fastq_path}")
        return None

    avg_length = sum(lengths) / len(lengths)
    print(f"{fastq_path}: sampled {len(lengths)} read, avg length = {avg_length:.1f} bp")
    return avg_length


def run_fastp(r1_path, r2_path, out_dir, sample_name="sample"):
    """
    Runs fastp ro trim adapters and filter low-quality reads.

    Args:
        r1_path (str): Path to R1 fastq file.
        r2_path (str): Path to R2 fastq file.
        out_dir (str): Directory wo write trimmed output into.
        sample_name (str): Prefix for output filenames (default "sample").
    
        Returns:
            dict: Paths to trimmed R1/R2 and the fastp report, or None if fastp failed.
    """

    # 1. Prepare output paths
    os.makedirs(out_dir, exist_ok=True)
    out_r1 = os.path.join(out_dir, f"{sample_name}_R1.trim.fastq.gz")
    out_r2 = os.path.join(out_dir, f"{sample_name}_R2.trim.fastq.gz")
    report_html = os.path.join(out_dir, f"{sample_name}.fastp.html")
    report_json = os.path.join(out_dir, f"{sample_name}.fastp.json")

    # 2. Build the command as a list of arguments (not a single string)
    cmd = [
        "fastp",
        "-i", r1_path, "-I", r2_path,
        "-o", out_r1, "-O", out_r2,
        "-h", report_html, "-j", report_json,
    ]

    # 3. Run it and check the result
    print(f"Running fastp on {sample_name}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: fastp failed for {sample_name}")
        print(e.stderr)
        return None

    print(f"fastp finished. Trimmed reads: {out_r1}, {out_r2}")
    return {"r1": out_r1, "r2": out_r2, "html": report_html, "json": report_json}


def check_qc_report(report_path, min_q30_rate=0.8, max_duplication_rate=0.2):
    """
    Checks a fastp JSON report against quality thresholds.

    Args:
        report_path: path to fastp's report json (output of run_fastp)
        min_q30_rate: minimum acceptable Q30 rate after filtering (default: 0.8)
        max_duplication_rate: maximum acceptable duplication rate (default: 0.2)

    Returns:
        True if the reads pass both checks, False otherwise.
    """
    try:
        with open(report_path) as f:
            data = json.load(f)
        q30_rate = data["summary"]["after_filtering"]["q30_rate"]
        duplication_rate = data["duplication"]["rate"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: could not read fastp report - {e}")
        return False

    print(f"Q30 rate (after filtering) = {q30_rate} (threshold: >= {min_q30_rate})")
    print(f"Duplication rate = {duplication_rate} (threshold: <= {max_duplication_rate})")

    q30_ok = q30_rate >= min_q30_rate
    duplication_ok = duplication_rate <= max_duplication_rate

    if q30_ok and duplication_ok:
        print("PASS: reads meet quality gate, safe to continue to assembly")
        return True

    print("FAIL: reads did not meet quality gate")
    if not q30_ok:
        print(f"  - Q30 rate too low ({q30_rate} < {min_q30_rate})")
    if not duplication_ok:
        print(f"  - Duplication rate too high ({duplication_rate} > {max_duplication_rate})")
    return False
    

def run_quast(assembly_path, out_dir, threads=10):
    """
    Runs QUAST to compute assembly statistics (N50, number of contigs, total length...).

    Args:
        assembly_path: path to the assembled genome (assembly.fasta)
        out_dir: directory where QUAST writes its report
        threads: number of threads to use (default: 10)

    Returns:
        Path to report.tsv, or None if QUAST failed.
    """
    cmd = ["quast.py", assembly_path, "-o", out_dir, "-t", str(threads)]
    print(f"Running QUAST on {assembly_path} ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: QUAST failed - {e.stderr}")
        return None
    report_path = os.path.join(out_dir, "report.tsv")
    if not os.path.exists(report_path):
        print(f"Error: QUAST finished but no report.tsv found in {out_dir}")
        return None
    print(f"QUAST report: {report_path}")
    return report_path


def run_busco(assembly_path, out_dir, lineage="bacillales_odb10", threads=10):
    """
    Runs BUSCO to estimate genome completeness against a conserved single-copy gene set.

    Args:
        assembly_path: path to the assembled genome (assembly.fasta)
        out_dir: parent directory where BUSCO writes its "busco_result" run folder
        lineage: BUSCO lineage dataset to compare against (default: bacillales_odb10,
                 matches S35's 16S hit - Bacillus velezensis)
        threads: number of threads to use (default: 10)

    Returns:
        Path to the short_summary JSON file, or None if BUSCO failed.
    """
    cmd = ["busco", "-i", assembly_path, "-o", "busco_result", "--out_path", out_dir,
           "-l", lineage, "-m", "genome", "-c", str(threads), "-f"]
    print(f"Running BUSCO on {assembly_path} (lineage={lineage}) ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: BUSCO failed - {e.stderr}")
        return None
    matches = glob.glob(os.path.join(out_dir, "busco_result", "short_summary*.json"))
    if not matches:
        print(f"Error: BUSCO finished but no short_summary json found in {out_dir}/busco_result")
        return None
    summary_path = matches[0]
    print(f"BUSCO summary: {summary_path}")
    return summary_path


def check_assembly_quality(quast_report_path, busco_summary_path, min_n50=50000, min_busco_completeness=95.0):
    """
    Go/no-go gate for a genome assembly (advisor's rule): N50 must be close to
    the expected genome size AND BUSCO completeness must be high enough.

    Args:
        quast_report_path: path to QUAST's report.tsv
        busco_summary_path: path to BUSCO's short_summary JSON file
        min_n50: minimum acceptable N50 in bp (default: 4200000, ~S35 genome size)
        min_busco_completeness: minimum acceptable BUSCO completeness in % (default: 95.0)

    Returns:
        True if the assembly passes both checks, False otherwise.
    """
    n50 = None
    try:
        with open(quast_report_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[0] == "N50":
                    n50 = int(parts[1])
                    break
    except OSError as e:
        print(f"Error: could not read QUAST report - {e}")
        return False
    if n50 is None:
        print(f"Error: N50 not found in {quast_report_path}")
        return False

    try:
        with open(busco_summary_path) as f:
            busco_data = json.load(f)
        completeness = busco_data["results"]["Complete percentage"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: could not read BUSCO completeness - {e}")
        return False

    print(f"N50 = {n50} bp (threshold: >= {min_n50} bp)")
    print(f"BUSCO completeness = {completeness}% (threshold: >= {min_busco_completeness}%)")

    n50_ok = n50 >= min_n50
    completeness_ok = completeness >= min_busco_completeness

    if n50_ok and completeness_ok:
        print("PASS: assembly meets go/no-go gate, safe to continue to annotation")
        return True

    print("FAIL: assembly did not meet go/no-go gate")
    if not n50_ok:
        print(f"  - N50 too low ({n50} < {min_n50})")
    if not completeness_ok:
        print(f"  - BUSCO completeness too low ({completeness} < {min_busco_completeness})")
    return False


def run_assembly_qc_gate(assembly_path, out_dir, lineage="bacillales_odb10",
                          min_n50=50000, min_busco_completeness=95.0, threads=10):
    """
    Orchestrator: runs QUAST + BUSCO on a finished assembly, then checks the
    go/no-go gate. Call this right after wgs_assembly.py finishes.

    Returns:
        True if the assembly passes, False if it fails or any step errors out.
    """
    quast_report_path = run_quast(assembly_path, out_dir, threads=threads)
    if quast_report_path is None:
        return False
    busco_summary_path = run_busco(assembly_path, out_dir, lineage=lineage, threads=threads)
    if busco_summary_path is None:
        return False
    return check_assembly_quality(quast_report_path, busco_summary_path,
                                   min_n50=min_n50, min_busco_completeness=min_busco_completeness)


if __name__ == "__main__":
    fastq_file_1 = "resources/raw/G13_1.fq.gz"
    fastq_file_2 = "resources/raw/G13_2.fq.gz"
    peek_read_length(fastq_file_1)
    peek_read_length(fastq_file_2)

    result = run_fastp(fastq_file_1, fastq_file_2, out_dir="results/qc", sample_name="G13")
    print(result)

    if result:
        check_qc_report(result["json"])