#wgs_qc.py
import gzip
import subprocess
import os

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


if __name__ == "__main__":
    fastq_file_1 = "resources/raw/G1X_1.fq.gz"
    fastq_file_2 = "resources/raw/G1X_2.fq.gz"
    peek_read_length(fastq_file_1)
    peek_read_length(fastq_file_2)

    result = run_fastp(fastq_file_1, fastq_file_2, out_dir="results/qc", sample_name="G1X")
    print(result)