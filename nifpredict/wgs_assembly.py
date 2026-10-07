#wgs_assembly.py
import os
import subprocess
import argparse


def run_unicycler_short(r1_path, r2_path, out_dir, threads=10):
    """
    Assemblles a short-read genome using Unicycler (wraps SPAdes internally).

    Args: 
        r1_path (str): Path to trimmed forward reads (fastq.gz)
        r2_path (str): Path to trimmed reverse reads (fastq.gz)
        out_dir (str): Directory to write assembly output
        threads (int): Number of CPU threads to use
    
        Returns:
            str: Path to the assembled contifs fasta, or None if assembly failed.
    """
    cmd = [
        "unicycler",
        "-1", r1_path,
        "-2", r2_path,
        "-o", out_dir,
        "-t", str(threads),
    ]

    print(f"Running Unicycler (short) on {r1_path} + {r2_path} ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Unicycler failed - {e.stderr}")
        return None

    assembly_path = os.path.join(out_dir, "assembly.fasta")
    if not os.path.exists(assembly_path):
        print(f"Error: Unicycler finished but no assembly.fasta found in {out_dir}")
        return None

    print(f"Assembly done: {assembly_path}")
    return assembly_path

def run_flye_long(long_read_path, out_dir, genome_size = "4.2m", threads=10):
    """
    Assembles a long-read genome using Flye.

    Args:
        long_reads_path (str): Path to long reads (Nanopore fastq/fastq.gz).
        out_dir (str): Directory to write assembly output.
        genome_size (str): Expected genome size for Flye, e.g. "4.2m" for Bacillus (default: "4.2m").
        threads (int): Number of CPU threads to use (default: 10).

    Returns: 
        str: Path to the assembled contigs fasta or None if assembly failed.
    """
    cmd = [
        "flye",
        "--nano-raw", long_read_path,
        "--out-dir", out_dir,
        "--genome-size", genome_size,
        "--threads", str(threads),
    ]

    print(f"Running Flye on {long_read_path} (excepted genome size {genome_size}) ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Flye failed - {e.stderr}")
        return None

    assembly_path = os.path.join(out_dir, "assembly.fasta")
    if not os.path.exists(assembly_path):
        print(f"Error: Flye finished but no assembly.fasta found in {out_dir}")
        return None
    
    print(f"Assembly done: {assembly_path}")
    return assembly_path

def run_unicycler_hybrid(r1_path, r2_path, long_reads_path, out_dir, threads=10):
    """
    Assembles a genome from both short and long reads using Unicycler's
    hybrid mode (short reads for accuracy, long reads for contiguity).

    Args:
        r1_path (str): Path to trimmed forward short reads (fastq.gz).
        r2_path (str): Path to trimmed reverse short reads (fastq.gz).
        long_reads_path (str): Path to long reads (fastq.gz).
        out_dir (str): Directory to write assembly output.
        threads (int): Number of CPU threads to use (default: 10).

    Returns:
        str: Path to the assembled contigs fasta, or None if assembly failed.
    """
    cmd = [
        "unicycler",
        "-1", r1_path,
        "-2", r2_path,
        "-l", long_reads_path,
        "-o", out_dir,
        "-t", str(threads),
    ]

    print(f"Running Unicycler (hybrid) on {r1_path} + {r2_path} + {long_reads_path} ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Unicycler (hybrid) failed - {e.stderr}")
        return None

    assembly_path = os.path.join(out_dir, "assembly.fasta")
    if not os.path.exists(assembly_path):
        print(f"Error: Unicycler (hybrid) finished but no assembly.fasta found in {out_dir}")
        return None

    print(f"Assembly done: {assembly_path}")
    return assembly_path


def assemble_genome(platform, out_dir, r1_path=None, r2_path=None, long_reads_path=None, threads=10):
    """
    Dispatches to the correct assembler based on the detected sequencing
    platform (see wgs_qc.peek_read_length for how platform is detected).

    Args:
        platform (str): "short", "long", or "hybrid".
        out_dir (str): Directory to write assembly output.
        r1_path (str): Forward short reads - required for "short"/"hybrid".
        r2_path (str): Reverse short reads - required for "short"/"hybrid".
        long_reads_path (str): Long reads - required for "long"/"hybrid".
        threads (int): Number of CPU threads to use (default: 10).

    Returns:
        str: Path to assembled contigs fasta, or None if assembly failed
            or required inputs for the chosen platform are missing.
    """
    if platform == "short":
        if not r1_path or not r2_path:
            print("Error: platform='short' requires both r1_path and r2_path")
            return None
        return run_unicycler_short(r1_path, r2_path, out_dir, threads=threads)

    if platform == "long":
        if not long_reads_path:
            print("Error: platform='long' requires long_reads_path")
            return None
        return run_flye_long(long_reads_path, out_dir, threads=threads)

    if platform == "hybrid":
        if not (r1_path and r2_path and long_reads_path):
            print("Error: platform='hybrid' requires r1_path, r2_path and long_reads_path")
            return None
        return run_unicycler_hybrid(r1_path, r2_path, long_reads_path, out_dir, threads=threads)

    print(f"Error: unknown platform '{platform}' (expected 'short', 'long' or 'hybrid')")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble a bacterial genome from QC'd reads.")
    parser.add_argument("-p", "--platform", required=True, choices=["short", "long", "hybrid"],
                         help="Sequencing platform branch to use")
    parser.add_argument("-1", "--r1", default=None, help="Forward short reads, fastq.gz (default: None)")
    parser.add_argument("-2", "--r2", default=None, help="Reverse short reads, fastq.gz (default: None)")
    parser.add_argument("-l", "--long", default=None, help="Long reads, fastq.gz (default: None)")
    parser.add_argument("-o", "--out-dir", default="results/assembly",
                         help="Output directory (default: results/assembly)")
    parser.add_argument("-t", "--threads", type=int, default=10, help="Number of threads (default: 10)")
    args = parser.parse_args()

    result = assemble_genome(
        args.platform,
        args.out_dir,
        r1_path=args.r1,
        r2_path=args.r2,
        long_reads_path=args.long,
        threads=args.threads,
    )

    if result:
        print(f"Final assembly: {result}")
    else:
        print("Assembly failed - see errors above.")