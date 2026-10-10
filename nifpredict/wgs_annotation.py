# wgs_annotation.py
import os
import glob
import subprocess
import argparse

def run_bakta(assembly_path, out_dir, db_path, threads=10):
    """
    Annotates a bacterial genome using Bakta. Requires a pre-downloaded
    Bakta reference database (db_path) - Bakta does not fetch it itself.

    Args:
        assembly_path: path to the assembled genome (assembly.fasta)
        out_dir: output directory for annotation results
        db_path: path to the local Bakta database
        threads: number of threads to use (default: 10)

    Returns:
        (faa_path, gff_path) tuple, or None if Bakta failed.
    """
    cmd = ["bakta", "--db", db_path, "--output", out_dir,
           "--threads", str(threads), "--force", assembly_path]
    print(f"Running Bakta on {assembly_path} (db={db_path}) ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Bakta failed - {e.stderr}")
        return None

    faa_matches = glob.glob(os.path.join(out_dir, "*.faa"))
    gff_matches = glob.glob(os.path.join(out_dir, "*.gff3"))
    if not faa_matches or not gff_matches:
        print(f"Error: Bakta finished but no .faa/.gff3 found in {out_dir}")
        return None
    print(f"Annotation done: {faa_matches[0]}, {gff_matches[0]}")
    return faa_matches[0], gff_matches[0]


def run_prokka(assembly_path, out_dir, prefix="genome", threads=10):
    """
    Annotates a bacterial genome using Prokka - lighter than Bakta,
    no separate database download needed.

    Args:
        assembly_path: path to the assembled genome (assembly.fasta)
        out_dir: output directory for annotation results
        prefix: prefix for Prokka's output files (default: "genome")
        threads: number of threads to use (default: 10)

    Returns:
        (faa_path, gff_path) tuple, or None if Prokka failed.
    """
    cmd = ["prokka", "--outdir", out_dir, "--prefix", prefix,
           "--cpus", str(threads), "--force", assembly_path]
    print(f"Running Prokka on {assembly_path} ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Prokka failed - {e.stderr}")
        return None

    faa_path = os.path.join(out_dir, f"{prefix}.faa")
    gff_path = os.path.join(out_dir, f"{prefix}.gff")
    if not os.path.exists(faa_path) or not os.path.exists(gff_path):
        print(f"Error: Prokka finished but no {prefix}.faa/.gff found in {out_dir}")
        return None
    print(f"Annotation done: {faa_path}, {gff_path}")
    return faa_path, gff_path


def annotate_genome(tool, assembly_path, out_dir, db_path=None, prefix="genome", threads=10):
    """
    Dispatches to the correct annotation tool.

    Args:
        tool: "bakta" or "prokka"
        assembly_path: path to the assembled genome (assembly.fasta)
        out_dir: output directory for annotation results
        db_path: path to Bakta database (required if tool="bakta")
        prefix: output file prefix (used by Prokka)
        threads: number of threads to use (default: 10)

    Returns:
        (faa_path, gff_path) tuple, or None if annotation failed.
    """
    if tool == "bakta":
        if not db_path:
            print("Error: tool='bakta' requires db_path")
            return None
        return run_bakta(assembly_path, out_dir, db_path, threads=threads)
    if tool == "prokka":
        return run_prokka(assembly_path, out_dir, prefix=prefix, threads=threads)
    print(f"Error: unknown tool '{tool}' (expected 'bakta' or 'prokka')")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate a bacterial genome assembly.")
    parser.add_argument("-i", "--assembly", required=True, help="Path to assembly fasta")
    parser.add_argument("--tool", default="bakta", choices=["bakta", "prokka"],
                         help="Annotation tool to use (default: bakta)")
    parser.add_argument("--db", default=None, help="Path to Bakta database (required if --tool bakta)")
    parser.add_argument("-p", "--prefix", default="genome", help="Output file prefix for Prokka (default: genome)")
    parser.add_argument("-o", "--out-dir", default="results/annotation",
                         help="Output directory (default: results/annotation)")
    parser.add_argument("-t", "--threads", type=int, default=10, help="Number of threads (default: 10)")
    args = parser.parse_args()

    result = annotate_genome(args.tool, args.assembly, args.out_dir,
                              db_path=args.db, prefix=args.prefix, threads=args.threads)
    if result:
        faa_path, gff_path = result
        print(f"Final annotation: {faa_path}, {gff_path}")
    else:
        print("Annotation failed - see errors above.")