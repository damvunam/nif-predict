# wgs_loader.py
import argparse
import os
import glob

def find_raw_reads(data_dir, sample_name):
    """
    Finds raw fastq files for a sample in a directory.
    
    Args:
        data_dir (str): Folder containing raw fastq/fastq.gz files.
        sample_name (str): Sample prefix to match (e.g. "X35").

    Returns:
        list: Sorted list of matching fastq file paths, or None if none found.
    """
    # 1. Build search pattern (accept .fastq / .fq, gzipped or not)
    patterns = [
        os.path.join(data_dir, f"{sample_name}*.fastq"),
        os.path.join(data_dir, f"{sample_name}*.fastq.gz"),
        os.path.join(data_dir, f"{sample_name}*.fq"),
        os.path.join(data_dir, f"{sample_name}*.fq.gz")
    ]

    # 2. Collect all matches across patterns
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern))

    # 3. Report results
    if not found_files:
        print(f"Error: No fastq files found for '{sample_name}' in {data_dir}")
        return None

    found_files = sorted(found_files)
    print(f"Found {len(found_files)} file(s) for sample '{sample_name}'")
    return found_files

if __name__ == "__main__":
    # FIX (W11): replace hardcoded G1X sample and data directory with CLI arguments
    # # Test with your own data folder
    # test_dir = "resources/raw"
    # test_sample = "G1X"
    #
    # reads = find_raw_reads(test_dir, test_sample)
    parser = argparse.ArgumentParser(description="Find raw fastq files for a given sample.")
    parser.add_argument("-s", "--sample", required=True, help="Sample name prefix (required)")
    parser.add_argument("-d", "--data_dir", default="resources/raw", help="Folder containing raw fastq files (default: resources/raw)")
    args = parser.parse_args()

    reads = find_raw_reads(args.data_dir, args.sample)

    if reads:
        print("\nFiles found:")
        for f in reads:
            print(f" - {f}")
    else:
        print("\nNo files returned")