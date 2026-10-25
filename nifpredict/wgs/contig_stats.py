# nifpredict/wgs/contig_stats.py
import argparse
import os
import re
import statistics


def process_contig(header, seq):
    """Extracts metadata from header and computes GC% from sequence.

    Args:
        header: FASTA header line including '>'.
        seq: Nucleotide sequence string in uppercase.

    Returns:
        Dict with contig, length, gc_pct, depth, circular keys.
    """
    # 1. Extract contig identifier
    first_token = header[1:].split()[0]
    contig_name = first_token

    # 2. Extract length and compute GC% on A/C/G/T
    length = len(seq)
    g_count = seq.count("G")
    c_count = seq.count("C")
    a_count = seq.count("A")
    t_count = seq.count("T")
    acgt_total = a_count + c_count + g_count + t_count
    gc_pct = round(((g_count + c_count) / acgt_total) * 100, 2) if acgt_total > 0 else 0.0

    # 3. Extract depth from header (e.g. depth=0.96x)
    depth_match = re.search(r"depth=([0-9.]+)x?", header)
    if depth_match:
        try:
            depth = float(depth_match.group(1))
        except ValueError:
            depth = "NA"
    else:
        depth = "NA"

    # 4. Check circularity flag
    circular = bool(re.search(r"circular=true", header, re.IGNORECASE))

    return {
        "contig": contig_name,
        "length": length,
        "gc_pct": gc_pct,
        "depth": depth,
        "circular": circular,
    }


def parse_fasta(fasta_path):
    """Parses assembly FASTA and computes statistics per contig.

    Args:
        fasta_path: Path to the input FASTA file.

    Returns:
        List of dicts containing per-contig stats.
    """
    records = []
    current_header = None
    seq_chunks = []

    # 1. Read FASTA line by line
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    seq = "".join(seq_chunks).upper()
                    records.append(process_contig(current_header, seq))
                    seq_chunks = []
                current_header = line
            else:
                seq_chunks.append(line)

        # 2. Process final contig
        if current_header is not None:
            seq = "".join(seq_chunks).upper()
            records.append(process_contig(current_header, seq))

    return records


def write_tsv(records, out_path):
    """Writes contig records to a TSV file.

    Args:
        records: List of contig dicts.
        out_path: Path to output TSV file.
    """
    # 1. Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 2. Write headers and records
    with open(out_path, "w") as f:
        f.write("contig\tlength\tgc_pct\tdepth\tcircular\n")
        for r in records:
            depth_str = f"{r['depth']:.2f}" if isinstance(r["depth"], float) else str(r["depth"])
            f.write(f"{r['contig']}\t{r['length']}\t{r['gc_pct']:.2f}\t{depth_str}\t{r['circular']}\n")


def print_summary(records):
    """Prints contig statistics summaries to stdout.

    Args:
        records: List of contig dicts.
    """
    # 1. Total contigs and length
    total_contigs = len(records)
    total_length = sum(r["length"] for r in records)
    print(f"\nTotal contigs: {total_contigs}")
    print(f"Total length: {total_length:,} bp")

    # 2. Top 30 longest contigs
    sorted_records = sorted(records, key=lambda x: x["length"], reverse=True)
    top_30 = sorted_records[:30]
    print(f"\nTop {min(30, len(top_30))} longest contigs:")
    print(f"{'Contig':<10} {'Length (bp)':<14} {'GC (%)':<10} {'Depth':<10} {'Circular':<10}")
    print("-" * 54)
    for r in top_30:
        depth_str = f"{r['depth']:.2f}x" if isinstance(r["depth"], float) else str(r["depth"])
        print(f"{r['contig']:<10} {r['length']:<14,} {r['gc_pct']:<10.2f} {depth_str:<10} {str(r['circular']):<10}")

    # 3. Group by GC: <40, 40-43, 43-50, >=50
    gc_groups = {
        "<40": [r for r in records if r["gc_pct"] < 40.0],
        "40-43": [r for r in records if 40.0 <= r["gc_pct"] < 43.0],
        "43-50": [r for r in records if 43.0 <= r["gc_pct"] < 50.0],
        ">=50": [r for r in records if r["gc_pct"] >= 50.0],
    }
    print("\nContig grouping by GC%:")
    print(f"{'GC Group':<12} {'Contigs':<10} {'Total Length (bp)':<20} {'Median Depth':<14}")
    print("-" * 56)
    for grp_name, grp_list in gc_groups.items():
        grp_count = len(grp_list)
        grp_len = sum(r["length"] for r in grp_list)
        numeric_depths = [r["depth"] for r in grp_list if isinstance(r["depth"], float)]
        med_depth_str = f"{statistics.median(numeric_depths):.2f}x" if numeric_depths else "NA"
        print(f"{grp_name:<12} {grp_count:<10} {grp_len:<20,} {med_depth_str:<14}")

    # 4. Group by depth: <0.5x, 0.5-1.5x, >1.5x
    depth_groups = {
        "<0.5x": [r for r in records if isinstance(r["depth"], float) and r["depth"] < 0.5],
        "0.5-1.5x": [r for r in records if isinstance(r["depth"], float) and 0.5 <= r["depth"] <= 1.5],
        ">1.5x": [r for r in records if isinstance(r["depth"], float) and r["depth"] > 1.5],
    }
    na_depths = [r for r in records if not isinstance(r["depth"], float)]
    if na_depths:
        depth_groups["NA"] = na_depths

    print("\nContig grouping by Depth:")
    print(f"{'Depth Group':<14} {'Contigs':<10} {'Total Length (bp)':<20} {'Median GC (%)':<14}")
    print("-" * 58)
    for grp_name, grp_list in depth_groups.items():
        grp_count = len(grp_list)
        grp_len = sum(r["length"] for r in grp_list)
        gcs = [r["gc_pct"] for r in grp_list]
        med_gc_str = f"{statistics.median(gcs):.2f}%" if gcs else "NA"
        print(f"{grp_name:<14} {grp_count:<10} {grp_len:<20,} {med_gc_str:<14}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-contig statistics (length, GC%, depth, circularity).")
    parser.add_argument("-i", "--assembly", required=True, help="Path to input assembly FASTA file")
    # Default is resolved dynamically if None
    parser.add_argument("-o", "--out_path", default=None, help="Path to output TSV file (default: <assembly_dir>/contig_stats.tsv)")
    args = parser.parse_args()

    assembly_path = args.assembly
    if not os.path.exists(assembly_path):
        print(f"Error: assembly file not found at {assembly_path}")
        exit(1)

    out_path = args.out_path
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(assembly_path))
        out_path = os.path.join(out_dir, "contig_stats.tsv")

    records = parse_fasta(assembly_path)
    write_tsv(records, out_path)
    print_summary(records)
    print(f"\nOutput TSV saved to: {out_path}")