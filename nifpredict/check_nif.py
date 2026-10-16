# check_nif.py
import os
import csv
import argparse
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime

NIF_GENES = ["nifH", "nifD", "nifK"]


def write_log(log_path, text):
    """Prints a line and appends it (with timestamp) to the evidence log."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, log_path):
    """
    Runs an external command and records it in the log.

    Args:
        cmd (list): Command and arguments.
        log_path (str): Evidence log file.

    Returns:
        bool: True if the command finished with exit code 0, False otherwise.
    """
    write_log(log_path, "CMD: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        write_log(log_path, "  -> exit 0")
        return True
    except FileNotFoundError:
        write_log(log_path, f"Error: '{cmd[0]}' not found. Is the conda env activated?")
        return False
    except subprocess.CalledProcessError as e:
        write_log(log_path, f"Error: exit {e.returncode}\n{e.stderr}")
        return False


def download_nif_references(ref_path, log_path):
    """
    Downloads reviewed (Swiss-Prot) NifH/NifD/NifK proteins from UniProt.

    Only the query (gene names) is sent out; no sample data leaves the machine.

    Args:
        ref_path (str): Output protein FASTA.
        log_path (str): Evidence log file.

    Returns:
        str: ref_path, or None on failure.
    """
    # 1. Reuse an existing download so reruns give identical references
    if os.path.exists(ref_path) and os.path.getsize(ref_path) > 0:
        write_log(log_path, f"Reusing references: {ref_path}")
        return ref_path

    # 2. Download from UniProt REST
    # FIX: gene_exact:nifH missed Swiss-Prot entries whose primary gene name carries a copy number
    # (e.g. P00459 NIFH1_AZOVI = nifH1, Q9AKT4 NIFH2_PAEDU = nifH2); load_gene_map already accepts these
    # query = "(gene_exact:nifH OR gene_exact:nifD OR gene_exact:nifK) AND (reviewed:true)"
    query = (
        "(gene_exact:nifH OR gene_exact:nifH1 OR gene_exact:nifH2 OR gene_exact:nifH3"
        " OR gene_exact:nifD OR gene_exact:nifD1 OR gene_exact:nifD2"
        " OR gene_exact:nifK OR gene_exact:nifK1 OR gene_exact:nifK2) AND (reviewed:true)"
    )
    url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=" + urllib.parse.quote(query)
    write_log(log_path, f"Downloading references: {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            fasta_text = response.read().decode("utf-8")
    except Exception as e:
        write_log(log_path, f"Error: UniProt download failed: {e}")
        return None

    # 3. Save and report how many sequences per gene
    with open(ref_path, "w") as f:
        f.write(fasta_text)
    gene_map = load_gene_map(ref_path)
    counts = {g: list(gene_map.values()).count(g) for g in NIF_GENES}
    write_log(log_path, f"Saved {len(gene_map)} reference proteins {counts} -> {ref_path}")
    return ref_path


def load_gene_map(ref_path):
    """
    Maps each reference sequence ID to nifH / nifD / nifK using the UniProt 'GN=' tag.

    Returns:
        dict: {seq_id: gene}, e.g. {"sp|P00459|NIFH1_AZOVI": "nifH"}.
    """
    gene_map = {}
    with open(ref_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            seq_id = line[1:].split()[0]
            gene = "other"
            for token in line.split():
                if token.startswith("GN="):
                    name = token[3:].lower()
                    for g in NIF_GENES:
                        # FIX: was name.startswith(g) -> "nifHD" counted as nifH.
                        # Accept only the gene name plus an optional copy number (nifH, nifH1, nifH2).
                        # if name.startswith(g.lower()):
                        if name == g.lower() or (name.startswith(g.lower()) and name[len(g):].isdigit()):
                            gene = g
            gene_map[seq_id] = gene
    return gene_map


def run_tblastn(ref_path, assembly_path, out_dir, threads, log_path):
    """
    Searches reference Nif proteins against the assembly contigs with tblastn.

    Returns:
        str: Path to the tabular hits file, or None on failure.
    """
    # 1. Build nucleotide database from the contigs
    db_prefix = os.path.join(out_dir, "assembly_db")
    if not run_cmd(["makeblastdb", "-in", assembly_path, "-dbtype", "nucl", "-out", db_prefix], log_path):
        return None

    # 2. Search
    hits_path = os.path.join(out_dir, "tblastn_hits.tsv")
    cmd = [
        "tblastn", "-query", ref_path, "-db", db_prefix,
        "-outfmt", "6 qseqid sseqid pident length qlen evalue bitscore sstart send",
        "-evalue", "1e-5",
        "-max_target_seqs", "5",
        "-num_threads", str(threads),
        "-out", hits_path,
    ]
    if not run_cmd(cmd, log_path):
        return None
    return hits_path


def summarize_tblastn(hits_path, gene_map):
    """
    Keeps the best hit (highest bitscore) per gene.

    Returns:
        dict: {gene: {"contig", "pident", "coverage", "start", "end", "ref"}} or {} if no hits.
    """
    best = {}
    with open(hits_path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 9:
                continue
            qseqid, sseqid, pident, length, qlen, evalue, bitscore, sstart, send = row[:9]
            gene = gene_map.get(qseqid, "other")
            if gene not in NIF_GENES:
                continue
            bitscore = float(bitscore)
            if gene in best and best[gene]["bitscore"] >= bitscore:
                continue
            best[gene] = {
                "ref": qseqid,
                "contig": sseqid,
                "pident": float(pident),
                "coverage": int(length) / int(qlen),
                "start": min(int(sstart), int(send)),
                "end": max(int(sstart), int(send)),
                "bitscore": bitscore,
            }
    return best


def run_diamond_reads(ref_path, reads_paths, out_dir, threads, log_path):
    """
    Translates trimmed reads and searches them against the Nif references (diamond blastx).

    Used when no assembly exists yet. Reads stay on this machine.

    Returns:
        str: Path to the combined hits file, or None on failure.
    """
    # 1. Protein database from the references
    db_prefix = os.path.join(out_dir, "nif_ref")
    if not run_cmd(["diamond", "makedb", "--in", ref_path, "-d", db_prefix], log_path):
        return None

    # 2. One search per read file (R1, R2), then concatenate
    hits_path = os.path.join(out_dir, "diamond_read_hits.tsv")
    part_paths = []
    for i, reads_path in enumerate(reads_paths):
        part_path = os.path.join(out_dir, f"diamond_part{i + 1}.tsv")
        cmd = [
            "diamond", "blastx", "-d", db_prefix, "-q", reads_path,
            "-o", part_path,
            "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "evalue", "bitscore",
            "--evalue", "1e-5",
            "--max-target-seqs", "1",
            "--threads", str(threads),
        ]
        if not run_cmd(cmd, log_path):
            return None
        part_paths.append(part_path)

    # 3. Merge
    with open(hits_path, "w") as out:
        for part_path in part_paths:
            with open(part_path) as f:
                out.write(f.read())
    return hits_path


def summarize_reads(hits_path, gene_map,
                    min_pident=60.0,   # true Nif homologs; paralogs (BchL/ChlL, BchN/B) sit ~25-40%
                    min_aln_len=40):    # 150bp read -> max 50 aa, so 40 aa = near full-length alignment
    """
    Counts reads per gene that pass identity/length filters.

    Returns:
        dict: {gene: {"reads": int, "reads_loose": int, "median_pident": float}}.
    """
    pidents = {g: [] for g in NIF_GENES}
    loose = {g: 0 for g in NIF_GENES}
    with open(hits_path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 6:
                continue
            gene = gene_map.get(row[1], "other")
            if gene not in NIF_GENES:
                continue
            loose[gene] += 1
            if float(row[2]) >= min_pident and int(row[3]) >= min_aln_len:
                pidents[gene].append(float(row[2]))

    summary = {}
    for g in NIF_GENES:
        values = sorted(pidents[g])
        median = values[len(values) // 2] if values else 0.0
        summary[g] = {"reads": len(values), "reads_loose": loose[g], "median_pident": median}
    return summary


def is_nif_present_assembly(best,
                            min_pident=50.0,     # NifH/D/K vs nearest Swiss-Prot ortholog usually >60%
                            min_coverage=0.7,    # allow partial genes at contig edges
                            max_gap=20000):      # nifHDK sit in one operon, a few kb apart
    """
    Decides presence from assembly hits.

    Returns:
        tuple: (verdict str, list of reason strings).
    """
    reasons = []
    passed = []
    for g in NIF_GENES:
        hit = best.get(g)
        if hit is None:
            reasons.append(f"{g}: no hit")
        elif hit["pident"] < min_pident or hit["coverage"] < min_coverage:
            reasons.append(f"{g}: weak hit ({hit['pident']:.1f}% id, {hit['coverage']:.0%} cov) - likely paralog")
        else:
            passed.append(g)
            reasons.append(f"{g}: OK ({hit['pident']:.1f}% id, {hit['coverage']:.0%} cov, {hit['contig']}:{hit['start']}-{hit['end']})")

    if len(passed) == 0:
        return "ABSENT", reasons
    if len(passed) < 3:
        return "PARTIAL - check manually", reasons

    # FIX: nifD and nifK are paralogs (same Pfam PF00148); make sure their best hits
    # are two different loci, not the same gene matched twice.
    for g1, g2 in [("nifH", "nifD"), ("nifH", "nifK"), ("nifD", "nifK")]:
        a, b = best[g1], best[g2]
        min_overlap = 50  # neighbouring genes in an operon can share a few bp at start/stop codons
        if a["contig"] == b["contig"] and a["start"] + min_overlap < b["end"] and b["start"] + min_overlap < a["end"]:
            reasons.append(f"{g1} and {g2} best hits overlap on {a['contig']} - same locus counted twice")
            return "PARTIAL - check manually", reasons

    # all three found: are they together like an operon?
    contigs = {best[g]["contig"] for g in NIF_GENES}
    if len(contigs) == 1:
        positions = [best[g]["start"] for g in NIF_GENES] + [best[g]["end"] for g in NIF_GENES]
        if max(positions) - min(positions) <= max_gap:
            return "PRESENT (nifHDK clustered)", reasons
    reasons.append("nifH/D/K not on one contig within max_gap - could be assembly break")
    return "PRESENT (not clustered) - check manually", reasons


def is_nif_present_reads(summary,
                         min_reads=20):  # a real single-copy gene at ~100x gives hundreds of reads
    """Decides presence from read hits. Returns (verdict str, list of reason strings)."""
    reasons = []
    passed = []
    for g in NIF_GENES:
        s = summary[g]
        reasons.append(f"{g}: {s['reads']} reads >=60% id (median {s['median_pident']:.1f}%), {s['reads_loose']} reads any identity")
        if s["reads"] >= min_reads:
            passed.append(g)
    if len(passed) == 0:
        return "ABSENT", reasons
    if len(passed) < 3:
        return "PARTIAL - check manually", reasons
    return "PRESENT (reads only - confirm with assembly)", reasons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether a WGS sample carries nifH/nifD/nifK.")
    parser.add_argument("-s", "--sample", default="G13", help="Sample name (default: G13)")
    parser.add_argument("-r", "--reads_dir", default="results/qc", help="Folder with {sample}_R1/_R2.trim.fastq.gz (default: results/qc)")
    parser.add_argument("-a", "--assembly", default=None, help="Assembly FASTA; if given, uses tblastn (default: none -> reads mode)")
    parser.add_argument("-o", "--out_dir", default="results/nif_check", help="Output folder (default: results/nif_check)")
    # FIX: default was a fixed 10; never ask for more threads than the machine has
    parser.add_argument("-t", "--threads", type=int, default=min(10, os.cpu_count() or 1), help="Threads (default: min(10, nproc))")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, args.sample)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{args.sample}_nif_check.log")
    write_log(log_path, f"=== nif check for {args.sample} ===")

    ref_path = download_nif_references(os.path.join(args.out_dir, "nif_ref_swissprot.faa"), log_path)
    if ref_path:
        gene_map = load_gene_map(ref_path)

        # FIX: guard against a silent false ABSENT - if any gene has 0 references
        # (bad UniProt query / header parsing), no hit can ever be counted for it.
        ref_counts = {g: list(gene_map.values()).count(g) for g in NIF_GENES}
        write_log(log_path, f"Reference proteins per gene: {ref_counts}")
        if min(ref_counts.values()) == 0:
            write_log(log_path, "Error: a nif gene has 0 reference proteins - result would be meaningless, stopping")
            ref_path = None

    if ref_path:

        if args.assembly and os.path.exists(args.assembly):
            write_log(log_path, f"Mode: assembly ({args.assembly})")
            hits_path = run_tblastn(ref_path, args.assembly, out_dir, args.threads, log_path)
            if hits_path:
                verdict, reasons = is_nif_present_assembly(summarize_tblastn(hits_path, gene_map))
        else:
            if args.assembly:
                write_log(log_path, f"Warning: assembly not found ({args.assembly}), falling back to reads")
            reads_paths = [os.path.join(args.reads_dir, f"{args.sample}_R{i}.trim.fastq.gz") for i in (1, 2)]
            reads_paths = [p for p in reads_paths if os.path.exists(p)]
            write_log(log_path, f"Mode: reads ({reads_paths})")
            hits_path = run_diamond_reads(ref_path, reads_paths, out_dir, args.threads, log_path) if reads_paths else None
            if hits_path:
                verdict, reasons = is_nif_present_reads(summarize_reads(hits_path, gene_map))
            elif not reads_paths:
                write_log(log_path, f"Error: no trimmed reads found in {args.reads_dir}")

        if hits_path:
            for reason in reasons:
                write_log(log_path, "  " + reason)
            write_log(log_path, f"VERDICT: {verdict}")
            write_log(log_path, f"Evidence: {hits_path}, {ref_path}, {log_path}")
