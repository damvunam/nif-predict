#!/usr/bin/env bash
# check_baseline.sh — regression check after any change to the WGS / nif code.
#   1. pytest (project venv: .venv/bin/python, override with PYTEST_PYTHON=...)
#   2. PSON runner on the existing outputs in results/wgs/PSON/: must print DONE, exit 0 and create no file
#      (stops BEFORE running if an output that would trigger a heavy step - fastp, assembly, QUAST/BUSCO, skani - is missing)
#   3. check_nif on the 5 controls: per-gene lines + VERDICT and tblastn_hits.tsv must equal tests/baseline/<sample>/
# Run from anywhere in the conda env that has the tools (fastp, unicycler, quast, busco, skani, blast):
#   bash scripts/check_baseline.sh
# Output: results/checks/<timestamp>/ (summary.txt + logs). Exit 0 only if every check passes.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
PYTEST_PYTHON="${PYTEST_PYTHON:-.venv/bin/python}"
OUT="results/checks/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
exec > >(tee "$OUT/summary.txt") 2>&1
echo "DATE: $(date -Is)  BRANCH: $(git branch --show-current)  HEAD: $(git rev-parse --short HEAD)"
git status --porcelain

if pgrep -af "unicycler|spades|busco|prokka|fastp|wget|aria2c" > "$OUT/pgrep_before.log"; then
  echo "STOP: heavy job running (see $OUT/pgrep_before.log)"; exit 2
fi

# Paths come from nifpredict/common/paths.py only
eval "$(python - <<'PY'
from nifpredict.common import paths
pson = paths.WGS_RESULTS / "PSON"
print(f'READS_DIR="{paths.reads_dir("SRR17868980")}"')
print(f'PSON_REF="{paths.genome_fna("GCF_021312615.1")}"')
print(f'PSON_DIR="{pson}"')
print(f'PSON_ASM="{pson / "assembly" / "assembly.fasta"}"')
print(f'BASELINE="{paths.BASELINE_DIR}"')
for name, acc in (("CTRL_AVIN", "GCF_000021045.1"), ("CTRL_PWLY", "GCF_000463565.1"), ("CTRL_BSUB", "GCF_000009045.1")):
    print(f'{name}="{paths.genome_fna(acc)}"')
needed = [pson / "qc" / "PSON_R1.trim.fastq.gz", pson / "qc" / "PSON_R2.trim.fastq.gz", pson / "qc" / "PSON.fastp.json",
          pson / "assembly" / "assembly.fasta", pson / "assembly" / "contig_stats.tsv", pson / "assembly_qc" / "report.tsv",
          pson / "identity" / f"skani_vs_{paths.genome_fna('GCF_021312615.1').stem}.tsv"]
print("MISSING=" + ",".join(str(p) for p in needed if not p.exists()))
PY
)"
fail=0

echo "== 1. pytest"
set +e
"$PYTEST_PYTHON" -m pytest -p no:cacheprovider -o addopts="-ra" > "$OUT/pytest.log" 2>&1; rc=$?
set -e
echo "pytest: $(grep -E '[0-9]+ (passed|failed)' "$OUT/pytest.log" | tail -1)  EXIT=$rc"
grep -q -E '[0-9]+ skipped' "$OUT/pytest.log" && echo "WARNING: some tests were skipped"
[ "$rc" -eq 0 ] || fail=1

echo "== 2. PSON runner (existing outputs only)"
if [ -n "$MISSING" ]; then
  echo "STOP-PSON: missing outputs, the runner would start a heavy step: $MISSING"; fail=1
else
  find "$PSON_DIR" -type f | sort > "$OUT/pson_files_before.txt"
  set +e
  python -m nifpredict.wgs.run_sample -s PSON -d "$READS_DIR" -r "$PSON_REF" > "$OUT/runner_pson.log" 2>&1; rc=$?
  set -e
  find "$PSON_DIR" -type f | sort > "$OUT/pson_files_after.txt"
  grep -E '^(COVERAGE|IDENTITY|DONE)' "$OUT/runner_pson.log" || true
  echo "runner EXIT=$rc"
  if [ "$rc" -ne 0 ] || ! grep -q '^DONE: PSON' "$OUT/runner_pson.log"; then echo "FAIL: runner"; fail=1; fi
  if ! diff -q "$OUT/pson_files_before.txt" "$OUT/pson_files_after.txt" > /dev/null; then echo "FAIL: runner created/removed files"; fail=1; fi
fi

echo "== 3. check_nif on 5 controls vs tests/baseline"
declare -A ASM=([CTRL_AVIN]="$CTRL_AVIN" [CTRL_PWLY]="$CTRL_PWLY" [CTRL_BSUB]="$CTRL_BSUB" [PSON]="$PSON_ASM" [PSON_REF]="$PSON_REF")
for s in CTRL_AVIN CTRL_PWLY CTRL_BSUB PSON PSON_REF; do
  set +e; python -m nifpredict.nif.check_nif -s "$s" -a "${ASM[$s]}" -o "$OUT/nif" > /dev/null 2>&1; rc=$?; set -e
  d="$OUT/nif/$s/assembly"
  grep -E "Reference proteins per gene|^\[[^]]*\]   nif[HDK]: |VERDICT:" "$d/${s}_nif_check.log" 2>/dev/null | cut -c23- > "$d/verdict.txt" || true
  v=same; h=same
  cmp -s "$BASELINE/$s/verdict.txt" "$d/verdict.txt" || { v=DIFFERENT; fail=1; }
  cmp -s "$BASELINE/$s/tblastn_hits.tsv" "$d/tblastn_hits.tsv" || { h=DIFFERENT; fail=1; }
  echo "$s EXIT=$rc  verdict+genes: $v  tblastn_hits: $h  ($(grep VERDICT "$d/verdict.txt" 2>/dev/null || echo 'no VERDICT'))"
done

[ "$fail" -eq 0 ] && echo "ALL CHECKS PASSED ($OUT)" || { echo "SOME CHECKS FAILED ($OUT)"; exit 1; }
