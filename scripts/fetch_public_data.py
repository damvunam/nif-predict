#!/usr/bin/env python3
"""Re-download public data listed in data/public/MANIFEST.tsv and verify md5.

Each manifest row carries a fetch_cmd template with placeholders:
  {root}            destination directory for the row (default: <repo>/<local_root>)
  {accessions_file} file listing every accession of the row's group (group download)
Rows of the same group that share one fetch_cmd are fetched once.

Run from the repo root: PYTHONPATH=. .venv/bin/python scripts/fetch_public_data.py ...

Examples:
  python scripts/fetch_public_data.py --list
  python scripts/fetch_public_data.py --verify                      # md5 of files on disk; absent files = MISSING, not FAIL
  python scripts/fetch_public_data.py --group controls --dest /tmp/fetch_test --dry-run
  python scripts/fetch_public_data.py --id PSON_RAW_1 PSON_RAW_2 PSON_1 PSON_2
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from nifpredict.common import paths

REPO = paths.REPO_ROOT
MANIFEST = paths.MANIFEST


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_gz_content(path: Path) -> str:
    h = hashlib.md5()
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_file(row: dict, root: Path) -> Path:
    """Where the row's file lands under root (same path relative to local_root)."""
    rel = Path(row["local_file"]).relative_to(row["local_root"])
    return root / rel


def check_row(row: dict, path: Path) -> tuple[bool, str]:
    """Verify one file against the manifest md5 (md5_content for derived .gz)."""
    if not path.exists():
        return False, "missing"
    if row["md5_content"]:
        got = md5_gz_content(path)
        return got == row["md5_content"], f"md5_content {got}"
    got = md5_file(path)
    return got == row["md5"], f"md5 {got}"


def select(rows: list[dict], args) -> list[dict]:
    out = rows
    if args.id:
        out = [r for r in out if r["id"] in set(args.id)]
    if args.group:
        out = [r for r in out if r["group"] in set(args.group)]
    if args.kind:
        out = [r for r in out if r["kind"] in set(args.kind)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--id", nargs="+")
    ap.add_argument("--group", nargs="+")
    ap.add_argument("--kind", nargs="+")
    ap.add_argument("--dest", type=Path, default=None,
                    help="write under DEST/<local_root> instead of the repo (recommended for tests)")
    ap.add_argument("--list", action="store_true", help="list rows and exit")
    ap.add_argument("--verify", action="store_true", help="only verify md5 of existing files, do not download")
    ap.add_argument("--dry-run", action="store_true", help="print commands, do not run them")
    args = ap.parse_args()

    rows = select(read_manifest(args.manifest), args)
    if not rows:
        print("Error: no manifest row selected")
        return 2
    if args.list:
        for r in rows:
            print(f"{r['id']}\t{r['group']}\t{r['kind']}\t{r['accession']}\t{r['local_file']}")
        return 0

    base = args.dest if args.dest else REPO
    done_cmds: set[str] = set()
    failed = 0
    missing = 0
    for r in rows:
        root = base / r["local_root"]
        if not args.verify:
            cmd = r["fetch_cmd"]
            acc_file = ""
            if "{accessions_file}" in cmd:
                group_accs = [x["accession"] for x in read_manifest(args.manifest) if x["group"] == r["group"]]
                tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
                tmp.write("\n".join(group_accs) + "\n")
                tmp.close()
                acc_file = tmp.name
            key = r["group"] + "|" + cmd
            if key not in done_cmds:
                done_cmds.add(key)
                full = cmd.format(root=root, accessions_file=acc_file)
                print(f"[{r['id']}] {full}")
                if not args.dry_run:
                    root.mkdir(parents=True, exist_ok=True)
                    # no pipefail: in "zcat | head" zcat is stopped by SIGPIPE on purpose; the md5 check below catches real failures
                    rc = subprocess.run(["bash", "-c", full]).returncode
                    if rc != 0:
                        print(f"Error: command failed for {r['id']} (exit {rc})")
                        failed += 1
                        continue
        if args.dry_run:
            continue
        if r["verified"].startswith("not_downloaded") and args.verify:
            print(f"  {r['id']}: not downloaded yet (skip)")
            continue
        path = target_file(r, root)
        if args.verify and not path.exists():
            print(f"  {r['id']}: MISSING (re-download: --id {r['id']})")
            missing += 1
            continue
        ok, info = check_row(r, path)
        print(f"  {r['id']}: {'OK' if ok else 'FAIL'} ({info})")
        failed += 0 if ok else 1
    if failed:
        print(f"FAILED: {failed}  missing: {missing}")
        return 1
    print(f"DONE  missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
