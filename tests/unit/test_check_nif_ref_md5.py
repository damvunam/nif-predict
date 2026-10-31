"""Unit tests for check_nif reference-md5 checking (MU8: items 5, 7).

Style follows tests/unit/test_check_nif_exit.py: call main([...]), check
BOTH the return code AND the log lines. No network, no real BLAST.

The new function is only reached through getattr(check_nif, "<name>")
inside the test body (never imported at the top), so each new-behaviour
test fails on code that does not have the function yet.

A temporary MANIFEST is built through monkeypatching paths.MANIFEST and
paths.REPO_ROOT (both read at call time, never copied at import time).
"""

from __future__ import annotations

import hashlib

import pytest

from nifpredict.common import paths
from nifpredict.nif import check_nif
from nifpredict.nif.check_nif import main

SAMPLE = "TESTX"

REF_FULL = (
    ">sp|T00001|NIFH_TEST Test NifH OS=Test OX=1 GN=nifH PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
    ">sp|T00002|NIFD_TEST Test NifD OS=Test OX=1 GN=nifD PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
    ">sp|T00003|NIFK_TEST Test NifK OS=Test OX=1 GN=nifK PE=1 SV=1\n"
    "MAAAAAAAAAA\n"
)

EXPECTED_REAL_MD5 = "e4a328f996ce5709fe8ec9751152f086"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _assembly(tmp_path):
    return _write(tmp_path / "asm.fna", ">contig1\nACGTACGTACGTACGT\n")


def _log_text(out_dir):
    logs = sorted(out_dir.rglob("*_nif_check.log"))
    assert logs, f"no nif_check log under {out_dir}"
    return logs[0].read_text()


def _valid_hits(tmp_path):
    rows = []
    for ref_id, start, end in (
        ("sp|T00001|NIFH_TEST", 100, 1000),
        ("sp|T00002|NIFD_TEST", 1200, 2100),
        ("sp|T00003|NIFK_TEST", 2300, 3200),
    ):
        rows.append("\t".join(str(x) for x in (
            ref_id, "contig1", "99.0", 300, 300, "1e-50", "500", start, end, 1, 300)))
    return _write(tmp_path / "hits11.tsv", "\n".join(rows) + "\n")


def _tmp_manifest_setup(tmp_path, manifest_md5):
    """Builds a fake repo root with a ref file + MANIFEST row for it.

    Returns (ref_path, manifest_path, repo_root, actual_md5).
    """
    repo_root = tmp_path / "repo"
    ref_dir = repo_root / "data" / "public" / "proteins" / "test_ref"
    ref_path = _write(ref_dir / "ref.faa", REF_FULL)
    actual_md5 = hashlib.md5(ref_path.read_bytes()).hexdigest()
    if manifest_md5 is None:
        manifest_md5 = actual_md5
    header = ("id\tgroup\tkind\taccession\tsource\tlocal_root\tlocal_file\t"
              "download_date\tmd5\tmd5_content\tverified\treproducible\tfetch_cmd\tnotes\n")
    row = ("TEST_REF\ttest\tprotein_ref\t-\t-\tdata/public/proteins/test_ref\t"
           f"data/public/proteins/test_ref/ref.faa\t-\t{manifest_md5}\t\t-\t-\t-\t-\n")
    manifest_path = _write(repo_root / "data" / "public" / "MANIFEST.tsv", header + row)
    return ref_path, manifest_path, repo_root, actual_md5


def _patch_manifest(monkeypatch, manifest_path, repo_root):
    monkeypatch.setattr(paths, "MANIFEST", manifest_path)
    monkeypatch.setattr(paths, "REPO_ROOT", repo_root)


def _run_with_tmp_ref(tmp_path, monkeypatch, ref_path):
    hits = _valid_hits(tmp_path)
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = ["-s", SAMPLE, "--ref", str(ref_path), "-o", str(tmp_path / "out"),
            "-a", str(_assembly(tmp_path))]
    return main(args)


# --- Item 5: md5 mismatch / match / unlisted -------------------------------

def test_ref_md5_mismatch(tmp_path, monkeypatch):
    ref_path, manifest_path, repo_root, actual_md5 = _tmp_manifest_setup(
        tmp_path, manifest_md5="0" * 32)
    _patch_manifest(monkeypatch, manifest_path, repo_root)
    rc = _run_with_tmp_ref(tmp_path, monkeypatch, ref_path)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    assert "VERDICT:" not in log
    assert f"Reference md5: {actual_md5}" in log
    assert f"Error: reference md5 {actual_md5} != MANIFEST {'0' * 32} (TEST_REF)" in log
    assert "ERROR: no VERDICT - reference md5 mismatch" in log


def test_ref_md5_match(tmp_path, monkeypatch):
    ref_path, manifest_path, repo_root, actual_md5 = _tmp_manifest_setup(
        tmp_path, manifest_md5=None)
    _patch_manifest(monkeypatch, manifest_path, repo_root)
    rc = _run_with_tmp_ref(tmp_path, monkeypatch, ref_path)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert f"Reference md5: {actual_md5}" in log
    assert "Reference verified against MANIFEST: TEST_REF" in log
    assert "ERROR: no VERDICT" not in log
    assert "VERDICT: PRESENT (nifHDK clustered)" in log


def test_ref_not_in_manifest_warns_and_continues(tmp_path, monkeypatch):
    # MANIFEST with only an unrelated row: the tmp ref is unlisted.
    repo_root = tmp_path / "repo"
    ref_path = _write(repo_root / "elsewhere" / "ref.faa", REF_FULL)
    header = ("id\tgroup\tkind\taccession\tsource\tlocal_root\tlocal_file\t"
              "download_date\tmd5\tmd5_content\tverified\treproducible\tfetch_cmd\tnotes\n")
    row = ("OTHER\tother\tprotein_ref\t-\t-\tdata/public/proteins/other\t"
           "data/public/proteins/other/ref.faa\t-\t" + "1" * 32 + "\t\t-\t-\t-\t-\n")
    manifest_path = _write(repo_root / "data" / "public" / "MANIFEST.tsv", header + row)
    _patch_manifest(monkeypatch, manifest_path, repo_root)
    rc = _run_with_tmp_ref(tmp_path, monkeypatch, ref_path)
    log = _log_text(tmp_path / "out")
    assert rc == 0
    assert "Warning: reference not listed in MANIFEST, md5 not checked" in log
    assert "VERDICT: PRESENT (nifHDK clustered)" in log
    assert "ERROR: no VERDICT" not in log


def test_ref_md5_unreadable_manifest_becomes_error(tmp_path, monkeypatch):
    ref_path = _write(tmp_path / "ref.faa", REF_FULL)
    _patch_manifest(monkeypatch, tmp_path / "does_not_exist.tsv", tmp_path / "repo")
    hits = _valid_hits(tmp_path)
    monkeypatch.setattr(check_nif, "run_tblastn", lambda *a, **k: str(hits))
    args = ["-s", SAMPLE, "--ref", str(ref_path), "-o", str(tmp_path / "out"),
            "-a", str(_assembly(tmp_path))]
    rc = main(args)
    log = _log_text(tmp_path / "out")
    assert rc == 1
    assert "VERDICT:" not in log
    assert "ERROR: no VERDICT - unexpected FileNotFoundError:" in log


def test_expected_ref_md5_direct(tmp_path, monkeypatch):
    expected_ref_md5 = getattr(check_nif, "expected_ref_md5")
    ref_path, manifest_path, repo_root, actual_md5 = _tmp_manifest_setup(
        tmp_path, manifest_md5=None)
    _patch_manifest(monkeypatch, manifest_path, repo_root)
    assert expected_ref_md5(str(ref_path)) == ("TEST_REF", actual_md5)
    assert expected_ref_md5(str(tmp_path / "nope.faa")) is None


def test_expected_ref_md5_resolves_symlink(tmp_path, monkeypatch):
    # Pins resolve(): a symlinked ref path must still match the MANIFEST
    # row, whose local_file points at the real path.
    expected_ref_md5 = getattr(check_nif, "expected_ref_md5")
    ref_path, manifest_path, repo_root, actual_md5 = _tmp_manifest_setup(
        tmp_path, manifest_md5=None)
    _patch_manifest(monkeypatch, manifest_path, repo_root)
    link = tmp_path / "link.faa"
    link.symlink_to(ref_path)
    assert expected_ref_md5(str(link)) == ("TEST_REF", actual_md5)


# --- Item 7: MANIFEST row + real reference file -----------------------------

def test_manifest_nif_ref_171_row_and_real_file():
    manifest_path = paths.MANIFEST
    if not manifest_path.exists():
        pytest.skip(f"MANIFEST not found: {manifest_path}")
    rows = {}
    with open(manifest_path, newline="") as f:
        import csv
        for row in csv.DictReader(f, delimiter="\t"):
            rows[row["id"]] = row
    row = rows.get("NIF_REF_171")
    assert row is not None, "NIF_REF_171 row missing from MANIFEST"
    assert row["md5"] == EXPECTED_REAL_MD5
    assert "19 nifK" in row["verified"]
    ref_path = paths.NIF_REF_FAA
    if not ref_path.exists():
        pytest.skip(f"real reference file not found: {ref_path}")
    actual_md5 = hashlib.md5(ref_path.read_bytes()).hexdigest()
    assert actual_md5 == row["md5"] == EXPECTED_REAL_MD5
    gene_map = check_nif.load_gene_map(str(ref_path))
    counts = {g: list(gene_map.values()).count(g) for g in ("nifH", "nifD", "nifK", "other")}
    assert counts == {"nifH": 113, "nifD": 27, "nifK": 19, "other": 12}
