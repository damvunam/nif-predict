"""run_busco must opt out of BUSCO run-stats upload (MU6 C).

BUSCO 6.1.0 --offline does not stop the POST to busco-data.ezlab.org;
--opt-out-run-stats does.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nifpredict.wgs import qc


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    busco_dir = tmp_path / "busco"
    lineage = busco_dir / "lineages" / "bacillales_odb10"
    lineage.mkdir(parents=True)
    (lineage / "dataset.cfg").write_text("name=bacillales_odb10\n")
    monkeypatch.setattr(qc.paths, "BUSCO_DIR", busco_dir)
    out_dir = tmp_path / "out"
    res = out_dir / "busco_result"
    res.mkdir(parents=True)
    (res / "short_summary.specific.bacillales_odb10.busco_result.json").write_text(
        json.dumps({"results": {"Complete": 99.0}})
    )
    return busco_dir, out_dir


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    cmd_seen: list[str] = []

    def _fake_run(cmd, *a, **k):
        cmd_seen.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(qc.subprocess, "run", _fake_run)
    return cmd_seen


def test_run_busco_cmd_has_opt_out_run_stats(tmp_path, monkeypatch) -> None:
    _, out_dir = _setup(tmp_path, monkeypatch)
    cmd = _capture(monkeypatch)
    qc.run_busco(str(tmp_path / "a.fasta"), str(out_dir), lineage="bacillales_odb10", threads=2)
    assert cmd.count("--opt-out-run-stats") == 1


def test_run_busco_cmd_keeps_offline_and_download_path(tmp_path, monkeypatch) -> None:
    busco_dir, out_dir = _setup(tmp_path, monkeypatch)
    cmd = _capture(monkeypatch)
    qc.run_busco(str(tmp_path / "a.fasta"), str(out_dir), lineage="bacillales_odb10", threads=2)
    assert "--offline" in cmd
    assert cmd[cmd.index("--download_path") + 1] == str(busco_dir)


def test_run_busco_opt_out_flag_is_a_standalone_token(tmp_path, monkeypatch) -> None:
    _, out_dir = _setup(tmp_path, monkeypatch)
    cmd = _capture(monkeypatch)
    qc.run_busco(str(tmp_path / "a.fasta"), str(out_dir), lineage="bacillales_odb10", threads=2)
    i = cmd.index("--opt-out-run-stats")
    # flag takes no value: next token (if any) must be another option
    assert i == len(cmd) - 1 or cmd[i + 1].startswith("--")
