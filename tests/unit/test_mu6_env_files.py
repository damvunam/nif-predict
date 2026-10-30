"""Static checks for environment.yml and pyproject.toml (MU6 A+B).

These do not solve the conda env; they only guard against the regressions
found in MU6: missing runtime deps in pyproject, unpinned tools, and a pip
entry without the [dev] extra (bare `pytest` then fails on --cov).
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "environment.yml"
PYPROJECT = ROOT / "pyproject.toml"

REQUIRED_TOOLS = {
    "fastp", "unicycler", "spades", "quast", "busco", "skani",
    "blast", "diamond", "seqtk", "seqkit", "ncbi-datasets-cli", "pytest",
}


def _env_lines() -> list[str]:
    return [ln.rstrip() for ln in ENV.read_text().splitlines()]


def _conda_deps() -> dict[str, str]:
    deps: dict[str, str] = {}
    in_deps = False
    for ln in _env_lines():
        if ln.startswith("dependencies:"):
            in_deps = True
            continue
        if in_deps and re.match(r"^  - [^\s:]+$", ln):
            item = ln[4:]
            name, _, ver = item.partition("=")
            deps[name] = ver
    return deps


def _project_dep_names() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    names = set()
    for d in data["project"]["dependencies"]:
        names.add(re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower())
    return names


def test_pyproject_lists_runtime_imports_found_in_s0() -> None:
    names = _project_dep_names()
    assert "pyhmmer" in names
    assert "tqdm" in names


def test_environment_yml_exists_and_is_named_nifwgs() -> None:
    assert ENV.is_file()
    assert "name: nifwgs" in _env_lines()


def test_environment_yml_channels() -> None:
    text = ENV.read_text()
    assert "- conda-forge" in text
    assert "- bioconda" in text


def test_environment_yml_pins_every_required_tool_with_version() -> None:
    deps = _conda_deps()
    missing = REQUIRED_TOOLS - set(deps)
    assert not missing, f"missing from environment.yml: {sorted(missing)}"
    unpinned = sorted(t for t in REQUIRED_TOOLS if not deps[t])
    assert not unpinned, f"no version pin: {unpinned}"


def test_environment_yml_python_range_matches_numpy_pin() -> None:
    deps = _conda_deps()
    assert deps["python"].startswith("3.12")


def test_environment_yml_pip_entry_installs_dev_extra() -> None:
    pip_lines = [ln.strip() for ln in _env_lines() if ln.strip().startswith("- -e")]
    assert pip_lines == ["- -e .[dev]"]


def test_pytest_addopts_cov_requires_pytest_cov_in_dev_extra() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    addopts = data["tool"]["pytest"]["ini_options"]["addopts"]
    if "--cov" in addopts:
        dev = data["project"]["optional-dependencies"]["dev"]
        assert any(d.lower().startswith("pytest-cov") for d in dev)
