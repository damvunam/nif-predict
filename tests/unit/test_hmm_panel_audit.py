"""Unit tests for the reproducible nitrogenase HMM-panel audit."""

from __future__ import annotations

import gzip
import hashlib
import io
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from nifpredict.evaluation.hmm_panel_audit import (
    AcceptedProteinHit,
    HMMPanelAuditError,
    NearDuplicatePolicy,
    accession_from_fasta_path,
    audit_exact_seed_leakage,
    audit_near_duplicate_seed_leakage,
    audit_panel_inputs,
    calculate_pairwise_similarity,
    download_panel_inputs,
    iter_fasta_records,
    load_panel_spec,
    parse_hmm_header,
    parse_stockholm,
    sequence_sha256,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSIONED_PANEL_CONFIG = PROJECT_ROOT / "config" / "hmm_panel_pgap_19.yaml"


def _model_mapping(
    *,
    role: str = "nifD",
    accession: str = "TIGR00001.1",
    hmm_name: str = "nifD",
    hmm_raw_sha256: str = "0" * 64,
    seed_raw_sha256: str = "1" * 64,
    seed_sequence_count: int = 2,
) -> dict[str, object]:
    return {
        "role": role,
        "accession": accession,
        "source_identifier": accession.removesuffix(".1"),
        "hmm_name": hmm_name,
        "gene_symbols": [role],
        "hmm_length": 4,
        "sequence_cutoff": 10.0,
        "domain_cutoff": 11.0,
        "noise_cutoff": 5.0,
        "family_type": "equivalog",
        "product": f"Test product for {role}",
        "seed_sequence_count": seed_sequence_count,
        "checksums": {
            "hmm_raw_sha256": hmm_raw_sha256,
            "hmmconvert_sha256": "2" * 64,
            "seed_raw_sha256": seed_raw_sha256,
        },
    }


def _panel_mapping(
    models: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "panel": {
            "id": "test_panel",
            "provider": "Test provider",
            "release_name": "test/1.0",
            "release_date": "2026-01-01",
            "registry_url": "https://example.test/registry.tsv",
            "model_base_url": "https://example.test/models",
            "seed_base_url": "https://example.test/seeds",
            "cutoff_policy": "trusted",
            "normalized_with": "HMMER test",
        },
        "models": models if models is not None else [_model_mapping()],
    }


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _valid_hmm_text() -> str:
    return """HMMER3/f [3.4 | Aug 2023]
NAME  nifD
ACC   TIGR00001.1
DESC  Test HMM
LENG  4
NSEQ  2
GA    10.0 11.0
TC    10.0 11.0
NC    5.0 5.0
HMM          A        C
//
"""


def _valid_seed_text() -> str:
    return """# STOCKHOLM 1.0
seed_one    AC-D
seed_two    A.ED
//
"""


def test_versioned_panel_manifest_is_valid() -> None:
    panel = load_panel_spec(VERSIONED_PANEL_CONFIG)

    assert panel.panel_id == "pgap_nitrogenase_19_0"
    assert len(panel.models) == 12
    assert len({model.role for model in panel.models}) == 12
    assert len({model.accession for model in panel.models}) == 12
    assert sum(model.seed_sequence_count for model in panel.models) == 71


def test_load_panel_spec_rejects_duplicate_roles(tmp_path: Path) -> None:
    first = _model_mapping()
    second = _model_mapping(
        role="nifD",
        accession="TIGR00002.1",
        hmm_name="other",
    )
    path = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([first, second]))

    with pytest.raises(HMMPanelAuditError, match="duplicate roles"):
        load_panel_spec(path)


def test_load_panel_spec_rejects_invalid_checksum(tmp_path: Path) -> None:
    data = _panel_mapping()
    model = data["models"][0]
    model["checksums"]["hmm_raw_sha256"] = "not-a-checksum"
    path = _write_yaml(tmp_path / "panel.yaml", data)

    with pytest.raises(HMMPanelAuditError, match="64-character SHA-256"):
        load_panel_spec(path)


def test_load_panel_spec_rejects_unknown_schema_version(tmp_path: Path) -> None:
    data = _panel_mapping()
    data["schema_version"] = 2
    path = _write_yaml(tmp_path / "panel.yaml", data)

    with pytest.raises(HMMPanelAuditError, match="Unsupported panel schema_version"):
        load_panel_spec(path)


def test_parse_stockholm_reads_and_ungaps_sequences(tmp_path: Path) -> None:
    path = tmp_path / "model.SEED"
    path.write_text(_valid_seed_text(), encoding="utf-8")

    records = parse_stockholm(path)

    assert [record.identifier for record in records] == ["seed_one", "seed_two"]
    assert [record.aligned_sequence for record in records] == ["AC-D", "A.ED"]
    assert [record.sequence for record in records] == ["ACD", "AED"]


def test_parse_stockholm_concatenates_interleaved_blocks(tmp_path: Path) -> None:
    path = tmp_path / "interleaved.SEED"
    path.write_text(
        """# STOCKHOLM 1.0
seed_one    AC-
seed_two    A.D

#=GC RF     xxx
seed_one    EFG
seed_two    E-G
//
""",
        encoding="utf-8",
    )

    records = parse_stockholm(path)

    assert records[0].aligned_sequence == "AC-EFG"
    assert records[0].sequence == "ACEFG"
    assert records[1].aligned_sequence == "A.DE-G"
    assert records[1].sequence == "ADEG"


def test_parse_stockholm_rejects_fasta_instead_of_returning_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incorrect.SEED"
    path.write_text(">seed_one\nACDE\n", encoding="utf-8")

    with pytest.raises(HMMPanelAuditError, match="not Stockholm 1.0"):
        parse_stockholm(path)


def test_parse_stockholm_rejects_missing_terminator(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.SEED"
    path.write_text("# STOCKHOLM 1.0\nseed_one ACDE\n", encoding="utf-8")

    with pytest.raises(HMMPanelAuditError, match="incomplete"):
        parse_stockholm(path)


def test_parse_stockholm_rejects_inconsistent_alignment_lengths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unequal.SEED"
    path.write_text(
        "# STOCKHOLM 1.0\nseed_one ACDE\nseed_two ACD\n//\n",
        encoding="utf-8",
    )

    with pytest.raises(HMMPanelAuditError, match="inconsistent aligned lengths"):
        parse_stockholm(path)


def test_parse_hmm_header_reads_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "TIGR00001.1.HMM"
    path.write_text(_valid_hmm_text(), encoding="utf-8")

    header = parse_hmm_header(path)

    assert header.name == "nifD"
    assert header.accession == "TIGR00001.1"
    assert header.length == 4
    assert header.sequence_count == 2
    assert header.gathering_cutoff == (10.0, 11.0)
    assert header.trusted_cutoff == (10.0, 11.0)
    assert header.noise_cutoff == (5.0, 5.0)


def test_parse_hmm_header_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "missing.HMM"
    path.write_text(
        _valid_hmm_text().replace("NSEQ  2\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(HMMPanelAuditError, match="missing header field.*NSEQ"):
        parse_hmm_header(path)


def test_sha256_helpers_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "content.bin"
    path.write_bytes(b"NifPredict\n")

    assert sha256_file(path) == _sha256_bytes(b"NifPredict\n")
    assert sequence_sha256("acde") == sequence_sha256("ACDE")


def test_download_panel_inputs_writes_only_manifest_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hmm_content = b"versioned HMM\n"
    seed_content = b"versioned SEED\n"
    model = _model_mapping(
        seed_sequence_count=1,
        hmm_raw_sha256=_sha256_bytes(hmm_content),
        seed_raw_sha256=_sha256_bytes(seed_content),
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)
    responses = {
        "https://example.test/models/TIGR00001.1.HMM": hmm_content,
        "https://example.test/seeds/TIGR00001.1.SEED": seed_content,
    }
    calls: list[tuple[str, float]] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(responses[url])

    monkeypatch.setattr(
        "nifpredict.evaluation.hmm_panel_audit.urllib.request.urlopen",
        fake_urlopen,
    )

    downloads = download_panel_inputs(
        panel,
        tmp_path / "panel-inputs",
        timeout=17.0,
    )

    assert len(downloads) == 1
    assert downloads[0].hmm_downloaded
    assert downloads[0].seed_downloaded
    assert downloads[0].hmm_path.read_bytes() == hmm_content
    assert downloads[0].seed_path.read_bytes() == seed_content
    assert calls == [
        ("https://example.test/models/TIGR00001.1.HMM", 17.0),
        ("https://example.test/seeds/TIGR00001.1.SEED", 17.0),
    ]


def test_download_panel_inputs_reuses_checksum_valid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hmm_content = b"versioned HMM\n"
    seed_content = b"versioned SEED\n"
    model = _model_mapping(
        seed_sequence_count=1,
        hmm_raw_sha256=_sha256_bytes(hmm_content),
        seed_raw_sha256=_sha256_bytes(seed_content),
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)
    root = tmp_path / "panel-inputs"
    (root / "models").mkdir(parents=True)
    (root / "seeds").mkdir(parents=True)
    (root / "models" / "TIGR00001.1.HMM").write_bytes(hmm_content)
    (root / "seeds" / "TIGR00001.1.SEED").write_bytes(seed_content)

    def unexpected_urlopen(url: str, timeout: float) -> io.BytesIO:
        raise AssertionError(f"Unexpected download: {url}, timeout={timeout}")

    monkeypatch.setattr(
        "nifpredict.evaluation.hmm_panel_audit.urllib.request.urlopen",
        unexpected_urlopen,
    )

    downloads = download_panel_inputs(panel, root)

    assert not downloads[0].hmm_downloaded
    assert not downloads[0].seed_downloaded


def test_download_panel_inputs_removes_checksum_failed_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model_mapping(
        seed_sequence_count=1,
        hmm_raw_sha256=_sha256_bytes(b"expected HMM"),
        seed_raw_sha256=_sha256_bytes(b"expected SEED"),
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        return io.BytesIO(b"unexpected content")

    monkeypatch.setattr(
        "nifpredict.evaluation.hmm_panel_audit.urllib.request.urlopen",
        fake_urlopen,
    )
    root = tmp_path / "panel-inputs"

    with pytest.raises(HMMPanelAuditError, match="failed SHA-256"):
        download_panel_inputs(panel, root)

    assert not (root / "models" / "TIGR00001.1.HMM").exists()
    assert list(root.rglob("*.download")) == []


def test_download_panel_inputs_refuses_mismatched_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model_mapping(
        seed_sequence_count=1,
        hmm_raw_sha256=_sha256_bytes(b"expected HMM"),
        seed_raw_sha256=_sha256_bytes(b"expected SEED"),
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)
    root = tmp_path / "panel-inputs"
    hmm_path = root / "models" / "TIGR00001.1.HMM"
    hmm_path.parent.mkdir(parents=True)
    hmm_path.write_bytes(b"wrong HMM")

    def unexpected_urlopen(url: str, timeout: float) -> io.BytesIO:
        raise AssertionError(f"Unexpected download: {url}, timeout={timeout}")

    monkeypatch.setattr(
        "nifpredict.evaluation.hmm_panel_audit.urllib.request.urlopen",
        unexpected_urlopen,
    )

    with pytest.raises(HMMPanelAuditError, match="unexpected SHA-256"):
        download_panel_inputs(panel, root)

    assert hmm_path.read_bytes() == b"wrong HMM"


def test_iter_fasta_records_reads_plain_and_gzip_files(tmp_path: Path) -> None:
    plain = tmp_path / "GCF_000000001.1_protein.faa"
    plain.write_text(">P1 description\nAC\nDE\n", encoding="utf-8")
    compressed = tmp_path / "GCF_000000002.1_protein.faa.gz"
    with gzip.open(compressed, mode="wt", encoding="utf-8") as handle:
        handle.write(">P2\nFGHI\n")

    records = list(iter_fasta_records([plain, compressed]))

    assert [(record.identifier, record.sequence) for record in records] == [
        ("P1", "ACDE"),
        ("P2", "FGHI"),
    ]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("GCF_000000001.1_protein.faa", "GCF_000000001.1"),
        ("GCA_000000002.1_protein.faa.gz", "GCA_000000002.1"),
        ("sample.faa", "sample"),
    ],
)
def test_accession_from_fasta_path(filename: str, expected: str) -> None:
    assert accession_from_fasta_path(filename) == expected


def test_audit_panel_inputs_passes_without_external_tool_checks(
    tmp_path: Path,
) -> None:
    hmm_text = _valid_hmm_text()
    seed_text = _valid_seed_text()
    hmm_path = tmp_path / "TIGR00001.1.HMM"
    seed_path = tmp_path / "TIGR00001.1.SEED"
    hmm_path.write_text(hmm_text, encoding="utf-8")
    seed_path.write_text(seed_text, encoding="utf-8")

    model = _model_mapping(
        hmm_raw_sha256=_sha256_bytes(hmm_text.encode()),
        seed_raw_sha256=_sha256_bytes(seed_text.encode()),
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)

    audits = audit_panel_inputs(
        panel,
        hmm_roots=tmp_path,
        seed_roots=tmp_path,
        check_normalized=False,
        check_hmmstat=False,
    )

    assert len(audits) == 1
    assert audits[0].valid
    assert audits[0].errors == ()
    assert audits[0].seed_sequence_count == 2
    assert audits[0].normalized_hmm_sha256 is None
    assert audits[0].hmmstat_valid is None


def test_audit_panel_inputs_reports_checksum_mismatch(tmp_path: Path) -> None:
    hmm_path = tmp_path / "TIGR00001.1.HMM"
    seed_path = tmp_path / "TIGR00001.1.SEED"
    hmm_path.write_text(_valid_hmm_text(), encoding="utf-8")
    seed_path.write_text(_valid_seed_text(), encoding="utf-8")
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping())
    panel = load_panel_spec(config)

    audits = audit_panel_inputs(
        panel,
        hmm_roots=tmp_path,
        seed_roots=tmp_path,
        check_normalized=False,
        check_hmmstat=False,
    )

    assert not audits[0].valid
    assert "Raw HMM SHA-256 mismatch" in audits[0].errors
    assert "Raw SEED SHA-256 mismatch" in audits[0].errors


def test_exact_leakage_audit_detects_cross_model_seed_match(
    tmp_path: Path,
) -> None:
    first_model = _model_mapping(
        role="nifD",
        accession="TIGR00001.1",
        hmm_name="nifD",
        seed_sequence_count=1,
    )
    second_model = _model_mapping(
        role="vnfD",
        accession="TIGR00002.1",
        hmm_name="VNFD",
        seed_sequence_count=1,
    )
    config = _write_yaml(
        tmp_path / "panel.yaml",
        _panel_mapping([first_model, second_model]),
    )
    panel = load_panel_spec(config)
    for accession in ("TIGR00001.1", "TIGR00002.1"):
        (tmp_path / f"{accession}.SEED").write_text(
            "# STOCKHOLM 1.0\nseed ACDE\n//\n",
            encoding="utf-8",
        )
    fasta = tmp_path / "GCF_000000001.1_protein.faa"
    fasta.write_text(">MATCH\nACDE\n>OTHER\nFGHI\n", encoding="utf-8")

    audit = audit_exact_seed_leakage(
        panel,
        seed_roots=tmp_path,
        protein_fastas=[fasta],
    )

    assert audit.total_seed_sequences == 2
    assert audit.unique_seed_hashes == 1
    assert audit.cross_model_duplicate_hashes == 1
    assert audit.local_proteins_scanned == 2
    assert len(audit.matches) == 1
    assert audit.matches[0].protein_id == "MATCH"
    assert {reference.role for reference in audit.matches[0].seed_references} == {
        "nifD",
        "vnfD",
    }


def test_exact_leakage_audit_reports_no_match(tmp_path: Path) -> None:
    model = deepcopy(_model_mapping(seed_sequence_count=1))
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)
    (tmp_path / "TIGR00001.1.SEED").write_text(
        "# STOCKHOLM 1.0\nseed ACDE\n//\n",
        encoding="utf-8",
    )
    fasta = tmp_path / "GCF_000000001.1_protein.faa"
    fasta.write_text(">NO_MATCH\nFGHI\n", encoding="utf-8")

    audit = audit_exact_seed_leakage(
        panel,
        seed_roots=tmp_path,
        protein_fastas=[fasta],
    )

    assert audit.local_proteins_scanned == 1
    assert audit.matches == ()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("min_sequence_identity", 0.0),
        ("min_query_coverage", 101.0),
        ("min_seed_coverage", float("nan")),
    ],
)
def test_near_duplicate_policy_rejects_invalid_thresholds(
    field_name: str,
    value: float,
) -> None:
    parameters = {
        "min_sequence_identity": 95.0,
        "min_query_coverage": 90.0,
        "min_seed_coverage": 90.0,
    }
    parameters[field_name] = value

    with pytest.raises(HMMPanelAuditError, match=field_name):
        NearDuplicatePolicy(**parameters)


def test_pairwise_similarity_reports_perfect_match() -> None:
    policy = NearDuplicatePolicy(
        min_sequence_identity=95.0,
        min_query_coverage=90.0,
        min_seed_coverage=90.0,
    )

    similarity = calculate_pairwise_similarity(
        "ACDEFG",
        "ACDEFG",
        policy=policy,
    )

    assert similarity.matches == 6
    assert similarity.alignment_columns == 6
    assert similarity.sequence_identity == pytest.approx(100.0)
    assert similarity.query_coverage == pytest.approx(100.0)
    assert similarity.seed_coverage == pytest.approx(100.0)


def test_pairwise_similarity_reports_bidirectional_fragment_coverage() -> None:
    policy = NearDuplicatePolicy(
        min_sequence_identity=95.0,
        min_query_coverage=50.0,
        min_seed_coverage=90.0,
    )

    similarity = calculate_pairwise_similarity(
        "XXACDEFGXX",
        "ACDEFG",
        policy=policy,
    )

    assert similarity.sequence_identity == pytest.approx(100.0)
    assert similarity.query_coverage == pytest.approx(60.0)
    assert similarity.seed_coverage == pytest.approx(100.0)


def test_near_duplicate_audit_compares_only_same_model_seeds(
    tmp_path: Path,
) -> None:
    nifd = _model_mapping(
        role="nifD",
        accession="TIGR00001.1",
        hmm_name="nifD",
        seed_sequence_count=1,
    )
    vnfd = _model_mapping(
        role="vnfD",
        accession="TIGR00002.1",
        hmm_name="VNFD",
        seed_sequence_count=1,
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([nifd, vnfd]))
    panel = load_panel_spec(config)
    (tmp_path / "TIGR00001.1.SEED").write_text(
        "# STOCKHOLM 1.0\nnifd_seed ACDEFG\n//\n",
        encoding="utf-8",
    )
    (tmp_path / "TIGR00002.1.SEED").write_text(
        "# STOCKHOLM 1.0\nvnfd_seed HIKLMN\n//\n",
        encoding="utf-8",
    )
    policy = NearDuplicatePolicy(95.0, 90.0, 90.0)

    audit = audit_near_duplicate_seed_leakage(
        panel,
        seed_roots=tmp_path,
        accepted_hits=[AcceptedProteinHit("GCF_000000001.1", "P1", "nifD", "ACDEFG")],
        policy=policy,
    )

    assert audit.accepted_hit_count == 1
    assert audit.unique_protein_count == 1
    assert len(audit.comparisons) == 1
    assert audit.comparisons[0].model_role == "nifD"
    assert audit.comparisons[0].seed_identifier == "nifd_seed"


def test_near_duplicate_flag_requires_identity_and_both_coverages(
    tmp_path: Path,
) -> None:
    model = _model_mapping(seed_sequence_count=1)
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)
    (tmp_path / "TIGR00001.1.SEED").write_text(
        "# STOCKHOLM 1.0\nseed ACDEFG\n//\n",
        encoding="utf-8",
    )
    policy = NearDuplicatePolicy(95.0, 90.0, 90.0)

    audit = audit_near_duplicate_seed_leakage(
        panel,
        seed_roots=tmp_path,
        accepted_hits=[
            AcceptedProteinHit("GCF_000000001.1", "EXACT", "nifD", "ACDEFG"),
            AcceptedProteinHit("GCF_000000001.1", "FRAGMENT", "nifD", "ACDE"),
        ],
        policy=policy,
    )

    by_protein = {comparison.protein_id: comparison for comparison in audit.comparisons}
    assert by_protein["EXACT"].exact_sequence_match
    assert by_protein["EXACT"].flagged
    assert not by_protein["FRAGMENT"].exact_sequence_match
    assert not by_protein["FRAGMENT"].flagged
    assert by_protein["FRAGMENT"].query_coverage == pytest.approx(100.0)
    assert by_protein["FRAGMENT"].seed_coverage == pytest.approx(4 / 6 * 100)
    assert len(audit.flagged_comparisons) == 1


def test_near_duplicate_audit_reports_multi_model_proteins(
    tmp_path: Path,
) -> None:
    nifd = _model_mapping(
        role="nifD",
        accession="TIGR00001.1",
        hmm_name="nifD",
        seed_sequence_count=1,
    )
    vnfd = _model_mapping(
        role="vnfD",
        accession="TIGR00002.1",
        hmm_name="VNFD",
        seed_sequence_count=1,
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([nifd, vnfd]))
    panel = load_panel_spec(config)
    for accession in ("TIGR00001.1", "TIGR00002.1"):
        (tmp_path / f"{accession}.SEED").write_text(
            "# STOCKHOLM 1.0\nseed ACDEFG\n//\n",
            encoding="utf-8",
        )
    hits = [
        AcceptedProteinHit("GCF_000000001.1", "P1", "nifD", "ACDEFG"),
        AcceptedProteinHit("GCF_000000001.1", "P1", "vnfD", "ACDEFG"),
    ]

    audit = audit_near_duplicate_seed_leakage(
        panel,
        seed_roots=tmp_path,
        accepted_hits=hits,
        policy=NearDuplicatePolicy(95.0, 90.0, 90.0),
    )

    assert audit.accepted_hit_count == 2
    assert audit.unique_protein_count == 1
    assert len(audit.comparisons) == 2
    assert len(audit.multi_model_proteins) == 1
    assert audit.multi_model_proteins[0].model_roles == ("nifD", "vnfD")


def test_near_duplicate_audit_rejects_duplicate_assignments(
    tmp_path: Path,
) -> None:
    model = _model_mapping(seed_sequence_count=1)
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)
    (tmp_path / "TIGR00001.1.SEED").write_text(
        "# STOCKHOLM 1.0\nseed ACDEFG\n//\n",
        encoding="utf-8",
    )
    hit = AcceptedProteinHit("GCF_000000001.1", "P1", "nifD", "ACDEFG")

    with pytest.raises(HMMPanelAuditError, match="Duplicate accepted-hit"):
        audit_near_duplicate_seed_leakage(
            panel,
            seed_roots=tmp_path,
            accepted_hits=[hit, hit],
            policy=NearDuplicatePolicy(95.0, 90.0, 90.0),
        )


def test_near_duplicate_audit_rejects_conflicting_protein_sequences(
    tmp_path: Path,
) -> None:
    nifd = _model_mapping(
        role="nifD",
        accession="TIGR00001.1",
        hmm_name="nifD",
        seed_sequence_count=1,
    )
    vnfd = _model_mapping(
        role="vnfD",
        accession="TIGR00002.1",
        hmm_name="VNFD",
        seed_sequence_count=1,
    )
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([nifd, vnfd]))
    panel = load_panel_spec(config)
    for accession in ("TIGR00001.1", "TIGR00002.1"):
        (tmp_path / f"{accession}.SEED").write_text(
            "# STOCKHOLM 1.0\nseed ACDEFG\n//\n",
            encoding="utf-8",
        )
    hits = [
        AcceptedProteinHit("GCF_000000001.1", "P1", "nifD", "ACDEFG"),
        AcceptedProteinHit("GCF_000000001.1", "P1", "vnfD", "HIKLMN"),
    ]

    with pytest.raises(HMMPanelAuditError, match="Conflicting sequences"):
        audit_near_duplicate_seed_leakage(
            panel,
            seed_roots=tmp_path,
            accepted_hits=hits,
            policy=NearDuplicatePolicy(95.0, 90.0, 90.0),
        )


def test_near_duplicate_audit_rejects_unknown_model_role(
    tmp_path: Path,
) -> None:
    model = _model_mapping(seed_sequence_count=1)
    config = _write_yaml(tmp_path / "panel.yaml", _panel_mapping([model]))
    panel = load_panel_spec(config)
    (tmp_path / "TIGR00001.1.SEED").write_text(
        "# STOCKHOLM 1.0\nseed ACDEFG\n//\n",
        encoding="utf-8",
    )

    with pytest.raises(HMMPanelAuditError, match="unknown model role"):
        audit_near_duplicate_seed_leakage(
            panel,
            seed_roots=tmp_path,
            accepted_hits=[
                AcceptedProteinHit("GCF_000000001.1", "P1", "unknown", "ACDEFG")
            ],
            policy=NearDuplicatePolicy(95.0, 90.0, 90.0),
        )
