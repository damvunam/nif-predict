"""Validated bridge from trusted-cutoff panel scans to feature extraction."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from nifpredict.features.cluster_filter import ClusterFilter
from nifpredict.features.gff_mapping import merge_hits_with_gff, parse_gff3


PANEL_HIT_COLUMNS = (
    "accession",
    "protein_id",
    "protein_source",
    "protein_length",
    "protein_sha256",
    "model_role",
    "model_accession",
    "model_name",
    "sequence_score",
    "sequence_evalue",
    "best_domain_score",
    "best_domain_i_evalue",
    "accepted_by",
)
FEATURE_HIT_COLUMNS = (
    "target_protein",
    "gene_family",
    "effective_score",
    "seq_evalue",
)

_ACCESSION_PATTERN = re.compile(r"^(?:GCA|GCF)_\d+\.\d+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PanelFeatureInputError(ValueError):
    """Raised when panel-scan artifacts cannot safely feed feature extraction."""


@dataclass(frozen=True)
class PanelFeatureInput:
    """One validated panel scan and its deterministic feature-hit rows."""

    scan_directory: Path
    manifest: Mapping[str, Any]
    accessions: tuple[str, ...]
    model_roles: tuple[str, ...]
    hits: pd.DataFrame

    def hits_for_accession(self, accession: str) -> pd.DataFrame:
        """Return an accession-specific frame with the stable panel schema."""
        if accession not in self.accessions:
            raise PanelFeatureInputError(
                f"Accession is absent from the panel scan: {accession}"
            )
        selected = self.hits[self.hits["accession"] == accession].copy()
        return selected.reset_index(drop=True)

    def feature_hits_for_accession(self, accession: str) -> pd.DataFrame:
        """Adapt one accession without relabeling scientific model roles."""
        return adapt_panel_hits(self.hits_for_accession(accession))


def _require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PanelFeatureInputError(f"{location} must be a mapping")
    return value


def _require_integer(
    mapping: Mapping[str, Any], key: str, location: str, *, minimum: int = 0
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PanelFeatureInputError(
            f"{location}.{key} must be an integer >= {minimum}"
        )
    return value


def _require_nonempty_string(
    mapping: Mapping[str, Any], key: str, location: str
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PanelFeatureInputError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _parse_integer(value: str, location: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise PanelFeatureInputError(f"{location} must be an integer") from error
    if parsed < minimum:
        raise PanelFeatureInputError(f"{location} must be >= {minimum}")
    return parsed


def _parse_float(value: str, location: str, *, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise PanelFeatureInputError(f"{location} must be numeric") from error
    if not math.isfinite(parsed):
        raise PanelFeatureInputError(f"{location} must be finite")
    if minimum is not None and parsed < minimum:
        raise PanelFeatureInputError(f"{location} must be >= {minimum}")
    return parsed


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PanelFeatureInputError(
            f"Cannot read scan manifest '{path}': {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise PanelFeatureInputError(
            f"Invalid scan manifest JSON '{path}': {error}"
        ) from error

    manifest = _require_mapping(raw, "scan_manifest")
    schema_version = _require_integer(
        manifest, "schema_version", "scan_manifest", minimum=1
    )
    if schema_version != 1:
        raise PanelFeatureInputError(
            f"Unsupported scan schema_version={schema_version}; expected 1"
        )
    if manifest.get("cutoff_policy") != "trusted":
        raise PanelFeatureInputError("scan_manifest.cutoff_policy must be 'trusted'")
    if manifest.get("additional_hit_thresholds") != []:
        raise PanelFeatureInputError(
            "scan_manifest.additional_hit_thresholds must be an empty list"
        )
    _require_nonempty_string(manifest, "panel_id", "scan_manifest")
    _require_nonempty_string(manifest, "panel_release", "scan_manifest")
    _require_integer(manifest, "accepted_hit_count", "scan_manifest")
    _require_integer(manifest, "unique_hit_protein_count", "scan_manifest")
    _require_integer(manifest, "multi_model_protein_count", "scan_manifest")
    _require_integer(manifest, "genome_count", "scan_manifest")
    _require_integer(manifest, "protein_count", "scan_manifest")
    return manifest


def _manifest_models(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, tuple[str, str]]]:
    raw_models = manifest.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise PanelFeatureInputError("scan_manifest.models must be a non-empty list")

    roles: list[str] = []
    provenance: dict[str, tuple[str, str]] = {}
    for index, raw_model in enumerate(raw_models):
        location = f"scan_manifest.models[{index}]"
        model = _require_mapping(raw_model, location)
        role = _require_nonempty_string(model, "role", location)
        model_accession = _require_nonempty_string(model, "accession", location)
        model_name = _require_nonempty_string(model, "name", location)
        if role in provenance:
            raise PanelFeatureInputError(f"Duplicate model role in manifest: {role}")
        roles.append(role)
        provenance[role] = (model_accession, model_name)
    return tuple(roles), provenance


def _manifest_protein_inputs(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, int]]:
    raw_inputs = manifest.get("protein_inputs")
    if not isinstance(raw_inputs, list):
        raise PanelFeatureInputError("scan_manifest.protein_inputs must be a list")

    accessions: list[str] = []
    hit_counts: dict[str, int] = {}
    for index, raw_input in enumerate(raw_inputs):
        location = f"scan_manifest.protein_inputs[{index}]"
        protein_input = _require_mapping(raw_input, location)
        accession = _require_nonempty_string(protein_input, "accession", location)
        if _ACCESSION_PATTERN.fullmatch(accession) is None:
            raise PanelFeatureInputError(f"Invalid assembly accession: {accession}")
        if accession in hit_counts:
            raise PanelFeatureInputError(
                f"Duplicate protein-input accession in manifest: {accession}"
            )
        _require_nonempty_string(protein_input, "path", location)
        sha256 = _require_nonempty_string(protein_input, "sha256", location)
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise PanelFeatureInputError(
                f"{location}.sha256 must be a lowercase SHA-256"
            )
        _require_integer(protein_input, "protein_count", location)
        hit_counts[accession] = _require_integer(
            protein_input, "accepted_hit_count", location
        )
        accessions.append(accession)

    expected_genomes = _require_integer(manifest, "genome_count", "scan_manifest")
    if len(accessions) != expected_genomes:
        raise PanelFeatureInputError(
            "scan_manifest.genome_count does not match protein_inputs"
        )
    return tuple(accessions), hit_counts


def _read_hit_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != PANEL_HIT_COLUMNS:
                raise PanelFeatureInputError(
                    "accepted_hits.tsv header does not match the panel-hit contract"
                )
            return [dict(row) for row in reader]
    except OSError as error:
        raise PanelFeatureInputError(
            f"Cannot read accepted hits '{path}': {error}"
        ) from error


def _validate_hit_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    accessions: Sequence[str],
    model_roles: Sequence[str],
    model_provenance: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    accession_order = {accession: index for index, accession in enumerate(accessions)}
    model_order = {role: index for index, role in enumerate(model_roles)}
    assignments: set[tuple[str, str, str]] = set()
    parsed_rows: list[dict[str, object]] = []

    for index, row in enumerate(rows, start=2):
        location = f"accepted_hits.tsv:{index}"
        accession = str(row.get("accession", "")).strip()
        protein_id = str(row.get("protein_id", "")).strip()
        model_role = str(row.get("model_role", "")).strip()
        if accession not in accession_order:
            raise PanelFeatureInputError(
                f"{location} references undeclared accession '{accession}'"
            )
        if not protein_id:
            raise PanelFeatureInputError(f"{location}.protein_id must not be empty")
        if model_role not in model_order:
            raise PanelFeatureInputError(
                f"{location} references unknown model role '{model_role}'"
            )
        assignment = (accession, protein_id, model_role)
        if assignment in assignments:
            raise PanelFeatureInputError(
                "Duplicate accepted-hit assignment: " + "/".join(assignment)
            )
        assignments.add(assignment)

        expected_model_accession, expected_model_name = model_provenance[model_role]
        model_accession = str(row.get("model_accession", "")).strip()
        model_name = str(row.get("model_name", "")).strip()
        if (model_accession, model_name) != (
            expected_model_accession,
            expected_model_name,
        ):
            raise PanelFeatureInputError(
                f"{location} model provenance does not match role '{model_role}'"
            )
        protein_source = str(row.get("protein_source", "")).strip()
        protein_sha256 = str(row.get("protein_sha256", "")).strip()
        if not protein_source:
            raise PanelFeatureInputError(f"{location}.protein_source must not be empty")
        if _SHA256_PATTERN.fullmatch(protein_sha256) is None:
            raise PanelFeatureInputError(
                f"{location}.protein_sha256 must be a lowercase SHA-256"
            )
        if str(row.get("accepted_by", "")).strip() != "trusted":
            raise PanelFeatureInputError(f"{location}.accepted_by must be 'trusted'")

        parsed_rows.append(
            {
                "accession": accession,
                "protein_id": protein_id,
                "protein_source": protein_source,
                "protein_length": _parse_integer(
                    row.get("protein_length", ""),
                    f"{location}.protein_length",
                    minimum=1,
                ),
                "protein_sha256": protein_sha256,
                "model_role": model_role,
                "model_accession": model_accession,
                "model_name": model_name,
                "sequence_score": _parse_float(
                    row.get("sequence_score", ""), f"{location}.sequence_score"
                ),
                "sequence_evalue": _parse_float(
                    row.get("sequence_evalue", ""),
                    f"{location}.sequence_evalue",
                    minimum=0.0,
                ),
                "best_domain_score": _parse_float(
                    row.get("best_domain_score", ""),
                    f"{location}.best_domain_score",
                ),
                "best_domain_i_evalue": _parse_float(
                    row.get("best_domain_i_evalue", ""),
                    f"{location}.best_domain_i_evalue",
                    minimum=0.0,
                ),
                "accepted_by": "trusted",
                "_accession_order": accession_order[accession],
                "_model_order": model_order[model_role],
            }
        )

    if not parsed_rows:
        return pd.DataFrame(columns=PANEL_HIT_COLUMNS)

    frame = pd.DataFrame.from_records(parsed_rows)
    frame = frame.sort_values(
        ["_accession_order", "_model_order", "protein_id"], kind="stable"
    ).drop(columns=["_accession_order", "_model_order"])
    return frame.loc[:, PANEL_HIT_COLUMNS].reset_index(drop=True)


def load_panel_feature_input(scan_directory: Path | str) -> PanelFeatureInput:
    """Load and cross-validate one complete trusted-cutoff scan directory."""
    directory = Path(scan_directory)
    manifest_path = directory / "scan_manifest.json"
    hits_path = directory / "accepted_hits.tsv"
    if not manifest_path.is_file():
        raise PanelFeatureInputError(f"Missing scan manifest: {manifest_path}")
    if not hits_path.is_file():
        raise PanelFeatureInputError(f"Missing accepted hits: {hits_path}")

    manifest = _load_manifest(manifest_path)
    model_roles, model_provenance = _manifest_models(manifest)
    accessions, expected_accession_counts = _manifest_protein_inputs(manifest)
    rows = _read_hit_rows(hits_path)
    hits = _validate_hit_rows(
        rows,
        accessions=accessions,
        model_roles=model_roles,
        model_provenance=model_provenance,
    )

    expected_hits = _require_integer(manifest, "accepted_hit_count", "scan_manifest")
    if len(hits) != expected_hits:
        raise PanelFeatureInputError(
            "scan_manifest.accepted_hit_count does not match accepted_hits.tsv"
        )
    observed_accession_counts = Counter(hits["accession"])
    for accession, expected_count in expected_accession_counts.items():
        if observed_accession_counts[accession] != expected_count:
            raise PanelFeatureInputError(
                f"Accepted-hit count does not match manifest for {accession}"
            )

    roles_by_protein: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in hits.itertuples(index=False):
        roles_by_protein[(row.accession, row.protein_id)].add(row.model_role)
    if len(roles_by_protein) != manifest["unique_hit_protein_count"]:
        raise PanelFeatureInputError(
            "scan_manifest.unique_hit_protein_count does not match accepted hits"
        )
    observed_multi_model = sum(len(roles) > 1 for roles in roles_by_protein.values())
    if observed_multi_model != manifest["multi_model_protein_count"]:
        raise PanelFeatureInputError(
            "scan_manifest.multi_model_protein_count does not match accepted hits"
        )

    return PanelFeatureInput(
        scan_directory=directory.resolve(),
        manifest=manifest,
        accessions=accessions,
        model_roles=model_roles,
        hits=hits,
    )


def adapt_panel_hits(panel_hits: pd.DataFrame) -> pd.DataFrame:
    """Map panel columns to feature columns while preserving model-role semantics."""
    missing = [
        column for column in PANEL_HIT_COLUMNS if column not in panel_hits.columns
    ]
    if missing:
        raise PanelFeatureInputError(
            f"Panel hit frame is missing required columns: {missing}"
        )
    adapted = panel_hits.rename(
        columns={
            "protein_id": "target_protein",
            "model_role": "gene_family",
            "sequence_score": "effective_score",
            "sequence_evalue": "seq_evalue",
        }
    )
    return adapted.loc[:, FEATURE_HIT_COLUMNS].reset_index(drop=True)


def join_panel_hits_with_gff(
    feature_hits: pd.DataFrame,
    gff_frame: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Join adapted hits to GFF loci and expose unmatched/ambiguous mappings."""
    if tuple(feature_hits.columns) != FEATURE_HIT_COLUMNS:
        raise PanelFeatureInputError(
            "Feature hit frame does not match the panel adapter schema"
        )
    if feature_hits.empty:
        empty = pd.DataFrame()
        empty.attrs["match_stats"] = {
            "hit_rows": 0,
            "matched_hit_rows": 0,
            "unmatched_protein_ids": [],
            "ambiguous_aliases": [],
            "direct_multimatch_aliases": [],
        }
        return empty

    try:
        merged = merge_hits_with_gff(feature_hits, gff_frame)
    except ValueError as error:
        raise PanelFeatureInputError(str(error)) from error
    stats = merged.attrs.get("match_stats", {})
    unmatched = list(stats.get("unmatched_protein_ids", []))
    if require_complete and unmatched:
        ambiguous = sorted(
            set(unmatched).intersection(stats.get("ambiguous_aliases", []))
        )
        raise PanelFeatureInputError(
            "Panel hits could not be mapped uniquely to GFF CDS records; "
            f"unmatched={unmatched[:5]}, ambiguous={ambiguous[:5]}"
        )
    return merged


def build_panel_raw_records(
    panel_input: PanelFeatureInput,
    gff_paths: Mapping[str, Path],
    *,
    cluster_filter: ClusterFilter,
) -> list[dict[str, Any]]:
    """Build deterministic raw feature records from precomputed panel hits."""
    records: list[dict[str, Any]] = []
    for accession in panel_input.accessions:
        gff_path = gff_paths.get(accession)
        if gff_path is None or not Path(gff_path).is_file():
            raise PanelFeatureInputError(f"Missing GFF input for {accession}")
        feature_hits = panel_input.feature_hits_for_accession(accession)
        gff_frame = parse_gff3(Path(gff_path))
        join_panel_hits_with_gff(feature_hits, gff_frame, require_complete=True)
        clusters = (
            cluster_filter.group_into_clusters(feature_hits, gff_frame)
            if not feature_hits.empty
            else []
        )
        records.append(
            {
                "accession": accession,
                "status": "SUCCESS",
                "df_all_hits": feature_hits,
                "clusters": clusters,
                "pfam_domains": feature_hits["gene_family"].astype(str).tolist(),
                "metadata": {},
            }
        )
    return records


__all__ = [
    "FEATURE_HIT_COLUMNS",
    "PANEL_HIT_COLUMNS",
    "PanelFeatureInput",
    "PanelFeatureInputError",
    "adapt_panel_hits",
    "build_panel_raw_records",
    "join_panel_hits_with_gff",
    "load_panel_feature_input",
]
