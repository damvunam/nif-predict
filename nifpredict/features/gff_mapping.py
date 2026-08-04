"""Parse GFF3 CDS records and map HMM protein hits to genomic loci."""

from __future__ import annotations

import logging
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger("nifpredict.features.gff_mapping")

_PRODIGAL_ID_RE = re.compile(r"^\d+_(\d+)$")
_ACCESSION_LIKE_RE = re.compile(r"^[A-Za-z]{1,10}_?\d+(?:\.\d+)?$")


def _attribute_values(value: Optional[str]) -> Iterable[str]:
    """Yield non-empty values from a possibly comma-separated attribute."""
    if not value:
        return
    for item in value.split(","):
        cleaned = item.strip()
        if cleaned:
            yield cleaned


def _dbxref_accessions(value: Optional[str]) -> Iterable[str]:
    """Yield sequence accessions from trusted GFF3 Dbxref namespaces."""
    for item in _attribute_values(value):
        namespace, separator, accession = item.partition(":")
        if separator and namespace.casefold() in {"genbank", "refseq"}:
            accession = accession.strip()
            if accession:
                yield accession


def _build_id_aliases(
    seqid: str,
    source: str,
    attributes: Dict[str, str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return direct and inferred aliases for one CDS.

    Direct aliases identify the translated sequence itself and may legitimately
    point to multiple genomic CDS loci. Inferred aliases identify a feature or
    are reconstructed from annotation conventions, so they are only safe when
    unique within the GFF3 file.
    """
    direct: Set[str] = set(_attribute_values(attributes.get("protein_id")))
    direct.update(_dbxref_accessions(attributes.get("Dbxref")))

    name = attributes.get("Name", "").strip()
    protein_ids = set(_attribute_values(attributes.get("protein_id")))
    if name and (name in protein_ids or _ACCESSION_LIKE_RE.fullmatch(name)):
        direct.add(name)

    inferred: Set[str] = set()
    for key in ("locus_tag", "ID", "Parent"):
        for value in _attribute_values(attributes.get(key)):
            inferred.add(value)
            if value.startswith("cds-") and len(value) > 4:
                inferred.add(value[4:])

    raw_id = attributes.get("ID", "")
    prodigal_match = _PRODIGAL_ID_RE.fullmatch(raw_id)
    looks_like_prodigal = (
        source.casefold() == "prodigal"
        or "partial" in attributes
        or "start_type" in attributes
    )
    if prodigal_match and looks_like_prodigal:
        inferred.add(f"{seqid}_{prodigal_match.group(1)}")

    inferred.difference_update(direct)
    return tuple(sorted(direct)), tuple(sorted(inferred))


def parse_gff3(gff_path: Path) -> pd.DataFrame:
    """Parse CDS coordinates and match aliases from an NCBI/Prodigal GFF3."""
    gff_path = Path(gff_path)
    columns = [
        "seqid",
        "feature_type",
        "start",
        "end",
        "strand",
        "protein_id",
        "locus_tag",
        "gff_id",
        "parent_id",
        "direct_id_aliases",
        "inferred_id_aliases",
        "id_aliases",
    ]
    if not gff_path.is_file():
        return pd.DataFrame(columns=columns)

    records: List[Dict[str, Any]] = []
    with gff_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#") or not line.strip():
                continue

            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 9 or parts[2] != "CDS":
                continue

            seqid, source, _, start_raw, end_raw, _, strand, _, attributes_raw = parts
            attributes: Dict[str, str] = {}
            for item in attributes_raw.split(";"):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                attributes[key.strip()] = urllib.parse.unquote(value.strip())

            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Tọa độ CDS không hợp lệ tại {gff_path.name}:{line_number}"
                ) from exc
            if start < 1 or end < start:
                raise ValueError(
                    f"Khoảng CDS không hợp lệ tại {gff_path.name}:{line_number}: "
                    f"start={start}, end={end}"
                )

            direct_aliases, inferred_aliases = _build_id_aliases(
                seqid, source, attributes
            )
            raw_id = attributes.get("ID")
            protein_id = attributes.get("protein_id")

            if not protein_id and raw_id:
                prodigal_match = _PRODIGAL_ID_RE.fullmatch(raw_id)
                reconstructed_id = (
                    f"{seqid}_{prodigal_match.group(1)}" if prodigal_match else None
                )
                if reconstructed_id and reconstructed_id in inferred_aliases:
                    protein_id = reconstructed_id
                else:
                    protein_id = raw_id.removeprefix("cds-")

            all_aliases = tuple(sorted(set(direct_aliases) | set(inferred_aliases)))
            records.append(
                {
                    "seqid": seqid,
                    "feature_type": "CDS",
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "protein_id": protein_id,
                    "locus_tag": attributes.get("locus_tag"),
                    "gff_id": raw_id,
                    "parent_id": attributes.get("Parent"),
                    "direct_id_aliases": direct_aliases,
                    "inferred_id_aliases": inferred_aliases,
                    "id_aliases": all_aliases,
                }
            )

    result = pd.DataFrame.from_records(records, columns=columns)
    result.attrs["source_path"] = str(gff_path)
    return result


def _row_aliases(row: Any, column: str) -> Set[str]:
    aliases: Set[str] = set()
    values = getattr(row, column, ())
    if isinstance(values, str):
        values = (values,)
    if values is None:
        return aliases
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip():
            aliases.add(str(value).strip())
    return aliases


def _aliases_for_feature(row: Any) -> Tuple[Set[str], Set[str]]:
    """Read provenance-aware aliases, with compatibility for older DataFrames."""
    has_provenance = hasattr(row, "direct_id_aliases") or hasattr(
        row, "inferred_id_aliases"
    )
    direct = _row_aliases(row, "direct_id_aliases")
    inferred = _row_aliases(row, "inferred_id_aliases")

    if not has_provenance:
        protein_id = getattr(row, "protein_id", None)
        if (
            protein_id is not None
            and not pd.isna(protein_id)
            and str(protein_id).strip()
        ):
            direct.add(str(protein_id).strip())
        inferred.update(_row_aliases(row, "id_aliases"))
        for column in ("locus_tag", "gff_id", "parent_id"):
            value = getattr(row, column, None)
            if value is not None and not pd.isna(value) and str(value).strip():
                inferred.add(str(value).strip())

    inferred.difference_update(direct)
    return direct, inferred


def _empty_match(hits: pd.DataFrame) -> pd.DataFrame:
    merged = pd.DataFrame()
    merged.attrs["match_stats"] = {
        "hit_rows": len(hits),
        "matched_hit_rows": 0,
        "unmatched_protein_ids": sorted(set(hits["protein_id"])),
        "ambiguous_aliases": [],
        "direct_multimatch_aliases": [],
    }
    return merged


def merge_hits_with_gff(df_hits: pd.DataFrame, df_gff: pd.DataFrame) -> pd.DataFrame:
    """Map HMM hit rows to every valid genomic CDS locus."""
    hits = df_hits.copy()
    if "target_protein" in hits.columns:
        hits = hits.rename(columns={"target_protein": "protein_id"})
    elif "target_name" in hits.columns and "protein_id" not in hits.columns:
        hits = hits.rename(columns={"target_name": "protein_id"})

    if "protein_id" not in hits.columns:
        raise ValueError("HMM hits thiếu cột target_protein/target_name/protein_id")
    if df_gff.empty:
        raise ValueError("GFF3 không chứa bản ghi CDS để khớp với HMM hits")

    hits = hits.dropna(subset=["protein_id"]).copy()
    hits["protein_id"] = hits["protein_id"].astype(str).str.strip()
    hits = hits[hits["protein_id"] != ""].reset_index(drop=True)
    hits["_hit_row_id"] = hits.index

    gff = df_gff.reset_index(drop=True).copy()
    gff["_feature_row_id"] = gff.index
    alias_records: List[Dict[str, Any]] = []
    for feature_row_id, row in enumerate(gff.itertuples(index=False)):
        direct, inferred = _aliases_for_feature(row)
        alias_records.extend(
            {
                "_feature_row_id": feature_row_id,
                "_match_id": alias,
                "_alias_kind": "direct",
            }
            for alias in direct
        )
        alias_records.extend(
            {
                "_feature_row_id": feature_row_id,
                "_match_id": alias,
                "_alias_kind": "inferred",
            }
            for alias in inferred
        )

    if not alias_records:
        return _empty_match(hits)

    alias_table = pd.DataFrame.from_records(alias_records).drop_duplicates()
    direct_table = alias_table[alias_table["_alias_kind"] == "direct"].copy()
    inferred_table = alias_table[alias_table["_alias_kind"] == "inferred"].copy()

    direct_aliases = set(direct_table["_match_id"])
    inferred_table = inferred_table[~inferred_table["_match_id"].isin(direct_aliases)]
    inferred_counts = inferred_table.groupby("_match_id")["_feature_row_id"].nunique()
    ambiguous_aliases = set(inferred_counts[inferred_counts > 1].index)
    inferred_table = inferred_table[
        ~inferred_table["_match_id"].isin(ambiguous_aliases)
    ]

    direct_counts = direct_table.groupby("_match_id")["_feature_row_id"].nunique()
    direct_multimatch_aliases = set(direct_counts[direct_counts > 1].index)
    if ambiguous_aliases:
        logger.debug(
            "Bỏ qua %d alias GFF3 suy diễn không duy nhất.", len(ambiguous_aliases)
        )
    if direct_multimatch_aliases:
        logger.debug(
            "Mở rộng %d protein ID trực tiếp sang nhiều locus CDS.",
            len(direct_multimatch_aliases),
        )

    resolved_aliases = pd.concat([direct_table, inferred_table], ignore_index=True)
    if resolved_aliases.empty:
        merged = _empty_match(hits)
        merged.attrs["match_stats"]["ambiguous_aliases"] = sorted(ambiguous_aliases)
        return merged

    resolved_aliases = resolved_aliases.drop(columns="_alias_kind").drop_duplicates()
    gff_for_merge = gff.rename(columns={"protein_id": "gff_protein_id"})
    merged = hits.merge(
        resolved_aliases,
        left_on="protein_id",
        right_on="_match_id",
        how="inner",
    ).merge(gff_for_merge, on="_feature_row_id", how="inner")
    merged = merged.drop_duplicates(subset=["_hit_row_id", "_feature_row_id"])

    matched_row_ids = set(merged["_hit_row_id"]) if not merged.empty else set()
    unmatched = hits[~hits["_hit_row_id"].isin(matched_row_ids)]
    merged.attrs["match_stats"] = {
        "hit_rows": len(hits),
        "matched_hit_rows": len(matched_row_ids),
        "unmatched_protein_ids": sorted(set(unmatched["protein_id"])),
        "ambiguous_aliases": sorted(ambiguous_aliases),
        "direct_multimatch_aliases": sorted(direct_multimatch_aliases),
    }
    return merged