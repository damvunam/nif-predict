"""Dataset and split-level leakage checks for grouped cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nifpredict.models.trainer import LABEL_MAPPING


class GroupedCVValidationError(ValueError):
    """Raised when grouped cross-validation would be invalid or leaky."""


@dataclass(frozen=True)
class GroupedDatasetSummary:
    """Validated row, group, and class counts used for provenance."""

    row_count: int
    accession_count: int
    group_count: int
    class_row_counts: dict[str, int]
    class_group_counts: dict[str, int]


def validate_grouped_cv_input(
    frame: pd.DataFrame,
    *,
    target_column: str,
    positive_label: str,
    accession_column: str,
    group_column: str,
    n_splits: int,
    n_repeats: int,
) -> GroupedDatasetSummary:
    """Validate dataset-level requirements before generating any folds."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Grouped CV input must be a pandas DataFrame")
    if frame.empty:
        raise GroupedCVValidationError("Grouped CV dataset contains no rows")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise GroupedCVValidationError(
            f"n_splits={n_splits!r} is invalid; grouped CV requires n_splits >= 2"
        )
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise GroupedCVValidationError(
            f"n_repeats={n_repeats!r} is invalid; repeated CV requires n_repeats >= 1"
        )

    required_columns = {target_column, accession_column, group_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise GroupedCVValidationError(
            "Grouped CV dataset is missing required column(s): "
            + ", ".join(missing_columns)
        )
    if frame.columns.duplicated().any():
        duplicates = sorted(
            set(frame.columns[frame.columns.duplicated()].astype(str))
        )
        raise GroupedCVValidationError(
            "Grouped CV dataset contains duplicate column name(s): "
            + ", ".join(duplicates)
        )

    if positive_label not in LABEL_MAPPING:
        raise GroupedCVValidationError(
            f"Positive label '{positive_label}' is unsupported; expected 'positive'"
        )
    if LABEL_MAPPING[positive_label] != 1:
        raise GroupedCVValidationError(
            f"Positive label '{positive_label}' must map to encoded class 1"
        )

    target = frame[target_column]
    if target.isna().any():
        raise GroupedCVValidationError(f"Column '{target_column}' contains null labels")
    observed_labels = set(target.astype(str))
    expected_labels = set(LABEL_MAPPING)
    if observed_labels != expected_labels:
        raise GroupedCVValidationError(
            f"Column '{target_column}' must contain exactly negative and positive; "
            f"observed {sorted(observed_labels)}"
        )

    if frame[group_column].isna().any():
        raise GroupedCVValidationError(
            f"Group column '{group_column}' contains null values"
        )
    if frame[accession_column].isna().any():
        raise GroupedCVValidationError(
            f"Accession column '{accession_column}' contains null values"
        )
    if frame[accession_column].duplicated().any():
        duplicate_accessions = sorted(
            frame.loc[
                frame[accession_column].duplicated(keep=False), accession_column
            ]
            .astype(str)
            .unique()
        )
        preview = ", ".join(duplicate_accessions[:5])
        raise GroupedCVValidationError(
            f"Accession column '{accession_column}' must be unique; duplicate(s): "
            f"{preview}"
        )

    labels_per_group = frame.groupby(group_column, dropna=False)[
        target_column
    ].nunique()
    mixed_groups = sorted(labels_per_group.index[labels_per_group > 1].astype(str))
    if mixed_groups:
        preview = ", ".join(mixed_groups[:10])
        raise GroupedCVValidationError(
            f"Group column '{group_column}' contains mixed target labels for group(s): "
            f"{preview}. Curate each group to one label before CV."
        )

    group_labels = frame[[group_column, target_column]].drop_duplicates(group_column)
    class_group_counts = {
        label: int((group_labels[target_column] == label).sum())
        for label in LABEL_MAPPING
    }
    insufficient = {
        label: count
        for label, count in class_group_counts.items()
        if count < n_splits
    }
    if insufficient:
        details = ", ".join(
            f"{label}={count}" for label, count in sorted(insufficient.items())
        )
        raise GroupedCVValidationError(
            f"n_splits={n_splits} requires at least {n_splits} independent groups "
            f"in each class; observed {details}"
        )

    return GroupedDatasetSummary(
        row_count=len(frame),
        accession_count=int(frame[accession_column].nunique()),
        group_count=int(frame[group_column].nunique()),
        class_row_counts={
            label: int((target == label).sum()) for label in LABEL_MAPPING
        },
        class_group_counts=class_group_counts,
    )


def validate_fold_split(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    repeat: int,
    fold: int,
) -> None:
    """Reject empty, single-class, or group-overlapping fold partitions."""
    if training_frame.empty or validation_frame.empty:
        raise GroupedCVValidationError(
            f"Repeat {repeat}, fold {fold} produced an empty train or validation split"
        )

    expected_labels = set(LABEL_MAPPING)
    for split_name, split_frame in (
        ("training", training_frame),
        ("validation", validation_frame),
    ):
        labels = set(split_frame[target_column].astype(str))
        if labels != expected_labels:
            raise GroupedCVValidationError(
                f"Repeat {repeat}, fold {fold} {split_name} split must contain both "
                f"classes; observed {sorted(labels)}"
            )

    training_groups = set(training_frame[group_column])
    validation_groups = set(validation_frame[group_column])
    overlap = sorted(str(group) for group in training_groups & validation_groups)
    if overlap:
        raise GroupedCVValidationError(
            f"Repeat {repeat}, fold {fold} has group leakage: " + ", ".join(overlap)
        )


def validate_fold_manifest(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    target_column: str,
    accession_column: str,
    group_column: str,
    n_splits: int,
    n_repeats: int,
) -> None:
    """Validate validation-only assignments and their complete coverage."""
    required_columns = {
        "repeat",
        "fold",
        "split",
        accession_column,
        group_column,
        target_column,
    }
    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise GroupedCVValidationError(
            "Fold manifest is missing column(s): " + ", ".join(missing)
        )
    if manifest.empty:
        raise GroupedCVValidationError("Fold manifest contains no assignments")
    if set(manifest["split"].astype(str)) != {"validation"}:
        raise GroupedCVValidationError(
            "Fold manifest must contain validation-only assignments"
        )

    expected_accessions = set(frame[accession_column])
    expected_repeats = set(range(1, n_repeats + 1))
    observed_repeats = set(manifest["repeat"])
    if observed_repeats != expected_repeats:
        raise GroupedCVValidationError(
            f"Fold manifest repeats must be {sorted(expected_repeats)}; "
            f"observed {sorted(observed_repeats)}"
        )

    expected_folds = set(range(1, n_splits + 1))
    for repeat in sorted(expected_repeats):
        repeat_rows = manifest.loc[manifest["repeat"] == repeat]
        observed_folds = set(repeat_rows["fold"])
        if observed_folds != expected_folds:
            raise GroupedCVValidationError(
                f"Repeat {repeat} fold IDs must be {sorted(expected_folds)}; "
                f"observed {sorted(observed_folds)}"
            )
        if repeat_rows[accession_column].duplicated().any():
            raise GroupedCVValidationError(
                f"Repeat {repeat} assigns an accession to validation more than once"
            )
        if set(repeat_rows[accession_column]) != expected_accessions:
            raise GroupedCVValidationError(
                f"Repeat {repeat} does not cover every accession exactly once"
            )

        group_fold_counts = repeat_rows.groupby(group_column)["fold"].nunique()
        split_groups = sorted(
            group_fold_counts.index[group_fold_counts != 1].astype(str)
        )
        if split_groups:
            raise GroupedCVValidationError(
                f"Repeat {repeat} splits derivative group(s) across validation folds: "
                + ", ".join(split_groups)
            )

        for fold in sorted(expected_folds):
            validation_accessions = set(
                repeat_rows.loc[
                    repeat_rows["fold"] == fold, accession_column
                ]
            )
            validation_frame = frame.loc[
                frame[accession_column].isin(validation_accessions)
            ]
            validation_groups = set(validation_frame[group_column])
            training_frame = frame.loc[
                ~frame[group_column].isin(validation_groups)
            ]
            validate_fold_split(
                training_frame,
                validation_frame,
                target_column=target_column,
                group_column=group_column,
                repeat=repeat,
                fold=fold,
            )

    manifest_lookup = manifest.set_index(["repeat", accession_column])
    for repeat in expected_repeats:
        for _, source_row in frame.iterrows():
            manifest_row = manifest_lookup.loc[(repeat, source_row[accession_column])]
            if manifest_row[group_column] != source_row[group_column]:
                raise GroupedCVValidationError(
                    f"Fold manifest changes group identity for accession "
                    f"'{source_row[accession_column]}'"
                )
            if manifest_row[target_column] != source_row[target_column]:
                raise GroupedCVValidationError(
                    f"Fold manifest changes target label for accession "
                    f"'{source_row[accession_column]}'"
                )
