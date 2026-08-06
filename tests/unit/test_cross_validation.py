"""Unit tests for repeated stratified grouped cross-validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import nifpredict.evaluation.cross_validation as cv_module
from nifpredict.evaluation import (
    GroupedCVConfig,
    GroupedCVValidationError,
    generate_fold_manifest,
    run_grouped_cross_validation,
)
from nifpredict.models import (
    LABEL_MAPPING,
    TrainingResult,
    prepare_training_data,
)

FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "grouped_training_dataset.csv"
)


@pytest.fixture
def grouped_frame() -> pd.DataFrame:
    """Load a deterministic offline grouped binary dataset."""
    return pd.read_csv(FIXTURE_PATH)


def _config(**overrides: Any) -> GroupedCVConfig:
    values: dict[str, Any] = {
        "model_names": ("logistic_regression",),
        "n_splits": 2,
        "n_repeats": 2,
        "random_seed": 17,
    }
    values.update(overrides)
    return GroupedCVConfig(**values)


def test_same_seed_produces_identical_manifest(
    grouped_frame: pd.DataFrame,
) -> None:
    first, _ = generate_fold_manifest(grouped_frame, _config())
    second, _ = generate_fold_manifest(grouped_frame, _config())

    pd.testing.assert_frame_equal(first, second)


def test_derivatives_stay_together_and_groups_do_not_overlap(
    grouped_frame: pd.DataFrame,
) -> None:
    manifest, _ = generate_fold_manifest(grouped_frame, _config())

    assert (
        manifest.groupby(["repeat", "source_genome_id"])["fold"].nunique() == 1
    ).all()
    for (repeat, fold), validation_rows in manifest.groupby(["repeat", "fold"]):
        validation_groups = set(validation_rows["source_genome_id"])
        training_groups = set(grouped_frame["source_genome_id"]) - validation_groups
        assert validation_groups.isdisjoint(training_groups), (repeat, fold)


def test_each_group_is_validation_once_per_repeat_and_folds_have_both_classes(
    grouped_frame: pd.DataFrame,
) -> None:
    manifest, _ = generate_fold_manifest(grouped_frame, _config())

    for repeat, repeat_rows in manifest.groupby("repeat"):
        assert set(repeat_rows["assembly_accession"]) == set(
            grouped_frame["assembly_accession"]
        )
        assert not repeat_rows["assembly_accession"].duplicated().any(), repeat
        for _, fold_rows in repeat_rows.groupby("fold"):
            assert set(fold_rows["target_label"]) == {"negative", "positive"}


def test_repeats_use_valid_complete_assignments(
    grouped_frame: pd.DataFrame,
) -> None:
    manifest, _ = generate_fold_manifest(
        grouped_frame,
        _config(n_repeats=3),
    )

    assert set(manifest["repeat"]) == {1, 2, 3}
    assert len(manifest) == len(grouped_frame) * 3


def test_mixed_label_group_is_rejected(grouped_frame: pd.DataFrame) -> None:
    invalid = grouped_frame.copy()
    invalid.loc[1, "target_label"] = "positive"

    with pytest.raises(GroupedCVValidationError, match="mixed target labels"):
        generate_fold_manifest(invalid, _config())


def test_insufficient_class_groups_are_rejected(grouped_frame: pd.DataFrame) -> None:
    with pytest.raises(
        GroupedCVValidationError,
        match="requires at least 5 independent groups",
    ):
        generate_fold_manifest(grouped_frame, _config(n_splits=5))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_splits", 1, "n_splits=1"),
        ("n_repeats", 0, "n_repeats=0"),
    ],
)
def test_invalid_cv_counts_are_actionable(
    grouped_frame: pd.DataFrame,
    field: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(GroupedCVValidationError, match=message):
        generate_fold_manifest(grouped_frame, _config(**{field: value}))


def test_all_models_share_folds_and_oof_coverage(
    grouped_frame: pd.DataFrame,
) -> None:
    config = _config(
        model_names=("logistic_regression", "random_forest", "svm"),
        n_repeats=1,
    )
    result = run_grouped_cross_validation(grouped_frame, config)

    expected_pairs = set(
        result.fold_manifest[["repeat", "fold"]].itertuples(index=False, name=None)
    )
    for model_name, model_rows in result.oof_predictions.groupby("model_name"):
        observed_pairs = set(
            model_rows[["repeat", "fold"]].itertuples(index=False, name=None)
        )
        assert observed_pairs == expected_pairs, model_name
        assert len(model_rows) == len(grouped_frame)
        assert not model_rows.duplicated(
            ["repeat", "assembly_accession"]
        ).any()

    assert tuple(result.oof_predictions.columns) == cv_module.OOF_COLUMNS
    assert tuple(result.fold_metrics.columns) == cv_module.FOLD_METRIC_COLUMNS
    assert set(result.fold_metrics["model_name"]) == set(config.model_names)
    assert set(result.aggregate_metrics["model_name"]) == set(config.model_names)
    assert result.selection.selected_model in config.model_names


def test_new_pipeline_and_scaler_are_fit_per_training_fold(
    grouped_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_train = cv_module.train_classifier
    estimators: list[Any] = []
    training_sizes: list[int] = []

    def tracking_train(frame: pd.DataFrame, **kwargs: Any) -> TrainingResult:
        result = original_train(frame, **kwargs)
        estimators.append(result.estimator)
        training_sizes.append(len(frame))
        return result

    monkeypatch.setattr(cv_module, "train_classifier", tracking_train)
    run_grouped_cross_validation(grouped_frame, _config())

    assert len(estimators) == 4
    assert len({id(estimator) for estimator in estimators}) == 4
    for estimator, training_size in zip(estimators, training_sizes, strict=True):
        assert isinstance(estimator, Pipeline)
        assert estimator.named_steps["scaler"].n_samples_seen_ == training_size


def test_decision_scores_are_not_labeled_as_probabilities(
    grouped_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def train_decision_model(
        frame: pd.DataFrame,
        *,
        model_name: str,
        target_column: str,
        **_: Any,
    ) -> TrainingResult:
        prepared = prepare_training_data(frame, target_column=target_column)
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(random_state=5)),
            ]
        ).fit(prepared.features, prepared.target)
        return TrainingResult(
            estimator=estimator,
            model_name=model_name,
            feature_names=prepared.feature_names,
            label_mapping=dict(LABEL_MAPPING),
            class_distribution=prepared.class_distribution,
        )

    monkeypatch.setattr(cv_module, "train_classifier", train_decision_model)
    result = run_grouped_cross_validation(
        grouped_frame,
        _config(model_names=("decision_only",), n_repeats=1),
    )

    assert result.oof_predictions["positive_probability"].isna().all()
    assert result.oof_predictions["decision_score"].notna().all()


def test_external_manifest_is_validated_before_training(
    grouped_frame: pd.DataFrame,
) -> None:
    manifest, _ = generate_fold_manifest(grouped_frame, _config())
    invalid = manifest.iloc[1:].copy()

    with pytest.raises(GroupedCVValidationError, match="cover every accession"):
        run_grouped_cross_validation(
            grouped_frame,
            _config(),
            fold_manifest=invalid,
        )