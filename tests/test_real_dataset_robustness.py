from pathlib import Path

import numpy as np
import pandas as pd

from research.run_real_dataset_robustness import (
    PERTURBATIONS,
    load_real_datasets,
    perturb_pair,
    run_experiment,
    select_numeric_pairs,
)


def test_loader_returns_real_numeric_feature_frames_and_excludes_titanic_target():
    root = Path(__file__).resolve().parents[1]
    datasets = load_real_datasets(root)

    assert {"diabetes", "wine_recognition", "breast_cancer_diagnostic", "iris"}.issubset(datasets)
    assert "titanic_sample" not in datasets  # unverified provenance is excluded by default
    assert all(all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes) for frame in datasets.values())

    exploratory = load_real_datasets(root, include_titanic_sample=True)
    assert "titanic_sample" in exploratory
    assert "PassengerId" not in exploratory["titanic_sample"].columns
    assert "Survived" not in exploratory["titanic_sample"].columns
    assert "Name" not in exploratory["titanic_sample"].columns


def test_pair_selection_is_deterministic_and_filters_degenerate_pairs():
    frame = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6] * 6,
        "y": [2, 3, 4, 5, 6, 7] * 6,
        "constant": [1] * 36,
    })
    first = select_numeric_pairs(frame, max_pairs=5, seed=42, min_complete=30)
    second = select_numeric_pairs(frame, max_pairs=5, seed=42, min_complete=30)
    assert first == second
    assert all("constant" not in pair for pair in first)


def test_perturbations_are_reproducible_and_do_not_mutate_source():
    frame = pd.DataFrame({
        "x": np.linspace(0.0, 1.0, 100),
        "y": np.linspace(0.0, 1.0, 100) ** 2,
    })
    original = frame.copy(deep=True)
    for perturbation in PERTURBATIONS:
        first = perturb_pair(frame, "x", "y", perturbation, seed=12)
        second = perturb_pair(frame, "x", "y", perturbation, seed=12)
        pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(frame, original)


def test_small_real_data_experiment_produces_baselines_and_heldout_folds():
    root = Path(__file__).resolve().parents[1]
    result = run_experiment(
        project_root=root,
        seeds=[7],
        max_pairs_per_dataset=2,
        perturbation_replicates=2,
        label_margin=0.001,
    )
    methods = {row["method"] for row in result["method_summary"]}
    assert methods == {
        "fixed_pearson",
        "fixed_spearman",
        "skew_heuristic",
        "task_majority_fallback",
        "contextual_memory",
    }
    assert result["n_heldout_datasets"] >= 4
    assert result["n_test_cases"] > 0
    assert len(result["per_dataset_seed_summary"]) == result["n_heldout_datasets"]
    assert all(row["heldout_dataset"] in result["datasets"] for row in result["per_dataset_seed_summary"])
    assert {row["perturbation"] for row in result["per_perturbation_summary"]} == set(PERTURBATIONS)
    assert all("label_decisive" in row for row in result["cases"])
