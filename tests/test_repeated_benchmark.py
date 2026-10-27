import pytest

from research.run_repeated_benchmark import aggregate_runs


def _fake_run(seed, contextual_accuracy, heuristic_accuracy):
    methods = [
        ("fixed_pearson", 0.50),
        ("fixed_spearman", 0.50),
        ("skew_heuristic", heuristic_accuracy),
        ("task_majority_fallback", heuristic_accuracy),
        ("contextual_memory", contextual_accuracy),
    ]
    return {
        "metadata": {
            "seed": seed,
            "train_per_family": 2,
            "test_per_family": 3,
            "n_train_experiences": 12,
            "n_test_cases": 18,
        },
        "method_summary": [
            {
                "method": name,
                "selection_accuracy": accuracy,
                "mean_absolute_coefficient_error": 0.1,
                "historical_recommendation_coverage": None,
            }
            for name, accuracy in methods
        ],
        "per_scenario_summary": [{
            "scenario": "scenario_a",
            "reference_method": "pearson",
            "n_test_cases": 3,
            "fixed_pearson_accuracy": 1.0,
            "fixed_spearman_accuracy": 0.0,
            "skew_heuristic_accuracy": heuristic_accuracy,
            "task_majority_fallback_accuracy": heuristic_accuracy,
            "contextual_memory_accuracy": contextual_accuracy,
            "contextual_memory_coverage": 1.0,
        }],
        "duplicate_context_stress_test": {"contextual_recommendation_unchanged": True},
    }


def test_repeated_summary_aggregates_across_seeds_and_paired_differences():
    summary = aggregate_runs([
        _fake_run(101, 0.8, 0.6),
        _fake_run(102, 0.9, 0.7),
    ])
    methods = {row["method"]: row for row in summary["method_summary"]}

    assert summary["n_seeds"] == 2
    assert methods["contextual_memory"]["mean_selection_accuracy"] == pytest.approx(0.85)
    assert methods["contextual_memory"]["min_selection_accuracy"] == pytest.approx(0.8)
    assert methods["contextual_memory"]["max_selection_accuracy"] == pytest.approx(0.9)
    assert summary["per_scenario_summary"][0]["contextual_memory_accuracy_mean"] == pytest.approx(0.85)
    difference = summary["paired_seed_accuracy_differences"]["contextual_vs_heuristic"]
    assert difference["mean"] == pytest.approx(0.2)
    assert set(difference["differences_by_seed"]) == {"101", "102"}
