import math

from research.benchmark_strategy_selection import (
    SCENARIOS,
    generate_scenario,
    run_benchmark,
)


def test_synthetic_scenario_generation_is_reproducible():
    left = generate_scenario("linear_clean", seed=31415)
    right = generate_scenario("linear_clean", seed=31415)
    assert left.equals(right)


def test_benchmark_contains_all_baselines_and_finite_metrics():
    result = run_benchmark(
        train_per_family=2,
        test_per_family=3,
        seed=1212,
    )

    expected_methods = {
        "fixed_pearson",
        "fixed_spearman",
        "skew_heuristic",
        "task_majority_fallback",
        "contextual_memory",
    }
    summaries = {row["method"]: row for row in result["method_summary"]}

    assert set(summaries) == expected_methods
    assert result["metadata"]["n_train_experiences"] == 2 * len(SCENARIOS)
    assert result["metadata"]["n_test_cases"] == 3 * len(SCENARIOS)
    assert len(result["test_cases"]) == 3 * len(SCENARIOS)
    for summary in summaries.values():
        assert 0.0 <= summary["selection_accuracy"] <= 1.0
        assert math.isfinite(summary["mean_absolute_coefficient_error"])


def test_benchmark_uses_non_overlapping_train_and_test_seed_ranges():
    result = run_benchmark(
        train_per_family=1,
        test_per_family=2,
        seed=987,
    )
    test_seeds = {row["seed"] for row in result["test_cases"]}
    training_seed_start = 987 + 10_000
    training_seed_end = training_seed_start + len(SCENARIOS) * 10_000
    assert not any(training_seed_start <= test_seed < training_seed_end for test_seed in test_seeds)


def test_duplicate_history_stress_suppresses_replay_votes():
    result = run_benchmark(
        train_per_family=2,
        test_per_family=2,
        seed=4567,
    )
    stress = result["duplicate_context_stress_test"]
    assert stress["replay_count"] == 20
    assert stress["raw_task_majority_before"] is None
    assert stress["raw_task_majority_after"] == "pearson"
    assert stress["contextual_recommendation_unchanged"] is True
