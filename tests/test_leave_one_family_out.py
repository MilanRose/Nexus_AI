from research.benchmark_strategy_selection import SCENARIOS
from research.run_leave_one_family_out import run_experiment, run_fold


def test_leave_one_family_out_excludes_held_out_family_from_training():
    held_out = SCENARIOS[0].name
    result = run_fold(
        held_out_family=held_out,
        train_per_family=1,
        test_per_family=2,
        seed=101,
    )
    summary = result["fold_summary"]
    assert held_out not in summary["training_families"]
    assert summary["n_train_experiences"] == len(SCENARIOS) - 1
    assert summary["n_test_cases"] == 2
    assert all(row["held_out_family"] == held_out for row in result["test_cases"])
    assert all(
        0.0 <= accuracy <= 1.0
        for accuracy in summary["method_accuracy"].values()
    )
    # A new dataset has no exact-dataset history in this fold, so the
    # production-compatible history baseline must fall back to the heuristic.
    for row in result["test_cases"]:
        assert row["dataset_history_recommendation"] is None
        assert row["dataset_history_fallback_selected"] == row["heuristic_method"]
        assert row["dataset_history_fallback_correct"] == row["skew_heuristic_correct"]


def test_leave_one_family_out_runs_each_family_and_reports_fallback_coverage():
    result = run_experiment(
        seeds=[202],
        train_per_family=1,
        test_per_family=2,
    )
    assert result["metadata"]["n_fold_runs"] == len(SCENARIOS)
    assert result["metadata"]["n_total_test_cases"] == len(SCENARIOS) * 2
    assert {row["held_out_family"] for row in result["per_family_summary"]} == {
        scenario.name for scenario in SCENARIOS
    }
    assert all(
        0.0 <= row["contextual_coverage"] <= 1.0
        for row in result["per_family_summary"]
    )
    # Distinguish dataset-scoped production fallback from the intentionally
    # naive global majority diagnostic. They answer different questions.
    for row in result["per_family_summary"]:
        assert row["dataset_history_fallback_accuracy"] == row["skew_heuristic_accuracy"]
    for row in result["test_cases"]:
        assert row["dataset_history_fallback_selected"] == row["heuristic_method"]
