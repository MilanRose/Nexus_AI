"""Repeat the synthetic strategy benchmark across independent random seeds.

Example:
    python -m research.run_repeated_benchmark --seeds 20261025 20261026 20261027 20261028 20261029
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from research.benchmark_strategy_selection import run_benchmark


def _mean_t_interval(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95_lower": None, "ci95_upper": None}
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"mean": mean, "std": 0.0, "ci95_lower": None, "ci95_upper": None}
    std = float(np.std(array, ddof=1))
    margin = float(stats.t.ppf(0.975, df=len(array) - 1) * std / math.sqrt(len(array)))
    return {
        "mean": mean,
        "std": std,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate method metrics and paired accuracy differences across seeds."""
    if not runs:
        raise ValueError("At least one benchmark run is required.")
    method_names = [row["method"] for row in runs[0]["method_summary"]]
    method_summary = []
    for method in method_names:
        per_seed_rows = []
        for result in runs:
            match = next((row for row in result["method_summary"] if row["method"] == method), None)
            if match is None:
                raise ValueError(f"Method {method!r} is missing from one benchmark run.")
            per_seed_rows.append(match)
        accuracy = _mean_t_interval([row["selection_accuracy"] for row in per_seed_rows])
        coefficient_errors = [
            row["mean_absolute_coefficient_error"]
            for row in per_seed_rows
            if row.get("mean_absolute_coefficient_error") is not None
        ]
        coverage_values = [
            row["historical_recommendation_coverage"]
            for row in per_seed_rows
            if row.get("historical_recommendation_coverage") is not None
        ]
        method_summary.append({
            "method": method,
            "n_seeds": len(runs),
            "mean_selection_accuracy": accuracy["mean"],
            "std_selection_accuracy": accuracy["std"],
            "ci95_lower": accuracy["ci95_lower"],
            "ci95_upper": accuracy["ci95_upper"],
            "min_selection_accuracy": min(row["selection_accuracy"] for row in per_seed_rows),
            "max_selection_accuracy": max(row["selection_accuracy"] for row in per_seed_rows),
            "mean_absolute_coefficient_error": (
                float(np.mean(coefficient_errors)) if coefficient_errors else None
            ),
            "mean_historical_recommendation_coverage": (
                float(np.mean(coverage_values)) if coverage_values else None
            ),
            "accuracy_by_seed": {
                str(result["metadata"]["seed"]): row["selection_accuracy"]
                for result, row in zip(runs, per_seed_rows)
            },
        })

    comparison_specs = (
        ("contextual_memory", "skew_heuristic", "contextual_vs_heuristic"),
        ("contextual_memory", "task_majority_fallback", "contextual_vs_task_majority_fallback"),
    )
    paired_differences = {}
    for method_a, method_b, label in comparison_specs:
        differences = []
        for result in runs:
            table = {row["method"]: row for row in result["method_summary"]}
            differences.append(
                table[method_a]["selection_accuracy"]
                - table[method_b]["selection_accuracy"]
            )
        paired_differences[label] = _mean_t_interval(differences)
        paired_differences[label]["differences_by_seed"] = {
            str(result["metadata"]["seed"]): difference
            for result, difference in zip(runs, differences)
        }

    scenario_names = [row["scenario"] for row in runs[0]["per_scenario_summary"]]
    per_scenario_summary = []
    scenario_metric_names = [
        "fixed_pearson_accuracy",
        "fixed_spearman_accuracy",
        "skew_heuristic_accuracy",
        "task_majority_fallback_accuracy",
        "contextual_memory_accuracy",
        "contextual_memory_coverage",
    ]
    for scenario_name in scenario_names:
        scenario_rows_by_seed = []
        for result in runs:
            match = next(
                row for row in result["per_scenario_summary"]
                if row["scenario"] == scenario_name
            )
            scenario_rows_by_seed.append(match)
        base = scenario_rows_by_seed[0]
        record = {
            "scenario": scenario_name,
            "reference_method": base["reference_method"],
            "n_test_cases_per_seed": base["n_test_cases"],
        }
        for metric in scenario_metric_names:
            interval = _mean_t_interval([row[metric] for row in scenario_rows_by_seed])
            lower = interval["ci95_lower"]
            upper = interval["ci95_upper"]
            if metric.endswith("_accuracy") or metric.endswith("_coverage"):
                # Accuracy and coverage are bounded proportions. Clamp the
                # approximate t interval to the valid range [0, 1].
                lower = max(0.0, lower) if lower is not None else None
                upper = min(1.0, upper) if upper is not None else None
            record[f"{metric}_mean"] = interval["mean"]
            record[f"{metric}_std"] = interval["std"]
            record[f"{metric}_ci95_lower"] = lower
            record[f"{metric}_ci95_upper"] = upper
        per_scenario_summary.append(record)

    return {
        "n_seeds": len(runs),
        "seeds": [result["metadata"]["seed"] for result in runs],
        "train_per_family": runs[0]["metadata"]["train_per_family"],
        "test_per_family": runs[0]["metadata"]["test_per_family"],
        "n_train_experiences_per_seed": runs[0]["metadata"]["n_train_experiences"],
        "n_test_cases_per_seed": runs[0]["metadata"]["n_test_cases"],
        "n_total_test_cases_across_seeds": sum(result["metadata"]["n_test_cases"] for result in runs),
        "method_summary": method_summary,
        "per_scenario_summary": per_scenario_summary,
        "paired_seed_accuracy_differences": paired_differences,
        "stress_tests_by_seed": {
            str(result["metadata"]["seed"]): result["duplicate_context_stress_test"]
            for result in runs
        },
        "interpretation_note": (
            "Means and t-based intervals describe variability across the specified random seeds "
            "under this synthetic benchmark protocol. They do not establish generalization to "
            "unseen scientific domains or real datasets. Training history uses oracle-labelled "
            "scenario experiences; repeat on held-out real datasets and independent quality labels."
        ),
    }


def write_repeated_report(aggregate: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "repeated_seed_summary.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "repeated_seed_per_scenario.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate["per_scenario_summary"][0].keys()))
        writer.writeheader()
        writer.writerows(aggregate["per_scenario_summary"])

    with (output_dir / "repeated_seed_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "method", "n_seeds", "mean_selection_accuracy", "std_selection_accuracy",
            "ci95_lower", "ci95_upper", "min_selection_accuracy", "max_selection_accuracy",
            "mean_absolute_coefficient_error", "mean_historical_recommendation_coverage",
            "accuracy_by_seed",
        ])
        writer.writeheader()
        for row in aggregate["method_summary"]:
            writer.writerow({
                **row,
                "accuracy_by_seed": json.dumps(row["accuracy_by_seed"], sort_keys=True),
            })

    lines = [
        "# Repeated-Seed Synthetic Benchmark Summary",
        "",
        f"- Independent random seeds: `{aggregate['seeds']}`",
        f"- Training experiences per seed: {aggregate['n_train_experiences_per_seed']}",
        f"- Test cases per seed: {aggregate['n_test_cases_per_seed']}",
        f"- Total held-out synthetic evaluations: {aggregate['n_total_test_cases_across_seeds']}",
        "- The interval is a two-sided 95% Student-t interval across run-level accuracies; it is descriptive and based on the number of seeds, not the individual test rows.",
        "",
        "## Results",
        "",
        "| Method | Mean accuracy | SD across seeds | 95% t interval | Range | Mean coefficient error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["method_summary"]:
        lower = row["ci95_lower"]
        upper = row["ci95_upper"]
        interval = "N/A" if lower is None else f"[{lower:.1%}, {upper:.1%}]"
        error = row["mean_absolute_coefficient_error"]
        lines.append(
            f"| {row['method']} | {row['mean_selection_accuracy']:.1%} | "
            f"{row['std_selection_accuracy']:.1%} | {interval} | "
            f"[{row['min_selection_accuracy']:.1%}, {row['max_selection_accuracy']:.1%}] | "
            f"{error:.4f} |"
        )
    lines += [
        "",
        "## Per-scenario accuracy across seeds",
        "",
        "| Scenario | Reference | Pearson | Spearman | Skew heuristic | Contextual memory | Contextual coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["per_scenario_summary"]:
        lines.append(
            f"| {row['scenario']} | {row['reference_method']} | "
            f"{row['fixed_pearson_accuracy_mean']:.1%} | "
            f"{row['fixed_spearman_accuracy_mean']:.1%} | "
            f"{row['skew_heuristic_accuracy_mean']:.1%} | "
            f"{row['contextual_memory_accuracy_mean']:.1%} | "
            f"{row['contextual_memory_coverage_mean']:.1%} |"
        )
    lines += ["", "## Paired seed-level accuracy differences", "", "| Comparison | Mean difference | 95% t interval |", "|---|---:|---:|"]
    for label, row in aggregate["paired_seed_accuracy_differences"].items():
        interval = "N/A" if row["ci95_lower"] is None else f"[{row['ci95_lower']:+.1%}, {row['ci95_upper']:+.1%}]"
        lines.append(f"| {label} | {row['mean']:+.1%} | {interval} |")
    lines += [
        "",
        "## Interpretation limits",
        "",
        aggregate["interpretation_note"],
        "",
        "The benchmark is based on a controlled synthetic scenario policy. The next required step before a conference performance claim is replication with held-out real datasets and quality labels validated independently of the strategy selector.",
        "",
    ]
    (output_dir / "repeated_seed_report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = aggregate["method_summary"]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    labels = [row["method"].replace("_", "\n") for row in summary]
    means = [row["mean_selection_accuracy"] * 100 for row in summary]
    err_lower = [
        (row["mean_selection_accuracy"] - row["ci95_lower"]) * 100
        if row["ci95_lower"] is not None else 0.0
        for row in summary
    ]
    err_upper = [
        (row["ci95_upper"] - row["mean_selection_accuracy"]) * 100
        if row["ci95_upper"] is not None else 0.0
        for row in summary
    ]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=np.array([err_lower, err_upper]), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Selection accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Repeated-Seed Synthetic Strategy Benchmark")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "repeated_seed_accuracy.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[20261025, 20261026, 20261027, 20261028, 20261029],
    )
    parser.add_argument("--train-per-family", type=int, default=12)
    parser.add_argument("--test-per-family", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/research_evaluation/multi_seed"))
    args = parser.parse_args(argv)
    if len(args.seeds) < 2:
        parser.error("Provide at least two seeds to estimate between-run variability.")

    runs = []
    seed_output = args.output / "seed_results"
    seed_output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"Running seed {seed} ...", flush=True)
        result = run_benchmark(
            train_per_family=args.train_per_family,
            test_per_family=args.test_per_family,
            seed=seed,
        )
        runs.append(result)
        (seed_output / f"benchmark_seed_{seed}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )

    aggregate = aggregate_runs(runs)
    write_repeated_report(aggregate, args.output)
    print(f"Wrote repeated-seed report to: {args.output.resolve()}")
    for row in aggregate["method_summary"]:
        print(
            f"{row['method']:<25} mean_accuracy={row['mean_selection_accuracy']:.3f} "
            f"sd={row['std_selection_accuracy']:.3f} "
            f"range=[{row['min_selection_accuracy']:.3f}, {row['max_selection_accuracy']:.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
