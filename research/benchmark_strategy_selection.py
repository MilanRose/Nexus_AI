"""Reproducible synthetic benchmark for correlation-strategy selection.

This benchmark evaluates strategy selection against a declared, scenario-specific
reference policy. It is not a claim that either Pearson or Spearman is universally
superior, and the synthetic training memory uses oracle-labelled prior experiences.
Real-dataset generalization and quality-verified online learning remain separate
follow-up experiments.

Run from the repository root:
    python -m research.benchmark_strategy_selection

Optional arguments:
    --train-per-family 12 --test-per-family 30 --seed 20261025
    --output outputs/research_evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.knowledge_graph.graph import KnowledgeGraph
from src.analysis import (
    build_correlation_context,
    execute_correlation_task,
    select_correlation_strategy,
)

TASK = "investigate_correlation"
STRATEGY_NAMES = {
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
}
DISPLAY_TO_METHOD = {value: key for key, value in STRATEGY_NAMES.items()}


@dataclass(frozen=True)
class Scenario:
    name: str
    reference_method: str
    description: str
    sample_size: int = 180


SCENARIOS = (
    Scenario(
        "linear_clean", "pearson",
        "Approximately linear association with light-tailed noise.", 180,
    ),
    Scenario(
        "linear_missing", "pearson",
        "Linear association with MCAR missingness in both columns.", 220,
    ),
    Scenario(
        "linear_small_sample", "pearson",
        "Linear association with a smaller sample and noisier profile estimates.", 45,
    ),
    Scenario(
        "monotonic_exponential", "spearman",
        "Monotonic nonlinear association with a strongly skewed response.", 180,
    ),
    Scenario(
        "monotonic_cubic", "spearman",
        "Monotonic cubic association; the declared target is rank association.", 180,
    ),
    Scenario(
        "monotonic_with_outliers", "spearman",
        "Underlying monotonic association with symmetric extreme response contamination; the target is robust rank association to the uncontaminated process.", 220,
    ),
)


def generate_scenario(
    scenario_name: str,
    seed: int,
    *,
    include_perturbations: bool = True,
    sample_size_override: int | None = None,
) -> pd.DataFrame:
    """Generate one reproducible scenario with known intended analysis objective."""
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    rng = np.random.default_rng(seed)
    if sample_size_override is not None:
        n = sample_size_override
    else:
        # Randomize sample size so the benchmark is not testing only six fixed
        # profile points. Keep small-sample cases large enough for correlation.
        spread = max(5, int(round(scenario.sample_size * 0.15)))
        n = max(30, scenario.sample_size + int(rng.integers(-spread, spread + 1)))
    x = rng.normal(size=n)

    if scenario_name in {"linear_clean", "linear_missing", "linear_small_sample"}:
        slope = {"linear_clean": 0.85, "linear_missing": 0.65,
                 "linear_small_sample": 0.8}[scenario_name]
        noise_ranges = {
            "linear_clean": (0.55, 0.95),
            "linear_missing": (0.65, 1.05),
            "linear_small_sample": (0.45, 0.95),
        }
        noise = rng.uniform(*noise_ranges[scenario_name])
        y = slope * x + rng.normal(0.0, noise, size=n)
    elif scenario_name == "monotonic_exponential":
        noise = rng.uniform(0.25, 0.65)
        y = np.exp(0.9 * x) + rng.normal(0.0, noise, size=n)
    elif scenario_name == "monotonic_cubic":
        noise = rng.uniform(0.25, 0.75)
        y = x ** 3 + rng.normal(0.0, noise, size=n)
    elif scenario_name == "monotonic_with_outliers":
        noise = rng.uniform(0.35, 0.75)
        y = 0.75 * x + rng.normal(0.0, noise, size=n)
        if include_perturbations:
            # Inject a balanced set of positive and negative extreme response
            # perturbations, independent of x, to make the robustness target clear.
            contamination_fraction = rng.uniform(0.04, 0.08)
            count = max(4, int(round(n * contamination_fraction)))
            positions = rng.choice(n, size=count, replace=False)
            signs = np.resize(np.array([-1.0, 1.0]), count)
            rng.shuffle(signs)
            y[positions] += signs * rng.uniform(8.0, 18.0, size=count)
    else:  # pragma: no cover - protects future scenario additions
        raise ValueError(f"Unknown scenario: {scenario_name}")

    frame = pd.DataFrame({"x": x, "y": y})
    if include_perturbations and scenario_name == "linear_missing":
        # Missing Completely At Random: the mask is independent of values.
        mask_x = rng.random(n) < rng.uniform(0.08, 0.16)
        mask_y = rng.random(n) < rng.uniform(0.08, 0.16)
        frame.loc[mask_x, "x"] = np.nan
        frame.loc[mask_y, "y"] = np.nan
    return frame


def _reference_coefficients(scenario: Scenario, seed: int) -> dict[str, float]:
    """Approximate clean-process population coefficients with a large sample."""
    population = generate_scenario(
        scenario.name,
        seed,
        include_perturbations=False,
        sample_size_override=50_000,
    )
    clean = population[["x", "y"]].dropna()
    return {
        "pearson": float(stats.pearsonr(clean["x"], clean["y"]).statistic),
        "spearman": float(stats.spearmanr(clean["x"], clean["y"]).statistic),
    }


def _seed_experience(
    graph: KnowledgeGraph,
    *,
    execution_id: str,
    dataset: str,
    method: str,
    context: dict[str, Any],
) -> None:
    """Insert a synthetic prior experience with a known scenario reference label.

    The prior label is oracle-labelled from the scenario protocol. This is a
    controlled retrieval benchmark, not a simulation of autonomous quality
    evaluation. That distinction is documented in the generated report.
    """
    graph.add_node(
        execution_id,
        "execution",
        agent="EDAAgent",
        task=TASK,
        dataset=dataset,
        successful=True,
        confidence=0.9,
        model_performance=None,
        verification_passed=True,
        strategies=[STRATEGY_NAMES[method]],
        strategy_reasons=["Synthetic benchmark scenario reference label."],
        strategy_context=context,
        benchmark_reference_label=True,
    )


def _method_from_display(value: str | None) -> str | None:
    if value in DISPLAY_TO_METHOD:
        return DISPLAY_TO_METHOD[value]
    if value in STRATEGY_NAMES:
        return value
    return None


def _bootstrap_accuracy_difference(
    rows: list[dict[str, Any]],
    method_a: str,
    method_b: str,
    *,
    seed: int,
    replicates: int = 2_000,
) -> dict[str, float]:
    """Stratified paired bootstrap CI for accuracy(A)-accuracy(B)."""
    rng = np.random.default_rng(seed)
    scenario_names = sorted({row["scenario"] for row in rows})
    by_scenario = {
        name: [row for row in rows if row["scenario"] == name]
        for name in scenario_names
    }
    differences = np.empty(replicates, dtype=float)
    for index in range(replicates):
        scenario_differences = []
        for name in scenario_names:
            candidates = by_scenario[name]
            selected = rng.integers(0, len(candidates), size=len(candidates))
            sample = [candidates[int(i)] for i in selected]
            acc_a = np.mean([
                item[f"{method_a}_selected"] == item["reference_method"]
                for item in sample
            ])
            acc_b = np.mean([
                item[f"{method_b}_selected"] == item["reference_method"]
                for item in sample
            ])
            scenario_differences.append(acc_a - acc_b)
        differences[index] = float(np.mean(scenario_differences))
    return {
        "mean_difference": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(differences, 0.025)),
        "ci95_upper": float(np.quantile(differences, 0.975)),
        "bootstrap_replicates": replicates,
    }


def run_benchmark(
    *,
    train_per_family: int = 12,
    test_per_family: int = 30,
    seed: int = 20261025,
) -> dict[str, Any]:
    """Run the baselines on held-out synthetic random seeds.

    Family is represented in training and testing by different random seeds;
    this measures generalization to new samples of known scenario families,
    not transfer to entirely unseen scientific domains.
    """
    if train_per_family < 1 or test_per_family < 1:
        raise ValueError("Train and test repetitions must be positive integers.")

    started = time.perf_counter()
    graph = KnowledgeGraph()
    engine = SelfEvolutionEngine(graph)
    training_labels: list[str] = []
    execution_number = 0
    replay_example: dict[str, Any] | None = None

    # Seed per-scenario histories using training seeds disjoint from test seeds.
    for scenario_index, scenario in enumerate(SCENARIOS):
        for repetition in range(train_per_family):
            sample_seed = seed + 10_000 + scenario_index * 10_000 + repetition
            frame = generate_scenario(scenario.name, sample_seed)
            context = build_correlation_context(frame, "x", "y")
            execution_number += 1
            train_dataset = f"synthetic_train_{scenario.name}_{repetition}.csv"
            _seed_experience(
                graph,
                execution_id=f"benchmark_execution_{execution_number}",
                dataset=train_dataset,
                method=scenario.reference_method,
                context=context,
            )
            if scenario.name == "linear_clean" and repetition == 0:
                replay_example = {
                    "dataset": train_dataset,
                    "method": scenario.reference_method,
                    "context": context,
                }
            training_labels.append(scenario.reference_method)

    label_counts = {
        method: training_labels.count(method)
        for method in ("pearson", "spearman")
    }
    winning_count = max(label_counts.values())
    majority_leaders = [
        method for method, count in label_counts.items()
        if count == winning_count
    ]
    global_majority = majority_leaders[0] if len(majority_leaders) == 1 else None

    # Stress-test whether repeated writes of the same context can bias a plain
    # execution-count majority while contextual retrieval collapses duplicates.
    stress_query_frame = generate_scenario(
        "monotonic_with_outliers", seed + 800_123
    )
    stress_query_context = build_correlation_context(
        stress_query_frame, "x", "y"
    )
    stress_graph = KnowledgeGraph()
    stress_graph.graph = graph.graph.copy()
    stress_engine = SelfEvolutionEngine(stress_graph)
    contextual_before = stress_engine.recommend_strategy(
        task=TASK,
        dataset="stress_target.csv",
        context=stress_query_context,
    )
    replay_count = 20
    if replay_example is not None:
        for replay_index in range(replay_count):
            _seed_experience(
                stress_graph,
                execution_id=f"benchmark_replaycopy{replay_index}",
                dataset=replay_example["dataset"],
                method=replay_example["method"],
                context=replay_example["context"],
            )
    contextual_after = SelfEvolutionEngine(stress_graph).recommend_strategy(
        task=TASK,
        dataset="stress_target.csv",
        context=stress_query_context,
    )
    raw_counts_before = dict(label_counts)
    raw_counts_after = dict(label_counts)
    raw_counts_after["pearson"] += replay_count
    raw_majority_before = (
        majority_leaders[0] if len(majority_leaders) == 1 else None
    )
    max_after = max(raw_counts_after.values())
    raw_leaders_after = [
        method for method, count in raw_counts_after.items()
        if count == max_after
    ]
    raw_majority_after = raw_leaders_after[0] if len(raw_leaders_after) == 1 else None
    duplicate_context_stress = {
        "query_scenario": "monotonic_with_outliers",
        "replayed_context_dataset": replay_example["dataset"] if replay_example else None,
        "replay_count": replay_count,
        "raw_strategy_counts_before": raw_counts_before,
        "raw_strategy_counts_after": raw_counts_after,
        "raw_task_majority_before": raw_majority_before,
        "raw_task_majority_after": raw_majority_after,
        "contextual_recommendation_before": contextual_before.get("recommended_strategy"),
        "contextual_recommendation_after": contextual_after.get("recommended_strategy"),
        "contextual_recommendation_unchanged": (
            contextual_before.get("recommended_strategy")
            == contextual_after.get("recommended_strategy")
            and contextual_before.get("strategy_support")
            == contextual_after.get("strategy_support")
            and contextual_before.get("contextual_matches")
            == contextual_after.get("contextual_matches")
        ),
        "note": (
            "Stress test only: repeated execution records share the exact same dataset, "
            "column pair, and rounded context. Contextual retrieval groups these records; "
            "a raw task-level vote counts each replay. This is not an independent sample."
        ),
    }

    # Compute large-sample, uncontaminated reference coefficients for each
    # scenario. These are used only to report coefficient-recovery error.
    reference_coefficients = {
        scenario.name: _reference_coefficients(
            scenario, seed + 900_000 + index * 19
        )
        for index, scenario in enumerate(SCENARIOS)
    }

    rows: list[dict[str, Any]] = []
    method_names = (
        "fixed_pearson",
        "fixed_spearman",
        "skew_heuristic",
        "task_majority_fallback",
        "contextual_memory",
    )

    for scenario_index, scenario in enumerate(SCENARIOS):
        for repetition in range(test_per_family):
            test_seed = seed + 1_000_000 + scenario_index * 10_000 + repetition
            frame = generate_scenario(scenario.name, test_seed)
            context = build_correlation_context(frame, "x", "y")
            task = {
                "task_id": f"TEST-{scenario.name}-{repetition}",
                "task": TASK,
                "columns": ["x", "y"],
            }
            heuristic = select_correlation_strategy(frame, "x", "y")[0]
            recommendation = engine.recommend_strategy(
                task=TASK,
                dataset=f"synthetic_test_{scenario.name}_{repetition}.csv",
                context=context,
            )
            contextual_display = recommendation.get("recommended_strategy")
            contextual = _method_from_display(contextual_display)
            contextual_used_memory = contextual is not None

            # Task-level majority deliberately ignores the context. If the
            # history is tied, it uses the same skew-based fallback as the app.
            majority = global_majority or heuristic

            selected = {
                "fixed_pearson": "pearson",
                "fixed_spearman": "spearman",
                "skew_heuristic": heuristic,
                "task_majority_fallback": majority,
                "contextual_memory": contextual or heuristic,
            }

            true_reference = scenario.reference_method
            task_result = dict(task)
            method_outputs = {}
            for name, chosen in selected.items():
                result = execute_correlation_task(frame, task_result, method=chosen)
                coefficient = result.get("value")
                reference = reference_coefficients[scenario.name][chosen]
                method_outputs[name] = {
                    "selected": chosen,
                    "coefficient": coefficient,
                    "abs_coefficient_error": (
                        abs(float(coefficient) - reference)
                        if coefficient is not None and math.isfinite(float(coefficient))
                        else None
                    ),
                }

            row: dict[str, Any] = {
                "scenario": scenario.name,
                "scenario_description": scenario.description,
                "seed": test_seed,
                "sample_size": int(len(frame)),
                "reference_method": true_reference,
                "skew_heuristic_method": heuristic,
                "task_majority_method": majority,
                "contextual_recommendation": contextual,
                "contextual_reason": recommendation.get("reason", ""),
                "contextual_matches": recommendation.get("contextual_matches", 0),
                "contextual_similarity": recommendation.get("nearest_similarity"),
                "contextual_support": (
                    recommendation.get("strategy_support", {}).get(
                        STRATEGY_NAMES.get(contextual, ""), 0.0
                    ) if contextual else None
                ),
                "contextual_used_memory": contextual_used_memory,
            }
            for name, output in method_outputs.items():
                row[f"{name}_selected"] = output["selected"]
                row[f"{name}_coefficient"] = output["coefficient"]
                row[f"{name}_abs_coefficient_error"] = output["abs_coefficient_error"]
            rows.append(row)

    summaries: list[dict[str, Any]] = []
    for name in method_names:
        correctness = [
            row[f"{name}_selected"] == row["reference_method"]
            for row in rows
        ]
        errors = [
            row[f"{name}_abs_coefficient_error"]
            for row in rows
            if row[f"{name}_abs_coefficient_error"] is not None
        ]
        uses_memory = name in {"task_majority_fallback", "contextual_memory"}
        used_count = sum(
            row.get("contextual_used_memory", False)
            if name == "contextual_memory"
            else (
                global_majority is not None if name == "task_majority_fallback" else False
            )
            for row in rows
        )
        summaries.append({
            "method": name,
            "n_test_cases": len(rows),
            "selection_accuracy": float(np.mean(correctness)),
            "correct_cases": int(sum(correctness)),
            "mean_absolute_coefficient_error": (
                float(np.mean(errors)) if errors else None
            ),
            "coefficient_error_n": len(errors),
            "historical_recommendation_coverage": (
                float(used_count / len(rows)) if uses_memory else None
            ),
            "fallback_count": (
                int(len(rows) - used_count) if uses_memory else None
            ),
            "reference_labels": label_counts,
        })

    per_scenario: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario.name]
        record: dict[str, Any] = {
            "scenario": scenario.name,
            "reference_method": scenario.reference_method,
            "n_test_cases": len(scenario_rows),
        }
        for name in method_names:
            record[f"{name}_accuracy"] = float(np.mean([
                row[f"{name}_selected"] == row["reference_method"]
                for row in scenario_rows
            ]))
        record["contextual_memory_coverage"] = float(np.mean([
            row["contextual_used_memory"] for row in scenario_rows
        ]))
        record["contextual_memory_fallbacks"] = int(sum(
            not row["contextual_used_memory"] for row in scenario_rows
        ))
        per_scenario.append(record)

    ci_comparisons = {
        "contextual_vs_heuristic": _bootstrap_accuracy_difference(
            rows, "contextual_memory", "skew_heuristic", seed=seed + 17
        ),
        "contextual_vs_task_majority_fallback": _bootstrap_accuracy_difference(
            rows, "contextual_memory", "task_majority_fallback", seed=seed + 23
        ),
    }

    return {
        "metadata": {
            "benchmark_name": "Context-aware correlation-strategy selection",
            "seed": seed,
            "train_per_family": train_per_family,
            "test_per_family": test_per_family,
            "n_train_experiences": train_per_family * len(SCENARIOS),
            "n_test_cases": test_per_family * len(SCENARIOS),
            "scenario_families": [
                {
                    "name": item.name,
                    "reference_method": item.reference_method,
                    "description": item.description,
                    "sample_size": item.sample_size,
                }
                for item in SCENARIOS
            ],
            "training_strategy_counts": label_counts,
            "task_level_majority": global_majority,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "method_summary": summaries,
        "per_scenario_summary": per_scenario,
        "paired_bootstrap_95pct_ci": ci_comparisons,
        "duplicate_context_stress_test": duplicate_context_stress,
        "test_cases": rows,
        "interpretation_note": (
            "Reference methods are specified per scenario as a benchmark protocol, not universal truths. Training experiences use oracle-labelled synthetic scenarios. The experiment does not establish real-world generalization or statistical superiority; validate separately on held-out real datasets and quality-based outcomes."
        ),
    }


def write_results(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )

    for key, filename in (
        ("method_summary", "method_summary.csv"),
        ("per_scenario_summary", "per_scenario_summary.csv"),
        ("test_cases", "test_cases.csv"),
    ):
        rows = results[key]
        if not rows:
            continue
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metadata = results["metadata"]
    summary_lines = [
        "# Synthetic Correlation Strategy Benchmark",
        "",
        "> **Scope:** This is a synthetic proof-of-concept benchmark. It is not a claim of real-world superiority.",
        "",
        "## Protocol",
        "",
        f"- Random seed: `{metadata['seed']}`",
        f"- Oracle-labelled training experiences: {metadata['n_train_experiences']}",
        f"- Held-out synthetic test cases: {metadata['n_test_cases']}",
        f"- Scenario families: {len(metadata['scenario_families'])}",
        f"- Training label counts: `{metadata['training_strategy_counts']}`",
        "- Train/test use disjoint random seeds. Scenario families are represented in both splits; this is held-out-sample testing, not unseen-family transfer.",
        "- Each family randomizes sample size by approximately +/-15% around its base size. Noise scale varies by seed, and missingness/outlier contamination varies where applicable.",
        "- For each family, the reference method encodes the declared analytical objective. Pearson and Spearman are not treated as universally interchangeable or universally optimal.",
        "- Task-level majority voting ignores profile features. A tied majority abstains and falls back to the skew-based heuristic; this is labelled `task_majority_fallback`.",
        "- Contextual memory is the project's actual `SelfEvolutionEngine` recommender using synthetic oracle-labelled prior experiences.",
        "",
        "## Aggregate results",
        "",
        "| Method | Selection accuracy | Mean absolute coefficient error | Historical coverage | Fallback cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["method_summary"]:
        error = row["mean_absolute_coefficient_error"]
        error_text = f"{error:.4f}" if error is not None else "N/A"
        coverage = row["historical_recommendation_coverage"]
        coverage_text = f"{coverage:.1%}" if coverage is not None else "N/A"
        fallback = row["fallback_count"] if row["fallback_count"] is not None else "N/A"
        summary_lines.append(
            f"| {row['method']} | {row['selection_accuracy']:.1%} "
            f"({row['correct_cases']}/{row['n_test_cases']}) | {error_text} | "
            f"{coverage_text} | {fallback} |"
        )
    summary_lines += [
        "",
        "## Per-scenario selection accuracy",
        "",
        "| Scenario | Reference method | Pearson | Spearman | Skew heuristic | Task-majority fallback | Contextual memory | Contextual coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["per_scenario_summary"]:
        summary_lines.append(
            f"| {row['scenario']} | {row['reference_method']} | "
            f"{row['fixed_pearson_accuracy']:.1%} | "
            f"{row['fixed_spearman_accuracy']:.1%} | "
            f"{row['skew_heuristic_accuracy']:.1%} | "
            f"{row['task_majority_fallback_accuracy']:.1%} | "
            f"{row['contextual_memory_accuracy']:.1%} | "
            f"{row['contextual_memory_coverage']:.1%} |"
        )
    summary_lines += [
        "",
        "## Paired, stratified bootstrap comparisons",
        "",
        "The confidence intervals below are for the absolute difference in selection accuracy, with test cases resampled within scenario families.",
        "",
        "| Comparison | Mean accuracy difference | 95% bootstrap interval |",
        "|---|---:|---:|",
    ]
    for comparison, row in results["paired_bootstrap_95pct_ci"].items():
        summary_lines.append(
            f"| {comparison} | {row['mean_difference']:+.1%} | "
            f"[{row['ci95_lower']:+.1%}, {row['ci95_upper']:+.1%}] |"
        )
    summary_lines += [
        "",
        "## Limitations and interpretation",
        "",
        results["interpretation_note"],
        "",
        "The training memory is seeded with scenario-reference labels. This gives the contextual retrieval algorithm labelled prior experiences; it does not test whether the application's current structural-verification success flag independently identifies statistically high-quality strategies.",
        "",
        "Coefficient error is calculated against a large, uncontaminated reference population for the selected estimator. It supplements strategy-selection accuracy and should not be interpreted as a universal quality ordering between Pearson and Spearman, since they estimate different association quantities.",
        "",
        "The next experiment should repeat this protocol across random seeds, include additional scenario families and held-out licensed real datasets, and validate labels through an objective-specific evaluation protocol.",
        "",
        f"Runtime: {metadata['elapsed_seconds']} seconds. Versions: Python {metadata['python_version']}, NumPy {metadata['numpy_version']}, pandas {metadata['pandas_version']}, SciPy {metadata['scipy_version']}.",
        "",
    ]
    stress = results["duplicate_context_stress_test"]
    summary_lines += [
        "",
        "## Duplicate-history stress test",
        "",
        f"The stress-test corpus starts with strategy counts `{stress['raw_strategy_counts_before']}`. It then adds {stress['replay_count']} copies of one existing Pearson experience with the exact same dataset, column pair, and context.",
        f"Raw task-level majority changes from `{stress['raw_task_majority_before'] or 'tie / abstain'}` to `{stress['raw_task_majority_after'] or 'tie / abstain'}`.",
        f"Contextual recommendation before replay: `{stress['contextual_recommendation_before']}`; after replay: `{stress['contextual_recommendation_after']}`.",
        f"Contextual recommendation and support unchanged: `{stress['contextual_recommendation_unchanged']}`.",
        "",
        "This stress test illustrates duplicate suppression under a deliberately repeated history entry; it is not evidence about real-world performance and the replay copies are not independent observations.",
        "",
    ]
    (output_dir / "benchmark_report.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    # Publication-friendly figures, kept separate from numerical source data.
    summary = results["method_summary"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [row["method"].replace("_", "\n") for row in summary]
    values = [row["selection_accuracy"] * 100 for row in summary]
    bars = ax.bar(labels, values)
    ax.set_ylabel("Selection accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Correlation Strategy Selection — Synthetic Benchmark")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_accuracy.png", dpi=180)
    plt.close(fig)

    per_scenario = results["per_scenario_summary"]
    comparison_methods = [
        "fixed_pearson", "fixed_spearman", "skew_heuristic",
        "task_majority_fallback", "contextual_memory",
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x_positions = np.arange(len(per_scenario))
    width = 0.15
    for method_index, method in enumerate(comparison_methods):
        offsets = x_positions + (method_index - (len(comparison_methods) - 1) / 2) * width
        metric_key = f"{method}_accuracy"
        if method == "task_majority_fallback":
            metric_key = "task_majority_fallback_accuracy"
        ax.bar(
            offsets,
            [row[metric_key] * 100 for row in per_scenario],
            width=width,
            label=method.replace("_", " "),
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([row["scenario"].replace("_", "\n") for row in per_scenario])
    ax.set_ylabel("Selection accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Scenario Strategy Selection Accuracy")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_scenario_accuracy.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-family", type=int, default=12)
    parser.add_argument("--test-per-family", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/research_evaluation"),
    )
    args = parser.parse_args(argv)
    if args.train_per_family < 1 or args.test_per_family < 1:
        parser.error("--train-per-family and --test-per-family must be positive")

    results = run_benchmark(
        train_per_family=args.train_per_family,
        test_per_family=args.test_per_family,
        seed=args.seed,
    )
    write_results(results, args.output)

    print(f"Wrote benchmark artifacts to: {args.output.resolve()}")
    print(f"Held-out synthetic cases: {results['metadata']['n_test_cases']}")
    print("\nMethod summary:")
    for row in results["method_summary"]:
        print(
            f"{row['method']:<22} "
            f"accuracy={row['selection_accuracy']:.3f} "
            f"mean_abs_coef_error={row['mean_absolute_coefficient_error']:.4f}"
        )
    print("\nPaired bootstrap comparisons:")
    for name, comparison in results["paired_bootstrap_95pct_ci"].items():
        print(
            f"{name}: diff={comparison['mean_difference']:+.3f}, "
            f"95% CI=[{comparison['ci95_lower']:+.3f}, "
            f"{comparison['ci95_upper']:+.3f}]"
        )
    print("\nCaution: synthetic proof-of-concept, not real-world superiority evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
