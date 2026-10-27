"""Leave-one-scenario-family-out evaluation for contextual strategy memory.

For each fold, one entire synthetic scenario family is excluded from the
historical memory. Contextual recommendations are tested only on that held-out
family. Other folds use independent graphs so no fold can leak experience into
another fold.

Example:
    python -m research.run_leave_one_family_out \
      --seeds 20261025 20261026 20261027 20261028 20261029 \
      --train-per-family 12 --test-per-family 30 \
      --output outputs/research_evaluation/leave_one_family_out

Labels remain oracle-labelled from the predeclared synthetic protocol. Results
measure transfer across these defined synthetic scenario families, not real-
world superiority or autonomous quality discovery.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.knowledge_graph.graph import KnowledgeGraph
from research.benchmark_strategy_selection import (
    SCENARIOS,
    STRATEGY_NAMES,
    TASK,
    _method_from_display,
    _seed_experience,
    generate_scenario,
)
from src.analysis import build_correlation_context, select_correlation_strategy

METHODS = (
    "fixed_pearson",
    "fixed_spearman",
    "skew_heuristic",
    "dataset_history_fallback",
    "global_majority_heuristic_fallback",
    "contextual_memory",
)


def _majority_method(labels: list[str]) -> str | None:
    counts = {"pearson": labels.count("pearson"), "spearman": labels.count("spearman")}
    maximum = max(counts.values())
    leaders = [method for method, count in counts.items() if count == maximum]
    return leaders[0] if len(leaders) == 1 else None


def run_fold(
    *,
    held_out_family: str,
    train_per_family: int,
    test_per_family: int,
    seed: int,
) -> dict[str, Any]:
    """Train on every family except one and evaluate only on the held-out family."""
    if held_out_family not in {scenario.name for scenario in SCENARIOS}:
        raise ValueError(f"Unknown scenario family: {held_out_family}")
    if train_per_family < 1 or test_per_family < 1:
        raise ValueError("Train/test repetitions must be positive integers.")

    held_out = next(s for s in SCENARIOS if s.name == held_out_family)
    graph = KnowledgeGraph()
    engine = SelfEvolutionEngine(graph)
    training_families = [s for s in SCENARIOS if s.name != held_out_family]
    training_labels: list[str] = []
    experience_id = 0

    for family_index, scenario in enumerate(training_families):
        # The excluded scenario family is never seeded into this fold's graph.
        for repetition in range(train_per_family):
            train_seed = seed + 20_000 + family_index * 20_000 + repetition
            frame = generate_scenario(scenario.name, train_seed)
            context = build_correlation_context(frame, "x", "y")
            experience_id += 1
            _seed_experience(
                graph,
                execution_id=f"lofo_{held_out_family}_{experience_id}",
                dataset=f"lofo_train_{scenario.name}_{repetition}.csv",
                method=scenario.reference_method,
                context=context,
            )
            training_labels.append(scenario.reference_method)

    global_majority = _majority_method(training_labels)
    rows: list[dict[str, Any]] = []
    for repetition in range(test_per_family):
        test_seed = seed + 2_000_000 + list(s.name for s in SCENARIOS).index(held_out_family) * 20_000 + repetition
        frame = generate_scenario(held_out.name, test_seed)
        context = build_correlation_context(frame, "x", "y")
        heuristic = select_correlation_strategy(frame, "x", "y")[0]
        test_dataset = f"lofo_test_{held_out_family}_{repetition}.csv"

        # Faithful task-and-dataset history baseline: as the held-out dataset
        # has no matching history, the legacy recommender abstains and the
        # production policy falls back to the current-data heuristic.
        dataset_recommendation = engine.recommend_strategy(
            task=TASK,
            dataset=test_dataset,
        )
        dataset_history_method = _method_from_display(
            dataset_recommendation.get("recommended_strategy")
        )

        # Separate naive baseline: ignore dataset identity and use the global
        # majority label from the remaining families. Under this deliberately
        # balanced six-family LOFO design, this can systematically predict the
        # opposite class for the held-out family; it is reported transparently
        # as a global-voting diagnostic, not the production fallback baseline.
        global_majority_fallback = global_majority or heuristic

        recommendation = engine.recommend_strategy(
            task=TASK,
            dataset=test_dataset,
            context=context,
        )
        contextual = _method_from_display(recommendation.get("recommended_strategy"))

        selected = {
            "fixed_pearson": "pearson",
            "fixed_spearman": "spearman",
            "skew_heuristic": heuristic,
            "dataset_history_fallback": dataset_history_method or heuristic,
            "global_majority_heuristic_fallback": global_majority_fallback,
            "contextual_memory": contextual or heuristic,
        }
        row = {
            "seed": seed,
            "held_out_family": held_out_family,
            "train_families": ";".join(s.name for s in training_families),
            "n_train_experiences": len(training_labels),
            "n_test_case": repetition,
            "test_sample_seed": test_seed,
            "sample_size": len(frame),
            "reference_method": held_out.reference_method,
            "training_label_counts": json.dumps({
                "pearson": training_labels.count("pearson"),
                "spearman": training_labels.count("spearman"),
            }, sort_keys=True),
            "global_majority_method": global_majority,
            "dataset_history_recommendation": dataset_history_method,
            "dataset_history_reason": dataset_recommendation.get("reason", ""),
            "heuristic_method": heuristic,
            "contextual_recommendation": contextual,
            "contextual_reason": recommendation.get("reason", ""),
            "contextual_matches": recommendation.get("contextual_matches", 0),
            "contextual_nearest_similarity": recommendation.get("nearest_similarity"),
            "contextual_support": (
                recommendation.get("strategy_support", {}).get(
                    STRATEGY_NAMES.get(contextual, ""), 0.0
                ) if contextual else None
            ),
            "contextual_coverage": contextual is not None,
        }
        for method_name, choice in selected.items():
            row[f"{method_name}_selected"] = choice
            row[f"{method_name}_correct"] = choice == held_out.reference_method
        rows.append(row)

    summary = {
        "seed": seed,
        "held_out_family": held_out_family,
        "reference_method": held_out.reference_method,
        "training_families": [s.name for s in training_families],
        "n_train_experiences": len(training_labels),
        "n_test_cases": len(rows),
        "training_label_counts": {
            "pearson": training_labels.count("pearson"),
            "spearman": training_labels.count("spearman"),
        },
        "global_majority_method": global_majority,
        "method_accuracy": {
            method: float(np.mean([row[f"{method}_correct"] for row in rows]))
            for method in METHODS
        },
        "contextual_coverage": float(np.mean([row["contextual_coverage"] for row in rows])),
        "contextual_fallbacks": sum(not row["contextual_coverage"] for row in rows),
    }
    return {"fold_summary": summary, "test_cases": rows}


def _mean_interval(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array)) if len(array) else float("nan")
    if len(array) < 2:
        return {"mean": mean, "sd": 0.0 if len(array) == 1 else float("nan"), "ci95_lower": None, "ci95_upper": None}
    sd = float(np.std(array, ddof=1))
    margin = float(stats.t.ppf(0.975, df=len(array) - 1) * sd / math.sqrt(len(array)))
    return {"mean": mean, "sd": sd, "ci95_lower": mean - margin, "ci95_upper": mean + margin}


def run_experiment(
    *, seeds: list[int], train_per_family: int = 12, test_per_family: int = 30
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one random seed is required.")
    started = time.perf_counter()
    all_folds = []
    all_rows = []
    for seed in seeds:
        for scenario in SCENARIOS:
            result = run_fold(
                held_out_family=scenario.name,
                train_per_family=train_per_family,
                test_per_family=test_per_family,
                seed=seed,
            )
            all_folds.append(result["fold_summary"])
            all_rows.extend(result["test_cases"])

    method_summary = []
    by_seed_method = []
    for seed in seeds:
        seed_rows = [row for row in all_rows if row["seed"] == seed]
        record = {"seed": seed, "n_test_cases": len(seed_rows)}
        for method in METHODS:
            record[f"{method}_accuracy"] = float(np.mean([
                row[f"{method}_correct"] for row in seed_rows
            ]))
        record["contextual_coverage"] = float(np.mean([
            row["contextual_coverage"] for row in seed_rows
        ]))
        by_seed_method.append(record)

    for method in METHODS:
        values = [row[f"{method}_accuracy"] for row in by_seed_method]
        interval = _mean_interval(values)
        method_summary.append({
            "method": method,
            "n_seeds": len(seeds),
            "mean_accuracy": interval["mean"],
            "sd_across_seeds": interval["sd"],
            "ci95_t_lower": interval["ci95_lower"],
            "ci95_t_upper": interval["ci95_upper"],
            "min_seed_accuracy": min(values),
            "max_seed_accuracy": max(values),
        })

    per_family_summary = []
    for scenario in SCENARIOS:
        folds = [fold for fold in all_folds if fold["held_out_family"] == scenario.name]
        rows = [row for row in all_rows if row["held_out_family"] == scenario.name]
        item = {
            "held_out_family": scenario.name,
            "reference_method": scenario.reference_method,
            "n_folds": len(folds),
            "n_test_cases": len(rows),
        }
        for method in METHODS:
            item[f"{method}_accuracy"] = float(np.mean([
                row[f"{method}_correct"] for row in rows
            ]))
        item["contextual_coverage"] = float(np.mean([
            row["contextual_coverage"] for row in rows
        ]))
        item["contextual_fallbacks"] = int(sum(
            not row["contextual_coverage"] for row in rows
        ))
        per_family_summary.append(item)

    return {
        "metadata": {
            "benchmark": "Leave-one-synthetic-scenario-family-out strategy selection",
            "seeds": seeds,
            "train_per_family": train_per_family,
            "test_per_held_out_family": test_per_family,
            "n_scenario_families": len(SCENARIOS),
            "n_total_test_cases": len(all_rows),
            "n_fold_runs": len(all_folds),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": __import__("pandas").__version__,
            "scipy_version": scipy.__version__,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "protocol": (
                "Each fold excludes the complete held-out family from the memory. Test cases are generated with separate seeds. "
                "Historical memory remains seeded with oracle-labelled reference methods for the non-held-out synthetic families. "
                "The dataset-scoped historical baseline receives a previously unseen test dataset name, so it has no eligible exact-dataset history and falls back to the skew heuristic. "
                "A separate global-majority-with-heuristic-fallback diagnostic intentionally ignores dataset identity and contextual profiles."
            ),
            "limitations": (
                "This tests transfer to an unseen family within six synthetic scenario families, not generalization to real datasets. "
                "Reference labels are defined by the benchmark protocol and are not universal statistical rules. "
                "Because the six families are balanced 3-vs-3 by reference method, excluding any one family leaves a 2-vs-3 family count; the global-majority diagnostic consequently chooses the opposite reference class in this designed protocol. "
                "This is a benchmark-design effect, not evidence that all task-level memory systems would achieve zero accuracy."
            ),
        },
        "method_summary": method_summary,
        "per_family_summary": per_family_summary,
        "per_seed_summary": by_seed_method,
        "fold_summary": all_folds,
        "test_cases": all_rows,
    }


def write_results(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leave_one_family_out_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    for key, filename in (
        ("method_summary", "leave_one_family_out_method_summary.csv"),
        ("per_family_summary", "leave_one_family_out_per_family.csv"),
        ("per_seed_summary", "leave_one_family_out_per_seed.csv"),
        ("fold_summary", "leave_one_family_out_folds.csv"),
        ("test_cases", "leave_one_family_out_test_cases.csv"),
    ):
        rows = results[key]
        if not rows:
            continue
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metadata = results["metadata"]
    lines = [
        "# Leave-One-Synthetic-Family-Out Results",
        "",
        f"- Seeds: `{metadata['seeds']}`",
        f"- Held-out scenario families: {metadata['n_scenario_families']}",
        f"- Total held-out test cases: {metadata['n_total_test_cases']}",
        f"- Fold runs: {metadata['n_fold_runs']}",
        f"- Elapsed seconds: {metadata['elapsed_seconds']}",
        "",
        "## Overall selection accuracy",
        "",
        "| Method | Mean across seeds | SD | 95% t interval across seeds | Range |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["method_summary"]:
        interval = "N/A" if row["ci95_t_lower"] is None else f"[{row['ci95_t_lower']:.1%}, {row['ci95_t_upper']:.1%}]"
        lines.append(
            f"| {row['method']} | {row['mean_accuracy']:.1%} | {row['sd_across_seeds']:.1%} | "
            f"{interval} | [{row['min_seed_accuracy']:.1%}, {row['max_seed_accuracy']:.1%}] |"
        )
    lines += [
        "",
        "## Performance for each excluded family",
        "",
        "| Held-out family | Reference label | Heuristic | Dataset-history fallback | Global-majority fallback | Contextual memory | Contextual coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["per_family_summary"]:
        lines.append(
            f"| {row['held_out_family']} | {row['reference_method']} | "
            f"{row['skew_heuristic_accuracy']:.1%} | {row['dataset_history_fallback_accuracy']:.1%} | "
            f"{row['global_majority_heuristic_fallback_accuracy']:.1%} | "
            f"{row['contextual_memory_accuracy']:.1%} | {row['contextual_coverage']:.1%} |"
        )
    lines += [
        "",
        "## Protocol and limitations",
        "",
        metadata["protocol"],
        "",
        metadata["limitations"],
        "",
        "The dataset-history fallback uses exact dataset isolation and therefore falls back to the data-based heuristic for new held-out dataset IDs. The global-majority baseline ignores dataset identity; its 0% accuracy is induced by the 3-vs-3 family label design and must not be interpreted as a realistic universal baseline. "
        "The confidence intervals summarize across-seed variation descriptively. The number of seeds is small, and held-out folds reuse overlapping training families; do not interpret these intervals as definitive evidence of superiority.",
    ]
    (output_dir / "leave_one_family_out_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    methods = [
        "fixed_pearson", "fixed_spearman", "skew_heuristic",
        "dataset_history_fallback", "global_majority_heuristic_fallback",
        "contextual_memory",
    ]
    x = np.arange(len(results["per_family_summary"]))
    width = 0.12
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, method in enumerate(methods):
        values = [row[f"{method}_accuracy"] * 100 for row in results["per_family_summary"]]
        ax.bar(x + (index - (len(methods) - 1) / 2) * width, values, width, label=method.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([row["held_out_family"].replace("_", "\n") for row in results["per_family_summary"]])
    ax.set_ylabel("Selection accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Leave-One-Synthetic-Family-Out Strategy Selection")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "leave_one_family_out_accuracy.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261025, 20261026, 20261027, 20261028, 20261029])
    parser.add_argument("--train-per-family", type=int, default=12)
    parser.add_argument("--test-per-family", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/research_evaluation/leave_one_family_out"))
    args = parser.parse_args(argv)
    if args.train_per_family < 1 or args.test_per_family < 1:
        parser.error("--train-per-family and --test-per-family must be positive")

    results = run_experiment(
        seeds=args.seeds,
        train_per_family=args.train_per_family,
        test_per_family=args.test_per_family,
    )
    write_results(results, args.output)
    print(f"Wrote leave-one-family-out artifacts to: {args.output.resolve()}")
    print(f"Held-out test cases: {results['metadata']['n_total_test_cases']}")
    print("\nMean accuracy across seeds:")
    for row in results["method_summary"]:
        print(
            f"{row['method']:<24} mean_accuracy={row['mean_accuracy']:.3f} "
            f"sd={row['sd_across_seeds']:.3f} "
            f"range=[{row['min_seed_accuracy']:.3f}, {row['max_seed_accuracy']:.3f}]"
        )
    print("\nInterpretation notes:")
    print("- dataset_history_fallback matches the task-and-dataset recommender's exact dataset isolation; unseen test datasets use the skew heuristic.")
    print("- global_majority_heuristic_fallback is a separate naive cross-dataset baseline; its result is affected by the balanced family-label design.")
    print("- Held-out synthetic-family evaluation is not real-world validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
