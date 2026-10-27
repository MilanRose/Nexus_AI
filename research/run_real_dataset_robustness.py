"""Cross-dataset robustness evaluation using real tabular datasets.

This experiment does NOT claim to know the scientifically correct correlation
method for every real pair. It evaluates a narrower, explicit objective:
which selected coefficient is more stable relative to its own clean-data
estimate under prespecified MCAR-missingness and extreme-outlier perturbations.

For each held-out dataset, historical contexts are built only from other
real datasets. A method-level target is computed independently on each held-out
pair by comparing Pearson and Spearman's mean absolute coefficient shift under
the perturbation family. If neither method has a meaningfully lower shift, the
case is marked ambiguous and excluded from selection-accuracy scoring, but is
retained for stability-shift summaries.

Included data:
- Diabetes, Wine Recognition, Breast Cancer Wisconsin Diagnostic, Iris:
  bundled scikit-learn datasets (see source citations in generated report).
- Titanic sample: data/titanic.csv distributed with this project; its upstream
  provenance must be verified separately before publication.

Run from the repository root:
    python -m research.run_real_dataset_robustness \
      --seeds 20261025 20261026 20261027 20261028 20261029 \
      --max-pairs-per-dataset 40 --perturbation-replicates 4 \
      --output outputs/research_evaluation/real_dataset_robustness

This is a controlled real-data perturbation benchmark, not a benchmark of
scientific truth or unconditional superiority of Pearson or Spearman.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)

from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.knowledge_graph.graph import KnowledgeGraph
from src.analysis import build_correlation_context, select_correlation_strategy

TASK = "investigate_correlation"
DISPLAY = {"pearson": "Pearson correlation", "spearman": "Spearman correlation"}
FROM_DISPLAY = {value: key for key, value in DISPLAY.items()}
METHODS = (
    "fixed_pearson",
    "fixed_spearman",
    "skew_heuristic",
    "task_majority_fallback",
    "contextual_memory",
)
PERTURBATIONS = ("mcar_10pct", "extreme_outliers_5pct")
DATASET_SOURCES = {
    "diabetes": (
        "scikit-learn load_diabetes dataset documentation",
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "442 rows; 10 features; bundled by scikit-learn; documentation notes that some feature meanings are not fully explicit.",
    ),
    "wine_recognition": (
        "UCI Wine Recognition dataset; Aeberhard and Forina (1992)",
        "https://doi.org/10.24432/C5PC7J",
        "178 rows; 13 numeric features; class label excluded from correlation pairs; CC BY 4.0 according to UCI attribution records.",
    ),
    "breast_cancer_diagnostic": (
        "UCI Breast Cancer Wisconsin (Diagnostic); Wolberg, Mangasarian, Street, and Street (1993)",
        "https://doi.org/10.24432/C5DW2B",
        "569 rows; 30 numeric features; diagnosis label excluded; UCI states CC BY 4.0.",
    ),
    "iris": (
        "UCI Iris dataset; Fisher (1936)",
        "https://doi.org/10.24432/C56C76",
        "150 rows; 4 numeric measurements; species label excluded; UCI states CC BY 4.0.",
    ),
    "titanic_sample": (
        "Titanic sample CSV bundled with this project; upstream source not documented in the repository",
        "",
        "Numeric feature pairs exclude PassengerId and Survived. Verify provenance and licensing before using this dataset in a paper.",
    ),
}


def load_real_datasets(
    project_root: Path, *, include_titanic_sample: bool = False
) -> dict[str, pd.DataFrame]:
    """Return clean numeric-feature frames without class/target labels."""
    datasets: dict[str, pd.DataFrame] = {
        "diabetes": load_diabetes(as_frame=True).data.copy(),
        "wine_recognition": load_wine(as_frame=True).data.copy(),
        "breast_cancer_diagnostic": load_breast_cancer(as_frame=True).data.copy(),
        "iris": load_iris(as_frame=True).data.copy(),
    }

    titanic_path = project_root / "data" / "titanic.csv"
    if include_titanic_sample and titanic_path.exists():
        raw = pd.read_csv(titanic_path)
        # ID and the supervised outcome are excluded to avoid spurious
        # correlations driven by row order or target leakage.
        requested = [
            column for column in ("Pclass", "Age", "SibSp", "Parch", "Fare")
            if column in raw.columns
        ]
        numeric = raw[requested].apply(pd.to_numeric, errors="coerce")
        datasets["titanic_sample"] = numeric

    return {
        name: frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        for name, frame in datasets.items()
    }


def select_numeric_pairs(
    frame: pd.DataFrame,
    *,
    max_pairs: int,
    seed: int,
    min_complete: int = 30,
) -> list[tuple[str, str]]:
    """Select a reproducible, nonconstant subset of numeric feature pairs."""
    candidates: list[tuple[str, str]] = []
    for left, right in combinations(frame.columns.tolist(), 2):
        pair = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < min_complete:
            continue
        if pair[left].nunique() < 2 or pair[right].nunique() < 2:
            continue
        candidates.append((str(left), str(right)))
    candidates.sort()
    if len(candidates) <= max_pairs:
        return candidates
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(candidates), size=max_pairs, replace=False).tolist())
    return [candidates[index] for index in chosen]


def correlate(frame: pd.DataFrame, left: str, right: str, method: str) -> float:
    """Compute a finite Pearson or Spearman coefficient, otherwise NaN."""
    pair = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "pearson":
                result = stats.pearsonr(pair[left].to_numpy(), pair[right].to_numpy()).statistic
            elif method == "spearman":
                result = stats.spearmanr(pair[left].to_numpy(), pair[right].to_numpy()).statistic
            else:
                raise ValueError(f"Unknown correlation method: {method}")
        value = float(result)
        return value if math.isfinite(value) else float("nan")
    except (ValueError, TypeError, FloatingPointError):
        return float("nan")


def perturb_pair(
    frame: pd.DataFrame,
    left: str,
    right: str,
    perturbation: str,
    seed: int,
) -> pd.DataFrame:
    """Create a reproducible perturbed copy of a real-data feature pair."""
    rng = np.random.default_rng(seed)
    result = frame[[left, right]].apply(pd.to_numeric, errors="coerce").astype(float).copy()
    if perturbation == "mcar_10pct":
        for column in (left, right):
            valid = np.flatnonzero(result[column].notna().to_numpy())
            count = int(round(0.10 * len(valid)))
            if count:
                selected = rng.choice(valid, size=count, replace=False)
                result.loc[result.index[selected], column] = np.nan
        return result

    if perturbation == "extreme_outliers_5pct":
        valid_positions = np.flatnonzero(
            result[[left, right]].notna().all(axis=1).to_numpy()
        )
        if len(valid_positions) < 5:
            return result
        count = max(1, int(round(0.05 * len(valid_positions))))
        positions = rng.choice(valid_positions, size=count, replace=False)
        values = pd.to_numeric(result[right], errors="coerce")
        scale = float(values.std(ddof=0))
        if not math.isfinite(scale) or scale <= 1e-12:
            return result
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=count, replace=True)
        # Contamination is applied to one variable at randomly selected rows.
        result.loc[result.index[positions], right] = (
            values.iloc[positions].to_numpy(dtype=float) + signs * 6.0 * scale
        )
        return result

    raise ValueError(f"Unsupported perturbation: {perturbation}")


def _best_stability_method(
    shifts: dict[str, list[float]], *, label_margin: float
) -> tuple[str | None, dict[str, float | None]]:
    means: dict[str, float | None] = {}
    for method in ("pearson", "spearman"):
        valid = [value for value in shifts[method] if math.isfinite(value)]
        means[method] = float(np.mean(valid)) if valid else None
    if means["pearson"] is None or means["spearman"] is None:
        return None, means
    difference = abs(means["pearson"] - means["spearman"])
    if difference < label_margin:
        return None, means
    return ("pearson" if means["pearson"] < means["spearman"] else "spearman"), means


def _build_pair_trial(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    seed: int,
    perturbation_replicates: int,
    label_margin: float,
) -> dict[str, Any]:
    pair = frame[[left, right]].copy()
    clean = {method: correlate(pair, left, right, method) for method in ("pearson", "spearman")}
    if not all(math.isfinite(value) for value in clean.values()):
        return {"usable": False, "reason": "undefined_clean_correlation"}

    by_perturbation = {}
    for p_index, perturbation in enumerate(PERTURBATIONS):
        samples = []
        shifts = {"pearson": [], "spearman": []}
        for replicate in range(perturbation_replicates):
            perturb_seed = int(seed + 1_000_003 * p_index + 9_176 * replicate)
            altered = perturb_pair(pair, left, right, perturbation, perturb_seed)
            values = {
                method: correlate(altered, left, right, method)
                for method in ("pearson", "spearman")
            }
            for method in ("pearson", "spearman"):
                if math.isfinite(values[method]):
                    shifts[method].append(abs(values[method] - clean[method]))
            samples.append({"frame": altered, "values": values, "replicate": replicate})

        target, mean_shifts = _best_stability_method(shifts, label_margin=label_margin)
        by_perturbation[perturbation] = {
            "target": target,
            "mean_abs_shift": mean_shifts,
            "samples": samples,
        }

    return {"usable": True, "clean": clean, "by_perturbation": by_perturbation}


def _seed_experience(
    graph: KnowledgeGraph,
    *,
    experience_id: str,
    dataset: str,
    method: str,
    context: dict[str, Any],
    perturbation: str,
) -> None:
    graph.add_node(
        experience_id,
        "execution",
        agent="EDAAgent",
        task=TASK,
        dataset=dataset,
        successful=True,
        confidence=0.9,
        model_performance=None,
        verification_passed=True,
        strategies=[DISPLAY[method]],
        strategy_reasons=[
            "Reference label: lower mean absolute coefficient shift under "
            f"the declared {perturbation} stress test on this training pair."
        ],
        strategy_context=context,
        benchmark_reference_label=True,
        perturbation_type=perturbation,
    )


def _method_from_display(value: Any) -> str | None:
    if value in FROM_DISPLAY:
        return FROM_DISPLAY[value]
    if value in DISPLAY:
        return value
    return None


def _majority_method(labels: list[str]) -> str | None:
    counts = {method: labels.count(method) for method in ("pearson", "spearman")}
    high = max(counts.values(), default=0)
    winners = [method for method, count in counts.items() if count == high and count > 0]
    return winners[0] if len(winners) == 1 else None


def _seed_training_memory(
    graph: KnowledgeGraph,
    datasets: dict[str, pd.DataFrame],
    pairs_by_dataset: dict[str, list[tuple[str, str]]],
    *,
    heldout_dataset: str,
    seed: int,
    perturbation_replicates: int,
    label_margin: float,
) -> tuple[list[str], int, dict[str, int]]:
    labels: list[str] = []
    experience_count = 0
    label_counts = {"pearson": 0, "spearman": 0, "ambiguous": 0}

    for dataset_index, (dataset_name, frame) in enumerate(datasets.items()):
        if dataset_name == heldout_dataset:
            continue
        for pair_index, (left, right) in enumerate(pairs_by_dataset[dataset_name]):
            pair_seed = seed + 30_000 + dataset_index * 100_000 + pair_index * 101
            trial = _build_pair_trial(
                frame, left, right, seed=pair_seed,
                perturbation_replicates=perturbation_replicates,
                label_margin=label_margin,
            )
            if not trial.get("usable"):
                continue
            for perturbation, result in trial["by_perturbation"].items():
                target = result["target"]
                if target is None:
                    label_counts["ambiguous"] += 1
                    continue
                labels.append(target)
                label_counts[target] += 1
                # Store one representative context per source pair and
                # perturbation family. The target label is still computed
                # across all replicates, so reruns do not manufacture votes.
                representative = result["samples"][0]
                context = build_correlation_context(
                    representative["frame"], left, right
                )
                experience_count += 1
                _seed_experience(
                    graph,
                    experience_id=f"realexp_{seed}_{experience_count}",
                    dataset=dataset_name,
                    method=target,
                    context=context,
                    perturbation=perturbation,
                )
    return labels, experience_count, label_counts


def run_seed(
    datasets: dict[str, pd.DataFrame],
    pairs_by_dataset: dict[str, list[tuple[str, str]]],
    *,
    seed: int,
    perturbation_replicates: int,
    label_margin: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_cases: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for heldout_index, heldout_dataset in enumerate(datasets):
        graph = KnowledgeGraph()
        engine = SelfEvolutionEngine(graph)
        training_labels, experience_count, training_label_counts = _seed_training_memory(
            graph, datasets, pairs_by_dataset,
            heldout_dataset=heldout_dataset,
            seed=seed,
            perturbation_replicates=perturbation_replicates,
            label_margin=label_margin,
        )
        global_majority = _majority_method(training_labels)
        dataset_rows: list[dict[str, Any]] = []
        frame = datasets[heldout_dataset]

        for pair_index, (left, right) in enumerate(pairs_by_dataset[heldout_dataset]):
            pair_seed = seed + 2_000_000 + heldout_index * 100_000 + pair_index * 107
            trial = _build_pair_trial(
                frame, left, right, seed=pair_seed,
                perturbation_replicates=perturbation_replicates,
                label_margin=label_margin,
            )
            if not trial.get("usable"):
                continue

            for p_index, perturbation in enumerate(PERTURBATIONS):
                outcome = trial["by_perturbation"][perturbation]
                target_method = outcome["target"]
                target_shifts = outcome["mean_abs_shift"]
                for sample in outcome["samples"]:
                    altered = sample["frame"]
                    values = sample["values"]
                    heuristic_method, heuristic_reason = select_correlation_strategy(
                        altered, left, right
                    )
                    context = build_correlation_context(altered, left, right)
                    recommendation = engine.recommend_strategy(
                        task=TASK,
                        dataset=heldout_dataset,
                        context=context,
                        top_k=5,
                        min_similarity=0.60,
                    )
                    contextual_raw = _method_from_display(
                        recommendation.get("recommended_strategy")
                    )
                    predictions = {
                        "fixed_pearson": "pearson",
                        "fixed_spearman": "spearman",
                        "skew_heuristic": heuristic_method,
                        "task_majority_fallback": global_majority or heuristic_method,
                        "contextual_memory": contextual_raw or heuristic_method,
                    }
                    row: dict[str, Any] = {
                        "seed": seed,
                        "heldout_dataset": heldout_dataset,
                        "pair_index": pair_index,
                        "column_1": left,
                        "column_2": right,
                        "perturbation": perturbation,
                        "replicate": sample["replicate"],
                        "reference_stable_method": target_method,
                        "label_decisive": target_method is not None,
                        "pearson_clean_coefficient": trial["clean"]["pearson"],
                        "spearman_clean_coefficient": trial["clean"]["spearman"],
                        "pearson_reference_mean_shift": target_shifts["pearson"],
                        "spearman_reference_mean_shift": target_shifts["spearman"],
                        "heuristic_reason": heuristic_reason,
                        "contextual_raw_recommendation": contextual_raw,
                        "contextual_reason": recommendation.get("reason", ""),
                        "contextual_matches": recommendation.get("contextual_matches", 0),
                        "contextual_nearest_similarity": recommendation.get("nearest_similarity"),
                        "contextual_support": (
                            recommendation.get("strategy_support", {}).get(
                                DISPLAY.get(contextual_raw, ""), 0.0
                            ) if contextual_raw else None
                        ),
                        "contextual_coverage": contextual_raw is not None,
                        "training_experiences": experience_count,
                        "training_label_counts": json.dumps(training_label_counts, sort_keys=True),
                    }
                    for method_name, prediction in predictions.items():
                        row[f"{method_name}_selected"] = prediction
                        row[f"{method_name}_correct"] = (
                            prediction == target_method if target_method is not None else None
                        )
                        chosen_value = values.get(prediction)
                        clean_value = trial["clean"].get(prediction)
                        row[f"{method_name}_abs_shift"] = (
                            abs(chosen_value - clean_value)
                            if chosen_value is not None and clean_value is not None
                            and math.isfinite(chosen_value) and math.isfinite(clean_value)
                            else None
                        )
                    dataset_rows.append(row)
                    all_cases.append(row)

        per_method = {}
        for method in METHODS:
            decisive = [row for row in dataset_rows if row["label_decisive"]]
            per_method[method] = {
                "robustness_target_accuracy": (
                    float(np.mean([row[f"{method}_correct"] for row in decisive]))
                    if decisive else None
                ),
                "mean_abs_coefficient_shift": float(np.nanmean([
                    row[f"{method}_abs_shift"] if row[f"{method}_abs_shift"] is not None else np.nan
                    for row in dataset_rows
                ])),
            }
        fold_summaries.append({
            "seed": seed,
            "heldout_dataset": heldout_dataset,
            "n_pairs": len(pairs_by_dataset[heldout_dataset]),
            "n_query_cases": len(dataset_rows),
            "n_decisive_cases": sum(row["label_decisive"] for row in dataset_rows),
            "decisive_label_coverage": (
                float(np.mean([row["label_decisive"] for row in dataset_rows]))
                if dataset_rows else 0.0
            ),
            "contextual_recommendation_coverage": (
                float(np.mean([row["contextual_coverage"] for row in dataset_rows]))
                if dataset_rows else 0.0
            ),
            "n_training_experiences": experience_count,
            "training_label_counts": training_label_counts,
            "method_metrics": per_method,
        })

    return all_cases, fold_summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    project_root: Path,
    seeds: list[int],
    max_pairs_per_dataset: int = 40,
    perturbation_replicates: int = 4,
    label_margin: float = 0.005,
    include_titanic_sample: bool = False,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if max_pairs_per_dataset < 1:
        raise ValueError("max_pairs_per_dataset must be at least 1.")
    if perturbation_replicates < 1:
        raise ValueError("perturbation_replicates must be at least 1.")
    if not 0 <= label_margin < 1:
        raise ValueError("label_margin must be in [0, 1).")

    started = time.perf_counter()
    datasets = load_real_datasets(project_root, include_titanic_sample=include_titanic_sample)
    pairs_by_dataset = {
        name: select_numeric_pairs(
            frame,
            max_pairs=max_pairs_per_dataset,
            seed=20261025 + index,
        )
        for index, (name, frame) in enumerate(datasets.items())
    }
    all_cases: list[dict[str, Any]] = []
    all_folds: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"Running real-data perturbation evaluation, seed {seed} ...", flush=True)
        cases, folds = run_seed(
            datasets,
            pairs_by_dataset,
            seed=seed,
            perturbation_replicates=perturbation_replicates,
            label_margin=label_margin,
        )
        all_cases.extend(cases)
        all_folds.extend(folds)

    per_dataset_seed_rows: list[dict[str, Any]] = []
    for fold in all_folds:
        row = {
            "seed": fold["seed"],
            "heldout_dataset": fold["heldout_dataset"],
            "n_pairs": fold["n_pairs"],
            "n_query_cases": fold["n_query_cases"],
            "n_decisive_cases": fold["n_decisive_cases"],
            "decisive_label_coverage": fold["decisive_label_coverage"],
            "contextual_recommendation_coverage": fold["contextual_recommendation_coverage"],
            "n_training_experiences": fold["n_training_experiences"],
        }
        for method in METHODS:
            row[f"{method}_accuracy"] = fold["method_metrics"][method]["robustness_target_accuracy"]
            row[f"{method}_mean_abs_shift"] = fold["method_metrics"][method]["mean_abs_coefficient_shift"]
        per_dataset_seed_rows.append(row)

    per_seed_rows = []
    for seed in seeds:
        current = [row for row in per_dataset_seed_rows if row["seed"] == seed]
        row = {"seed": seed}
        for method in METHODS:
            accuracies = [r[f"{method}_accuracy"] for r in current if r[f"{method}_accuracy"] is not None]
            shifts = [r[f"{method}_mean_abs_shift"] for r in current if r[f"{method}_mean_abs_shift"] is not None and math.isfinite(r[f"{method}_mean_abs_shift"])]
            row[f"{method}_macro_accuracy"] = float(np.mean(accuracies)) if accuracies else None
            row[f"{method}_macro_mean_abs_shift"] = float(np.mean(shifts)) if shifts else None
        row["mean_decisive_label_coverage"] = float(np.mean([r["decisive_label_coverage"] for r in current]))
        row["mean_contextual_recommendation_coverage"] = float(np.mean([r["contextual_recommendation_coverage"] for r in current]))
        per_seed_rows.append(row)

    per_perturbation_rows: list[dict[str, Any]] = []
    for perturbation in PERTURBATIONS:
        for method in METHODS:
            seed_accuracies: list[float] = []
            seed_shifts: list[float] = []
            for seed in seeds:
                heldout_accuracies: list[float] = []
                heldout_shifts: list[float] = []
                for dataset_name in datasets:
                    cases = [
                        row for row in all_cases
                        if row["seed"] == seed
                        and row["heldout_dataset"] == dataset_name
                        and row["perturbation"] == perturbation
                    ]
                    decisive = [row for row in cases if row["label_decisive"]]
                    if decisive:
                        heldout_accuracies.append(float(np.mean([
                            row[f"{method}_correct"] for row in decisive
                        ])))
                    shifts = [
                        row[f"{method}_abs_shift"] for row in cases
                        if row[f"{method}_abs_shift"] is not None
                        and math.isfinite(float(row[f"{method}_abs_shift"]))
                    ]
                    if shifts:
                        heldout_shifts.append(float(np.mean(shifts)))
                if heldout_accuracies:
                    seed_accuracies.append(float(np.mean(heldout_accuracies)))
                if heldout_shifts:
                    seed_shifts.append(float(np.mean(heldout_shifts)))
            per_perturbation_rows.append({
                "perturbation": perturbation,
                "method": method,
                "mean_dataset_macro_accuracy": float(np.mean(seed_accuracies)) if seed_accuracies else None,
                "sd_across_random_seeds": float(np.std(seed_accuracies, ddof=1)) if len(seed_accuracies) > 1 else 0.0,
                "mean_dataset_macro_abs_coefficient_shift": float(np.mean(seed_shifts)) if seed_shifts else None,
            })

    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        accuracies = [row[f"{method}_macro_accuracy"] for row in per_seed_rows if row[f"{method}_macro_accuracy"] is not None]
        shifts = [row[f"{method}_macro_mean_abs_shift"] for row in per_seed_rows if row[f"{method}_macro_mean_abs_shift"] is not None]
        summary_rows.append({
            "method": method,
            "mean_dataset_macro_robustness_target_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "sd_across_random_seeds": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
            "min_across_random_seeds": float(np.min(accuracies)) if accuracies else None,
            "max_across_random_seeds": float(np.max(accuracies)) if accuracies else None,
            "mean_dataset_macro_abs_coefficient_shift": float(np.mean(shifts)) if shifts else None,
            "mean_contextual_recommendation_coverage": float(np.mean([row["mean_contextual_recommendation_coverage"] for row in per_seed_rows])),
            "mean_decisive_label_coverage": float(np.mean([row["mean_decisive_label_coverage"] for row in per_seed_rows])),
        })

    output = {
        "experiment": "cross_dataset_real_data_perturbation_robustness",
        "seeds": seeds,
        "max_pairs_per_dataset": max_pairs_per_dataset,
        "perturbation_replicates": perturbation_replicates,
        "perturbations": list(PERTURBATIONS),
        "label_margin": label_margin,
        "n_heldout_datasets": len(datasets),
        "n_total_numeric_pairs": int(sum(len(pairs) for pairs in pairs_by_dataset.values())),
        "n_test_cases": len(all_cases),
        "datasets": {
            name: {
                "rows": int(len(frame)),
                "numeric_features": frame.columns.tolist(),
                "pairs_evaluated": pairs_by_dataset[name],
                "source": DATASET_SOURCES[name],
            }
            for name, frame in datasets.items()
        },
        "method_summary": summary_rows,
        "per_perturbation_summary": per_perturbation_rows,
        "per_dataset_seed_summary": per_dataset_seed_rows,
        "per_seed_summary": per_seed_rows,
        "folds": all_folds,
        "cases": all_cases,
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "limitations": [
            "The label is the method with lower mean absolute coefficient shift relative to its own clean-data estimate under a declared perturbation family; it is not scientific ground truth.",
            "Perturbations are controlled stress tests applied to real observations, not naturally occurring distribution shifts.",
            "Datasets differ in domain and size; a small number of datasets limits domain-level inference.",
            "Contextual memory is trained only on pairs from datasets other than the held-out dataset in each fold.",
            "The Titanic sample's upstream provenance is not documented in this repository and should be verified before publication.",
        ],
    }
    return output


def write_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_dataset_robustness_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(output_dir / "real_dataset_robustness_cases.csv", result["cases"])
    _write_csv(output_dir / "real_dataset_robustness_summary.csv", result["method_summary"])
    _write_csv(output_dir / "real_dataset_robustness_per_dataset.csv", result["per_dataset_seed_summary"])
    _write_csv(output_dir / "real_dataset_robustness_per_perturbation.csv", result["per_perturbation_summary"])

    report_lines = [
        "# Cross-Dataset Real-Data Perturbation Robustness Evaluation",
        "",
        "## Scope and metric",
        "",
        "This benchmark uses real tabular observations and applies controlled MCAR-missingness and extreme-outlier perturbations. Each feature pair has two clean-data reference coefficients (one Pearson, one Spearman). For each perturbation family, the reference strategy is whichever method has the lower mean absolute change from its own clean-data coefficient across the perturbation replicates. Cases where the difference is below the configured label margin are marked ambiguous and excluded from selection-accuracy scoring, but remain in coefficient-shift summaries.",
        "",
        "This is a measure of perturbation stability under a declared objective, not a measure of scientific truth, causal correctness, or universal superiority of one correlation measure.",
        "",
        "## Methods compared",
        "",
        "- `fixed_pearson`: always Pearson.",
        "- `fixed_spearman`: always Spearman.",
        "- `skew_heuristic`: the project's current skew-based heuristic.",
        "- `task_majority_fallback`: most common robust strategy label among training-dataset experiences; falls back to the skew heuristic when the training vote is tied or absent.",
        "- `contextual_memory`: graph-based similarity retrieval from other datasets; uses the production heuristic fallback when no context recommendation is made.",
        "",
        "## Cross-dataset protocol",
        "",
        f"- Dataset folds: {result['n_heldout_datasets']} (one complete dataset held out per fold).",
        f"- Distinct numeric feature pairs evaluated per run: {result['n_total_numeric_pairs']}",
        f"- Seeds: {', '.join(map(str, result['seeds']))}",
        f"- Perturbation families: {', '.join(result['perturbations'])}",
        f"- Perturbation replicates per feature-pair/family: {result['perturbation_replicates']}",
        f"- Label margin: {result['label_margin']:.4f} absolute coefficient shift.",
        "- For every fold, memory is populated only with feature pairs from datasets other than the held-out dataset. The held-out dataset's pairs are never inserted as training experiences.",
        "",
        "## Dataset sources and attribution",
        "",
    ]
    for name, metadata in result["datasets"].items():
        source, url, details = metadata["source"]
        link = f" [{url}]({url})" if url else ""
        report_lines.append(
            f"- **{name}** ({metadata['rows']} rows; {len(metadata['numeric_features'])} numeric features; {metadata['pairs_evaluated'].__len__()} pairs): {source}.{link} {details}"
        )
    report_lines.extend(["", "## Aggregate results", "", "| Method | Mean macro accuracy | SD across seeds | Seed range | Mean macro absolute coefficient shift |", "|---|---:|---:|---:|---:|"])
    for row in result["method_summary"]:
        acc = row["mean_dataset_macro_robustness_target_accuracy"]
        sd = row["sd_across_random_seeds"]
        low = row["min_across_random_seeds"]
        high = row["max_across_random_seeds"]
        shift = row["mean_dataset_macro_abs_coefficient_shift"]
        def fmt(value):
            return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.4f}"
        report_lines.append(
            f"| {row['method']} | {fmt(acc)} | {fmt(sd)} | {fmt(low)}–{fmt(high)} | {fmt(shift)} |"
        )
    coverage = float(np.mean([r["mean_contextual_recommendation_coverage"] for r in result["per_seed_summary"]]))
    report_lines.extend(["", "### Contextual-memory coverage", "", f"Mean decisive-label coverage: {np.mean([r['mean_decisive_label_coverage'] for r in result['per_seed_summary']]):.1%} of test cases.", f"Mean contextual recommendation coverage: {coverage:.1%} of queries."])
    if coverage >= 0.999:
        report_lines.append("The contextual system did not abstain on any query in this run, despite supporting fallback. Therefore this experiment does not validate out-of-distribution abstention or calibration.")
    report_lines.extend(["", "## Results by perturbation family", "", "| Perturbation | Method | Mean macro accuracy | SD across seeds | Mean macro absolute shift |", "|---|---|---:|---:|---:|"])
    for row in result["per_perturbation_summary"]:
        def fmt_pert(value):
            return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.4f}"
        report_lines.append(
            f"| {row['perturbation']} | {row['method']} | {fmt_pert(row['mean_dataset_macro_accuracy'])} | {fmt_pert(row['sd_across_random_seeds'])} | {fmt_pert(row['mean_dataset_macro_abs_coefficient_shift'])} |"
        )
    report_lines.extend(["", "## Per-dataset results", "", "| Held-out dataset | Method | Mean accuracy | Mean absolute shift | Decisive cases / queries | Contextual recommendation coverage |", "|---|---|---:|---:|---:|---:|"])
    for fold in result["folds"]:
        for method in METHODS:
            metrics = fold["method_metrics"][method]
            def fmt2(value):
                return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.4f}"
            report_lines.append(
                f"| {fold['heldout_dataset']} | {method} | {fmt2(metrics['robustness_target_accuracy'])} | {fmt2(metrics['mean_abs_coefficient_shift'])} | {fold['n_decisive_cases']}/{fold['n_query_cases']} | {fold['contextual_recommendation_coverage']:.1%} |"
            )
    report_lines.extend(["", "## Limitations and interpretation", ""])
    report_lines.extend(f"- {item}" for item in result["limitations"])
    if "titanic_sample" in result["datasets"]:
        report_lines.append("- The Titanic sample's upstream provenance is not documented in this repository. Do not include it in publication-grade results until the exact source, version, and license are verified.")
    report_lines.extend([
        "- The target method is computed from each held-out pair's perturbation stability and is used only for evaluation; it is not supplied to the held-out memory.",
        "- Correlation coefficients are not interchangeable: Pearson measures linear association, while Spearman measures rank-based monotonic association. Stability relative to each method's own clean estimate is a deliberately narrow robustness objective.",
        "- The held-out unit is an entire dataset, which reduces direct row/pair leakage. The number of datasets is still small, so results are descriptive rather than a definitive population-level inference.",
        "- The default run excludes the project's Titanic sample because its upstream provenance is undocumented; use `--include-titanic-sample` only for exploratory analysis until provenance is verified.",
        "- The benchmark should be complemented by domain-specific analysis goals and additional independent real datasets before a conference-paper claim is made.",
        "",
        "## Reproducibility metadata",
        "",
        f"- Python: {result['python_version']}",
        f"- pandas: {result['pandas_version']}",
        f"- NumPy: {result['numpy_version']}",
        f"- SciPy: {result['scipy_version']}",
        f"- scikit-learn: {result['scikit_learn_version']}",
        f"- Runtime: {result['runtime_seconds']:.2f} seconds",
        "",
    ])
    (output_dir / "real_dataset_robustness_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Plot uses macro-averaged metrics and keeps accuracy distinct from stability.
    names = [row["method"] for row in result["method_summary"]]
    accuracies = [row["mean_dataset_macro_robustness_target_accuracy"] for row in result["method_summary"]]
    shifts = [row["mean_dataset_macro_abs_coefficient_shift"] for row in result["method_summary"]]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    valid_acc = [(n, v) for n, v in zip(names, accuracies) if v is not None and math.isfinite(float(v))]
    if valid_acc:
        axes[0].bar([x[0] for x in valid_acc], [x[1] for x in valid_acc])
    axes[0].set_title("Agreement with empirical stability target")
    axes[0].set_ylabel("Macro accuracy on decisive cases")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", labelrotation=35)
    valid_shift = [(n, v) for n, v in zip(names, shifts) if v is not None and math.isfinite(float(v))]
    if valid_shift:
        axes[1].bar([x[0] for x in valid_shift], [x[1] for x in valid_shift])
    axes[1].set_title("Mean coefficient shift under perturbation")
    axes[1].set_ylabel("Absolute difference from same-method clean estimate")
    axes[1].tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "real_dataset_robustness.png", dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261025, 20261026, 20261027])
    parser.add_argument("--max-pairs-per-dataset", type=int, default=20)
    parser.add_argument("--perturbation-replicates", type=int, default=3)
    parser.add_argument("--label-margin", type=float, default=0.005)
    parser.add_argument("--include-titanic-sample", action="store_true", help="Include the project Titanic sample for exploratory analysis; its upstream provenance is not documented and it should be excluded from publication-grade results until verified.")
    parser.add_argument("--output", type=Path, default=Path("outputs/research_evaluation/real_dataset_robustness"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    output = run_experiment(
        project_root=args.project_root.resolve(),
        seeds=args.seeds,
        max_pairs_per_dataset=args.max_pairs_per_dataset,
        perturbation_replicates=args.perturbation_replicates,
        label_margin=args.label_margin,
        include_titanic_sample=args.include_titanic_sample,
    )
    write_artifacts(output, args.output)
    print(f"Wrote real-data robustness artifacts to: {args.output.resolve()}")
    print(f"Held-out datasets: {output['n_heldout_datasets']}")
    print(f"Unique numeric feature pairs per run: {output['n_total_numeric_pairs']}")
    print(f"Query cases across all seeds: {output['n_test_cases']}")
    print("\nMean dataset-macro robustness target accuracy:")
    for row in output["method_summary"]:
        print(
            f"{row['method']:<28} "
            f"accuracy={row['mean_dataset_macro_robustness_target_accuracy']:.3f} "
            f"sd={row['sd_across_random_seeds']:.3f} "
            f"mean_abs_shift={row['mean_dataset_macro_abs_coefficient_shift']:.4f}"
        )
    print("\nCaution: the labels measure relative stability under prescribed perturbations, not scientific truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
