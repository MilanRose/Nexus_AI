import math

from scipy import stats
import pandas as pd



def execute_correlation_task(
    df: pd.DataFrame,
    task: dict,
    method: str = "pearson",
) -> dict:
    """Compute Pearson or Spearman correlation and its p-value."""
    col1, col2 = task["columns"]
    clean_df = df[[col1, col2]].dropna()

    if len(clean_df) < 2:
        return {
            "task_id": task["task_id"],
            "task": task["task"],
            "columns": [col1, col2],
            "method": method,
            "error": "At least two complete rows are required.",
        }

    if method == "pearson":
        r_value, p_value = stats.pearsonr(
            clean_df[col1], clean_df[col2]
        )
        method_name = "Pearson correlation"
    elif method == "spearman":
        r_value, p_value = stats.spearmanr(
            clean_df[col1], clean_df[col2]
        )
        method_name = "Spearman correlation"
    else:
        raise ValueError(
            f"Unsupported correlation method: {method!r}"
        )

    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "columns": [col1, col2],
        "method": method_name,
        "value": round(float(r_value), 3),
        "p_value": round(float(p_value), 5),
        "sample_size": len(clean_df),
    }


def execute_missingness_task(df: pd.DataFrame, task: dict) -> dict:
    """
    Recomputes the missing-value count and percentage for the task's column.
    """
    column = task["columns"][0]
    n_missing = df[column].isnull().sum()
    pct_missing = round((n_missing / len(df)) * 100, 2)

    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "columns": [column],
        "method": "Missing value count",
        "n_missing": int(n_missing),
        "pct_missing": pct_missing,
        "sample_size": len(df),
    }


def execute_outlier_task(df: pd.DataFrame, task: dict) -> dict:
    """
    Recomputes outlier count/percentage for the task's column using IQR.
    """
    column = task["columns"][0]
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    pct_outliers = round((len(outliers) / len(df)) * 100, 2)

    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "columns": [column],
        "method": "IQR outlier detection",
        "n_outliers": len(outliers),
        "pct_outliers": pct_outliers,
        "bounds": [round(lower_bound, 2), round(upper_bound, 2)],
        "sample_size": len(df),
    }


def build_correlation_context(
    df: pd.DataFrame,
    col1: str,
    col2: str,
) -> dict:
    """Build a versioned profile describing the data for correlation choice.

    The profile contains only input/data characteristics. It deliberately
    excludes the selected method and the final finding so historical retrieval
    is based on context rather than copying a prior answer.
    """
    pair = df[[col1, col2]]
    complete = pair.dropna()
    sample_size = int(len(complete))
    total_rows = int(len(pair))
    missing_fraction = (
        float((total_rows - sample_size) / total_rows)
        if total_rows else 0.0
    )

    absolute_skews = []
    outlier_fractions = []
    constant_count = 0
    numeric_columns = []

    for column in (col1, col2):
        numeric = pd.to_numeric(complete[column], errors="coerce").dropna()
        numeric_columns.append(numeric)

        if numeric.nunique() < 2:
            constant_count += 1
            absolute_skews.append(0.0)
        elif len(numeric) >= 3:
            skew_value = float(stats.skew(numeric.to_numpy(), bias=False))
            absolute_skews.append(
                min(abs(skew_value), 8.0) if math.isfinite(skew_value) else 0.0
            )
        else:
            absolute_skews.append(0.0)

        if len(numeric) == 0:
            outlier_fractions.append(0.0)
        else:
            q1 = float(numeric.quantile(0.25))
            q3 = float(numeric.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            fraction = float(((numeric < lower) | (numeric > upper)).mean())
            outlier_fractions.append(
                fraction if math.isfinite(fraction) else 0.0
            )

    correlation_gap = 0.0
    correlation_gap_available = 0.0
    if (
        sample_size >= 3
        and len(numeric_columns) == 2
        and all(series.nunique() >= 2 for series in numeric_columns)
    ):
        try:
            pearson_value = float(stats.pearsonr(
                numeric_columns[0].to_numpy(dtype=float),
                numeric_columns[1].to_numpy(dtype=float),
            )[0])
            spearman_value = float(stats.spearmanr(
                numeric_columns[0].to_numpy(dtype=float),
                numeric_columns[1].to_numpy(dtype=float),
            )[0])
            if math.isfinite(pearson_value) and math.isfinite(spearman_value):
                correlation_gap = abs(pearson_value - spearman_value)
                correlation_gap_available = 1.0
        except (ValueError, TypeError, FloatingPointError):
            # This diagnostic is optional; retain the other profile features.
            pass

    features = {
        "log10_sample_size": math.log10(sample_size + 1),
        "missing_fraction": missing_fraction,
        "max_abs_skew": max(absolute_skews, default=0.0),
        "mean_abs_skew": (
            sum(absolute_skews) / len(absolute_skews)
            if absolute_skews else 0.0
        ),
        "max_iqr_outlier_fraction": max(outlier_fractions, default=0.0),
        "pearson_spearman_gap": correlation_gap,
        "correlation_gap_available": correlation_gap_available,
        "constant_column_fraction": constant_count / 2.0,
    }
    features = {
        name: (float(value) if math.isfinite(float(value)) else 0.0)
        for name, value in features.items()
    }

    return {
        "version": 1,
        "columns": [str(col1), str(col2)],
        "row_count": total_rows,
        "complete_sample_size": sample_size,
        "features": features,
    }



def execute_task(
    df: pd.DataFrame,
    task: dict,
    evolution_engine=None,
    dataset: str | None = None,
) -> dict:
    """Execute a task using an explicit, historical, or data-based strategy."""
    task_type = task["task"]

    if task_type == "investigate_correlation":
        col1, col2 = task["columns"]
        strategy_context = build_correlation_context(df, col1, col2)

        if "correlation_method" in task:
            method = task["correlation_method"]
            reason = "Method explicitly specified by the task."

        else:
            method = None
            reason = None
            recommendation = None

            # First consult historical strategy recommendations.
            if evolution_engine is not None and dataset is not None:
                recommendation = evolution_engine.recommend_strategy(
                    task=task_type,
                    dataset=dataset,
                    context=strategy_context,
                )
                historical_method = recommendation.get(
                    "recommended_strategy"
                )

                # History stores display names; the executor uses lowercase.
                method_map = {
                    "Pearson correlation": "pearson",
                    "Spearman correlation": "spearman",
                    "pearson": "pearson",
                    "spearman": "spearman",
                }

                if historical_method in method_map:
                    method = method_map[historical_method]
                    reason = (
                        "Using historical strategy recommendation: "
                        + recommendation.get("reason", "")
                    )

            # Fall back to the data-based heuristic if history is unavailable.
            if method is None:
                method, heuristic_reason = select_correlation_strategy(
                    df, col1, col2
                )
                if recommendation is not None and recommendation.get("reason"):
                    reason = (
                        "Contextual memory abstained: "
                        + str(recommendation["reason"])
                        + " Fallback heuristic: "
                        + heuristic_reason
                    )
                else:
                    reason = heuristic_reason

        result = execute_correlation_task(
            df, task, method=method
        )
        result["strategy_reason"] = reason
        result["strategy_context"] = strategy_context
        return result

    elif task_type == "investigate_missingness":
        return execute_missingness_task(df, task)

    elif task_type == "investigate_outliers":
        return execute_outlier_task(df, task)

    return {
        "task_id": task["task_id"],
        "error": f"Unknown task type: {task_type}",
    }

def select_correlation_strategy(df, col1, col2):
    """
    Select a correlation method using simple data diagnostics.
    Returns the method and an explanation.
    """
    from scipy.stats import skew

    clean = df[[col1, col2]].dropna()

    if len(clean) < 3:
        return "pearson", "Too few complete rows for reliable diagnostics; using default."

    # Constant columns cannot produce a meaningful correlation.
    if clean[col1].nunique() < 2 or clean[col2].nunique() < 2:
        return "pearson", "A column is constant; correlation may be undefined."

    skew_values = []
    for column in (col1, col2):
        value = float(skew(clean[column], bias=False))
        if value == value:  # Exclude NaN
            skew_values.append(abs(value))

    if any(value > 1.0 for value in skew_values):
        return (
            "spearman",
            "Strong skew detected; rank-based correlation is being used "
            "as a robustness-oriented alternative.",
        )

    return (
        "pearson",
        "No strong skew detected by the selection rule; using Pearson "
        "as the default for linear association.",
    )

def run_analysis(dataset_path="data/titanic.csv", evolution_engine=None):
    """Run the analysis pipeline and return its results."""

    if __package__:
        from .data_loader import load_dataset
        from .eda import compute_correlations, detect_outliers_iqr
        from .profiler import profile_dataset
        from .insights import (
            detect_correlation_insights,
            detect_missingness_insights,
            detect_outlier_insights,
            rank_insights,
        )
        from .planner import build_task_plan
    else:
        from data_loader import load_dataset
        from eda import compute_correlations, detect_outliers_iqr
        from profiler import profile_dataset
        from insights import (
            detect_correlation_insights,
            detect_missingness_insights,
            detect_outlier_insights,
            rank_insights,
        )
        from planner import build_task_plan

    df = load_dataset(dataset_path)
    profile = profile_dataset(df)
    correlations = compute_correlations(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_results = [
        detect_outliers_iqr(df, column)
        for column in numeric_cols
    ]

    corr_insights = detect_correlation_insights(
        correlations, threshold=0.3
    )
    missing_insights = detect_missingness_insights(
        profile["missing_pct"], threshold=20.0
    )
    outlier_insights = detect_outlier_insights(
        outlier_results, threshold=5.0
    )

    all_insights = (
        corr_insights + missing_insights + outlier_insights
    )
    ranked_insights = rank_insights(all_insights)
    task_plan = build_task_plan(ranked_insights)

    dataset_name = str(dataset_path).replace("\\", "/").rsplit("/", 1)[-1]
    
    execution_results = [
        execute_task(
            df,
            task,
            evolution_engine=evolution_engine,
            dataset=dataset_name,
        )
        for task in task_plan
    ]

    return {
        "dataset": str(dataset_path),
        "profile": profile,
        "insights": ranked_insights,
        "tasks": task_plan,
        "results": execution_results,
    }


if __name__ == "__main__":
    output = run_analysis()

    print(
        f"=== Executing {len(output['tasks'])} tasks ===\n"
    )

    for result in output["results"]:
        print(result)
