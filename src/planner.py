def create_task_from_insight(insight: dict) -> dict:
    """
    Converts one insight into a concrete analysis task,
    with a plain-English reason attached.
    """
    insight_type = insight["type"]
    columns = insight["columns"]

    if insight_type == "correlation":
        task_name = "investigate_correlation"
        reason = (
            f"Strong correlation ({insight['value']}) detected between "
            f"{columns[0]} and {columns[1]}."
        )

    elif insight_type == "missingness":
        task_name = "investigate_missingness"
        reason = (
            f"Column {columns[0]} has a high missing-value rate "
            f"({insight['value']}%)."
        )

    elif insight_type == "outlier":
        task_name = "investigate_outliers"
        reason = (
            f"Column {columns[0]} has a high percentage of outliers "
            f"({insight['value']}%)."
        )

    else:
        task_name = "unknown_task"
        reason = "No specific reason available."

    return {
        "task_id": f"T{insight['id'][1:]}",   # reuse the insight's number, e.g. I003 -> T003
        "task": task_name,
        "columns": columns,
        "reason": reason,
        "priority": insight["priority"],
        "source_insight_id": insight["id"],
    }


def build_task_plan(ranked_insights: list) -> list:
    """
    Converts a full ranked insight list into a ranked task plan.
    """
    tasks = [create_task_from_insight(insight) for insight in ranked_insights]
    tasks.sort(key=lambda t: t["priority"], reverse=True)
    return tasks


if __name__ == "__main__":
    from data_loader import load_dataset
    from eda import compute_correlations, detect_outliers_iqr
    from profiler import profile_dataset
    from insights import (
        detect_correlation_insights,
        detect_missingness_insights,
        detect_outlier_insights,
        rank_insights,
    )

    df = load_dataset("data/titanic.csv")
    profile = profile_dataset(df)
    correlations = compute_correlations(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_results = [detect_outliers_iqr(df, col) for col in numeric_cols]

    corr_insights = detect_correlation_insights(correlations, threshold=0.3)
    missing_insights = detect_missingness_insights(profile["missing_pct"], threshold=20.0)
    outlier_insights = detect_outlier_insights(outlier_results, threshold=5.0)

    all_insights = corr_insights + missing_insights + outlier_insights
    ranked_insights = rank_insights(all_insights)

    task_plan = build_task_plan(ranked_insights)

    print(f"=== Analysis Plan: {len(task_plan)} tasks ===\n")
    for task in task_plan:
        print(task)
