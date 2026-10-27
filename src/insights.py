def detect_correlation_insights(correlations: dict, threshold: float = 0.5) -> list:
    """
    Scans a correlation dictionary and returns pairs of columns
    whose correlation is strong enough to be worth noting.
    Each pair is only included once (X-Y, not also Y-X).
    """
    insights = []
    seen_pairs = set()

    for col1, related in correlations.items():
        for col2, value in related.items():
            if col1 == col2:
                continue  # skip self-correlation, always 1.0

            pair_key = tuple(sorted([col1, col2]))
            if pair_key in seen_pairs:
                continue  # already recorded this pair the other way around
            seen_pairs.add(pair_key)

            if abs(value) >= threshold:
                insights.append({
                    "id": f"I{len(insights) + 1:03d}",
                    "type": "correlation",
                    "columns": [col1, col2],
                    "value": value,
                    "priority": round(abs(value), 3),
                })

    return insights


def detect_missingness_insights(missing_pct: dict, threshold: float = 20.0) -> list:
    """
    Flags columns whose missing-value percentage is high enough to matter.
    """
    insights = []
    for column, pct in missing_pct.items():
        if pct >= threshold:
            insights.append({
                "id": f"I{len(insights) + 1:03d}",
                "type": "missingness",
                "columns": [column],
                "value": pct,
                "priority": round(pct / 100, 3),
            })
    return insights


def detect_outlier_insights(outlier_results: list, threshold: float = 5.0) -> list:
    """
    Flags columns whose outlier percentage is high enough to matter.
    outlier_results is expected to be a list of dicts, like the output
    of detect_outliers_iqr() for each column.
    """
    insights = []
    for result in outlier_results:
        if result["pct_outliers"] >= threshold:
            insights.append({
                "id": f"I{len(insights) + 1:03d}",
                "type": "outlier",
                "columns": [result["column"]],
                "value": result["pct_outliers"],
                "priority": round(result["pct_outliers"] / 100, 3),
            })
    return insights



def rank_insights(all_insights: list) -> list:
    """
    Sort insights by priority and assign globally unique IDs
    across correlation, missingness, and outlier insights.
    """
    ranked = sorted(
        (dict(insight) for insight in all_insights),
        key=lambda insight: insight["priority"],
        reverse=True
    )

    # Assign IDs after combining and ranking all insight types.
    for index, insight in enumerate(ranked, start=1):
        insight["id"] = f"I{index:03d}"

    return ranked



if __name__ == "__main__":
    from data_loader import load_dataset
    from eda import compute_correlations, detect_outliers_iqr
    from profiler import profile_dataset

    df = load_dataset("data/titanic.csv")

    profile = profile_dataset(df)
    correlations = compute_correlations(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_results = [detect_outliers_iqr(df, col) for col in numeric_cols]

    corr_insights = detect_correlation_insights(correlations, threshold=0.3)
    missing_insights = detect_missingness_insights(profile["missing_pct"], threshold=20.0)
    outlier_insights = detect_outlier_insights(outlier_results, threshold=5.0)

    all_insights = corr_insights + missing_insights + outlier_insights
    ranked = rank_insights(all_insights)

    print(f"=== Detected {len(ranked)} insights ===\n")
    for insight in ranked:
        print(insight)