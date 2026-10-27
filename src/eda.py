import os

import pandas as pd
import matplotlib.pyplot as plt


MAX_CORRELATION_COLUMNS = 200


def _bounded_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prevent correlation calculations on excessively wide datasets."""
    numeric_df = df.select_dtypes(include="number")

    if len(numeric_df.columns) > MAX_CORRELATION_COLUMNS:
        raise ValueError(
            "Correlation analysis supports at most "
            f"{MAX_CORRELATION_COLUMNS} numeric columns; "
            f"found {len(numeric_df.columns)}."
        )

    return numeric_df

def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Computes Pearson correlation between numeric columns.

    Returns:
        A nested dictionary: {col1: {col2: correlation_value}}
    """
    numeric_df = _bounded_numeric_df(df)
    corr_matrix = numeric_df.corr(method="pearson")
    return corr_matrix.round(3).to_dict()

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """
    Detects outliers in a single numeric column using the IQR method.
    IQR = Interquartile Range = the range of the 'middle 50%' of values.
    Any value far outside that range is flagged as an outlier.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return {
        "column": column,
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "n_outliers": len(outliers),
        "pct_outliers": round(len(outliers) / len(df) * 100, 2),
    }


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """
    Creates a correlation heatmap for all numeric columns and saves it as an image.
    """
    numeric_df = _bounded_numeric_df(df)
    corr_matrix = numeric_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_title("Correlation Heatmap", pad=20)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_histogram(df: pd.DataFrame, column: str, output_path: str):
    """
    Creates a histogram showing the distribution of one numeric column.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[column].dropna(), bins=20, color="steelblue", edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, column: str, output_path: str):
    """
    Creates a box plot for one numeric column, useful for visualizing outliers.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(df[column].dropna(), vert=True)
    ax.set_title(f"Box Plot of {column}")
    ax.set_ylabel(column)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset("data/titanic.csv")

    print("=== Correlations ===")
    correlations = compute_correlations(df)
    for col1, values in correlations.items():
        print(col1, ":", values)

    print("\n=== Outliers ===")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        result = detect_outliers_iqr(df, col)
        print(result)

    print("\n=== Generating charts ===")
    os.makedirs("outputs/charts", exist_ok=True)

    plot_correlation_heatmap(df, "outputs/charts/correlation_heatmap.png")
    print("Saved: correlation_heatmap.png")

    plot_histogram(df, "Age", "outputs/charts/age_histogram.png")
    print("Saved: age_histogram.png")

    plot_boxplot(df, "Fare", "outputs/charts/fare_boxplot.png")
    print("Saved: fare_boxplot.png")