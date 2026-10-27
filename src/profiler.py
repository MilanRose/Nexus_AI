import json
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Inspects a DataFrame and returns a structured profile:
    shape, column types, missing values, duplicates, and basic stats.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    profile = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df[numeric_cols].describe().to_dict(),
    }
    return profile


def save_profile(profile: dict, output_path: str):
    """
    Saves the profile dictionary as a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset("data/titanic.csv")
    profile = profile_dataset(df)
    save_profile(profile, "outputs/reports/dataset_profile.json")

    print("Profile generated successfully!")
    print("Rows:", profile["n_rows"])
    print("Columns:", profile["n_cols"])
    print("Numeric columns:", profile["numeric_columns"])
    print("Categorical columns:", profile["categorical_columns"])
    print("Missing values:", profile["missing_values"])
    print("Duplicate rows:", profile["duplicate_rows"])