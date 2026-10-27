import pandas as pd


DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MAX_COLUMNS = 500
DEFAULT_MAX_CELLS = 5_000_000


def load_dataset(
    file_path: str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    max_cells: int = DEFAULT_MAX_CELLS,
) -> pd.DataFrame:
    """
    Load a CSV file with configurable row, column, and cell limits.

    Reads at most one row beyond the effective limit to detect
    oversized datasets without loading the entire file first.
    """
    limits = {
        "max_rows": max_rows,
        "max_columns": max_columns,
        "max_cells": max_cells,
    }

    for name, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    # Read only the header before loading the data.
    column_count = len(pd.read_csv(file_path, nrows=0).columns)

    if column_count > max_columns:
        raise ValueError(
            f"Dataset exceeds the column limit "
            f"(max_columns={max_columns})."
        )

    # Apply both the row limit and the overall cell budget.
    effective_row_limit = min(
        max_rows,
        max_cells // max(column_count, 1),
    )

    df = pd.read_csv(
        file_path,
        nrows=effective_row_limit + 1,
    )

    if len(df) > max_rows:
        raise ValueError(
            f"Dataset exceeds the row limit (max_rows={max_rows})."
        )

    if len(df) * column_count > max_cells:
        raise ValueError(
            f"Dataset exceeds the cell limit (max_cells={max_cells})."
        )

    return df


if __name__ == "__main__":
    df = load_dataset("data/titanic.csv")
    print("Loaded dataset with shape:", df.shape)
    print(df.head())