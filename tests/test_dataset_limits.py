import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.eda import (
    MAX_CORRELATION_COLUMNS,
    compute_correlations,
    plot_correlation_heatmap,
)


def test_load_dataset_rejects_too_many_columns(tmp_path):
    csv_path = tmp_path / "wide.csv"
    csv_path.write_text(
        "a,b,c\n1,2,3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_columns=2"):
        load_dataset(
            str(csv_path),
            max_rows=10,
            max_columns=2,
            max_cells=100,
        )


def test_load_dataset_rejects_too_many_rows(tmp_path):
    csv_path = tmp_path / "large.csv"
    csv_path.write_text(
        "a,b\n1,2\n3,4\n5,6\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_rows=2"):
        load_dataset(
            str(csv_path),
            max_rows=2,
            max_columns=10,
            max_cells=100,
        )


def test_load_dataset_rejects_excessive_cell_count(tmp_path):
    csv_path = tmp_path / "cells.csv"
    csv_path.write_text(
        "a,b\n1,2\n3,4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_cells=2"):
        load_dataset(
            str(csv_path),
            max_rows=10,
            max_columns=10,
            max_cells=2,
        )


def test_compute_correlations_rejects_too_many_numeric_columns():
    df = pd.DataFrame({
        f"column_{i}": [i, i + 1]
        for i in range(MAX_CORRELATION_COLUMNS + 1)
    })

    with pytest.raises(ValueError, match="200 numeric columns"):
        compute_correlations(df)


def test_heatmap_rejects_too_many_numeric_columns(tmp_path):
    df = pd.DataFrame({
        f"column_{i}": [i, i + 1]
        for i in range(MAX_CORRELATION_COLUMNS + 1)
    })

    output_path = tmp_path / "heatmap.png"

    with pytest.raises(ValueError, match="200 numeric columns"):
        plot_correlation_heatmap(df, str(output_path))

    assert not output_path.exists()