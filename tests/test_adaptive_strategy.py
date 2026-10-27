
import numpy as np
import pandas as pd

from src.analysis import select_correlation_strategy, execute_task


def test_symmetric_data_selects_pearson():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 2, 3, 4, 5, 6],
    })

    method, reason = select_correlation_strategy(df, "x", "y")

    assert method == "pearson"
    assert "Pearson" in reason


def test_skewed_data_selects_spearman():
    # Exponential values produce strong positive skew.
    x = np.arange(1, 101, dtype=float)
    df = pd.DataFrame({
        "x": x,
        "y": np.exp(x / 10),
    })

    method, reason = select_correlation_strategy(df, "x", "y")

    assert method == "spearman"
    assert "skew" in reason.lower()


def test_explicit_method_overrides_automatic_selection():
    df = pd.DataFrame({
        "x": np.arange(1, 101, dtype=float),
        "y": np.exp(np.arange(1, 101, dtype=float) / 10),
    })
    task = {
        "task_id": "T001",
        "task": "investigate_correlation",
        "columns": ["x", "y"],
        "correlation_method": "pearson",
    }

    result = execute_task(df, task)

    assert result["method"] == "Pearson correlation"
    assert result["strategy_reason"] == (
        "Method explicitly specified by the task."
    )


def test_missing_values_are_handled():
    df = pd.DataFrame({
        "x": [1, 2, np.nan, 4, 5, 6],
        "y": [1, 2, 3, 4, 5, 6],
    })

    method, reason = select_correlation_strategy(df, "x", "y")

    assert method in {"pearson", "spearman"}
    assert isinstance(reason, str)
    assert reason
