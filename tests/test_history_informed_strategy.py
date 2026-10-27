
import pandas as pd

from src.analysis import execute_task


class FakeEvolutionEngine:
    def __init__(self, strategy, reason="Test historical recommendation."):
        self.strategy = strategy
        self.reason = reason

    def recommend_strategy(self, task, dataset=None, context=None):
        return {
            "recommended_strategy": self.strategy,
            "task": task,
            "dataset": dataset,
            "context_received": context is not None,
            "reason": self.reason,
        }


def test_historical_spearman_recommendation_is_executed():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 4, 2, 5, 3, 7],
    })
    task = {
        "task_id": "T001",
        "task": "investigate_correlation",
        "columns": ["x", "y"],
    }

    result = execute_task(
        df,
        task,
        evolution_engine=FakeEvolutionEngine("Spearman correlation"),
        dataset="customer_churn.csv",
    )

    assert result["method"] == "Spearman correlation"
    assert result["strategy_context"]["version"] == 1
    assert "historical strategy recommendation" in (
        result["strategy_reason"].lower()
    )


def test_explicit_method_overrides_historical_recommendation():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 4, 2, 5, 3, 7],
    })
    task = {
        "task_id": "T002",
        "task": "investigate_correlation",
        "columns": ["x", "y"],
        "correlation_method": "pearson",
    }

    result = execute_task(
        df,
        task,
        evolution_engine=FakeEvolutionEngine("Spearman correlation"),
        dataset="customer_churn.csv",
    )

    assert result["method"] == "Pearson correlation"
    assert result["strategy_reason"] == (
        "Method explicitly specified by the task."
    )


def test_contextual_memory_abstention_is_recorded_before_heuristic_fallback():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 4, 2, 5, 3, 7],
    })
    task = {
        "task_id": "T003",
        "task": "investigate_correlation",
        "columns": ["x", "y"],
    }

    result = execute_task(
        df,
        task,
        evolution_engine=FakeEvolutionEngine(
            None,
            reason="No similar context met the similarity threshold.",
        ),
        dataset="customer_churn.csv",
    )

    assert result["method"] in {
        "Pearson correlation",
        "Spearman correlation",
    }
    assert result["strategy_reason"].startswith("Contextual memory abstained:")
    assert "No similar context" in result["strategy_reason"]
    assert "Fallback heuristic:" in result["strategy_reason"]
