
from src.insights import rank_insights
from src.planner import build_task_plan


def test_insight_and_task_ids_are_unique_across_types():
    insights = [
        {
            "id": "I001",
            "type": "correlation",
            "columns": ["Age", "Fare"],
            "value": 0.4,
            "priority": 0.9,
        },
        {
            "id": "I001",
            "type": "missingness",
            "columns": ["Cabin"],
            "value": 77.0,
            "priority": 0.8,
        },
        {
            "id": "I001",
            "type": "outlier",
            "columns": ["Fare"],
            "value": 12.0,
            "priority": 0.7,
        },
    ]

    ranked = rank_insights(insights)
    insight_ids = [insight["id"] for insight in ranked]

    assert insight_ids == ["I001", "I002", "I003"]
    assert len(set(insight_ids)) == len(insight_ids)

    task_plan = build_task_plan(ranked)
    task_ids = [task["task_id"] for task in task_plan]

    assert task_ids == ["T001", "T002", "T003"]
    assert len(set(task_ids)) == len(task_ids)

    assert {
        task["source_insight_id"] for task in task_plan
    } == set(insight_ids)
