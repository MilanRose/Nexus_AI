
import pandas as pd
import pytest

from app.agents.explainability_agent import ExplainabilityAgent


def create_classification_dataset(path, rows=120):
    """Create a reproducible binary classification dataset."""

    df = pd.DataFrame(
        {
            "PassengerId": range(1, rows + 1),
            "signal": [i % 2 for i in range(rows)],
            "noise": [
                float((i * 17) % 23) / 23.0
                for i in range(rows)
            ],
            "category": [
                "even" if i % 2 == 0 else "odd"
                for i in range(rows)
            ],
            "Survived": [i % 2 for i in range(rows)],
        }
    )

    df.to_csv(path, index=False)
    return df


def test_explainability_agent_returns_feature_importance(tmp_path):
    dataset_path = tmp_path / "classification.csv"
    create_classification_dataset(dataset_path)

    agent = ExplainabilityAgent()

    agent_result, details = agent.run(
        dataset_path=dataset_path,
        target_column="Survived",
        top_n=5,
        n_repeats=3,
    )

    assert agent_result.agent == "ExplainabilityAgent"
    assert agent_result.task == "permutation_feature_importance"

    assert details["target_column"] == "Survived"
    assert details["model_name"] == "RandomForestClassifier"
    assert details["model_source"] == "refitted_reference_model"

    assert 0.0 <= details["baseline_accuracy"] <= 1.0
    assert 0.0 <= details["f1_macro"] <= 1.0

    assert details["train_samples"] > 0
    assert details["test_samples"] > 0

    assert details["feature_importance"]
    assert "PassengerId" in details["excluded_columns"]

    records = details["feature_importance"]

    assert len(records) <= 5
    assert records == sorted(
        records,
        key=lambda record: record["mean_accuracy_decrease"],
        reverse=True,
    )

    assert any(
        record["feature"] == "signal"
        for record in records
    )

    assert len(agent_result.findings) == len(records)
    assert len(agent_result.evidence) == len(records)

    for index, evidence_item in enumerate(agent_result.evidence):
        assert evidence_item.method == (
            "Permutation importance using held-out "
            "classification accuracy"
        )
        assert evidence_item.finding_indices == [index]
        assert "mean_accuracy_decrease" in evidence_item.value

    assert any(
        "causality" in limitation.lower()
        for limitation in details["limitations"]
    )


def test_explainability_agent_rejects_missing_target(tmp_path):
    dataset_path = tmp_path / "classification.csv"
    create_classification_dataset(dataset_path)

    agent = ExplainabilityAgent()

    with pytest.raises(ValueError, match="not found"):
        agent.run(
            dataset_path=dataset_path,
            target_column="UnknownTarget",
        )


def test_explainability_agent_rejects_invalid_target(tmp_path):
    dataset_path = tmp_path / "invalid_target.csv"

    pd.DataFrame(
        {
            "feature": range(20),
            "target": [1] * 20,
        }
    ).to_csv(dataset_path, index=False)

    agent = ExplainabilityAgent()

    with pytest.raises(ValueError, match="2 to 20"):
        agent.run(
            dataset_path=dataset_path,
            target_column="target",
        )


def test_explainability_agent_rejects_invalid_repeat_count(
    tmp_path,
):
    dataset_path = tmp_path / "classification.csv"
    create_classification_dataset(dataset_path)

    agent = ExplainabilityAgent()

    with pytest.raises(ValueError, match="n_repeats"):
        agent.run(
            dataset_path=dataset_path,
            target_column="Survived",
            n_repeats=0,
        )
