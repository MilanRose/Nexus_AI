import pandas as pd
import pytest

from app.agents.ml_agent import MLAgent


def test_ml_agent_trains_and_evaluates_classifier(tmp_path):
    dataset_path = tmp_path / "classification.csv"

    # Create a deterministic dataset with a learnable binary target.
    df = pd.DataFrame({
        "PassengerId": range(1, 121),
        "signal": [i % 2 for i in range(120)],
        "amount": [float(i % 2) * 10.0 + (i % 5) for i in range(120)],
        "category": [
            "high" if i % 2 else "low"
            for i in range(120)
        ],
        "Survived": [
            i % 2
            for i in range(120)
        ],
    })

    df.to_csv(dataset_path, index=False)

    agent = MLAgent()
    result, details = agent.run(
        str(dataset_path),
        target_column="Survived",
    )

    assert result.agent == "MLAgent"
    assert result.dataset == "classification.csv"
    assert result.task == "classification"
    assert result.model is not None
    assert result.model.name == "RandomForestClassifier"

    assert result.findings
    assert result.evidence

    assert result.model.metrics["accuracy"] == pytest.approx(1.0)
    assert 0.0 <= result.model.metrics["precision_macro"] <= 1.0
    assert 0.0 <= result.model.metrics["recall_macro"] <= 1.0
    assert 0.0 <= result.model.metrics["f1_macro"] <= 1.0

    assert details["target_column"] == "Survived"
    assert details["training_samples"] + details["test_samples"] == 120
    assert "signal" in details["feature_columns"]
    assert "PassengerId" in details["excluded_columns"]

    # Every metric must be attached to the finding as evidence.
    for evidence in result.evidence:
        assert evidence.finding_indices == [0]

    assert result.confidence > 0


def test_ml_agent_rejects_missing_target(tmp_path):
    dataset_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "feature": range(20),
        "label": [i % 2 for i in range(20)],
    }).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="was not found"):
        MLAgent().run(
            str(dataset_path),
            target_column="Survived",
        )


def test_ml_agent_rejects_target_with_one_class(tmp_path):
    dataset_path = tmp_path / "single_class.csv"

    pd.DataFrame({
        "feature": range(20),
        "label": [1] * 20,
    }).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="2-20 distinct classes"):
        MLAgent().run(
            str(dataset_path),
            target_column="label",
        )


def test_ml_agent_rejects_regression_target_for_version_one(tmp_path):
    dataset_path = tmp_path / "regression.csv"

    pd.DataFrame({
        "feature": range(30),
        "price": [float(i) for i in range(30)],
    }).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Regression is not yet implemented"):
        MLAgent().run(
            str(dataset_path),
            target_column="price",
        )