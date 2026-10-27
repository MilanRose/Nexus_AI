import pytest
from pydantic import ValidationError

from app.schemas import AgentResult, Evidence, ModelResult


def test_schemas():
    # 1. Create valid evidence
    evidence = Evidence(
        description="Pearson correlation = 0.42",
        method="Pearson correlation",
        value=0.42,
        finding_indices=[0]
    )

    assert evidence.description == "Pearson correlation = 0.42"
    assert evidence.method == "Pearson correlation"
    assert evidence.value == 0.42
    assert evidence.finding_indices == [0]

    # 2. Create a model result
    model = ModelResult(
        name="Random Forest",
        metrics={"accuracy": 0.87}
    )

    assert model.name == "Random Forest"
    assert model.metrics["accuracy"] == 0.87

    # 3. Create a complete agent result
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=[
            "MonthlyCharges is associated with churn"
        ],
        features=["MonthlyCharges"],
        evidence=[evidence],
        model=model,
        confidence=0.84
    )

    # 4. Verify the main fields
    assert result.agent == "EDAAgent"
    assert result.dataset == "customer_churn.csv"
    assert result.task == "correlation_analysis"
    assert result.findings == [
        "MonthlyCharges is associated with churn"
    ]
    assert result.features == ["MonthlyCharges"]
    assert result.confidence == 0.84

    # 5. Verify nested objects
    assert len(result.evidence) == 1
    assert result.evidence[0].finding_indices == [0]
    assert result.model is not None
    assert result.model.name == "Random Forest"

    # 6. Verify serialization
    # 6. Verify serialization using the supported Pydantic v2 API.
    serialized = result.model_dump()
    assert serialized["agent"] == "EDAAgent"
    assert serialized["confidence"] == 0.84
    assert serialized["evidence"][0]["finding_indices"] == [0]
    assert serialized["model"]["metrics"]["accuracy"] == 0.87

    # 7. Verify default values
    minimal_result = AgentResult(
        agent="EDAAgent",
        dataset="sample.csv",
        task="eda",
        confidence=0.75
    )

    assert minimal_result.findings == []
    assert minimal_result.features == []
    assert minimal_result.evidence == []
    assert minimal_result.model is None

    # 8. Confidence must remain between 0 and 1
    with pytest.raises(ValidationError):
        AgentResult(
            agent="EDAAgent",
            dataset="sample.csv",
            task="eda",
            confidence=1.2
        )

    with pytest.raises(ValidationError):
        AgentResult(
            agent="EDAAgent",
            dataset="sample.csv",
            task="eda",
            confidence=-0.1
        )

    print("SCHEMA TEST PASSED")