from app.schemas import AgentResult


def test_feature_schema():
    # 1. Test explicitly supplied feature names
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        features=["MonthlyCharges", "tenure"],
        confidence=0.84
    )

    assert result.features == ["MonthlyCharges", "tenure"]
    assert result.confidence == 0.84

    # 2. Test backward compatibility when features are omitted
    old_result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["A finding without explicit features"],
        confidence=0.80
    )

    assert old_result.features == []
    assert old_result.findings == [
        "A finding without explicit features"
    ]

    # 3. Verify features are preserved during serialization
    serialized = result.model_dump()

    assert serialized["features"] == ["MonthlyCharges", "tenure"]

    old_serialized = old_result.model_dump()
    assert old_serialized["features"] == []

    print("FEATURE SCHEMA TEST PASSED")