from app.schemas import AgentResult, Evidence, ModelResult
from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder


def create_result(agent, task, finding, evidence_description, confidence):
    return AgentResult(
        agent=agent,
        dataset="customer_churn.csv",
        task=task,
        findings=[finding],
        evidence=[
            Evidence(
                description=evidence_description,
                method="Test evidence",
                value=0.42,
                finding_indices=[0],
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": 0.87},
        ),
        confidence=confidence,
    )


def main():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    result1 = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="MonthlyCharges is associated with churn",
        evidence_description="Pearson correlation = 0.42",
        confidence=0.84,
    )

    created_finding_ids_1 = builder.add_agent_result(result1)
    print("First result findings:", created_finding_ids_1)

    result2 = create_result(
        agent="MLAgent",
        task="classification",
        finding="Random Forest predicts customer churn",
        evidence_description="Random Forest accuracy = 0.87",
        confidence=0.81,
    )

    created_finding_ids_2 = builder.add_agent_result(result2)
    print("Second result findings:", created_finding_ids_2)

    assert created_finding_ids_1 == ["finding_1_1"]
    assert created_finding_ids_2 == ["finding_2_1"]

    print("\nNODES:")
    for node_id, data in kg.graph.nodes(data=True):
        print(node_id, data)

    print("\nRELATIONSHIPS:")
    for source, target, data in kg.graph.edges(data=True):
        print(source, "->", target, data)

    print("\nBUILDER TEST PASSED")


def test_builder_adds_agent_result_with_evidence_and_model():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    result = create_result(
        agent="MLAgent",
        task="classification",
        finding="MonthlyCharges is associated with churn",
        evidence_description="Pearson correlation = 0.42",
        confidence=0.84,
    )

    finding_ids = builder.add_agent_result(result)

    assert finding_ids == ["finding_1_1"]

    feature_nodes = [
        data
        for _, data in kg.graph.nodes(data=True)
        if data.get("type") == "feature"
    ]

    assert len(feature_nodes) == 0  # No features were supplied.

    evidence_nodes = [
        data
        for _, data in kg.graph.nodes(data=True)
        if data.get("type") == "evidence"
    ]
    model_nodes = [
        data
        for _, data in kg.graph.nodes(data=True)
        if data.get("type") == "model"
    ]

    assert len(evidence_nodes) == 1
    assert len(model_nodes) == 1
    assert model_nodes[0]["model_name"] == "Random Forest"


def test_builder_does_not_match_feature_substrings():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    result = AgentResult(
        agent="EDAAgent",
        dataset="sales",
        task="feature_analysis",
        findings=["Revenue is associated with growth"],
        features=["Rev"],
        confidence=0.8,
    )

    builder.add_agent_result(result)

    feature_nodes = [
        node_id
        for node_id, data in kg.graph.nodes(data=True)
        if data.get("type") == "feature"
    ]

    assert feature_nodes == []


def test_feature_ids_do_not_collide_across_datasets():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    result1 = AgentResult(
        agent="EDAAgent",
        dataset="sales_region",
        task="feature_analysis",
        findings=["Revenue is associated with profit"],
        features=["profit"],
        confidence=0.8,
    )

    result2 = AgentResult(
        agent="EDAAgent",
        dataset="sales",
        task="feature_analysis",
        findings=[
            "The region_profit feature is associated with revenue"
        ],
        features=["region_profit"],
        confidence=0.8,
    )

    builder.add_agent_result(result1)
    builder.add_agent_result(result2)

    feature_nodes = [
        (node_id, data)
        for node_id, data in kg.graph.nodes(data=True)
        if data.get("type") == "feature"
    ]

    assert len(feature_nodes) == 2
    assert len({node_id for node_id, _ in feature_nodes}) == 2

    features_by_dataset = {
        data["dataset"]: data["name"]
        for _, data in feature_nodes
    }

    assert features_by_dataset == {
        "sales_region": "profit",
        "sales": "region_profit",
    }


if __name__ == "__main__":
    main()