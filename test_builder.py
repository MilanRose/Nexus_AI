from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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
                finding_indices=[0]
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": 0.87}
        ),
        confidence=confidence
    )


def test_builder():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    # First independent result
    result1 = create_result(
        agent="EDAAgent",
        task="correlation_analysis",
        finding="MonthlyCharges is associated with churn",
        evidence_description="Pearson correlation = 0.42",
        confidence=0.84
    )

    created_finding_ids_1 = builder.add_agent_result(result1)

    # Second independent result
    result2 = create_result(
        agent="MLAgent",
        task="classification",
        finding="Random Forest predicts customer churn",
        evidence_description="Random Forest accuracy = 0.87",
        confidence=0.81
    )

    created_finding_ids_2 = builder.add_agent_result(result2)

    # Verify unique finding IDs
    assert created_finding_ids_1 == ["finding_1_1"]
    assert created_finding_ids_2 == ["finding_2_1"]

    # Verify the findings exist
    assert "finding_1_1" in kg.graph.nodes
    assert "finding_2_1" in kg.graph.nodes

    # Verify evidence nodes and their associations
    for finding_id in ("finding_1_1", "finding_2_1"):
        evidence_targets = [
            target
            for _, target, data in kg.graph.out_edges(
                finding_id, data=True
            )
            if data.get("relation") == "supported_by"
        ]

        assert evidence_targets, (
            f"No supporting evidence connected to {finding_id}"
        )

        assert all(
            kg.graph.nodes[target].get("type") == "evidence"
            for target in evidence_targets
        )

    # Verify model relationships
    for finding_id in ("finding_1_1", "finding_2_1"):
        model_targets = [
            target
            for _, target, data in kg.graph.out_edges(
                finding_id, data=True
            )
            if data.get("relation") == "generated_by"
        ]

        assert model_targets, (
            f"No model connected to {finding_id}"
        )

    # Verify dataset relationships
    dataset_id = "dataset_customer_churn.csv"
    assert dataset_id in kg.graph.nodes

    for finding_id in ("finding_1_1", "finding_2_1"):
        assert any(
            target == finding_id
            and data.get("relation") == "contains_finding"
            for _, target, data in kg.graph.out_edges(
                dataset_id, data=True
            )
        )

    print("\nBUILDER TEST PASSED")


if __name__ == "__main__":
    test_builder()


def test_concurrent_builder_calls_create_unique_finding_ids():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    number_of_calls = 10
    barrier = Barrier(number_of_calls)

    def build_result(index):
        result = create_result(
            agent=f"Agent{index}",
            task="concurrent_test",
            finding=f"Finding number {index}",
            evidence_description=f"Evidence number {index}",
            confidence=0.8
        )

        barrier.wait()
        return builder.add_agent_result(result)

    with ThreadPoolExecutor(max_workers=number_of_calls) as executor:
        results = list(executor.map(build_result, range(number_of_calls)))

    finding_ids = [
        finding_id
        for result_ids in results
        for finding_id in result_ids
    ]

    assert len(finding_ids) == number_of_calls
    assert len(set(finding_ids)) == number_of_calls

    assert all(
        finding_id in kg.graph.nodes
        for finding_id in finding_ids
    )

    assert all(
        kg.graph.nodes[finding_id]["type"] == "finding"
        for finding_id in finding_ids
    )


def test_invalid_evidence_index_does_not_modify_graph():
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)

    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        features=["MonthlyCharges"],
        evidence=[
            Evidence(
                description="Evidence with an invalid finding index",
                method="Test",
                value=0.42,
                finding_indices=[5]
            )
        ],
        confidence=0.84
    )

    original_nodes = set(kg.graph.nodes)
    original_edges = list(kg.graph.edges(keys=True, data=True))

    try:
        builder.add_agent_result(result)
        assert False, "Expected ValueError for invalid finding index"
    except ValueError as exc:
        assert "invalid finding index" in str(exc).lower()

    assert set(kg.graph.nodes) == original_nodes
    assert list(kg.graph.edges(keys=True, data=True)) == original_edges
