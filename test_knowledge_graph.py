from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.nodes import (
    DatasetNode,
    FeatureNode,
    FindingNode,
    EvidenceNode,
    ModelNode,
)
from app.knowledge_graph.reasoning import GraphReasoner


def test_knowledge_graph():
    # 1. Create a fresh knowledge graph
    kg = KnowledgeGraph()

    # 2. Create graph nodes
    dataset = DatasetNode(
        "dataset_1",
        "customer_churn.csv"
    )

    feature = FeatureNode(
        "feature_1",
        "MonthlyCharges"
    )

    finding = FindingNode(
        "finding_1",
        "MonthlyCharges is associated with churn",
        confidence=0.84
    )

    evidence = EvidenceNode(
        "evidence_1",
        "Pearson correlation = 0.42",
        method="Pearson correlation"
    )

    model = ModelNode(
        "model_1",
        "Random Forest",
        accuracy=0.87
    )

    # 3. Add nodes to the graph
    kg.add_node_object(dataset)
    kg.add_node_object(feature)
    kg.add_node_object(finding)
    kg.add_node_object(evidence)
    kg.add_node_object(model)

    # Verify that all five nodes were added
    expected_node_ids = {
        "dataset_1",
        "feature_1",
        "finding_1",
        "evidence_1",
        "model_1",
    }

    assert expected_node_ids.issubset(set(kg.graph.nodes))

    # 4. Add relationships
    kg.add_relationship("dataset_1", "contains", "feature_1")
    kg.add_relationship("feature_1", "supports", "finding_1")
    kg.add_relationship("finding_1", "supported_by", "evidence_1")
    kg.add_relationship("finding_1", "generated_by", "model_1")

    # Verify that each relationship exists
    expected_relationships = [
        ("dataset_1", "feature_1", "contains"),
        ("feature_1", "finding_1", "supports"),
        ("finding_1", "evidence_1", "supported_by"),
        ("finding_1", "model_1", "generated_by"),
    ]

    for source, target, relation in expected_relationships:
        matching_edges = [
            data
            for _, _, data in kg.graph.edges(
                source,
                data=True
            )
            if data.get("relation") == relation
            and _ == target
        ]

        assert matching_edges, (
            f"Missing relationship: {source} -[{relation}]-> {target}"
        )

    # 5. Test graph reasoning
    reasoner = GraphReasoner(kg)

    evidence_results = reasoner.get_supporting_evidence("finding_1")

    assert evidence_results, "No supporting evidence was returned"
    assert any(
        item.get("evidence_id") == "evidence_1"
        for item in evidence_results
    ), "The expected evidence was not returned"

    # 6. Verify finding context
    context = reasoner.get_finding_context("finding_1")

    assert context["features"], "No feature context was returned"
    assert context["datasets"], "No dataset context was returned"
    assert "feature_1" in context["features"]
    assert "dataset_1" in context["datasets"]

    # 7. Verify provenance
    provenance = reasoner.get_provenance_chain("finding_1")

    assert provenance["evidence"], "Evidence provenance is missing"
    assert provenance["models"], "Model provenance is missing"

    print("KNOWLEDGE GRAPH TEST PASSED")