from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.reasoning import GraphReasoner
from app.schemas import AgentResult, Evidence, ModelResult


def test_builder_reasoning():
    # 1. Initialize the graph and components
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    reasoner = GraphReasoner(kg)

    # 2. Create sample agent output
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["MonthlyCharges is associated with churn"],
        features=["MonthlyCharges"],
        evidence=[
            Evidence(
                description="Pearson correlation = 0.42",
                method="Pearson correlation",
                value=0.42,
                finding_indices=[0]
            )
        ],
        model=ModelResult(
            name="Random Forest",
            metrics={"accuracy": 0.87}
        ),
        confidence=0.84
    )

    # 3. Build the graph
    finding_ids = builder.add_agent_result(result)

    assert finding_ids, "No finding was created"
    finding_id = finding_ids[0]

    # 4. Verify feature and dataset context
    context = reasoner.get_finding_context(finding_id)
    provenance = reasoner.get_provenance_chain(finding_id)
    evidence = reasoner.get_supporting_evidence(finding_id)

    assert context["features"], "No supporting features found"
    assert context["datasets"], "No dataset context found"

    feature_names = [
        kg.get_node(feature_id).get("name")
        for feature_id in context["features"]
    ]

    dataset_names = [
        kg.get_node(dataset_id).get("name")
        for dataset_id in context["datasets"]
    ]

    assert "MonthlyCharges" in feature_names
    assert "customer_churn.csv" in dataset_names

    provenance_dataset_ids = {
        item["id"] for item in provenance["datasets"]
    }
    assert context["datasets"][0] in provenance_dataset_ids

    # 5. Verify evidence and model provenance
    assert evidence, "No supporting evidence found"
    assert provenance["evidence"], "No evidence provenance found"
    assert provenance["models"], "No model provenance found"

    connected_evidence_ids = {
        target
        for _, target, edge_data in kg.graph.out_edges(
            finding_id, data=True
        )
        if edge_data.get("relation") == "supported_by"
    }

    assert any(
        item.get("evidence_id") in connected_evidence_ids
        for item in evidence
    ), "Returned evidence is not connected to the finding"

    # 6. Regression test: direct dataset provenance without features
    direct_kg = KnowledgeGraph()
    direct_reasoner = GraphReasoner(direct_kg)

    direct_kg.add_node(
        "dataset_direct",
        "dataset",
        name="direct_dataset.csv"
    )
    direct_kg.add_node(
        "finding_direct",
        "finding",
        description="Finding without explicit features",
        confidence=0.75
    )
    direct_kg.add_relationship(
        "dataset_direct",
        "contains_finding",
        "finding_direct"
    )

    direct_context = direct_reasoner.get_finding_context(
        "finding_direct"
    )
    direct_provenance = direct_reasoner.get_provenance_chain(
        "finding_direct"
    )

    assert direct_context["features"] == []
    assert direct_context["datasets"] == ["dataset_direct"]

    assert any(
        item["id"] == "dataset_direct"
        for item in direct_provenance["datasets"]
    ), "Direct dataset provenance is missing"

    # 7. Regression test: incoming and outgoing related relationships
    direct_kg.add_node("feature_direct", "feature", name="tenure")
    direct_kg.add_relationship(
        "feature_direct",
        "supports",
        "finding_direct"
    )
    direct_kg.add_relationship(
        "finding_direct",
        "supported_by",
        "evidence_direct"
    )
    direct_kg.add_node(
        "evidence_direct",
        "evidence",
        description="Supporting evidence"
    )

    related = direct_reasoner.get_related_nodes("finding_direct")

    assert any(
        item["source"] == "feature_direct"
        and item["relation"] == "supports"
        and item["target"] == "finding_direct"
        for item in related
    ), "Incoming relationship was not returned"

    assert any(
        item["source"] == "finding_direct"
        and item["relation"] == "supported_by"
        and item["target"] == "evidence_direct"
        for item in related
    ), "Outgoing relationship was not returned"

    print("BUILDER-REASONER AND PROVENANCE TEST PASSED")