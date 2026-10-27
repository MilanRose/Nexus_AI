from app.knowledge_graph.graph import KnowledgeGraph
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.reasoning import GraphReasoner
from app.knowledge_graph.evolution import SelfEvolutionEngine
from app.verification.fact_checker import FactChecker

from app.schemas import AgentResult, Evidence, ModelResult

def test_member3_integration():
    # 1. Create the knowledge graph and components
    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(kg)
    fact_checker = FactChecker(kg)
    evolution = SelfEvolutionEngine(kg)
    reasoner = GraphReasoner(kg)

    # 2. Create an agent result
    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=[
            "MonthlyCharges is associated with churn"
        ],
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

    # 3. Build the knowledge graph
    finding_ids = builder.add_agent_result(result)

    assert finding_ids, "No finding IDs were created"

    finding_id = finding_ids[0]

    assert finding_id in kg.graph.nodes
    assert (
        kg.get_node(finding_id).get("type") == "finding"
    ), "The created node is not a finding"

    # 4. Verify the finding
    verification = fact_checker.verify_finding(finding_id)

    assert verification["verified"] is False
    assert verification["structural_verified"] is True
    assert verification["checks"]["finding_exists"] is True
    assert verification["checks"]["is_finding"] is True
    assert verification["checks"]["has_evidence"] is True
    assert verification["checks"]["evidence_connected"] is True
    assert verification["checks"]["has_model"] is True
    assert verification["checks"]["valid_confidence"] is True
    assert verification["checks"]["valid_model_accuracy"] is True

    # 5. Check finding context and provenance
    context = reasoner.get_finding_context(finding_id)
    provenance = reasoner.get_provenance_chain(finding_id)

    assert context["features"], "Finding has no feature context"
    assert context["datasets"], "Finding has no dataset context"
    assert provenance["evidence"], "Finding has no evidence provenance"
    assert provenance["models"], "Finding has no model provenance"

    # 6. Evaluate the execution
    evaluation = evolution.evaluate_execution(
        result,
        verification
    )

    assert evaluation["successful"] is True
    assert evaluation["verification_passed"] is True
    assert evaluation["confidence"] == 0.84
    assert evaluation["model_performance"] == 0.87

    # 7. Store execution history
    execution_id = evolution.update_knowledge(
        result,
        evaluation,
        finding_ids
    )

    assert execution_id in kg.graph.nodes

    execution_node = kg.get_node(execution_id)

    assert execution_node.get("type") == "execution"
    assert execution_node.get("agent") == "EDAAgent"
    assert execution_node.get("task") == "correlation_analysis"
    assert execution_node.get("dataset") == "customer_churn.csv"
    assert execution_node.get("successful") is True

    # Verify the execution -> finding relationship
    produced_edge_exists = any(
        target == finding_id
        and data.get("relation") == "produced"
        for _, target, data in kg.graph.out_edges(
            execution_id,
            data=True
        )
    )

    assert produced_edge_exists, (
        "Execution is not linked to its finding "
        "by a produced relationship"
    )

    # 8. Check successful execution history
    successful = evolution.get_successful_executions()

    assert any(
        item["execution_id"] == execution_id
        for item in successful
    ), "Stored execution was not found in successful history"

    # 9. Check the workflow recommendation
    recommendation = evolution.recommend_workflow(
        "correlation_analysis",
        dataset="customer_churn.csv"
    )

    assert recommendation, "No workflow recommendation was returned"

    assert recommendation["recommended_agent"] == "EDAAgent", (
        "Expected EDAAgent to be recommended"
    )

    assert (
        evolution.get_recommended_agent(
            "correlation_analysis",
            dataset="customer_churn.csv"
        )
        == "EDAAgent"
    ), "The recommended agent does not match"

    # 10. Check related-node reasoning
    related_nodes = reasoner.get_related_nodes(finding_id)

    assert related_nodes, "No related nodes were returned"

    print("Member 3 integration test passed.")


def test_unverified_result_is_not_successful():
    kg = KnowledgeGraph()
    evolution = SelfEvolutionEngine(kg)

    result = AgentResult(
        agent="EDAAgent",
        dataset="customer_churn.csv",
        task="correlation_analysis",
        findings=["A test finding"],
        confidence=0.95
    )

    verification = {
        "verified": False,
        "checks": {"has_evidence": False}
    }

    evaluation = evolution.evaluate_execution(result, verification)

    assert evaluation["verification_passed"] is False
    assert evaluation["successful"] is False
